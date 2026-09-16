"""Native optimization on a real mock-backed RunEngine, with no hardware."""

from types import SimpleNamespace

import pytest
from bluesky import RunEngine
from geecs_schemas import OptimizerConfig
from ophyd_async.core import set_mock_value

from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace
from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans
from geecs_bluesky.plans.claim_scan import claim_scan_preprocessor
from geecs_bluesky.exceptions import GeecsConfigurationError
from tests.ca_mock_helpers import connect_mock, follow_setpoint, DocCollector
from tests.test_strict_plans import WRITES, FakeBox
from tests.test_namespace import row


@pytest.fixture
def setup(tmp_path):
    re = RunEngine()
    ns = GeecsNamespace(
        DeviceRoster(
            experiment="Test",
            variables={"Motor": [row("Current", settable=True, tolerance=0.01)]},
            types={},
            subscribed={"Motor": ["Current"]},
        ),
        file_plugin_hosts=set(),
    )
    motor = ns.resolve("Motor:Current")
    connect_mock(re, ns.resolve("Motor"))
    follow_setpoint(motor)
    box = FakeBox()
    sc = ShotControl(WRITES, experiment="Test", name="box", setter_factory=box)
    connect_mock(re, sc)
    profiles = TriggerProfiles({"test": sc}, default="test")
    cfg = OptimizerConfig(
        vocs={
            "variables": {"Motor:Current": [-1, 1]},
            "objectives": {"score": "MINIMIZE"},
        },
        measurements={"current": {"signal": "Motor:Current"}},
        derived={"score": "current**2"},
        generator={"name": "random"},
        run={"max_iterations": 3, "shots_per_step": 2},
    )
    resolver = SimpleNamespace(
        resolve_optimizer_config=lambda name: cfg,
        scan_variable_catalog=lambda: SimpleNamespace(variables={}),
        optimizer_config_path=lambda name: tmp_path / "optimizer.yaml",
    )
    plan = bind_plans(profiles, resolver=resolver, settables=ns)["optimize"]
    tag = SimpleNamespace(number=1, year=2026, month=1, day=1, experiment="Test")
    re.preprocessors.append(
        lambda p: claim_scan_preprocessor(
            p, experiment="Test", claim=lambda ex: (tag, str(tmp_path))
        )
    )
    docs = DocCollector()
    re.subscribe(docs)
    return re, ns, box, plan, cfg, docs, tmp_path


def test_scalar_optimization_records_and_dumps(setup):
    re, ns, box, plan, cfg, docs, folder = setup
    re(plan([ns.resolve("Motor")], optimizer_config="test"))
    assert box.fires == 6
    assert (folder / "xopt_dump.yaml").exists()
    import json

    assert (
        json.loads(docs.docs["start"][0]["geecs"]["optimizer_json"])["schema_version"]
        == 1
    )


def test_missing_device_refused_before_claim(setup):
    re, ns, box, plan, cfg, docs, folder = setup
    with pytest.raises(GeecsConfigurationError, match="essential device"):
        re(plan([], optimizer_config="test"))
    assert box.fires == 0
    assert not (folder / "xopt_dump.yaml").exists()


@pytest.mark.parametrize("defer", [False, True])
def test_pause_resume_produces_one_observation_per_iteration(setup, defer):
    from bluesky.utils import RunEngineInterrupted

    re, ns, box, plan, cfg, docs, folder = setup
    primary = set()
    requested = False

    def pause(name, doc):
        nonlocal requested
        if name == "descriptor" and doc["name"] == "primary":
            primary.add(doc["uid"])
        if name == "event" and doc["descriptor"] in primary and not requested:
            requested = True
            import threading

            threading.Thread(
                target=re.request_pause, kwargs={"defer": defer}, daemon=True
            ).start()

    token = re.subscribe(pause)
    with pytest.raises(RunEngineInterrupted):
        re(plan([ns.resolve("Motor")], optimizer_config="test", shot_period=0.02))
    re.unsubscribe(token)
    re.resume()
    optimization = {
        d["uid"] for d in docs.docs["descriptor"] if d["name"] == "optimization"
    }
    events = [e for e in docs.docs["event"] if e["descriptor"] in optimization]
    assert [e["data"]["iteration"] for e in events] == [1, 2, 3]
    assert all(e["data"]["n_valid_shots:current"] == 2 for e in events)
    from geecs_bluesky.optimization.inspection.dump_loader import load_xopt_dump

    _, data = load_xopt_dump(folder / "xopt_dump.yaml")
    assert len(data) == 3
    assert re.rewindable


def test_no_feasible_point_restores_start(setup):
    re, ns, box, plan, cfg, docs, folder = setup
    cfg.measurements["current"].min_shots = 3
    motor = ns.resolve("Motor:Current")
    set_mock_value(motor.position, 0.25)
    re(plan([ns.resolve("Motor")], optimizer_config="test"))
    import asyncio

    reading = asyncio.run_coroutine_threadsafe(motor.read(), re.loop).result(5)
    assert reading[motor.reading_key]["value"] == pytest.approx(0.25)


def test_relative_pseudo_restores_and_records_absolute_best(setup):
    import asyncio
    from gest_api.vocs import VOCS
    from geecs_schemas import ScanVariables
    from ophyd_async.core import set_mock_value

    re, ns, box, plan, cfg, docs, folder = setup
    catalog = ScanVariables.model_validate(
        {
            "variables": {
                "bump": {
                    "kind": "pseudo",
                    "mode": "relative",
                    "targets": [
                        {"target": "Motor:Current", "forward": "composite_var"}
                    ],
                }
            }
        }
    ).variables
    ns.add_pseudos(catalog)
    pseudo = ns.resolve("bump")
    connect_mock(re, pseudo)
    motor = ns.resolve("Motor:Current")
    set_mock_value(motor.position, 5.0)
    cfg.vocs = VOCS(variables={"bump": [-1, 1]}, objectives={"score": "MINIMIZE"})
    re(plan([ns.resolve("Motor")], optimizer_config="test"))
    reading = asyncio.run_coroutine_threadsafe(motor.read(), re.loop).result(5)
    assert reading[motor.reading_key]["value"] == pytest.approx(5.0)
    stream = {d["uid"] for d in docs.docs["descriptor"] if d["name"] == "optimization"}
    last = [e["data"] for e in docs.docs["event"] if e["descriptor"] in stream][-1]
    assert last["best_move:Motor:Current"] == pytest.approx(5.0 + last["best:bump"])


@pytest.mark.parametrize("connects", [True, False])
def test_camera_measurement_refire_and_preclaim_connection(
    setup, monkeypatch, connects
):
    import numpy as np
    import yaml
    from geecs_core.db.variable_types import LABVIEW_EPOCH_OFFSET
    from geecs_bluesky.optimization import measurements
    from geecs_bluesky.plans.optimize import optimize_plan
    from geecs_bluesky.exceptions import GeecsDeviceDownError
    from tests.test_strict_plans import _camera
    from ophyd_async.core import StaticFilenameProvider, StaticPathProvider

    re, ns, box, _, cfg, docs, folder = setup
    box.stamp += LABVIEW_EPOCH_OFFSET
    provider = StaticPathProvider(StaticFilenameProvider("f"), folder / "Camera")
    # Leave scheduler headroom beyond FakeBox's 20 ms fire latency; the dropped
    # frame still exercises timeout/refire deterministically.
    camera = _camera(re, box, "Camera", provider=provider, shot_timeout=0.5)
    box.drop.add((camera.name, 2))
    sc = ShotControl(WRITES, experiment="Test", name="camera_box", setter_factory=box)
    connect_mock(re, sc)
    profiles = TriggerProfiles({"test": sc}, default="test")
    path = folder / "analyzers" / "Test"
    path.mkdir(parents=True)
    (path / "Camera.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "name": "Camera",
                "image": {"type": "camera"},
                "analyzer": {"kind": "beam"},
            }
        )
    )
    cfg = OptimizerConfig(
        vocs={
            "variables": {"Motor:Current": [-1, 1]},
            "objectives": {"cam.image_total": "MAXIMIZE"},
        },
        measurements={"cam": {"diagnostic": "Camera", "min_shots": 2}},
        generator={"name": "random"},
        run={"max_iterations": 2, "shots_per_step": 2},
    )
    resolver = SimpleNamespace(
        resolve_optimizer_config=lambda name: cfg,
        scan_variable_catalog=lambda: SimpleNamespace(variables={}),
        optimizer_config_path=lambda name: folder / "test.yaml",
        analysis_config_dir=folder,
    )
    namespace = SimpleNamespace(
        resolve=lambda name: camera if name == "Camera" else ns.resolve(name),
        experiment="Test",
        roster=SimpleNamespace(variables={"Camera": [row("image", choices="image")]}),
    )
    opened = []
    closed = []

    class FakeFrameSource:
        def __init__(self, *args, **kwargs):
            pass

        def open(self):
            opened.append(True)

        def close(self):
            closed.append(True)

        def wait_connected(self, timeout):
            if not connects:
                raise GeecsDeviceDownError("Camera")

        def await_frames(self, stamps, timeout):
            assert all(1000 < stamp < 1100 for stamp in stamps)
            y, x = np.mgrid[:32, :32]
            frame = 100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 20)
            return {stamp: frame for stamp in stamps}

    compile_original = measurements.compile_measurements
    monkeypatch.setattr(
        measurements,
        "compile_measurements",
        lambda config, **kw: compile_original(
            config, source_factory=FakeFrameSource, **kw
        ),
    )
    plan = optimize_plan(profiles, resolver, namespace)
    if not connects:
        with pytest.raises(GeecsDeviceDownError):
            re(plan([camera], optimizer_config="test"))
        assert not docs.docs["start"]
        assert box.fires == 0
    else:
        re(plan([camera], optimizer_config="test"))
        assert box.fires == 5  # Four successes plus one partial refire row.
        streams = {
            d["uid"] for d in docs.docs["descriptor"] if d["name"] == "optimization"
        }
        events = [e["data"] for e in docs.docs["event"] if e["descriptor"] in streams]
        assert [e["n_valid_shots:cam"] for e in events] == [2, 2]
        assert all(np.isfinite(e["output:cam%2Eimage_total"]) for e in events)
    assert opened == closed == [True]
