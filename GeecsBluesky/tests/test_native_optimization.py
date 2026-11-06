"""Native optimization on a real mock-backed RunEngine, with no hardware."""

import math
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
            variables={
                "Motor": [row("Current", settable=True, tolerance=0.01)],
                "Gauge": [row("Pressure")],  # not in the run: background telemetry
            },
            types={},
            subscribed={"Motor": ["Current"], "Gauge": ["Pressure"]},
        ),
        file_plugin_hosts=set(),
    )
    motor = ns.resolve("Motor:Current")
    connect_mock(re, ns.resolve("Motor"), ns.resolve("Gauge"))
    set_mock_value(ns.resolve("Gauge").pressure, 2.5)
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
    # The background rides in every strict row of an optimize run too (#1016):
    # the gauge is not in the run, the motor is (its owner is left to the row).
    start = docs.docs["start"][0]
    assert start["background_telemetry"] is True and start["background_dropped"] == []
    # The shot-table layout, like a strict count.
    assert start["layout"] == "shot-table/1" and "non_essential" not in start
    # The motor has no shot stamp: a condition, read in every row.
    assert start["per_shot_streams"] == [] and start["required"] == []
    assert start["shot_clock_column"] == "fire_stamp" and "stream_meta" in start
    rows = docs.primary_events()
    assert len(rows) == 6 and all(e["data"]["gauge-pressure"] == 2.5 for e in rows)
    import json

    assert (
        json.loads(docs.docs["start"][0]["geecs"]["optimizer_json"])["schema_version"]
        == 1
    )


def test_missing_device_refused_before_claim(setup):
    re, ns, box, plan, cfg, docs, folder = setup
    with pytest.raises(GeecsConfigurationError, match="requires the device"):
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


def _assert_camera_bins(docs, box, asked, fired, plugin):
    """Two bins of two shots, one refire inside bin 1, frames matched by stamp."""
    import numpy as np

    assert box.fires == 5  # Four successes plus one partial refire row.
    # A refire inside bin 1: three rows, the dropped one incomplete; the
    # objective asked for the two complete shots' camera stamps.
    rows = [e["data"] for e in docs.primary_events()]
    assert [(r["bin_number"], r["complete"]) for r in rows] == [
        (1, True),
        (1, False),
        (1, True),
        (2, True),
        (2, True),
    ]
    assert [len(a) for a in asked] == [2, 2]
    assert all(s in fired for a in asked for s in a)
    names = {d["name"] for d in docs.docs["descriptor"]}
    assert ("camera_stamps_stream" in names) is plugin
    assert docs.docs["start"][0]["per_shot_streams"] == ["camera_stream"]
    streams = {d["uid"] for d in docs.docs["descriptor"] if d["name"] == "optimization"}
    events = [e["data"] for e in docs.docs["event"] if e["descriptor"] in streams]
    assert [e["n_valid_shots:cam"] for e in events] == [2, 2]
    assert all(np.isfinite(e["output:cam~2Eimage_total"]) for e in events)


@pytest.mark.parametrize(
    ("connects", "plugin"), [(True, False), (True, True), (False, False)]
)
def test_camera_measurement_refire_and_preclaim_connection(
    setup, monkeypatch, connects, plugin
):
    """A camera objective: its live frames matched by the shots' camera stamps.

    A saver camera's stamp is its own stream's; a plugin camera's stream is
    datum-only, so its stamp comes from the companion stream.
    """
    import numpy as np
    import yaml
    from geecs_core.db.variable_types import LABVIEW_EPOCH_OFFSET
    from geecs_bluesky.optimization import measurements
    from geecs_bluesky.plans.optimize import optimize_plan
    from geecs_bluesky.exceptions import GeecsDeviceDownError
    from tests.test_strict_plans import _camera, _plugin_camera
    from ophyd_async.core import StaticFilenameProvider, StaticPathProvider

    re, ns, box, _, cfg, docs, folder = setup
    # Real-time stamps, so the in-plan join places each record by fire.
    box.clock = True
    fired = []
    advance = box.advance

    def stamped(step=1.0):
        advance(step)
        if (camera.name, box.fires) not in box.drop:
            fired.append(box.stamp - LABVIEW_EPOCH_OFFSET)

    box.advance = stamped
    provider = StaticPathProvider(StaticFilenameProvider("f"), folder / "Camera")
    # Leave scheduler headroom beyond FakeBox's 20 ms fire latency; the dropped
    # frame still exercises timeout/refire deterministically.
    camera = (
        _plugin_camera(re, box, "Camera", folder, shot_timeout=0.5)[0]
        if plugin
        else _camera(re, box, "Camera", provider=provider, shot_timeout=0.5)
    )
    box.drop.add((camera.name, 2))
    sc = ShotControl(WRITES, experiment="Test", name="camera_box", setter_factory=box)
    connect_mock(re, sc)
    profiles = TriggerProfiles({"test": sc}, default="test")
    path = folder / "analyzers" / "Test"
    path.mkdir(parents=True)
    (path / "Camera.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "device": "Camera",
                "input": {"kind": "camera"},
                "measure": {"kind": "beam"},
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
    from geecs_bluesky.config_resolver import ConfigsRepoResolver

    monkeypatch.setattr(
        ConfigsRepoResolver, "analysis_config_dir", property(lambda _: folder)
    )
    diagnostic_resolver = ConfigsRepoResolver("Test", folder)
    resolver = SimpleNamespace(
        resolve_optimizer_config=lambda name: cfg,
        scan_variable_catalog=lambda: SimpleNamespace(variables={}),
        optimizer_config_path=lambda name: folder / "test.yaml",
        resolve_diagnostic=diagnostic_resolver.resolve_diagnostic,
    )
    namespace = SimpleNamespace(
        resolve=lambda name: camera if name == "Camera" else ns.resolve(name),
        experiment="Test",
        # A FROG-typed camera: the live frame source must follow the
        # devicetype's declared stream (frogTrace), not the one-image guess
        # (SpatialImage, alphabetically first, never pushed).
        roster=SimpleNamespace(
            variables={
                "Camera": [
                    row("SpatialImage", choices="image"),
                    row("frogTrace", choices="image"),
                ]
            },
            types={"Camera": "FROG"},
        ),
    )
    opened, asked, closed = [], [], []
    sources = []

    class FakeFrameSource:
        def __init__(self, *args, **kwargs):
            sources.append(args[0])

        def open(self):
            opened.append(True)

        def close(self):
            closed.append(True)

        def wait_connected(self, timeout):
            if not connects:
                raise GeecsDeviceDownError("Camera")

        def await_frames(self, stamps, timeout):
            # Matched by stamp: only the shots the camera took have a frame.
            asked.append(list(stamps))
            y, x = np.mgrid[:32, :32]
            frame = 100 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 20)
            return {s: frame for s in stamps if any(abs(s - f) < 1e-6 for f in fired)}

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
        re(plan([camera], optimizer_config="test", shot_period=0.15))
        _assert_camera_bins(docs, box, asked, fired, plugin)
    assert opened == closed == [True]
    assert sources == ["test:camera:frogtrace"]


@pytest.mark.parametrize(
    "outputs", [{"x" * 61: "current"}, {"score": "current", "Score": "current"}]
)
def test_unarchivable_columns_refused_before_claim(setup, outputs):
    re, ns, box, plan, cfg, docs, folder = setup
    cfg.derived = {"score": "current", **outputs}
    with pytest.raises(GeecsConfigurationError, match="column names"):
        re(plan([ns.resolve("Motor")], optimizer_config="test"))
    assert box.fires == 0
    assert not docs.docs["start"]


def _meter_plan(setup, *, plugin=False, late=None):
    """A meter reading the fire's number each shot, as optimize's one objective.

    *plugin*: the meter is a plugin-backed camera (its stream datum-only).
    *late*: ``{fire: seconds}`` — that fire's gateway stamp (and scalar)
    lands so long after its frame was counted.
    """
    import asyncio

    from geecs_bluesky.plans.optimize import optimize_plan
    from tests.test_strict_plans import _camera, _plugin_camera

    re, ns, box, _, _, docs, folder = setup
    box.clock = True
    meter = (
        _plugin_camera(re, box, "Meter", folder)[0]
        if plugin
        else _camera(re, box, "Meter")
    )
    advance, late = box.advance, dict(late or {})

    def advance_and_read(step=1.0):
        advance(step)
        fire, stamp = box.fires, box.stamp
        if fire not in late:
            set_mock_value(meter.meancounts, float(fire))
            return
        box.drop.add((meter.name, fire))  # the box counts the frame, not the stamp
        box.counts[meter.name] = box.counts.get(meter.name, 0) + 1
        set_mock_value(meter.hdf.num_captured, box.counts[meter.name])

        def land():
            set_mock_value(meter.meancounts, float(fire))
            set_mock_value(meter.acq_timestamp, stamp)

        asyncio.get_running_loop().call_later(late[fire], land)

    box.advance = advance_and_read
    sc = ShotControl(WRITES, experiment="Test", name="meter_box", setter_factory=box)
    connect_mock(re, sc)
    cfg = OptimizerConfig(
        vocs={
            "variables": {"Motor:Current": [-1, 1]},
            "objectives": {"meter": "MINIMIZE"},
        },
        measurements={"meter": {"signal": "Meter:MeanCounts"}},
        generator={"name": "random"},
        run={"max_iterations": 3, "shots_per_step": 2},
    )
    resolver = SimpleNamespace(
        resolve_optimizer_config=lambda name: cfg,
        scan_variable_catalog=lambda: SimpleNamespace(variables={}),
        optimizer_config_path=lambda name: folder / "optimizer.yaml",
    )
    names = {"Meter": meter, "Meter:MeanCounts": meter.meancounts}
    namespace = SimpleNamespace(
        resolve=lambda n: names.get(n) or ns.resolve(n), experiment="Test"
    )
    plan = optimize_plan(
        TriggerProfiles({"test": sc}, default="test"), resolver, namespace
    )
    return re, plan, meter, box, docs


def _optimization_events(docs):
    stream = {d["uid"] for d in docs.docs["descriptor"] if d["name"] == "optimization"}
    return [e["data"] for e in docs.docs["event"] if e["descriptor"] in stream]


@pytest.mark.parametrize("plugin", [False, True], ids=["stamped", "plugin"])
def test_a_scalar_objective_comes_from_the_in_plan_join(setup, plugin):
    """A per-shot device's scalar is in its own stream; each bin's value is the join's.

    The meter reads the fire's number on every shot, so a bin's mean is
    known by construction — and ``primary`` carries no meter column at all.
    A plugin camera's stream is datum-only until stop: its companion
    carries the scalar into the join.
    """
    re, plan, meter, box, docs = _meter_plan(setup, plugin=plugin)
    re(plan([meter], optimizer_config="test", shot_period=0.15))
    start = docs.docs["start"][-1]
    assert start["per_shot_streams"] == ["meter_stream"]
    assert start["required"] == ["meter"]
    assert all("meter-meancounts" not in e["data"] for e in docs.primary_events())
    events = _optimization_events(docs)
    assert [e["output:meter"] for e in events] == [1.5, 3.5, 5.5]
    assert [e["n_valid_shots:meter"] for e in events] == [2, 2, 2]


def test_a_stamp_landing_after_the_bins_last_collect_still_counts(setup, caplog):
    """The bin waits for the companion's stamp per frame counted; no expiry warning."""
    re, plan, meter, box, docs = _meter_plan(setup, plugin=True, late={2: 0.4})
    re(plan([meter], optimizer_config="test", shot_period=0.15))
    events = _optimization_events(docs)
    assert [e["output:meter"] for e in events] == [1.5, 3.5, 5.5]
    assert "without a gateway stamp" not in caplog.text


def test_a_stamp_that_never_lands_is_warned_once_and_not_waited_again(
    setup, caplog, monkeypatch
):
    """A frame whose stamp never comes: one WARNING, the shortfall carried."""
    from geecs_bluesky.plans import optimize

    monkeypatch.setattr(optimize, "SETTLE_S", 2.0)
    re, plan, meter, box, docs = _meter_plan(setup, plugin=True, late={2: 60.0})
    re(plan([meter], optimizer_config="test", shot_period=0.15))
    assert caplog.text.count("1 frame(s) without a gateway stamp") == 1
    events = [
        e
        for e in docs.docs["event"]
        if e["descriptor"]
        in {d["uid"] for d in docs.docs["descriptor"] if d["name"] == "optimization"}
    ]
    assert [e["data"]["n_valid_shots:meter"] for e in events] == [1, 2, 2]
    # One 2 s expiry (bin 1); bins 2 and 3 wait for nothing (~0.5 s each).
    assert events[-1]["time"] - events[0]["time"] < 2.0


def test_an_objective_device_that_is_not_required_is_refused_before_the_claim(
    setup,
):
    re, plan, meter, box, docs = _meter_plan(setup)
    with pytest.raises(GeecsConfigurationError, match="Meter required"):
        re(plan([meter], optimizer_config="test", required=[]))
    assert box.fires == 0 and not docs.docs["start"]


def test_a_bin_without_a_readback_records_nan_and_is_not_told(setup, caplog):
    """Never the proposal as measured: NaN, a WARNING, no tell."""
    import yaml

    re, ns, box, plan, cfg, docs, folder = setup
    motor = ns.resolve("Motor:Current")
    advance = box.advance

    def lose_readback(step=1.0):
        advance(step)
        set_mock_value(motor.position, float("nan"))

    box.advance = lose_readback
    re(plan([ns.resolve("Motor")], optimizer_config="test"))
    events = _optimization_events(docs)
    assert len(events) == 3
    assert all(math.isnan(e["measured:Motor:Current"]) for e in events)
    assert "no readback of Motor:Current" in caplog.text
    # Nothing told: the dump holds no evaluated point.
    assert not yaml.safe_load((folder / "xopt_dump.yaml").read_text())["data"]


def test_a_per_shot_movable_readback_is_read_in_the_row(setup):
    """A movable on a per-shot device: its readback is a row column, as before."""
    from tests.test_namespace import row as db_row

    re, _, box, _, cfg, docs, folder = setup
    ns = GeecsNamespace(
        DeviceRoster(
            experiment="Test",
            variables={"Stage": [db_row("Position", settable=True, tolerance=0.01)]},
            types={},
            subscribed={"Stage": ["Position"]},
            triggered={"Stage": True},
        ),
        file_plugin_hosts=set(),
    )
    stage = ns.resolve("Stage")
    connect_mock(re, stage)
    follow_setpoint(ns.resolve("Stage:Position"))
    box.cameras.append(stage)
    cfg = OptimizerConfig(
        vocs={
            "variables": {"Stage:Position": [-1, 1]},
            "objectives": {"current": "MINIMIZE"},
        },
        measurements={"current": {"signal": "Stage:Position"}},
        generator={"name": "random"},
        run={"max_iterations": 2, "shots_per_step": 2},
    )
    sc = ShotControl(WRITES, experiment="Test", name="stage_box", setter_factory=box)
    connect_mock(re, sc)
    resolver = SimpleNamespace(
        resolve_optimizer_config=lambda name: cfg,
        scan_variable_catalog=lambda: SimpleNamespace(variables={}),
        optimizer_config_path=lambda name: folder / "optimizer.yaml",
    )
    plan = bind_plans(
        TriggerProfiles({"test": sc}, default="test"), resolver=resolver, settables=ns
    )["optimize"]
    re(plan([stage], optimizer_config="test"))
    key = ns.resolve("Stage:Position").reading_key
    assert docs.docs["start"][-1]["per_shot_streams"] == ["stage_stream"]
    assert all(key in e["data"] for e in docs.primary_events())


def test_a_stamp_whose_read_failed_settles_rather_than_wedges(setup):
    """A stamp recorded as failed counts as settled: the bin never waits for it."""
    import asyncio

    from geecs_bluesky.devices.sampler import StampStream
    from geecs_bluesky.plans.optimize import _Settle
    from tests.test_strict_plans import _camera

    re, _, box, *_ = setup
    cam = _camera(re, box, "Cam")
    stream = StampStream(cam, name="cam_stamps")

    async def failing_read():
        raise RuntimeError("gateway read refused")

    async def scenario():
        await stream.kickoff()
        stream._readers = [(None, failing_read)]
        set_mock_value(cam.acq_timestamp, box.stamp + 5.0)
        for _ in range(100):
            if not stream.pending and stream.failed:
                break
            await asyncio.sleep(0.01)
        await stream.complete()

    asyncio.run_coroutine_threadsafe(scenario(), re.loop).result(5)
    assert (stream.published, stream.failed, stream.pending) == (0, 1, 0)
    settle = _Settle(SimpleNamespace(companions=[stream]))
    assert settle._missing(stream, (1.0,)) == 0
