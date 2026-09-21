"""The registration table: stock names, strict binding, queue-item validation (PR 2, #807)."""

from __future__ import annotations

import inspect
import logging

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
import bluesky.plans as bp  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from bluesky.utils import FailedStatus, is_plan  # noqa: E402
from ophyd_async.core import Device, callback_on_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaMotor  # noqa: E402
from geecs_bluesky.devices.shot_control import ShotControl  # noqa: E402
from geecs_bluesky.exceptions import (  # noqa: E402
    GeecsConfigurationError,
    GeecsDeviceDownError,
    failure_cause_text,
)
from geecs_bluesky.plan_names import (
    GEECS_PLAN_NAMES,
    NON_SCAN_PLAN_NAMES,
    NATIVE_SCAN_PLAN_NAMES,
)  # noqa: E402
from geecs_bluesky.plans.registry import (  # noqa: E402
    TriggerProfiles,
    bind_plans,
    strict_plan,
)
from tests.ca_mock_helpers import DocCollector, connect_mock, follow_setpoint  # noqa: E402
from tests.test_strict_plans import (  # noqa: E402
    WRITES,
    FakeBox,
    _RefusedPut,
    _camera,
    _plugin_camera,
)


class Magnet(Device):
    def __init__(self, name: str = "u_s1h") -> None:
        self.current = CaMotor("U_S1H", "Current", experiment="TestExp", tolerance=0.01)
        super().__init__(name=name)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> FakeBox:
    return FakeBox()


@pytest.fixture
def profiles(RE: RunEngine, box: FakeBox) -> TriggerProfiles:
    sc = ShotControl(WRITES, experiment="TestExp", name="htu_test", setter_factory=box)
    connect_mock(RE, sc)
    return TriggerProfiles({"HTU-Test": sc}, default="HTU-Test")


# ---------------------------------------------------------------- the table
def test_plan_names_are_the_three_scan_choices_and_utilities() -> None:
    assert set(GEECS_PLAN_NAMES) == {"count", "sweep", "optimize", *NON_SCAN_PLAN_NAMES}


def payload(axis="U_S1H.current", start=-1.0, stop=1.0, num=3, **extra):
    return {
        "trajectory": {
            "kind": "axes",
            "axes": [
                dict(kind="range", axis=axis, start=start, stop=stop, num=num, **extra)
            ],
        }
    }


def test_bound_plans_keep_the_stock_signature_minus_the_hook(profiles) -> None:
    bound = bind_plans(profiles)
    assert set(bound) == set(GEECS_PLAN_NAMES)
    assert bound["mv"].__wrapped__ is bps.mv  # the stock stub, its failure named
    assert inspect.signature(bound["mv"]) == inspect.signature(bps.mv)
    assert bound["mv"].__name__ == "mv"
    assert list(inspect.signature(bound["run_action"]).parameters) == ["name"]
    for name in GEECS_PLAN_NAMES:
        if name in (*NON_SCAN_PLAN_NAMES, *NATIVE_SCAN_PLAN_NAMES):
            continue
        plan = bound[name]
        assert is_plan(plan) and inspect.isgeneratorfunction(plan)
        assert plan.__name__ == name
        stock = inspect.signature(getattr(bp, name)).parameters
        params = inspect.signature(plan).parameters
        assert "per_step" not in params and "per_shot" not in params
        kept = [p for p in stock if p not in ("per_step", "per_shot")]
        assert list(params)[: len(kept)] == kept
        assert params["trigger_profile"].kind is inspect.Parameter.KEYWORD_ONLY
        assert ("shots_per_step" in params) == (name != "count")
        assert "trigger_profile" in plan.__doc__
        # phase 2: the acquisition mode, the non-essential list, the throttle;
        # #738: the LabVIEW-files switch
        for extra in (
            "acquisition",
            "non_essential",
            "shot_period",
            "native_image_save",
        ):
            assert params[extra].kind is inspect.Parameter.KEYWORD_ONLY
            assert extra in plan.__doc__
        assert params["acquisition"].default == "strict"
        assert params["non_essential"].default is None
        assert params["shot_period"].default is None
        assert params["native_image_save"].default is None


def test_strict_plan_refuses_a_plan_without_the_hook(profiles) -> None:
    with pytest.raises(ValueError, match="exactly one strict hook"):
        strict_plan(bp.tune_centroid, profiles)


# --------------------------------------------------------- manager contract
def test_queue_items_validate_against_the_bound_plans(RE, box, profiles) -> None:
    """The manager's own _process_plan/validate_plan accept a strict queue item."""
    pytest.importorskip("bluesky_queueserver")
    from bluesky_queueserver.manager.profile_ops import (
        _process_plan,
        existing_plans_and_devices_from_nspace,
        validate_plan,
    )

    cam = _camera(RE, box, "UC_Cam")
    ns = {"UC_Cam": cam, "U_S1H": Magnet(), **bind_plans(profiles)}
    plans, devices, plans_in_nspace, devices_in_nspace = (
        existing_plans_and_devices_from_nspace(nspace=ns)
    )
    assert set(plans) == set(GEECS_PLAN_NAMES)
    assert "scalars" in devices["UC_Cam"]["components"]
    assert (
        "acq_timestamp" in devices["UC_Cam"]["components"]
    )  # the preflight's trigger rule
    gated_item = {
        "name": "count",
        "args": [["UC_Cam"]],
        "kwargs": {"num": 3, "acquisition": "gated", "non_essential": ["UC_Cam"]},
        "item_type": "plan",
        "user_group": "admin",
    }
    ok, message = validate_plan(
        gated_item, allowed_plans=plans, allowed_devices=devices
    )
    assert ok, message
    from bluesky_queueserver.manager.profile_ops import prepare_plan

    # the manager resolves the non_essential names like the detectors'
    resolved = prepare_plan(
        gated_item,
        plans_in_nspace=plans_in_nspace,
        devices_in_nspace=devices_in_nspace,
        allowed_plans={"admin": plans},
        allowed_devices={"admin": devices},
        nspace=ns,
    )
    assert resolved["kwargs"]["non_essential"] == [cam]
    assert resolved["kwargs"]["acquisition"] == "gated"
    items = [
        ("count", [["UC_Cam"], 3], {"trigger_profile": "HTU-Test"}),
        ("count", [["UC_Cam.scalars"]], {"num": 2}),
        ("sweep", [["UC_Cam"]], {"sweep": payload(), "shots_per_step": 4}),
        ("run_action", ["Amp4_DUMP_HP"], {}),
    ]
    for name, args, kwargs in items:
        processed = _process_plan(ns[name], existing_devices={}, existing_plans={})
        ok, msg = validate_plan(
            {"name": name, "args": args, "kwargs": kwargs, "item_type": "plan"},
            allowed_plans={name: processed},
            allowed_devices=devices,
        )
        assert ok, (name, msg)
    processed = _process_plan(ns["count"], existing_devices={}, existing_plans={})
    ok, msg = validate_plan(
        {
            "name": "count",
            "args": [["UC_Cam"]],
            "kwargs": {"shots_per_step": 2},
            "item_type": "plan",
        },
        allowed_plans={"count": processed},
        allowed_devices=devices,
    )
    assert not ok and "shots_per_step" in msg


# ----------------------------------------------------------------- running
def test_bound_scan_runs_strict_with_shots_per_step_and_bins(RE, box, profiles):
    cam = _camera(RE, box, "UC_Cam")
    magnet = Magnet()
    connect_mock(RE, magnet)
    follow_setpoint(magnet.current)
    col = DocCollector()
    RE.subscribe(col)
    scan = bind_plans(profiles, settables={"U_S1H": magnet})["sweep"]
    RE(scan([cam], sweep=payload(), shots_per_step=2))
    events = col.primary_events()
    assert box.fires == 6 and len(events) == 6
    assert [e["data"]["bin_number"] for e in events] == [1, 1, 2, 2, 3, 3]
    assert [e["data"]["u_s1h-current-position"] for e in events] == pytest.approx(
        [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]
    )
    start = col.docs["start"][0]
    assert start["plan_name"] == "sweep"
    assert start["trigger_profile"] == "HTU-Test" and start["shots_per_step"] == 2
    assert start["num_points"] == 3
    # ARMED before the run, STANDBY after it — through the profile's device.
    states = [v for (_, var, v) in box.puts if var == "Trigger.Source"]
    assert states == ["single", "edges"]
    sc = profiles.resolve(None)
    assert sc.standing_state == "STANDBY"


def test_bound_count_is_one_bin_and_scalars_view_saves_nothing(
    RE, box, profiles, tmp_path
):
    folder = tmp_path / "Scan001"
    folder.mkdir()
    cam = _camera(RE, box, "UC_Cam", tmp_path=tmp_path)
    set_mock_value(cam.save, "on")  # a stale flag from a crash
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    RE(count([cam.scalars], 3))
    events = col.primary_events()
    assert box.fires == 3 and len(events) == 3
    assert [e["data"]["bin_number"] for e in events] == [1, 1, 1]
    assert [e["data"]["uc_cam-acq_timestamp"] for e in events] == [
        1001.0,
        1002.0,
        1003.0,
    ]
    assert not any("nonscalar_save_path" in k for k in events[0]["data"])
    assert not (folder / "UC_Cam").exists()
    assert col.docs["start"][0]["shots_per_step"] == 1
    assert col.docs["start"][0]["detectors"] == ["uc_cam-scalars"]


class _Defaults:
    """A resolver whose experiment defaults carry one native_image_save value."""

    def __init__(self, native_image_save: bool) -> None:
        self.value = native_image_save

    def resolve_experiment_defaults(self):
        from geecs_schemas import ExperimentDefaults

        return ExperimentDefaults(native_image_save=self.value)


def _dual_write_camera(RE, box, name, tmp_path):
    """A plugin-backed camera that also saves natively (today's dual-write)."""
    from ophyd_async.core import StaticFilenameProvider, StaticPathProvider

    native = StaticPathProvider(
        StaticFilenameProvider("frame"), tmp_path / "Scan001" / name
    )
    cam, _ = _plugin_camera(RE, box, name, tmp_path, path_provider=native)
    return cam


def test_native_image_save_off_reaches_plugin_cameras_only_and_restores(
    RE, box, profiles, tmp_path
):
    """#738: the switch skips the plugin-backed camera's PNGs, leaves the native-only one, restores."""
    (tmp_path / "Scan001").mkdir()
    plugin = _dual_write_camera(RE, box, "UC_Plugin", tmp_path)
    native = _camera(RE, box, "UC_Native", tmp_path=tmp_path)
    assert plugin.native_image_save and native.native_image_save
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    RE(count([plugin, native], 2, native_image_save=False))
    start = col.docs["start"][0]
    assert start["native_image_save"] is False
    data = col.primary_events()[0]["data"]
    assert "uc_native-nonscalar_save_path" in data  # no plugin: its only record
    assert "uc_plugin-nonscalar_save_path" not in data  # the stack is the record
    assert (tmp_path / "Scan001" / "UC_Native").is_dir()
    assert not (tmp_path / "Scan001" / "UC_Plugin").exists()
    assert plugin.native_image_save and native.native_image_save  # restored
    # Unset: the construction default (dual-write) is back for the next run.
    RE(count([plugin], 1))
    assert col.docs["start"][-1]["native_image_save"] is True
    assert "uc_plugin-nonscalar_save_path" in col.primary_events()[-1]["data"]
    assert (tmp_path / "Scan001" / "UC_Plugin").is_dir()


def test_native_image_save_default_comes_from_the_experiment_defaults_per_run(
    RE, box, profiles, tmp_path, caplog
):
    """Unset on the item → ExperimentDefaults.native_image_save, read at every run."""
    (tmp_path / "Scan001").mkdir()
    plugin = _dual_write_camera(RE, box, "UC_Plugin", tmp_path)
    col = DocCollector()
    RE.subscribe(col)
    defaults = _Defaults(False)
    count = bind_plans(profiles, resolver=defaults)["count"]
    RE(count([plugin], 1))
    assert col.docs["start"][-1]["native_image_save"] is False
    assert "uc_plugin-nonscalar_save_path" not in col.primary_events()[-1]["data"]
    defaults.value = True  # the file was edited: no rebind, no reopen
    RE(count([plugin], 1))
    assert col.docs["start"][-1]["native_image_save"] is True
    assert "uc_plugin-nonscalar_save_path" in col.primary_events()[-1]["data"]
    # The item's own value beats the default — recorded as the run's switch
    # even when nothing here can be switched (a view leaves the owner's
    # data logics unprepared), and the journal then names no camera.
    defaults.value = True
    with caplog.at_level(logging.INFO, logger="geecs_bluesky.plans.registry"):
        RE(count([plugin.scalars], 1, native_image_save=False))
    assert col.docs["start"][-1]["native_image_save"] is False
    assert "native saving off" not in caplog.text
    assert plugin.native_image_save  # never touched


def test_native_image_save_default_is_on_when_the_defaults_cannot_be_read(caplog):
    """Fail-open to the dual-write: a resolver that cannot read defaults keeps PNGs on."""
    from geecs_bluesky.plans.registry import resolve_native_image_save

    class Broken:
        def resolve_experiment_defaults(self):
            raise OSError("configs root unreadable")

    class Absent:
        def resolve_experiment_defaults(self):
            return None

    assert resolve_native_image_save(None, None) is True
    assert resolve_native_image_save(None, Absent()) is True
    assert resolve_native_image_save(False, Broken()) is False
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.plans.registry"):
        assert resolve_native_image_save(None, Broken()) is True
    assert "native saving stays on" in caplog.text


def test_scalars_view_of_a_scalar_only_device_reads_the_device(RE, box, profiles):
    from geecs_bluesky.devices.ca import CaSnapshotReadable

    cam = _camera(RE, box, "UC_Cam")
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    set_mock_value(gauge.pressure, 1.5e-6)
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    RE(count([cam, gauge.scalars], 2))
    events = col.primary_events()
    assert box.fires == 2 and len(events) == 2
    assert [e["data"]["u_gauge-pressure"] for e in events] == [1.5e-6, 1.5e-6]
    assert col.docs["start"][0]["detectors"] == ["uc_cam", "u_gauge-scalars"]
    assert (
        col.docs["start"][0]["geecs_scalar_headers"]
        if "geecs_scalar_headers" in col.docs["start"][0]
        else True
    )


def test_unknown_profile_is_refused_before_any_move(RE, box, profiles) -> None:
    cam = _camera(RE, box, "UC_Cam")
    count = bind_plans(profiles)["count"]
    with pytest.raises(GeecsConfigurationError, match="unknown trigger profile"):
        RE(count([cam], 1, trigger_profile="HTU-Nope"))
    assert box.puts == [] and box.fires == 0


def test_no_default_profile_makes_the_argument_mandatory(RE, box) -> None:
    sc = ShotControl(WRITES, experiment="TestExp", name="p", setter_factory=box)
    profiles = TriggerProfiles({"P": sc})
    with pytest.raises(GeecsConfigurationError, match="no default"):
        profiles.resolve(None)
    assert profiles.resolve("P") is sc
    with pytest.raises(GeecsConfigurationError, match="not one of the loaded"):
        TriggerProfiles({"P": sc}, default="Q")


def test_profiles_from_resolver_skip_unloadable_and_mark_namespace() -> None:
    from geecs_schemas import ExperimentDefaults, TriggerProfile

    good = TriggerProfile.model_validate(
        {
            "schema_version": 2,
            "name": "Good",
            "states": {
                "ARMED": [{"device": "DG", "variable": "Trigger.Source", "value": "s"}],
                "SINGLESHOT": [
                    {
                        "device": "DG",
                        "variable": "Trigger.ExecuteSingleShot",
                        "value": "on",
                    }
                ],
            },
        }
    )

    class Resolver:
        def list_trigger_profiles(self):
            return ["Good", "Broken"]

        def resolve_trigger_profile(self, name):
            if name == "Broken":
                raise GeecsConfigurationError("names no device")
            return good

        def resolve_experiment_defaults(self):
            return ExperimentDefaults(trigger_profile="Good")

    profiles = TriggerProfiles.from_resolver(Resolver(), experiment="TestExp")
    assert profiles.names == ["Good"] and profiles.default == "Good"
    sc = profiles.resolve(None)
    assert sc._geecs_namespace_member and sc.name == "good"


def test_scalars_view_yields_to_the_owners_scanned_child(RE, box, profiles):
    """``scan([X.scalars], X.current, …)``: the view already reads the child's readback."""
    from geecs_bluesky.devices.ca import CaMotor, CaSnapshotReadable

    magnet = CaSnapshotReadable(
        "U_S1H", ["Voltage"], experiment="TestExp", name="u_s1h"
    )
    magnet.current = CaMotor("U_S1H", "Current", experiment="TestExp", tolerance=0.01)
    magnet.add_readables(
        [magnet.current]
    )  # the DB subscribes Current: the namespace's rule
    cam = _camera(RE, box, "UC_Cam")
    cam.exposure = CaMotor("UC_Cam", "Exposure", experiment="TestExp", tolerance=0.01)
    cam.add_readables([cam.exposure])
    connect_mock(RE, magnet, cam.exposure)
    follow_setpoint(magnet.current)
    follow_setpoint(cam.exposure)
    col = DocCollector()
    RE.subscribe(col)
    scan = bind_plans(profiles, settables={"U_S1H": magnet})["sweep"]
    RE(scan([magnet.scalars], sweep=payload()))
    RE(
        bind_plans(profiles, settables={"UC_Cam": cam})["sweep"](
            [cam.scalars], sweep=payload("UC_Cam.exposure", 1, 3, 3)
        )
    )
    events = col.primary_events()
    assert len(events) == 6 and box.fires == 6
    assert [e["data"]["u_s1h-current-position"] for e in events[:3]] == pytest.approx(
        [-1.0, 0.0, 1.0]
    )
    assert "u_s1h-voltage" in events[0]["data"]
    assert [e["data"]["uc_cam-exposure-position"] for e in events[3:]] == pytest.approx(
        [1.0, 2.0, 3.0]
    )
    assert "uc_cam-acq_timestamp" in events[3]["data"]
    assert not any("nonscalar_save_path" in k for k in events[3]["data"])
    # a child the view does not read stays in the row
    other = CaMotor("U_S1H", "Other", experiment="TestExp", tolerance=0.01)
    magnet.other = other
    connect_mock(RE, other)
    follow_setpoint(other)
    assert not magnet.scalars.covers(other) and magnet.scalars.covers(magnet.current)
    assert cam.scalars.covers(cam.acq_timestamp) and cam.scalars.covers(cam.meancounts)


# ------------------------------------------------- the liveness gate (#852)
def test_a_dead_box_refuses_the_run_before_any_move(RE, box, profiles) -> None:
    """The profile's device reads Disconnected: refused, nothing driven, nothing opened."""
    cam = _camera(RE, box, "UC_Cam")
    sc = profiles.resolve(None)
    set_mock_value(sc.liveness_signals["DG"], "Disconnected")
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    with pytest.raises(GeecsDeviceDownError, match="DG") as info:
        RE(count([cam], 1))
    assert info.value.device_name == "DG"
    assert box.puts == [] and box.fires == 0  # the box was never driven
    assert col.docs["start"] == []  # no open_run → nothing claimed


def test_every_dead_device_of_the_scan_is_named(RE, box, profiles) -> None:
    """A detector, a scalars view and a non-essential device: all in one refusal."""
    a = _camera(RE, box, "UC_A")
    b = _camera(RE, box, "UC_B")
    c = _camera(RE, box, "UC_C")
    live = _camera(RE, box, "UC_Live")
    for cam in (a, b, c):
        set_mock_value(cam.connected_status, "Disconnected")
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    with pytest.raises(GeecsDeviceDownError) as info:
        RE(count([a, b.scalars, live], 1, non_essential=[c]))
    message = str(info.value)
    assert "UC_A, UC_B, UC_C" in message and "UC_Live" not in message
    assert "nothing claimed" in message
    assert box.puts == [] and col.docs["start"] == []


def test_a_dead_scalar_only_device_is_named_through_its_scalars_view(
    RE, box, profiles
) -> None:
    """Found on hardware: ``U_VS1H.scalars`` (a CaSnapshotReadable's view) must be judged."""
    from geecs_bluesky.devices.ca import CaSnapshotReadable

    cam = _camera(RE, box, "UC_Cam")
    magnet = CaSnapshotReadable(
        "U_VS1H", ["Current"], experiment="TestExp", name="u_vs1h"
    )
    connect_mock(RE, magnet)
    set_mock_value(magnet.connected_status, "Disconnected")
    count = bind_plans(profiles)["count"]
    with pytest.raises(GeecsDeviceDownError, match="U_VS1H"):
        RE(count([cam, magnet.scalars], 1))
    assert box.puts == []


def test_a_live_set_passes_the_gate_and_the_box_is_then_armed(
    RE, box, profiles
) -> None:
    """The gate reads but never drives: the first put is still the bracket's ARMED."""
    cam = _camera(RE, box, "UC_Cam")
    set_mock_value(cam.connected_status, "Connected")
    count = bind_plans(profiles)["count"]
    RE(count([cam], 1))
    assert box.puts[0] == ("DG", "Trigger.Source", "single")
    assert box.fires == 1


def test_an_unreadable_liveness_pv_is_fail_open_and_warns(
    RE, box, profiles, monkeypatch, caplog
) -> None:
    """A read that raises is not a verdict (the gateway serves CONNECTED for every device)."""
    cam = _camera(RE, box, "UC_Cam")
    sc = profiles.resolve(None)

    async def boom():
        raise OSError("CA timeout")

    monkeypatch.setattr(sc.liveness_signals["DG"], "read", boom)
    count = bind_plans(profiles)["count"]
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.ca.liveness"):
        RE(count([cam], 1))
    assert box.fires == 1
    (record,) = [r for r in caplog.records if "CONNECTED read failed" in r.message]
    assert (
        record.levelno == logging.WARNING
        and "OSError: CA timeout" in record.getMessage()
    )


# ------------------------------------------- the failure's name (#868/#894)
def test_a_refused_move_inside_the_run_names_its_cause(RE, box, profiles) -> None:
    """The stop document's reason is the cause by ``str``, not ``<AsyncStatus …>``.

    The cause is falsy and its repr is the bare code (the ``aioca.CANothing``
    shape): only ``str`` carries the PV.
    """
    cam = _camera(RE, box, "UC_Cam")
    magnet = Magnet()
    connect_mock(RE, magnet)
    text = "testexp:u_s1h:current:SP: Virtual circuit disconnect"

    def refuse(value, **kwargs):
        raise _RefusedPut(text)

    callback_on_mock_put(magnet.current._setpoint, refuse)
    col = DocCollector()
    RE.subscribe(col)
    scan = bind_plans(profiles, settables={"U_S1H": magnet})["sweep"]
    with pytest.raises(FailedStatus) as info:
        RE(scan([cam], sweep=payload()))
    assert isinstance(info.value.__cause__, _RefusedPut)
    assert str(info.value) == f"_RefusedPut: {text}"
    (stop,) = col.docs["stop"]
    assert stop["exit_status"] == "fail"
    assert stop["reason"] == f"_RefusedPut: {text}"


def test_a_refused_manual_move_names_its_cause(RE, profiles) -> None:
    """The registered ``mv`` (a queue item, no run): the manager's report reads the cause."""
    magnet = Magnet()
    connect_mock(RE, magnet)
    text = "testexp:u_s1h:current:SP: Channel write request failed"

    def refuse(value, **kwargs):
        raise _RefusedPut(text)

    callback_on_mock_put(magnet.current._setpoint, refuse)
    with pytest.raises(FailedStatus) as info:
        RE(bind_plans(profiles)["mv"](magnet.current, 0.5))
    assert str(info.value) == f"_RefusedPut: {text}"


def test_name_failed_status_leaves_a_cause_less_status_alone(RE) -> None:
    """Applied twice (hook + bracket): a status with no cause keeps its own text."""
    from geecs_bluesky.plans.strict import name_failed_status

    def failing():
        yield from bps.null()
        raise FailedStatus("<AsyncStatus …, done>")

    with pytest.raises(FailedStatus) as info:
        RE(name_failed_status(name_failed_status(failing())))
    assert str(info.value) == "<AsyncStatus …, done>"


def test_failure_cause_text_carries_the_causes_notes() -> None:
    """A note a device attached (the file plugin's WriteMessage, #894) is rendered."""
    cause = TimeoutError("uc_cam-hdf-capture didn't match True in 10.0s")
    cause.add_note("file plugin uc_cam-hdf: no frame from UC_Cam image within 8 s")
    failed = FailedStatus(
        "<AsyncStatus, task: <coroutine>, errored: TimeoutError(...)>"
    )
    failed.__cause__ = cause
    assert failure_cause_text(failed) == (
        "TimeoutError: uc_cam-hdf-capture didn't match True in 10.0s "
        "(file plugin uc_cam-hdf: no frame from UC_Cam image within 8 s)"
    )
    bare = _RefusedPut("pv:SP: refused")
    assert failure_cause_text(bare) == "_RefusedPut: pv:SP: refused"


def test_optimize_signature_and_classification(profiles):
    assert "optimize" not in NON_SCAN_PLAN_NAMES
    plan = bind_plans(profiles)["optimize"]
    assert is_plan(plan) and inspect.isgeneratorfunction(plan)
    parameters = inspect.signature(plan).parameters
    assert list(parameters) == [
        "detectors",
        "optimizer_config",
        "max_iterations",
        "shots_per_step",
        "trigger_profile",
        "shot_period",
        "non_essential",
        "native_image_save",
        "md",
    ]
    assert parameters["optimizer_config"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["optimizer_config"].default is inspect.Parameter.empty
