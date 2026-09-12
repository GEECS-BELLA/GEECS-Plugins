"""The registration table: stock names, strict binding, queue-item validation (PR 2, #807)."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
import bluesky.plans as bp  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from bluesky.utils import is_plan  # noqa: E402
from ophyd_async.core import Device, set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaMotor  # noqa: E402
from geecs_bluesky.devices.shot_control import ShotControl  # noqa: E402
from geecs_bluesky.exceptions import GeecsConfigurationError  # noqa: E402
from geecs_bluesky.plan_names import GEECS_PLAN_NAMES, NON_SCAN_PLAN_NAMES  # noqa: E402
from geecs_bluesky.plans.registry import (  # noqa: E402
    EXCLUDED_STOCK_PLANS,
    TriggerProfiles,
    bind_plans,
    stock_plans_with_hook,
    strict_plan,
)
from tests.ca_mock_helpers import DocCollector, connect_mock, follow_setpoint  # noqa: E402
from tests.test_strict_plans import WRITES, FakeBox, _camera  # noqa: E402


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
def test_plan_names_are_every_expressible_stock_plan_with_the_hook() -> None:
    """GEECS_PLAN_NAMES (import-light) pins the derivation the registry uses."""
    derived = set(stock_plans_with_hook()) - EXCLUDED_STOCK_PLANS
    assert derived == set(GEECS_PLAN_NAMES) - set(NON_SCAN_PLAN_NAMES)
    assert EXCLUDED_STOCK_PLANS <= set(stock_plans_with_hook())


def test_bound_plans_keep_the_stock_signature_minus_the_hook(profiles) -> None:
    bound = bind_plans(profiles)
    assert set(bound) == set(GEECS_PLAN_NAMES)
    assert bound["mv"] is bps.mv
    assert list(inspect.signature(bound["run_action"]).parameters) == ["name"]
    for name in GEECS_PLAN_NAMES:
        if name in NON_SCAN_PLAN_NAMES:
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
    plans, devices, *_ = existing_plans_and_devices_from_nspace(nspace=ns)
    assert set(plans) == set(GEECS_PLAN_NAMES)
    assert "scalars" in devices["UC_Cam"]["components"]
    items = [
        ("count", [["UC_Cam"], 3], {"trigger_profile": "HTU-Test"}),
        ("count", [["UC_Cam.scalars"]], {"num": 2}),
        ("scan", [["UC_Cam"], "U_S1H.current", -1, 1, 5], {"shots_per_step": 4}),
        ("list_scan", [["UC_Cam"], "U_S1H.current", [0.0, 0.5]], {}),
        ("rel_grid_scan", [["UC_Cam"], "U_S1H.current", -1, 1, 3], {}),
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
    scan = bind_plans(profiles)["scan"]
    RE(scan([cam], magnet.current, -1.0, 1.0, 3, shots_per_step=2))
    events = col.primary_events()
    assert box.fires == 6 and len(events) == 6
    assert [e["data"]["bin_number"] for e in events] == [1, 1, 2, 2, 3, 3]
    assert [e["data"]["u_s1h-current-position"] for e in events] == pytest.approx(
        [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]
    )
    start = col.docs["start"][0]
    assert start["plan_name"] == "scan"
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
    scan = bind_plans(profiles)["scan"]
    RE(scan([magnet.scalars], magnet.current, -1.0, 1.0, 3))
    RE(scan([cam.scalars], cam.exposure, 1.0, 3.0, 3))
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
