"""GeecsNamespace — the experiment's devices as nouns, composed from the ca layer (#807 phase 1).

Built from an explicit roster (no DB); the DB path is exercised with a fake
``GeecsDb`` so the loud-failure contract is pinned without a database.
Pins the hardware-derived rules of
``Planning/native_bluesky/01_device_namespace.md``: served-set children,
DB-derived types, protocol-name collisions, the triggerable shortcut.
"""

from __future__ import annotations

import pytest
from ophyd_async.core import SignalR

from geecs_bluesky.devices.ca.generic_detector import CaGenericDetector
from geecs_bluesky.devices.ca.motor import CaMotor
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.namespace import (
    DeviceRoster,
    GeecsNamespace,
    identifier_name,
    looks_triggerable,
    python_type,
)


def row(name, *, settable=False, variabletype=None, choices="numeric", tolerance=None):
    return {
        "name": name,
        "settable": settable,
        "variabletype": variabletype,
        "choices": choices,
        "tolerance": tolerance,
        "units": "",
        "min": None,
        "max": None,
    }


ROSTER = DeviceRoster(
    experiment="TestExp",
    variables={
        "UC_TestCam": [
            row("MeanCounts"),
            row("MaxCounts"),
            row("Unsubscribed"),  # served? no: not subscribed, not settable
            row("exposure", settable=True, tolerance=0.0),  # numeric settable, DB tol 0
            row(
                "trigger", settable=True, choices="on,off"
            ),  # enum; collides with trigger()
            row("localsavingpath", settable=True, choices="path"),
            row("image", choices="image"),  # non-scalar
        ],
        "U_S1H": [
            row(
                "Current", settable=True, tolerance=0.05
            ),  # untyped numeric with tolerance
            row("Voltage"),  # subscribed readback
            row(
                "Enable_Output", settable=True, variabletype="choice", choices="on,off"
            ),
            row("Position.Axis 1"),  # not subscribed → not served
        ],
        "U_ImagesOnly": [row("image", choices="image")],
        "U DG645 ShotControl": [
            row("Trigger.Source", settable=True, choices="a,b"),
            row("Trigger.Rate", settable=True),
        ],
    },
    types={
        "UC_TestCam": "Point Grey Camera",
        "U_S1H": "Magnet PS",
        "U DG645 ShotControl": "DG645",
    },
    subscribed={
        "UC_TestCam": ["MeanCounts", "MaxCounts"],
        "U_S1H": ["Current", "Voltage"],
    },
)


# ------------------------------------------------------------------ rules
def test_identifier_name_keeps_geecs_spelling_when_valid() -> None:
    assert identifier_name("U_S1H") == "U_S1H"
    assert identifier_name("Current") == "Current"
    assert identifier_name("Position.Axis 1") == "position_axis_1"


def test_python_type_follows_effective_vartype() -> None:
    assert python_type(row("x")) is float  # choices=numeric
    assert python_type(row("x", choices="on,off")) is str  # enum → label string
    assert python_type(row("x", choices="path")) is str  # long string
    # variabletype wins over an option list (the gateway's served type today;
    # the 18 such Undulator rows are a DB fix — see geecs_core.db.variable_types)
    assert python_type(row("x", variabletype="numeric", choices="1,2,3")) is float
    assert python_type(row("x", choices="image")) is None  # non-scalar
    assert python_type(row("x", choices="1darray")) is None


def test_looks_triggerable_heuristic() -> None:
    assert looks_triggerable([row("trigger"), row("MeanCounts")], "Point Grey Camera")
    assert looks_triggerable([row("EnableTrigger")], "PicoscopeV2")
    assert not looks_triggerable([row("Current")], "Magnet PS")
    dg645 = [row("Trigger.Source"), row("Trigger.ExecuteSingleShot")]
    assert not looks_triggerable(dg645, "DG645")
    assert not looks_triggerable(dg645, "DG645 Delay Generator")  # token match
    assert looks_triggerable(dg645 + [row("acq_timestamp")], "DG645")  # DB row wins


# -------------------------------------------------------------- composition
def test_namespace_composes_the_existing_device_classes() -> None:
    ns = GeecsNamespace(ROSTER)
    assert set(ns.devices) == {"UC_TestCam", "U_S1H", "u_dg645_shotcontrol"}
    cam, magnet, box = ns["UC_TestCam"], ns["U_S1H"], ns["U DG645 ShotControl"]
    assert isinstance(cam, CaGenericDetector)  # trigger-named variable → acquirer
    assert isinstance(magnet, CaSnapshotReadable)
    assert isinstance(box, CaSnapshotReadable)  # DG645 is a trigger *source*
    assert cam.name == "uc_testcam" and cam._geecs_device_name == "UC_TestCam"
    assert ns["u_dg645_shotcontrol"] is box and "uc_testcam" in ns and "nope" not in ns


def test_served_set_decides_which_children_exist() -> None:
    cam = GeecsNamespace(ROSTER)["UC_TestCam"]
    assert isinstance(cam.meancounts, SignalR) and isinstance(cam.maxcounts, SignalR)
    assert not hasattr(cam, "unsubscribed")  # not served
    assert not hasattr(cam, "image")  # non-scalar
    magnet = GeecsNamespace(ROSTER)["U_S1H"]
    assert not hasattr(magnet, "position_axis_1")


def test_settables_attach_as_movable_children_with_db_types() -> None:
    ns = GeecsNamespace(ROSTER)
    magnet, cam = ns["U_S1H"], ns["UC_TestCam"]
    assert isinstance(magnet.current, CaMotor)  # DB tolerance → convergence motor
    assert magnet.current._tolerance == pytest.approx(0.05)
    assert magnet.current.name == "u_s1h-current"  # named by the parent, safe_name
    assert isinstance(magnet.enable_output, CaSettable) and not isinstance(
        magnet.enable_output, CaMotor
    )
    # tol 0.0 → a plain setpoint, not a convergence motor (review #2)
    assert isinstance(cam.exposure, CaSettable) and not isinstance(
        cam.exposure, CaMotor
    )
    assert isinstance(cam.localsavingpath, CaSettable)
    # the settable's readback column header is the GEECS "Device Variable" form,
    # and the parent aggregates it (the s-file exporter reads top-level devices)
    assert magnet.current._column_headers == {"u_s1h-current-position": "U_S1H Current"}
    assert magnet._column_headers["u_s1h-current-position"] == "U_S1H Current"
    assert magnet._column_headers["u_s1h-voltage"] == "U_S1H Voltage"


def test_protocol_named_settable_binds_with_a_trailing_underscore() -> None:
    cam = GeecsNamespace(ROSTER)["UC_TestCam"]
    assert callable(cam.trigger) and cam.trigger.__func__ is CaGenericDetector.trigger
    assert isinstance(cam.trigger_, CaSettable)


async def test_read_returns_the_subscribed_list_plus_shot_stamp() -> None:
    ns = GeecsNamespace(ROSTER)
    cam, magnet = ns["UC_TestCam"], ns["U_S1H"]
    await cam.connect(mock=True)
    await magnet.connect(mock=True)
    assert set(await cam.read()) == {
        "uc_testcam-acq_timestamp",
        "uc_testcam-meancounts",
        "uc_testcam-maxcounts",
    }
    # Current is subscribed AND settable → its Movable child's readback is logged
    assert set(await magnet.read()) == {"u_s1h-current-position", "u_s1h-voltage"}
    assert hasattr(cam, "trigger") and not hasattr(magnet, "trigger")


def test_variable_and_resolve_accept_either_spelling() -> None:
    ns = GeecsNamespace(ROSTER)
    magnet = ns["U_S1H"]
    assert ns.variable("U_S1H", "Current") is magnet.current  # GEECS spelling
    assert ns.resolve("u_s1h:CURRENT") is magnet.current
    assert ns.resolve("U_S1H:voltage") is magnet.voltage
    assert ns.resolve("UC_TestCam:trigger") is ns["UC_TestCam"].trigger_
    assert ns.resolve("U_S1H") is magnet
    with pytest.raises(KeyError, match="no served scalar variable 'nope'"):
        ns.variable("U_S1H", "nope")
    with pytest.raises(KeyError, match="no device 'nope'"):
        ns["nope"]


def test_roster_triggered_override_wins() -> None:
    roster = DeviceRoster(
        experiment="TestExp",
        variables=ROSTER.variables,
        types=ROSTER.types,
        subscribed=ROSTER.subscribed,
        triggered={"U_S1H": True, "UC_TestCam": False},
    )
    ns = GeecsNamespace(roster)
    assert isinstance(ns["U_S1H"], CaGenericDetector)
    assert isinstance(ns["UC_TestCam"], CaSnapshotReadable)


def test_case_different_variable_names_are_refused() -> None:
    """GEECS names are case-insensitive downstream; two served variables that
    differ only by case would silently merge — refuse instead (review #7)."""
    roster = DeviceRoster(
        experiment="TestExp",
        variables={
            "UC_X": [row("Trigger"), row("trigger", settable=True, choices="on,off")]
        },
        types={"UC_X": "Point Grey Camera"},
        subscribed={"UC_X": ["Trigger"]},
    )
    with pytest.raises(GeecsConfigurationError, match="differ only by case"):
        GeecsNamespace(roster)


def test_settable_colliding_with_a_readable_child_is_refused() -> None:
    """'Position.Axis 1' (readable) and 'position_axis_1' (settable) both want one
    attribute; a rename would hide the readable — raise (review N1)."""
    roster = DeviceRoster(
        experiment="TestExp",
        variables={
            "U_X": [row("Position.Axis 1"), row("position_axis_1", settable=True)],
        },
        subscribed={"U_X": ["Position.Axis 1"]},
    )
    with pytest.raises(GeecsConfigurationError, match="collides with the child"):
        GeecsNamespace(roster)


def test_name_collision_between_devices_is_loud() -> None:
    roster = DeviceRoster(
        experiment="TestExp",
        variables={
            "U Foo": [row("a", settable=True)],
            "U-Foo": [row("a", settable=True)],
        },
    )
    with pytest.raises(
        GeecsConfigurationError, match="both normalise to the name 'u_foo'"
    ):
        GeecsNamespace(roster)


def test_export_into_binds_names_and_refuses_shadowing() -> None:
    ns = GeecsNamespace(ROSTER)
    target: dict = {"RE": object()}
    assert ns.export_into(target) == ["UC_TestCam", "U_S1H", "u_dg645_shotcontrol"]
    assert target["U_S1H"] is ns["U_S1H"]
    with pytest.raises(GeecsConfigurationError, match="would shadow"):
        ns.export_into({"U_S1H": object()})


# ----------------------------------------------------------------- DB path
class _FakeDb:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def _q(self, name, value):
        self.calls.append(name)
        if self.fail:
            raise ConnectionError("db down")
        return value

    def get_experiment_device_variables(self, experiment, *, enabled_only=True):
        return self._q("variables", ROSTER.variables)

    def get_experiment_device_types(self, experiment, *, enabled_only=True):
        return self._q("types", ROSTER.types)

    def get_subscribed_variables(self, experiment, *, enabled_only=True):
        return self._q("subscribed", ROSTER.subscribed)


def test_from_experiment_reuses_the_db_runtime_providers() -> None:
    db = _FakeDb()
    ns = GeecsNamespace.from_experiment("TestExp", geecs_db=db)
    assert (
        ns.roster.served
        == {  # GeecsDbServedSetProvider's rule: subscribed ∪ settable
            "UC_TestCam": {
                "MeanCounts",
                "MaxCounts",
                "exposure",
                "trigger",
                "localsavingpath",
            },
            "U_S1H": {"Current", "Voltage", "Enable_Output"},
            "U DG645 ShotControl": {"Trigger.Source", "Trigger.Rate"},
        }
    )
    assert ns.roster.types["UC_TestCam"] == "Point Grey Camera"
    assert set(ns.devices) == {"UC_TestCam", "U_S1H", "u_dg645_shotcontrol"}
    assert "variables" in db.calls and "subscribed" in db.calls and "types" in db.calls


def test_db_failure_at_build_is_loud_not_empty() -> None:
    with pytest.raises(
        GeecsConfigurationError, match="could not load the 'TestExp' device roster"
    ):
        GeecsNamespace.from_experiment("TestExp", geecs_db=_FakeDb(fail=True))
