"""GeecsNamespace — the experiment's devices as nouns (#807 phase 1).

Built from an explicit roster (no DB); the DB path is exercised with a fake
``GeecsDb`` so the loud-failure contract is pinned without a database.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.devices.geecs_device import GeecsDevice, GeecsTriggeredDevice
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace

ROSTER = DeviceRoster(
    experiment="TestExp",
    variables={
        "UC_TestCam": [
            {"name": "trigger", "settable": True, "variabletype": "choice"},
            {"name": "MeanCounts", "variabletype": "numeric"},
            {"name": "image", "variabletype": "image"},
        ],
        "U_DG645_Box": [
            {"name": "Trigger.Source", "settable": True, "variabletype": "choice"},
        ],
        "U_S1H": [
            {
                "name": "current",
                "settable": True,
                "variabletype": "numeric",
                "tolerance": 0.01,
            },
            {"name": "voltage", "variabletype": "numeric"},
        ],
        "U_ImagesOnly": [{"name": "image", "variabletype": "image"}],
        "U DG645 ShotControl": [
            {"name": "Trigger.Mode", "settable": True, "variabletype": "string"}
        ],
    },
    types={
        "UC_TestCam": "Point Grey Camera",
        "U_S1H": "Magnet PS",
        "U_DG645_Box": "DG645",
    },
    subscribed={"UC_TestCam": ["MeanCounts"]},
)


def test_namespace_builds_one_device_per_rostered_device() -> None:
    ns = GeecsNamespace(ROSTER)
    assert len(ns) == 4  # U_ImagesOnly has no scalars and is skipped
    assert set(ns.devices) == {
        "UC_TestCam",
        "U_S1H",
        "u_dg645_shotcontrol",
        "U_DG645_Box",
    }
    assert isinstance(
        ns["UC_TestCam"], GeecsTriggeredDevice
    )  # trigger variable → acquirer
    assert not isinstance(ns["U_DG645_Box"], GeecsTriggeredDevice)  # trigger *source*
    assert isinstance(ns["U_S1H"], GeecsDevice) and not isinstance(
        ns["U_S1H"], GeecsTriggeredDevice
    )
    assert ns["UC_TestCam"].devicetype == "Point Grey Camera"
    assert ns["UC_TestCam"].selected == ("MeanCounts",)
    assert ns["U DG645 ShotControl"].geecs_name == "U DG645 ShotControl"
    assert ns["u_dg645_shotcontrol"] is ns["U DG645 ShotControl"]
    assert "uc_testcam" in ns and "nope" not in ns
    with pytest.raises(KeyError, match="no device 'nope'"):
        ns["nope"]


def test_only_served_variables_become_children() -> None:
    """Served = subscribed ∪ settable ∪ acq_timestamp (the gateway's rule)."""
    ns = GeecsNamespace(ROSTER)
    magnet = ns["U_S1H"]
    # 'voltage' is neither subscribed nor settable → not served → no child
    assert magnet.variables == ("current",)
    assert not hasattr(magnet, "voltage")
    cam = ns["UC_TestCam"]
    assert cam.variables == ("trigger", "MeanCounts") and hasattr(cam, "acq_timestamp")
    assert isinstance(cam.trigger_, CaSettable) and callable(cam.trigger)
    everything = GeecsNamespace(ROSTER, include_unserved=True)
    assert everything["U_S1H"].variables == ("current", "voltage")


def test_served_variable_names_rule() -> None:
    from geecs_bluesky.namespace import served_variable_names

    rows = [
        {"name": "acq_timestamp"},
        {"name": "a", "settable": True},
        {"name": "b"},
        {"name": "c"},
    ]
    assert served_variable_names(rows, ["c"]) == {"acq_timestamp", "a", "c"}


def test_resolve_device_and_device_colon_variable() -> None:
    ns = GeecsNamespace(ROSTER)
    assert ns.resolve("U_S1H") is ns["U_S1H"]
    assert ns.resolve("U_S1H:current") is ns["U_S1H"].current
    assert ns.resolve("u_s1h:CURRENT") is ns["U_S1H"].current
    assert (
        ns.resolve("U DG645 ShotControl:Trigger.Mode")
        is ns["U DG645 ShotControl"].trigger_mode
    )


def test_catalog_motor_kind_promotes_a_settable() -> None:
    rows = {
        "U_Plain": [{"name": "current", "settable": True, "variabletype": "numeric"}]
    }
    roster = DeviceRoster(experiment="TestExp", variables=rows)
    from geecs_bluesky.devices.ca.motor import CaMotor

    assert not isinstance(GeecsNamespace(roster)["U_Plain"].current, CaMotor)
    promoted = GeecsNamespace(roster, motor_targets={"U_Plain": {"current"}})
    assert isinstance(promoted["U_Plain"].current, CaMotor)


def test_motor_targets_from_catalog_reads_kind_motor_entries() -> None:
    catalog = SimpleNamespace(
        variables={
            "jet_x": SimpleNamespace(
                kind="motor", target="U_ESP_JetXYZ:Position.Axis 1"
            ),
            "amp": SimpleNamespace(kind="setpoint", target="U_Amp:power"),
            "pseudo": SimpleNamespace(kind="pseudo", targets=[]),
        }
    )
    resolver = SimpleNamespace(scan_variable_catalog=lambda: catalog)
    assert GeecsNamespace.motor_targets_from_catalog(resolver) == {
        "U_ESP_JetXYZ": {"Position.Axis 1"}
    }
    broken = SimpleNamespace(
        scan_variable_catalog=lambda: (_ for _ in ()).throw(OSError("x"))
    )
    assert GeecsNamespace.motor_targets_from_catalog(broken) == {}


def test_name_collision_between_devices_is_loud() -> None:
    roster = DeviceRoster(
        experiment="TestExp",
        variables={
            "U Foo": [{"name": "a", "settable": True, "variabletype": "numeric"}],
            "U-Foo": [{"name": "a", "settable": True, "variabletype": "numeric"}],
        },
    )
    with pytest.raises(
        GeecsConfigurationError, match="both normalise to the name 'u_foo'"
    ):
        GeecsNamespace(roster)


def test_roster_triggered_override_wins(caplog) -> None:
    roster = DeviceRoster(
        experiment="TestExp",
        variables=ROSTER.variables,
        types=ROSTER.types,
        triggered={"U_S1H": True, "UC_TestCam": False},
        live_triggered=frozenset({"UC_TestCam"}),
    )
    with caplog.at_level("WARNING", logger="geecs_bluesky.namespace"):
        ns = GeecsNamespace(roster)
    assert isinstance(ns["U_S1H"], GeecsTriggeredDevice)
    assert not isinstance(ns["UC_TestCam"], GeecsTriggeredDevice)
    # the live probe disagrees with the classification → loud, not silent
    assert "pushing acq_timestamp but NOT classified" in caplog.text
    assert "UC_TestCam" in caplog.text


def test_export_into_binds_names_and_refuses_shadowing() -> None:
    ns = GeecsNamespace(ROSTER)
    target: dict = {"RE": object()}
    names = ns.export_into(target)
    assert names == ["UC_TestCam", "U_DG645_Box", "U_S1H", "u_dg645_shotcontrol"]
    assert target["U_S1H"] is ns["U_S1H"]
    with pytest.raises(GeecsConfigurationError, match="would shadow"):
        ns.export_into({"U_S1H": object()})


class _FakeDb:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def _q(self, name: str, value):
        self.calls.append(name)
        if self.fail:
            raise ConnectionError("db down")
        return value

    def get_experiment_device_variables(self, experiment):
        return self._q("variables", ROSTER.variables)

    def get_experiment_device_types(self, experiment):
        return self._q("types", ROSTER.types)

    def get_subscribed_variables(self, experiment):
        return self._q("subscribed", ROSTER.subscribed)

    def get_experiment_devices(self, experiment):
        return self._q("devices", {"U_S1H": ("192.168.0.1", 1234)})


def test_from_experiment_uses_the_four_batch_queries() -> None:
    db = _FakeDb()
    ns = GeecsNamespace.from_experiment("TestExp", geecs_db=db)
    assert db.calls == ["variables", "types", "subscribed", "devices"]
    assert ns.roster.endpoints == {"U_S1H": ("192.168.0.1", 1234)}
    assert len(ns) == 4


def test_db_failure_at_build_is_loud_not_empty() -> None:
    with pytest.raises(
        GeecsConfigurationError, match="could not load the 'TestExp' device roster"
    ):
        GeecsNamespace.from_experiment("TestExp", geecs_db=_FakeDb(fail=True))
