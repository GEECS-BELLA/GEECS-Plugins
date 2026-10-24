"""The positioner panel: the record's fields, the allow-list, the state words, the feed, the routes — no CA, no IOC."""

from __future__ import annotations

import json
import math
from typing import Optional

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme import STATES

from geecs_panels.app import create_app
from geecs_panels.feed import WriteFailed
from geecs_panels.motor import (
    ACTIONS,
    AXIS_FIELDS,
    MOTOR_KIND,
    NO_WAIT,
    WRITABLE,
    NotAllowed,
    NotWritable,
    allowed_actions,
    allowed_fields,
    axis_state,
    check_actions,
    check_write,
    device_state,
    msta_flags,
)
from geecs_panels.motor_feed import MotorFeed
from geecs_panels.registry import Device

STAGE = Device("SimMotor1", MOTOR_KIND, "undulator:simmotor1:", axes=("m1", "m2"))
CAM = Device("ADTEST_CAM1", "aravis_camera", "undulator:adtest_cam1:")

#: What simmotor1's m1 reports at rest, homed (MSTA 16514: DONE, HOME, HOMED).
AT_REST = {
    "desc": "Simulated axis 1", "egu": "mm", "prec": 5, "rbv": 0.0, "val": 0.0,
    "dmov": 1, "movn": 0, "hlm": 20000.0, "llm": -20000.0, "hls": 0, "lls": 0,
    "lvio": 0, "velo": 1.0, "vmax": 0.0, "vbas": 0.1, "accl": 0.2, "rdbd": 0.01,
    "twv": 1.0, "msta": 16514, "sevr": "NO_ALARM", "stat": "NO_ALARM",
    "set": "Use", "spmg": "Go", "off": 0.0,
}  # fmt: skip


class _Sub:
    def close(self) -> None:
        pass


def _feed(**m1) -> MotorFeed:
    feed = MotorFeed(STAGE.prefix, STAGE.axes, lambda pv, s, cb: _Sub())
    feed.start()
    for key, value in {**AT_REST, **m1}.items():
        feed.update(f"m1.{key}", value)
    return feed


class FakeMotors:
    def __init__(self, refuse: bool = False) -> None:
        self.feeds: dict[str, MotorFeed] = {}
        self.puts: list[tuple[str, str, object]] = []
        self.refuse = refuse

    def feed(self, device: Device) -> MotorFeed:
        if device.name not in self.feeds:
            self.feeds[device.name] = _feed()
        return self.feeds[device.name]

    async def put(self, device: Device, axis: str, field: str, value: object) -> None:
        if self.refuse:
            raise WriteFailed(f"{device.prefix}{axis}{field}: no write access")
        self.puts.append((axis, field, value))


def _client(motors: Optional[FakeMotors] = None, **kwargs) -> TestClient:
    return TestClient(
        create_app(
            "Undulator", [STAGE, CAM], object(), motors=motors or FakeMotors(), **kwargs
        )
    )


# ---- the record -------------------------------------------------------------


def test_every_field_the_page_shows_is_monitored_per_axis() -> None:
    feed = MotorFeed(STAGE.prefix, STAGE.axes, lambda pv, s, cb: _Sub())
    pvs: list[tuple[str, bool]] = []
    feed.subscribe = lambda pv, s, cb: (pvs.append((pv, s)), _Sub())[1]
    feed.start()
    assert ("undulator:simmotor1:m1.RBV", False) in pvs
    assert ("undulator:simmotor1:m2.SEVR", True) in pvs
    assert len(pvs) == 2 * len(AXIS_FIELDS)


def test_msta_decodes_the_records_status_bits() -> None:
    flags = msta_flags(16514)  # DONE (bit 2) + HOME (8) + HOMED (15)
    assert flags["done"] and flags["home"] and flags["homed"]
    assert not flags["moving"] and not flags["comm_err"]
    assert msta_flags(1 << 12)["comm_err"]  # bit 13
    assert not any(msta_flags(None).values())


@pytest.mark.parametrize(
    "fields, want",
    [
        ({}, ("ok", "stopped")),
        ({"dmov": 0, "movn": 1}, ("running", "moving")),
        ({"hls": 1}, ("degraded", "at the high limit switch")),
        ({"lvio": 1}, ("degraded", "target outside the soft limits")),
        ({"sevr": "MAJOR", "stat": "HWLIMIT"}, ("degraded", "alarm HWLIMIT")),
        ({"sevr": "INVALID", "stat": "COMM"}, ("failed", "no communication")),
        ({"sevr": "INVALID", "stat": "UDF"}, ("failed", "invalid (UDF)")),
        ({"msta": 16514 | (1 << 9)}, ("failed", "controller problem")),
        ({"spmg": "Pause"}, ("degraded", "motion held (SPMG Pause)")),
    ],
)
def test_the_state_word_is_judged_worst_first(fields, want) -> None:
    axis = _feed(**fields).axis("m1")
    assert (axis["state"], axis["status"]) == want
    assert axis["state"] in STATES


def test_an_axis_no_field_of_which_connected_is_unknown() -> None:
    feed = _feed()
    assert feed.axis("m2")["state"] == "unknown"
    assert feed.axis("m2")["reachable"] is False
    assert "rbv" in feed.axis("m2")["missing"]
    assert axis_state({"reachable": False}) == ("unknown", "connecting")


def test_the_devices_chip_is_the_worst_axis() -> None:
    assert device_state(["ok", "running"]) == "running"
    assert device_state(["ok", "unknown", "degraded"]) == "unknown"
    assert device_state(["degraded", "failed"]) == "failed"
    assert _feed().snapshot()["state"] == "unknown"  # m2 never connected


def test_a_disconnect_marks_the_field_missing_not_the_axis_dead() -> None:
    feed = _feed()
    feed.update("m1.twv", None)
    axis = feed.axis("m1")
    assert axis["missing"] == ["twv"] and axis["reachable"] and axis["twv"] == 1.0


# ---- the allow-list ---------------------------------------------------------


def test_a_move_inside_the_soft_limits_is_a_number() -> None:
    axis = _feed(hlm=10.0, llm=-10.0).axis("m1")
    assert check_write(".VAL", 3, axis) == 3.0
    assert check_write(".VAL", -10.0, axis) == -10.0


def test_a_move_outside_the_soft_limits_is_refused_naming_them() -> None:
    axis = _feed(hlm=10.0, llm=-10.0).axis("m1")
    with pytest.raises(NotWritable, match="outside the soft limits -10 to 10 mm"):
        check_write(".VAL", 10.5, axis)


def test_limits_that_read_equal_are_left_to_the_record() -> None:
    assert check_write(".VAL", 1e6, _feed(hlm=0.0, llm=0.0).axis("m1")) == 1e6
    # Limits not read yet: the record is left to judge (LVIO), not refused here.
    assert check_write(".VAL", 1e6, {"reachable": True}) == 1e6


@pytest.mark.parametrize(
    "field, value",
    [
        (".RBV", 1.0),  # a readback
        (".OFF", 1.0),  # the user offset: never from the panel
        (".SPMG", "Stop"),  # not in the allow-list
        (".VAL", math.nan),
        (".VAL", True),
        (".TWV", 0),  # a step must be positive
        (".TWV", -1),
        (".VELO", 0.0),
        (".STOP", 0),  # a press is 1
        (".HOMF", 2),
    ],
)
def test_everything_else_is_refused(field, value) -> None:
    with pytest.raises(NotWritable):
        check_write(field, value, _feed().axis("m1"))


def test_presses_take_one_and_velocity_respects_vmax() -> None:
    axis = _feed(vmax=5.0).axis("m1")
    assert check_write(".STOP", 1, axis) == 1 and check_write(".TWF", True, axis) == 1
    assert check_write(".VELO", 2.5, axis) == 2.5
    with pytest.raises(NotWritable, match="VMAX"):
        check_write(".VELO", 7.0, axis)
    assert check_write(".VELO", 7.0, _feed(vmax=0.0).axis("m1")) == 7.0  # no maximum


def test_moves_are_dispatched_without_waiting_for_the_motion() -> None:
    assert {".VAL", ".TWF", ".TWR", ".HOMF", ".HOMR"} <= NO_WAIT
    assert ".TWV" not in NO_WAIT and ".VELO" not in NO_WAIT


def test_the_live_put_waits_on_a_setting_and_not_on_a_move(monkeypatch) -> None:
    """``LiveMotors.put``: the one line that differs from the scope's, pinned on a fake aioca."""
    import asyncio

    from geecs_panels import ca
    from tests.test_writes import FakeCa

    fake = FakeCa()
    monkeypatch.setattr(ca, "aioca", fake)
    motors = ca.LiveMotors()
    asyncio.run(motors.put(STAGE, "m1", ".VAL", 3.0))
    asyncio.run(motors.put(STAGE, "m2", ".VELO", 2.0))
    assert fake.puts == [
        ("undulator:simmotor1:m1.VAL", 3.0, False),
        ("undulator:simmotor1:m2.VELO", 2.0, True),
    ]
    monkeypatch.setattr(ca, "aioca", FakeCa(refuse=True))
    with pytest.raises(WriteFailed, match="m1.STOP"):
        asyncio.run(ca.LiveMotors().put(STAGE, "m1", ".STOP", 1))


# ---- routes -----------------------------------------------------------------


def test_the_landing_page_links_the_positioner() -> None:
    assert 'href="/motor/SimMotor1"' in _client().get("/").text


def test_without_live_motors_the_positioner_is_listed_unlinked_not_a_500() -> None:
    page = TestClient(create_app("Undulator", [STAGE, CAM], object())).get("/")
    assert page.status_code == 200
    assert "SimMotor1" in page.text and 'href="/motor/SimMotor1"' not in page.text
    assert 'href="/camera/ADTEST_CAM1"' in page.text


def test_the_page_renders_a_card_per_axis_from_the_registry() -> None:
    html = _client().get("/motor/SimMotor1").text
    assert 'data-axis="m1"' in html and 'data-axis="m2"' in html
    assert 'data-field=".VAL"' in html and 'data-press=".STOP"' in html
    assert "undulator:simmotor1:m2" in html and 'data-writable="true"' in html


def test_a_camera_is_not_a_positioner() -> None:
    assert _client().get("/motor/ADTEST_CAM1").status_code == 404
    assert _client().get("/motor/NOPE").status_code == 404


def test_the_stream_carries_every_axis_with_its_state() -> None:
    with _client().stream("GET", "/motor/SimMotor1/events?once=1") as r:
        body = "".join(r.iter_text())
    data = json.loads(body.split("data: ", 1)[1])
    assert [a["name"] for a in data["axes"]] == ["m1", "m2"]
    m1 = data["axes"][0]
    assert m1["rbv"] == 0.0 and m1["state"] == "ok" and m1["flags"]["homed"]
    assert data["axes"][1]["state"] == "unknown"


def test_a_put_goes_through_the_allow_list_to_the_named_axis() -> None:
    motors = FakeMotors()
    client = _client(motors)
    ok = client.post(
        "/motor/SimMotor1/put", json={"axis": "m1", "field": ".VAL", "value": 3}
    )
    assert ok.json() == {"ok": True, "axis": "m1", "field": ".VAL", "value": 3.0}
    assert motors.puts == [("m1", ".VAL", 3.0)]
    far = client.post(
        "/motor/SimMotor1/put", json={"axis": "m1", "field": ".VAL", "value": 30000.0}
    )
    assert far.status_code == 422 and "soft limits" in far.json()["detail"]
    readback = client.post(
        "/motor/SimMotor1/put", json={"axis": "m1", "field": ".RBV", "value": 1}
    )
    assert readback.status_code == 422
    other = client.post(
        "/motor/SimMotor1/put", json={"axis": "m3", "field": ".STOP", "value": 1}
    )
    assert other.status_code == 404
    assert motors.puts == [("m1", ".VAL", 3.0)]


def test_the_ioc_refusing_is_a_502_and_read_only_a_403() -> None:
    put = {"axis": "m1", "field": ".STOP", "value": 1}
    assert (
        _client(FakeMotors(refuse=True))
        .post("/motor/SimMotor1/put", json=put)
        .status_code
        == 502
    )
    assert (
        _client(writable=False).post("/motor/SimMotor1/put", json=put).status_code
        == 403
    )


# ---- per-device actions -------------------------------------------------------

#: A positioner whose bench entry permits move, tweak and stop only (U_ESP302_01:
#: actuators on mirror mounts — no homing, no velocity change).
MIRRORS = Device(
    "U_ESP302_01",
    MOTOR_KIND,
    "undulator:esptest1:",
    axes=("m1",),
    allow=frozenset({"move", "tweak", "stop"}),
    note="Between moves the readback is the last reading.",
)


def _mirrors(motors: Optional[FakeMotors] = None) -> TestClient:
    return TestClient(
        create_app(
            "Undulator", [STAGE, MIRRORS], object(), motors=motors or FakeMotors()
        )
    )


def test_every_action_names_panel_fields_and_together_they_are_the_allow_list() -> None:
    assert frozenset().union(*ACTIONS.values()) == WRITABLE
    assert allowed_fields(None) == WRITABLE
    assert allowed_fields(["move", "tweak", "stop"]) == {
        ".VAL", ".TWV", ".TWF", ".TWR", ".STOP"
    }  # fmt: skip
    assert allowed_actions(None) == list(ACTIONS)
    assert allowed_actions({"stop", "move"}) == ["move", "stop"]


@pytest.mark.parametrize(
    "allow, message",
    [
        (["move", "fly"], "unknown positioner action"),
        (["move", "tweak"], "must permit stop"),
        (["home"], "must permit stop"),
    ],
)
def test_an_allow_list_names_known_actions_and_never_a_motion_without_stop(
    allow, message
) -> None:
    with pytest.raises(ValueError, match=message):
        check_actions(allow)
    check_actions(["stop"])
    check_actions(["velocity"])


@pytest.mark.parametrize("field", [".HOMF", ".HOMR", ".VELO"])
def test_a_field_the_device_does_not_permit_is_not_allowed(field) -> None:
    allowed = allowed_fields(["move", "tweak", "stop"])
    with pytest.raises(NotAllowed, match="not permitted on this device"):
        check_write(field, 1, AT_REST, allowed)
    assert check_write(".TWV", 0.001, AT_REST, allowed) == 0.001
    # The panel's own refusals come first and stay 422-shaped.
    with pytest.raises(NotWritable) as refused:
        check_write(".OFF", 1, AT_REST, allowed)
    assert not isinstance(refused.value, NotAllowed)


def test_a_restricted_device_refuses_homing_and_velocity_with_a_403() -> None:
    motors = FakeMotors()
    client = _mirrors(motors)
    for field, value in ((".TWV", 0.001), (".TWF", 1), (".TWR", 1), (".STOP", 1)):
        r = client.post(
            "/motor/U_ESP302_01/put",
            json={"axis": "m1", "field": field, "value": value},
        )
        assert r.status_code == 200, (field, r.text)
    for field, value in ((".HOMF", 1), (".HOMR", 1), (".VELO", 0.3)):
        r = client.post(
            "/motor/U_ESP302_01/put",
            json={"axis": "m1", "field": field, "value": value},
        )
        assert r.status_code == 403 and "not permitted" in r.json()["detail"]
    assert [f for _, f, _ in motors.puts] == [".TWV", ".TWF", ".TWR", ".STOP"]
    # A device without an allow list keeps every action (#1143's default).
    home = client.post(
        "/motor/SimMotor1/put", json={"axis": "m1", "field": ".HOMF", "value": 1}
    )
    assert home.status_code == 200


def test_the_page_draws_only_the_permitted_controls_and_says_which_are_held() -> None:
    client = _mirrors()
    html = client.get("/motor/U_ESP302_01").text
    for drawn in ('data-field=".VAL"', 'data-press=".TWF"', 'data-press=".STOP"'):
        assert drawn in html
    for held in ('data-press=".HOMF"', 'data-press=".HOMR"', 'data-field=".VELO"'):
        assert held not in html
    assert 'data-k="velo"' in html  # the velocity still reads, as a value
    assert "From this panel: move, tweak, stop." in html
    assert "Held for this device: home, velocity." in html
    assert "Between moves the readback is the last reading." in html
    full = client.get("/motor/SimMotor1").text
    assert 'data-press=".HOMF"' in full and 'data-field=".VELO"' in full
    assert "Held for this device" not in full


def test_the_banner_never_offers_what_the_panel_will_refuse() -> None:
    nothing = Device("Viewer", MOTOR_KIND, "p:", axes=("m1",), allow=frozenset())
    client = TestClient(
        create_app("Undulator", [nothing], object(), motors=FakeMotors())
    )
    assert "From this panel: nothing." in client.get("/motor/Viewer").text
    read_only = (
        TestClient(
            create_app(
                "Undulator", [MIRRORS], object(), motors=FakeMotors(), writable=False
            )
        )
        .get("/motor/U_ESP302_01")
        .text
    )
    assert "Read-only panel" in read_only and "From this panel" not in read_only
