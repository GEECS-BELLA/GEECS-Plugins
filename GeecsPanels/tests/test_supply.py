"""The power-supply panel: the roles, the allow-list, the state words, the feed, the routes — no CA, no IOC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme import STATES

from geecs_panels.app import PANELS, create_app
from geecs_panels.feed import WriteFailed
from geecs_panels.power import (
    ACTIONS,
    NotAllowed,
    NotWritable,
    POWER_KIND,
    REQUIRED_ROLES,
    allowed_actions,
    check_roles,
    check_write,
    monitored,
    supply_state,
)
from geecs_panels.registry import Device, RegistryError, load_bench
from geecs_panels.supply_feed import SupplyFeed

#: fpstest1's roles (the FAST-PS): the ramped setpoint, a loop mode, two interlocks.
FPS_ROLES = {
    "current_sp": "current_ramp:SP", "current": "current", "voltage": "voltage",
    "output_sp": "enable_output:SP", "output": "enable_output", "reset": "reset:SP",
    "loop_mode": "loop_mode",
}  # fmt: skip
FPS_BITS = {
    "fault": "fault",
    "interlock_1": "interlock_1",
    "interlock_2": "interlock_2",
}

#: The FAST-PS as the session permits it: current, output and reset only.
FPS = Device(
    "U_ChicaneInner",
    POWER_KIND,
    "undulator:fpstest1:",
    allow=frozenset({"current", "output", "reset"}),
    roles=tuple(FPS_ROLES.items()),
    bits=tuple(FPS_BITS.items()),
)
#: An Easy-Driver with no allow list: every action it has a role for.
EZ = Device(
    "U_EasyDriver",
    POWER_KIND,
    "undulator:eztest1:",
    roles=tuple({k: v for k, v in FPS_ROLES.items() if k != "loop_mode"}.items()),
    bits=(("fault", "fault"), ("interlock", "interlock")),
)

#: At rest: output off, 0 A, drive limits 0..1 A, no alarm limits armed, nothing tripped.
AT_REST = {
    "current_sp": 0.0, "current_sp.drvh": 1.0, "current_sp.drvl": 0.0,
    "current_sp.hihi": 0.0, "current_sp.lolo": 0.0,
    "current_sp.hhsv": "NO_ALARM", "current_sp.llsv": "NO_ALARM",
    "current_sp.egu": "A", "current_sp.prec": 4, "current_sp.sevr": "NO_ALARM",
    "current_sp.stat": "NO_ALARM", "current": 0.0, "current.egu": "A",
    "current.prec": 4, "current.sevr": "NO_ALARM", "current.stat": "NO_ALARM",
    "voltage": 0.0, "voltage.egu": "V", "voltage.prec": 3, "output_sp": 0,
    "output_sp.sevr": "NO_ALARM", "output_sp.stat": "NO_ALARM", "output": 0,
    "loop_mode": "current", "bit.fault": 0, "bit.interlock_1": 0, "bit.interlock_2": 0,
    "bit.interlock": 0,
}  # fmt: skip


class _Sub:
    def close(self) -> None:
        pass


def _feed(device: Device = FPS, **values) -> SupplyFeed:
    feed = SupplyFeed(
        device.prefix, dict(device.roles), dict(device.bits), lambda pv, s, cb: _Sub()
    )
    feed.start()
    for key, value in {**AT_REST, **values}.items():
        if key in feed.connected:
            feed.update(key, value)
    return feed


class FakeSupplies:
    def __init__(self, refuse: bool = False, **values) -> None:
        self.feeds: dict[str, SupplyFeed] = {}
        self.puts: list[tuple[str, object]] = []
        self.refuse = refuse
        self.values = values

    def feed(self, device: Device) -> SupplyFeed:
        if device.name not in self.feeds:
            self.feeds[device.name] = _feed(device, **self.values)
        return self.feeds[device.name]

    async def put(self, device: Device, role: str, value: object) -> None:
        if self.refuse:
            raise WriteFailed(f"{device.prefix}{role}: the supply did not accept it")
        self.puts.append((role, value))


def _client(supplies: Optional[FakeSupplies] = None, **kwargs) -> TestClient:
    return TestClient(
        create_app(
            "Undulator",
            [FPS, EZ],
            object(),
            supplies=supplies or FakeSupplies(),
            **kwargs,
        )
    )


# ---- the contract -------------------------------------------------------------


def test_every_role_its_fields_and_every_bit_are_monitored_but_never_the_reset() -> (
    None
):
    keys = monitored(FPS_ROLES, FPS_BITS)
    assert keys["current_sp"] == ("current_ramp:SP", False)
    assert keys["current_sp.drvh"] == ("current_ramp:SP.DRVH", False)
    assert keys["current_sp.sevr"] == ("current_ramp:SP.SEVR", True)
    assert keys["loop_mode"] == ("loop_mode", True)
    assert keys["bit.interlock_2"] == ("interlock_2", False)
    assert not any(suffix.startswith("reset") for suffix, _ in keys.values())


@pytest.mark.parametrize(
    "roles, allow, message",
    [
        ({k: v for k, v in FPS_ROLES.items() if k != "reset"}, None, "missing"),
        ({**FPS_ROLES, "frequency": "f"}, None, "unknown role"),
        (FPS_ROLES, ["current", "melt"], "unknown power-supply action"),
        (FPS_ROLES, ["voltage"], "need a role"),
    ],
)
def test_a_role_map_names_every_required_role_and_known_actions(
    roles, allow, message
) -> None:
    with pytest.raises(ValueError, match=message):
        check_roles(roles, allow)
    check_roles(FPS_ROLES, ["current", "output", "reset"])


def test_the_actions_a_device_has_are_its_roles_within_its_allow() -> None:
    assert allowed_actions(FPS_ROLES, None) == ["current", "output", "reset"]
    assert allowed_actions({**FPS_ROLES, "voltage_sp": "voltage:SP"}, None) == list(
        ACTIONS
    )
    assert allowed_actions(FPS_ROLES, {"output"}) == ["output"]


def test_a_setpoint_inside_the_drive_limits_goes_to_its_role() -> None:
    assert check_write("current", 0.5, AT_REST, FPS_ROLES) == ("current_sp", 0.5)
    assert check_write("output", 1, AT_REST, FPS_ROLES) == ("output_sp", 1)
    assert check_write("reset", 1, AT_REST, FPS_ROLES) == ("reset", 1)


@pytest.mark.parametrize(
    "values, message",
    [
        ({}, "outside the drive limits 0 to 1 A"),
        ({"current_sp.drvh": None}, "have not been read"),
        ({"current_sp.drvh": 0.0, "current_sp.drvl": 0.0}, "no drive limits"),
        (
            {
                "current_sp.drvh": 2.5,
                "current_sp.hihi": 1.5,
                "current_sp.hhsv": "INVALID",
            },
            "reaches the IOC's HIHI 1.5 A",
        ),
    ],
)
def test_a_setpoint_is_refused_outside_unread_or_absent_drive_limits(
    values, message
) -> None:
    with pytest.raises(NotWritable, match=message):
        check_write("current", 2.0, {**AT_REST, **values}, FPS_ROLES)


@pytest.mark.parametrize(
    "action, value",
    [("current", float("nan")), ("current", True), ("output", 2), ("output", 0.5),
     ("reset", 0), ("loop_mode", 1), ("voltage", 1.0)],
)  # fmt: skip
def test_everything_else_is_refused(action, value) -> None:
    with pytest.raises(NotWritable):
        check_write(action, value, AT_REST, FPS_ROLES)


def test_hihi_and_lolo_refuse_only_with_an_invalid_severity() -> None:
    limited = {**AT_REST, "current_sp.drvh": 2.0, "current_sp.drvl": -2.0,
               "current_sp.hihi": 1.5, "current_sp.lolo": -1.5}  # fmt: skip
    assert (
        check_write("current", 1.8, {**limited, "current_sp.hhsv": "MAJOR"}, FPS_ROLES)[
            1
        ]
        == 1.8
    )
    assert (
        check_write(
            "current", 1.4, {**limited, "current_sp.hhsv": "INVALID"}, FPS_ROLES
        )[1]
        == 1.4
    )
    with pytest.raises(NotWritable, match="LOLO"):
        check_write(
            "current", -1.5, {**limited, "current_sp.llsv": "INVALID"}, FPS_ROLES
        )


#: The IOC's refusal armed just inside the drive limits (the reviewer's case).
ARMED = {"current_sp.drvh": 6.01, "current_sp.drvl": -6.01,
         "current_sp.hihi": 6.0001, "current_sp.lolo": -6.0001,
         "current_sp.hhsv": "INVALID", "current_sp.llsv": "INVALID"}  # fmt: skip


@pytest.mark.parametrize(
    "dropped, message",
    [
        ("current_sp.hihi", "HIHI"),
        ("current_sp.hhsv", "HHSV"),
        ("current_sp.lolo", "LOLO"),
        ("current_sp.llsv", "LLSV"),
    ],
)
def test_a_setpoint_is_refused_while_an_alarm_bound_or_severity_is_unread(
    dropped, message
) -> None:
    """Like the drive limits, the HIHI/LOLO gate fails closed: a dropped channel refuses."""
    feed = _feed(**ARMED)
    assert check_write("current", 6.0, feed.current(), FPS_ROLES)[1] == 6.0
    feed.update(dropped, None)
    for value in (6.005, 0.5):  # past the IOC's HIHI, and well inside it
        with pytest.raises(NotWritable, match=f"{message} has not been read"):
            check_write("current", value, feed.current(), FPS_ROLES)


@pytest.mark.parametrize(
    "severity, bound, value",
    [("hhsv", "hihi", 6.005), ("llsv", "lolo", -6.005)],
)
def test_an_alarm_bound_whose_severity_is_not_invalid_need_not_be_read(
    severity, bound, value
) -> None:
    feed = _feed(**{**ARMED, f"current_sp.{severity}": "MAJOR"})
    feed.update(f"current_sp.{bound}", None)
    assert check_write("current", value, feed.current(), FPS_ROLES)[1] == value


def test_an_action_the_device_does_not_permit_is_not_allowed() -> None:
    with pytest.raises(NotAllowed):
        check_write("output", 1, AT_REST, FPS_ROLES, {"current", "reset"})


@pytest.mark.parametrize(
    "values, want",
    [
        ({}, ("ok", "output off")),
        ({"output": 1}, ("running", "output on")),
        ({"bit.interlock_1": 1, "output": 1}, ("failed", "interlock 1")),
        ({"current.sevr": "INVALID", "current.stat": "TIMEOUT"}, ("failed", "no reply (TIMEOUT)")),
        ({"current_sp.sevr": "INVALID", "current_sp.stat": "CALC"}, ("degraded", "setpoint refused (CALC)")),
        ({"output": None}, ("unknown", "not connected: output")),
        ({"current_sp.sevr": "INVALID", "current_sp.stat": "UDF"}, ("ok", "output off")),
    ],
)  # fmt: skip
def test_the_state_word_is_judged_worst_first(values, want) -> None:
    assert supply_state({**AT_REST, **values}, True, list(FPS_BITS)) == want
    assert want[0] in STATES


def test_nothing_connected_is_unknown_and_a_dropped_channel_shows_no_value() -> None:
    feed = SupplyFeed(FPS.prefix, FPS_ROLES, FPS_BITS, lambda pv, s, cb: _Sub())
    feed.start()
    assert feed.snapshot()["state"] == "unknown"
    feed = _feed()
    feed.update("current", None)
    snap = feed.snapshot()
    assert snap["values"]["current"] is None and "current" in snap["missing"]
    assert snap["state"] == "unknown" and snap["reachable"]


def test_the_reviewers_case_is_unknown_not_ok() -> None:
    """Output read, the readback, its alarm and the interlock not: not "ok / output off"."""
    state, status = supply_state({"output": 0}, True, ["interlock"])
    assert state == "unknown"
    assert all(k in status for k in ("current", "current.sevr", "bit.interlock"))


@pytest.mark.parametrize(
    "dropped",
    ["current", "current.sevr", "current_sp.sevr", "output_sp.sevr", "output",
     "bit.fault", "bit.interlock_2"],
)  # fmt: skip
def test_a_channel_the_state_is_judged_from_dropped_is_unknown(dropped) -> None:
    feed = _feed()
    assert feed.snapshot()["state"] == "ok"
    feed.update(dropped, None)
    snap = feed.snapshot()
    assert snap["reachable"] and (snap["state"], snap["status"]) == (
        "unknown",
        f"not connected: {dropped}",
    )


def test_an_invalid_command_whose_status_is_unread_is_unknown_not_refused() -> None:
    """UDF (never written) is not a refusal: until STAT is read, it cannot be told apart."""
    feed = _feed(**{"current_sp.sevr": "INVALID", "current_sp.stat": "UDF"})
    assert feed.snapshot()["state"] == "ok"
    feed.update("current_sp.stat", None)
    assert (feed.snapshot()["state"], feed.snapshot()["status"]) == (
        "unknown",
        "not connected: current_sp.stat",
    )


def test_a_tripped_supply_whose_fault_channel_drops_is_not_ok() -> None:
    feed = _feed(**{"bit.fault": 1})
    assert feed.snapshot()["state"] == "failed"
    feed.update("bit.fault", None)
    assert feed.snapshot()["state"] == "unknown"


def test_a_known_failure_outranks_an_unread_channel() -> None:
    feed = _feed(**{"bit.interlock_1": 1})
    feed.update("current", None)
    assert feed.snapshot()["state"] == "failed"


def test_a_supply_with_a_voltage_setpoint_waits_for_its_alarm_too() -> None:
    roles = {**FPS_ROLES, "voltage_sp": "voltage:SP"}
    device = Device(**{**FPS.__dict__, "roles": tuple(roles.items())})
    feed = _feed(device)
    assert feed.snapshot()["status"] == "not connected: voltage_sp.sevr"
    feed.update("voltage_sp.sevr", "NO_ALARM")
    assert feed.snapshot()["state"] == "ok"


# ---- the bench file -----------------------------------------------------------

BENCH = """\
devices:
  U_ChicaneInner:
    kind: power_supply
    prefix: "undulator:fpstest1:"
    roles: {current_sp: "current_ramp:SP", current: current, voltage: voltage,
            output_sp: "enable_output:SP", output: enable_output, reset: "reset:SP",
            loop_mode: loop_mode}
    bits: {fault: fault, interlock_1: interlock_1}
    allow: [current, output, reset]
"""


def test_a_bench_supply_carries_its_roles_bits_and_actions(tmp_path: Path) -> None:
    path = tmp_path / "bench.yaml"
    path.write_text(BENCH)
    (fps,) = load_bench(path)
    assert fps.kind == POWER_KIND and PANELS[fps.kind] == "supply"
    assert dict(fps.roles)["output_sp"] == "enable_output:SP"
    assert [label for label, _ in fps.bits] == ["fault", "interlock_1"]
    assert fps.allow == {"current", "output", "reset"}
    path.write_text(
        BENCH.replace("allow: [current, output, reset]", "allow: [voltage]")
    )
    with pytest.raises(RegistryError, match="need a role"):
        load_bench(path)
    assert set(REQUIRED_ROLES) <= set(dict(fps.roles))


# ---- the routes ---------------------------------------------------------------


def test_the_landing_page_links_the_supplies() -> None:
    html = _client().get("/").text
    assert (
        'href="/supply/U_ChicaneInner"' in html
        and 'href="/supply/U_EasyDriver"' in html
    )


def test_the_page_draws_the_permitted_controls_and_the_bits() -> None:
    html = _client().get("/supply/U_ChicaneInner").text
    for drawn in ('data-action="current"', 'data-press="output"', 'data-press="reset"',
                  'data-bit="interlock_2"', "undulator:fpstest1:enable_output:SP"):  # fmt: skip
        assert drawn in html
    assert 'data-action="voltage"' not in html
    assert "From this panel: current, output, reset." in html
    assert "Held for this device" not in html


def test_a_restricted_supply_draws_no_control_it_does_not_permit() -> None:
    held = Device(**{**FPS.__dict__, "name": "Held", "allow": frozenset({"output"})})
    client = TestClient(
        create_app("Undulator", [held], object(), supplies=FakeSupplies())
    )
    html = client.get("/supply/Held").text
    assert 'data-action="current"' not in html and 'data-press="reset"' not in html
    assert 'data-k="current-sp"' in html  # the setpoint still reads, as a value
    assert "Held for this device: current, reset." in html
    r = client.post("/supply/Held/put", json={"action": "current", "value": 0.5})
    assert r.status_code == 403


def test_the_stream_carries_the_values_bits_and_state() -> None:
    with _client(FakeSupplies(output=1)).stream(
        "GET", "/supply/U_ChicaneInner/events?once=1"
    ) as r:
        body = "".join(r.iter_text())
    data = json.loads(body.split("data: ", 1)[1])
    assert data["state"] == "running" and data["status"] == "output on"
    assert data["values"]["current_sp.drvh"] == 1.0
    assert [b["label"] for b in data["bits"]] == ["fault", "interlock_1", "interlock_2"]


def test_a_put_goes_through_the_allow_list_to_the_role() -> None:
    supplies = FakeSupplies()
    client = _client(supplies)
    for action, value in (
        ("output", 1),
        ("current", 0.5),
        ("current", 0),
        ("output", 0),
        ("reset", 1),
    ):
        r = client.post(
            "/supply/U_ChicaneInner/put", json={"action": action, "value": value}
        )
        assert r.status_code == 200, r.text
    assert supplies.puts == [
        ("output_sp", 1), ("current_sp", 0.5), ("current_sp", 0.0), ("output_sp", 0), ("reset", 1)
    ]  # fmt: skip
    far = client.post(
        "/supply/U_ChicaneInner/put", json={"action": "current", "value": 5}
    )
    assert far.status_code == 422 and "drive limits" in far.json()["detail"]
    loop = client.post(
        "/supply/U_ChicaneInner/put", json={"action": "loop_mode", "value": 1}
    )
    assert loop.status_code == 422
    assert len(supplies.puts) == 5


def test_the_supply_refusing_is_a_502_and_read_only_a_403() -> None:
    put = {"action": "reset", "value": 1}
    assert (
        _client(FakeSupplies(refuse=True))
        .post("/supply/U_ChicaneInner/put", json=put)
        .status_code
        == 502
    )
    assert (
        _client(writable=False).post("/supply/U_ChicaneInner/put", json=put).status_code
        == 403
    )


def test_a_supply_is_not_a_positioner_and_unknown_is_404() -> None:
    from tests.test_motor import STAGE, FakeMotors

    client = TestClient(
        create_app(
            "Undulator",
            [FPS, STAGE],
            object(),
            supplies=FakeSupplies(),
            motors=FakeMotors(),
        )
    )
    assert client.get("/supply/NOPE").status_code == 404
    assert client.get("/supply/SimMotor1").status_code == 404
    assert client.get("/motor/U_ChicaneInner").status_code == 404
    assert client.get("/supply/U_ChicaneInner").status_code == 200


def test_the_live_put_reads_the_alarm_after_the_supply_answers(monkeypatch) -> None:
    import asyncio

    from geecs_panels import ca

    calls: list[tuple] = []

    async def caput(pv, value, wait, timeout):
        calls.append(("put", pv, value, wait))

    async def caget(pv, datatype=None, timeout=None):
        calls.append(("get", pv))
        return "INVALID" if pv.endswith(".SEVR") else "CALC"

    monkeypatch.setattr(ca.aioca, "caput", caput)
    monkeypatch.setattr(ca.aioca, "caget", caget)
    with pytest.raises(WriteFailed, match="did not accept it"):
        asyncio.run(ca.LiveSupplies().put(FPS, "current_sp", 0.5))
    assert calls[0] == ("put", "undulator:fpstest1:current_ramp:SP", 0.5, True)
    assert ("get", "undulator:fpstest1:current_ramp:SP.SEVR") in calls
