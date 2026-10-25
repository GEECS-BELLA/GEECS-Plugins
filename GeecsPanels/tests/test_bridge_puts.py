"""Puts: the three gates, the hazard confirm, the stale-page refusal, the journal (Appendix A rules 6, 8, 11, 12)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import pytest

from geecs_panels.bridge.journal import Journal
from geecs_panels.bridge.puts import Puts
from geecs_panels.bridge.stream import DeviceStreams
from geecs_panels.contracts import PutRequest
from tests._bridge import (
    CAM,
    MOTOR,
    SUPPLY,
    Clock,
    FakeCA,
    devices,
    healthy_axis,
    healthy_supply,
)

OPERATOR = "S. Barber"


class FakeIO:
    """The put transport: records puts, answers reads from a table."""

    def __init__(self) -> None:
        self.puts: list[tuple[str, object, bool]] = []
        self.values: dict[str, object] = {}
        self.enums: dict[str, list[str]] = {}
        self.fail: Optional[Exception] = None
        self.gets: list[str] = []
        self.alarm_fails = False

    async def put(self, pv: str, value: object, wait: bool) -> None:
        if self.fail is not None:
            raise self.fail
        self.puts.append((pv, value, wait))

    async def get(self, pv: str, as_string: bool = False) -> object:
        self.gets.append(pv)
        record, _, field = pv.rpartition(".")
        if field and "." in record.rsplit(":", 1)[-1]:
            # rec.FIELD.SEVR: a field of a field is no CA channel (it times out).
            raise TimeoutError(f"{pv}: no such channel")
        if self.alarm_fails:
            raise TimeoutError(f"{pv}: timed out")
        if pv.endswith(".SEVR"):
            return self.values.get(pv, "NO_ALARM")
        if pv.endswith(".STAT"):
            return self.values.get(pv, "NO_ALARM")
        return self.values[pv]

    async def choices(self, pv: str) -> list[str]:
        return self.enums[pv]


class Bench:
    """A bridge with one device's stream open, over fakes."""

    def __init__(
        self,
        name: str,
        *,
        writable: bool = True,
        panels: Optional[dict] = None,
        listening: bool = True,
        journal: Optional[Journal] = None,
    ) -> None:
        self.ca, self.clock, self.io = FakeCA(), Clock(), FakeIO()
        self.streams = DeviceStreams(
            devices(panels),
            self.ca.subscribe,
            clock=self.clock.mono,
            wall=self.clock.wall,
        )
        self.hub = self.streams.hub(name, run=False)
        if listening:
            self.hub.listeners.add(asyncio.Queue())
        self.hub.tick()
        self.journal = journal or Journal(wall=self.clock.wall)
        self.puts = Puts(
            self.streams,
            self.io,
            self.journal,
            writable=writable,
            settle_s=0.3,
            poll_s=0.01,
            clock=self.clock.mono,
        )

    def put(self, device: str, role: str, value: object, **kw: object) -> tuple:
        request = PutRequest(
            device=device, role=role, value=value, operator=OPERATOR, **kw
        )
        return asyncio.run(self.puts.put(request))


def _supply(**kw: object) -> Bench:
    bench = Bench("PS1", **kw)
    healthy_supply(bench.ca)
    return bench


# ---- accepted -----------------------------------------------------------------------


def test_a_setpoint_inside_its_limits_is_put_waited_on_and_its_alarm_read() -> None:
    bench = _supply()
    status, result = bench.put("PS1", "current_sp", 0.5)
    assert status == 200 and result.outcome == "accepted"
    assert result.value == 0.5 and result.severity == "NO_ALARM"
    assert bench.io.puts == [(SUPPLY + "current_ramp:SP", 0.5, True)]
    (entry,) = bench.journal.last(10)
    assert entry.id == result.journal_id
    assert (entry.operator, entry.device, entry.role, entry.value) == (
        OPERATOR,
        "PS1",
        "current_sp",
        0.5,
    )


# ---- gate 1: denied ----------------------------------------------------------------------


def test_a_read_only_bridge_denies_and_journals_it() -> None:
    bench = _supply(writable=False)
    status, result = bench.put("PS1", "current_sp", 0.5)
    assert (status, result.outcome) == (403, "denied")
    assert bench.io.puts == []
    assert bench.journal.last(1)[0].outcome == "denied"


def test_a_read_only_role_or_an_unknown_one_is_denied() -> None:
    bench = _supply()
    assert bench.put("PS1", "current", 0.5)[1].outcome == "denied"
    assert bench.put("PS1", "nope", 0.5)[1].outcome == "denied"


def test_not_permitted_on_this_device_is_denied_before_the_value_is_judged() -> None:
    """Rule 8: 403 before 422 — an out-of-range value on a forbidden role is denied, not refused."""
    bench = Bench(
        "PS1",
        panels={"PS1": {"family": "fastps", "prefix": SUPPLY, "allow": ["output"]}},
    )
    healthy_supply(bench.ca)
    status, result = bench.put("PS1", "current_sp", 99.0)
    assert (status, result.outcome) == (403, "denied")
    assert "not permitted" in result.message


def test_a_hazard_role_needs_the_confirm() -> None:
    bench = _supply()
    status, result = bench.put("PS1", "output_sp", 1)
    assert (status, result.outcome) == (403, "denied")
    assert "hazard" in result.message and bench.io.puts == []
    status, result = bench.put("PS1", "output_sp", 1, confirm=True)
    assert (status, result.outcome) == (200, "accepted")


def test_a_device_hazard_from_panel_data_needs_the_confirm_too() -> None:
    bench = Bench(
        "PS1",
        panels={
            "PS1": {"family": "fastps", "prefix": SUPPLY, "hazard": ["current_sp"]}
        },
    )
    healthy_supply(bench.ca)
    assert bench.put("PS1", "current_sp", 0.5)[1].outcome == "denied"


def test_a_page_whose_stream_is_closed_cannot_write() -> None:
    bench = _supply(listening=False)
    status, result = bench.put("PS1", "current_sp", 0.5)
    assert (status, result.outcome) == (409, "denied")
    assert "no stream" in result.message


def test_a_stale_stream_cannot_write() -> None:
    bench = _supply()
    bench.clock.advance(10.0)  # the hub has not ticked since
    status, result = bench.put("PS1", "current_sp", 0.5)
    assert (status, result.outcome) == (409, "denied")
    assert "stale" in result.message


def test_a_disconnected_record_cannot_be_written() -> None:
    bench = _supply()
    bench.ca.drop(SUPPLY + "current_ramp:SP")
    status, result = bench.put("PS1", "current_sp", 0.5)
    assert (status, result.outcome) == (409, "denied")


def _camera(lock: Optional[str] = None) -> Bench:
    bench = Bench("Cam")
    bench.ca.post(CAM + "cam1:AcquireTime_RBV", 0.01)
    if lock is not None:
        bench.ca.post(CAM + "RUN:Owner", "worker:123 @10:00")
        bench.ca.post(CAM + "RUN:Lock", lock)
    return bench


def test_a_locked_role_is_denied_while_the_lock_is_unread() -> None:
    """Rule 6: an unreadable lock refuses acquisition writes."""
    status, result = _camera().put("Cam", "exposure", 0.02)
    assert (status, result.outcome) == (423, "denied")
    assert "cannot be read" in result.message


def test_a_locked_role_is_denied_naming_the_run_while_held() -> None:
    status, result = _camera("Locked").put("Cam", "exposure", 0.02)
    assert (status, result.outcome) == (423, "denied")
    assert "worker:123" in result.message


def test_a_locked_role_is_written_while_the_lock_is_free() -> None:
    bench = _camera("Free")
    status, result = bench.put("Cam", "exposure", 0.02)
    assert (status, result.outcome) == (200, "accepted")
    assert bench.io.puts == [(CAM + "cam1:AcquireTime", 0.02, True)]


def test_an_unlocked_role_ignores_the_lock() -> None:
    bench = _camera("Locked")
    bench.ca.post(CAM + "OVER1:1:CenterX_RBV", 10)
    assert bench.put("Cam", "crosshair1_x", 20)[1].outcome == "accepted"


# ---- gate 2: refused by the bridge --------------------------------------------------------


@pytest.mark.parametrize(
    "role, value",
    [
        ("exposure", 61.0),  # the family's limits
        ("crosshair1_x", 1.5),  # integer
        ("crosshair1_use", 2),  # a bit
        ("crosshair1_shape", "Circle"),  # the family's choices
        ("background_saving", 0),  # a press takes 1
        ("crosshair1_name", "x" * 40),  # text length
        ("exposure", True),  # a flag is no number
    ],
)
def test_the_familys_bounds_refuse(role: str, value: object) -> None:
    bench = _camera("Free")
    for suffix in ("OVER1:1:CenterX_RBV", "OVER1:1:Use_RBV", "OVER1:1:Shape_RBV",
                   "PROC1:SaveBackground_RBV", "OVER1:1:Name_RBV"):  # fmt: skip
        bench.ca.post(CAM + suffix, 0)
    status, result = bench.put("Cam", role, value)
    assert (status, result.outcome) == (422, "refused"), result.message
    assert bench.io.puts == []


def test_a_supply_setpoint_past_its_drive_limits_is_refused() -> None:
    """Rule 11: an ao clamps and writes DRVH, so the bridge is the gate that refuses."""
    bench = _supply()
    status, result = bench.put("PS1", "current_sp", 1.5)
    assert (status, result.outcome) == (422, "refused")
    assert "drive limits" in result.message
    assert bench.io.puts == []


def test_a_supply_setpoint_is_refused_while_its_limits_are_unread() -> None:
    bench = _supply()
    bench.ca.drop(SUPPLY + "current_ramp:SP.DRVH")
    status, result = bench.put("PS1", "current_sp", 0.5)
    assert (status, result.outcome) == (422, "refused")
    assert "not been read" in result.message


def test_a_supply_setpoint_at_an_invalid_hihi_is_refused() -> None:
    bench = _supply()
    bench.ca.post(SUPPLY + "current_ramp:SP.HIHI", 0.4)
    assert bench.put("PS1", "current_sp", 0.5)[1].outcome == "refused"


def _stage() -> Bench:
    bench = Bench("Stage")
    healthy_axis(bench.ca, "m1")
    return bench


def test_a_motor_target_outside_the_soft_limits_is_refused() -> None:
    bench = _stage()
    request = {"axis": "m1"}
    status, result = bench.put("Stage", "val", 20.0, **request)
    assert (status, result.outcome) == (422, "refused")
    assert "soft limits" in result.message


def test_a_per_axis_put_names_an_axis_the_device_has() -> None:
    bench = _stage()
    assert bench.put("Stage", "val", 1.0)[1].outcome == "denied"
    assert bench.put("Stage", "val", 1.0, axis="m9")[1].outcome == "denied"


def _scope() -> Bench:
    bench = Bench(
        "Scope", panels={"Scope": {"family": "scope_contract", "prefix": "b:sc:"}}
    )
    bench.ca.post("b:sc:Resolution_RBV", "8bit")
    return bench


def test_an_enum_without_family_choices_is_checked_against_the_records_own() -> None:
    bench = _scope()
    bench.io.enums["b:sc:Resolution"] = ["8bit", "12bit"]
    assert bench.put("Scope", "resolution", "12bit")[1].outcome == "accepted"
    status, result = bench.put("Scope", "resolution", "16bit")
    assert (status, result.outcome) == (422, "refused")
    assert "8bit, 12bit" in result.message


def test_an_enum_whose_choices_cannot_be_read_is_refused() -> None:
    """Fail closed: no choices read, no write."""
    status, result = _scope().put("Scope", "resolution", "12bit")
    assert (status, result.outcome) == (422, "refused")
    assert "cannot be read" in result.message


# ---- gate 3: the IOC ------------------------------------------------------------------------


def test_a_put_the_ioc_rejects_is_refused_never_a_success() -> None:
    bench = _supply()
    bench.io.fail = RuntimeError("Write access denied")
    status, result = bench.put("PS1", "current_sp", 0.5)
    assert (status, result.outcome) == (502, "refused")
    assert "Write access denied" in result.message


def test_a_record_left_invalid_after_the_put_is_refused() -> None:
    """Rule 11: a StreamDevice #NAK leaves the record INVALID: refused, never accepted."""
    bench = _supply()
    bench.io.values[SUPPLY + "current_ramp:SP.SEVR"] = "INVALID"
    bench.io.values[SUPPLY + "current_ramp:SP.STAT"] = "WRITE"
    status, result = bench.put("PS1", "current_sp", 0.5)
    assert (status, result.outcome) == (502, "refused")
    assert "INVALID WRITE" in result.message


def test_a_move_is_dispatched_without_waiting() -> None:
    """Rule 12: put-callback completes at the motion's end; the stream shows the move."""
    bench = _stage()
    status, result = bench.put("Stage", "val", 2.0, axis="m1")
    assert (status, result.outcome) == (200, "accepted")
    assert bench.io.puts == [(MOTOR + "m1.VAL", 2.0, False)]
    assert "dmov" in result.message


def test_a_setting_is_waited_on_and_its_records_alarm_read() -> None:
    """A field's put reads the record's alarm (m1.SEVR), never m1.VELO.SEVR, which is no channel."""
    bench = _stage()
    status, result = bench.put("Stage", "velo", 1.5, axis="m1")
    assert bench.io.puts == [(MOTOR + "m1.VELO", 1.5, True)]
    assert (status, result.outcome, result.severity) == (200, "accepted", "NO_ALARM")
    assert bench.io.gets == [MOTOR + "m1.SEVR", MOTOR + "m1.STAT"]


def test_stop_is_accepted_with_the_records_alarm() -> None:
    bench = _stage()
    status, result = bench.put("Stage", "stop", 1, axis="m1")
    assert (status, result.outcome) == (200, "accepted")
    assert bench.io.puts == [(MOTOR + "m1.STOP", 1, True)]


def test_a_put_that_took_is_never_journaled_refused_for_an_unread_alarm() -> None:
    bench = _supply()
    bench.io.alarm_fails = True
    status, result = bench.put("PS1", "current_sp", 0.5)
    assert (status, result.outcome, result.severity) == (200, "accepted", None)
    assert "could not be read" in result.message
    assert bench.journal.last(1)[0].outcome == "accepted"


def test_an_acquire_settles_on_its_readback() -> None:
    bench = _camera("Free")
    bench.ca.post(CAM + "cam1:Acquire_RBV", "Done")
    bench.io.values[CAM + "cam1:Acquire_RBV"] = 1
    status, result = bench.put("Cam", "acquire", 1)
    assert (status, result.outcome) == (200, "accepted")
    assert bench.io.puts == [(CAM + "cam1:Acquire", 1, False)]


def test_an_acquire_that_never_settles_is_refused() -> None:
    bench = _camera("Free")
    bench.ca.post(CAM + "cam1:Acquire_RBV", "Done")
    bench.io.values[CAM + "cam1:Acquire_RBV"] = 0
    status, result = bench.put("Cam", "acquire", 1)
    assert (status, result.outcome) == (502, "refused")


# ---- the run lock's release ----------------------------------------------------------------------


def test_releasing_the_lock_writes_it_free_and_nothing_else_journaled() -> None:
    bench = _camera("Locked")
    status, result = asyncio.run(bench.puts.release_lock("Cam", OPERATOR))
    assert (status, result.outcome) == (200, "accepted")
    assert bench.io.puts == [(CAM + "RUN:Lock", 0, True)]
    entry = bench.journal.last(1)[0]
    assert (entry.role, entry.operator, entry.value) == ("lock", OPERATOR, 0)


def test_a_device_without_a_lock_has_none_to_release() -> None:
    bench = _supply()
    status, result = asyncio.run(bench.puts.release_lock("PS1", OPERATOR))
    assert (status, result.outcome) == (403, "denied")


# ---- the journal ----------------------------------------------------------------------------------


def test_the_journal_survives_a_restart_newest_first_types_kept(tmp_path: Path) -> None:
    path = tmp_path / "journal.db"
    bench = _supply(journal=Journal(path))
    bench.put("PS1", "current_sp", 0.5)
    bench.put("PS1", "current_sp", 1.5)
    bench.put("PS1", "output_sp", True, confirm=True)
    bench.journal.close()
    entries = Journal(path).last(10)
    assert [e.outcome for e in entries] == ["accepted", "refused", "accepted"]
    assert [e.value for e in entries] == [True, 1.5, 0.5]
    assert entries[0].id > entries[1].id > entries[2].id >= 1
    assert len(Journal(path).last(2)) == 2


@pytest.mark.parametrize(
    "role, value",
    [
        ("exposure", 61.0),  # limits
        ("crosshair1_x", 70000),  # limits on a whole number
        ("transform", "Rot45"),  # an enum's choices
        ("crosshair1_shape", "Circle"),  # the family's own choices
    ],
)
def test_the_familys_bounds_hold_for_a_template_without_a_rule_of_its_own(
    monkeypatch: pytest.MonkeyPatch, role: str, value: object
) -> None:
    """A family whose template has no check of its own is still bounded by its file."""
    import geecs_panels.bridge.puts as puts_module
    from geecs_panels.bridge.rules import NO_RULES

    monkeypatch.setattr(puts_module, "rules_for", lambda template: NO_RULES)
    bench = _camera("Free")
    for suffix in ("OVER1:1:CenterX_RBV", "OVER1:1:Shape_RBV", "TRANS1:Type"):
        bench.ca.post(CAM + suffix, 0)
    bench.io.enums[CAM + "TRANS1:Type"] = ["None", "Rot90"]
    status, result = bench.put("Cam", role, value)
    assert (status, result.outcome) == (422, "refused"), result.message
    assert bench.io.puts == []
