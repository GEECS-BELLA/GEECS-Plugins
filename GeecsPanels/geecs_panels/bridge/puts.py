"""Puts with identity: the three gates of ``Planning/epics_estate.md`` §2.3, and the journal.

``POST /api/put`` takes a :class:`~geecs_panels.contracts.PutRequest` (a
device, a role, an axis for a per-axis family, a value, the operator, the
hazard confirm) and answers a :class:`~geecs_panels.contracts.PutResult`.
Every answer, whatever its outcome, is journaled
(:class:`~geecs_panels.bridge.journal.Journal`) before it is sent.

The gates, in order; the first that fails answers:

1. **Denied: the write itself** (the bridge's contextual allow-list,
   Appendix A rule 8: not permitted on this device comes *before* any
   check of the value).
   - a read-only bridge (``--read-only``);
   - a role the device does not have, or one no page may write; an axis
     it does not have;
   - a role outside the device's permitted actions (``allow``);
   - a ``confirm: hazard`` role without the operator's confirm;
   - **a stale page cannot write**: no stream of the device is open, its
     hub has not ticked within ``stale_after``, or the written role's
     record (for a command, any of the axis's records) is not connected;
   - a ``locked`` role while the run lock is held **or cannot be read**
     (rule 6; the IOC's access security refuses it too).
2. **Refused: the value** (still the bridge's gate, failing closed).
   - by the family: the role's type (a bit 0/1, a command 1, an enum one
     of its choices: the family's, else the record's state names as the
     stream carries them, refused while unread),
     ``integer``, ``limits``;
   - by the template's rule (:attr:`Rules.check`): a motor target inside
     the soft limits, a supply setpoint inside its drive limits and short
     of an INVALID HIHI/LOLO (rule 11: an ``ao`` clamps and writes past
     ``DRVH``, so this is the gate that refuses), a camera or scope
     setting within its table. Context that is not read refuses.
3. **Refused: the IOC** (rule 11). The put waits for the record and then
   reads its alarm: a put CA rejects (access security), or a record left
   INVALID (a StreamDevice ``#NAK``), is ``refused``, never a success.
   A write that completes only when its work ends (``settled_by``) is
   dispatched without waiting (rule 12): settled by its own readback
   (an acquire: until it reads the value, or a frame counter advances),
   or, for a motion, answered at dispatch (the stream shows the move and
   its done flag).

HTTP status, beside the body: accepted 200; denied 403 (the run lock
423, a stale stream 409); refused 422 by the bridge, 502 by the IOC.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol

from geecs_schemas.epics.family import Role

from geecs_panels.bridge.catalog import BridgeDevice
from geecs_panels.bridge.journal import Journal
from geecs_panels.bridge.rules import rules_for
from geecs_panels.bridge.stream import DeviceHub, DeviceStreams, UnknownDevice
from geecs_panels.bridge.feed import SEVERITIES, record_of
from geecs_panels.camera import MAX_TEXT, lock_word
from geecs_panels.contracts import PutOutcome, PutRequest, PutResult, Severity, Value

logger = logging.getLogger(__name__)

#: A write waits for the IOC this long, and a readback-settled one this long to
#: settle (the per-kind transports' too: :mod:`geecs_panels.ca` imports them).
PUT_TIMEOUT_S = 3.0
#: How often a readback-settled write re-reads its readback.
SETTLE_POLL_S = 0.1


class PutTransport(Protocol):
    """The channel writes and reads a put needs (real: :class:`geecs_panels.ca.CaPuts`)."""

    async def put(self, pv: str, value: object, wait: bool) -> None:
        """Write *pv*; with *wait*, until the record has processed. Raises on any CA failure."""

    async def get(self, pv: str, as_string: bool = False) -> object:
        """Read *pv* once, uncached. Raises on any CA failure."""

    async def choices(self, pv: str) -> list[str]:
        """An enum record's state names, from the record. Raises on any CA failure."""


class Denied(Exception):
    """The write is not permitted now; the message says why."""

    def __init__(self, message: str, status: int = 403) -> None:
        super().__init__(message)
        self.status = status


class Refused(Exception):
    """The value, or the IOC, refused; the message says why."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


class _AlarmUnread(Exception):
    """A written record's alarm could not be read after the put."""


@dataclass
class _Outcome:
    outcome: PutOutcome
    status: int
    value: Value = None
    severity: Optional[Severity] = None
    message: str = ""


@dataclass
class Puts:
    """The bridge's write path: gates, the put, the journal."""

    streams: DeviceStreams
    transport: PutTransport
    journal: Journal
    writable: bool = True
    stale_after: float = 3.0
    settle_s: float = PUT_TIMEOUT_S
    poll_s: float = SETTLE_POLL_S
    clock: Callable[[], float] = time.monotonic

    async def put(self, request: PutRequest) -> tuple[int, PutResult]:
        """Gate, write and journal one put; (HTTP status, result).

        Raises
        ------
        UnknownDevice
            No device of that name (nothing to journal it against).
        """
        device = self._device(request.device)
        try:
            done = await self._put(device, request)
        except Denied as exc:
            done = _Outcome("denied", exc.status, request.value, message=str(exc))
        except Refused as exc:
            done = _Outcome("refused", exc.status, request.value, message=str(exc))
        return done.status, self._journal(request, done)

    async def release_lock(self, name: str, operator: str) -> tuple[int, PutResult]:
        """Release a stuck run lock: write the lock free and nothing else, journaled as role ``lock``.

        Not gated by the stream or the lock itself: it exists for a lock a
        killed worker left held. It never writes the file writer or the
        driver, so misjudging a live run as stuck cannot end its file.
        """
        device = self._device(name)
        request = PutRequest(device=name, role="lock", value=0, operator=operator)
        try:
            done = await self._release(device, operator)
        except Denied as exc:
            done = _Outcome("denied", exc.status, 0, message=str(exc))
        except Refused as exc:
            done = _Outcome("refused", exc.status, 0, message=str(exc))
        return done.status, self._journal(request, done)

    async def _release(self, device: BridgeDevice, operator: str) -> _Outcome:
        if not self.writable:
            raise Denied("this bridge is read-only (--read-only)")
        if not device.has_role("lock"):
            raise Denied(f"{device.name} has no run lock")
        hub = self.streams.hubs.get(device.name)
        owner = hub.feed.value("owner") if hub is not None else None
        pv = device.prefix + device.role("lock").suffix
        try:
            await self.transport.put(pv, 0, wait=True)
        except Exception as exc:  # a CA failure of any kind is the IOC's refusal
            raise Refused(f"{pv}: {exc}", 502) from exc
        logger.warning(
            "run lock of %s released by %s (owner %r)", device.name, operator, owner
        )
        return _Outcome("accepted", 200, 0, message=f"released (owner {owner!r})")

    # ---- the gates -----------------------------------------------------------------

    def _device(self, name: str) -> BridgeDevice:
        device = self.streams.devices.get(name)
        if device is None:
            raise UnknownDevice(name)
        return device

    async def _put(self, device: BridgeDevice, request: PutRequest) -> _Outcome:
        if not self.writable:
            raise Denied("this bridge is read-only (--read-only)")
        role, axis = self._role(device, request)
        self._permitted(device, role)
        if role.confirm == "hazard" and not request.confirm:
            raise Denied(
                f"{request.role} on {device.name} needs the hazard confirm "
                f"(to {request.value!r})"
            )
        hub = self._live(device, role, axis)
        self._unlocked(hub, role)
        value = await self._checked(hub, role, axis, request.value)
        return await self._write(hub, role, axis, value)

    @staticmethod
    def _role(device: BridgeDevice, request: PutRequest) -> tuple[Role, str]:
        if not device.has_role(request.role):
            raise Denied(f"{device.name} has no role {request.role!r}")
        role = device.role(request.role)
        if not role.writable:
            raise Denied(f"{request.role} is read-only")
        if device.family.per_axis:
            if request.axis not in device.axes:
                raise Denied(f"{device.name} has no axis {request.axis!r}")
            return role, request.axis
        if request.axis is not None:
            raise Denied(f"{device.name} has no axes")
        return role, ""

    @staticmethod
    def _permitted(device: BridgeDevice, role: Role) -> None:
        """Rule 8: the role belongs to an action this device permits (a family without actions: every writable role)."""
        actions = device.family.actions
        if not actions:
            return
        if not any(role.name in actions[a].roles for a in device.actions):
            raise Denied(f"{role.name} is not permitted on {device.name}")

    def _live(self, device: BridgeDevice, role: Role, axis: str) -> DeviceHub:
        """A stale page cannot write: the device's stream open, ticking, the record connected."""
        hub = self.streams.hubs.get(device.name)
        if hub is None or not hub.listeners:
            raise Denied(f"no stream of {device.name} is open: reload the page", 409)
        if self.clock() - hub.ticked_at > self.stale_after:
            raise Denied(f"the stream of {device.name} is stale", 409)
        connected = (
            hub.feed.connected(role.name, axis)
            if role.read_mode == "monitor"
            else hub.feed.any_connected(axis or None)
        )
        if not connected:
            raise Denied(f"{role.name} on {device.name} is not connected", 409)
        return hub

    @staticmethod
    def _unlocked(hub: DeviceHub, role: Role) -> None:
        """Rule 6: a locked role only while the run lock reads free."""
        if not role.locked:
            return
        feed = hub.feed
        has = feed.device.has_role("lock")
        word = (
            lock_word(feed.connected("lock"), feed.value("lock")) if has else "unknown"
        )
        if word == "free":
            return
        if word == "locked":
            raise Denied(
                f"{role.name}: the device belongs to a run ({feed.value('owner') or 'unknown'})",
                423,
            )
        raise Denied(f"{role.name}: the run lock cannot be read", 423)

    async def _checked(
        self, hub: DeviceHub, role: Role, axis: str, value: Value
    ) -> object:
        """The value to put, or :class:`Refused`: the family's bounds, then the template's rule."""
        choices: list[str] = []
        if role.type == "enum" and not role.integer:
            choices = await self._enum_choices(hub, role, axis)
        checked = coerce(role, value, choices)
        rule = rules_for(hub.device.family.template).check
        if rule is None:
            return checked
        try:
            return rule(hub.feed, role, axis, checked, choices)
        except ValueError as exc:  # each kind's NotWritable
            raise Refused(str(exc)) from exc

    async def _enum_choices(self, hub: DeviceHub, role: Role, axis: str) -> list[str]:
        """The family's choices, else the record's: the names the stream shows (one reader, never a stale copy).

        Those are the monitored record's (an areaDetector ``X_RBV``), not
        the write record's (``X``): the two carry the same strings from one
        asyn parameter, and the IOC still refuses a name its record lacks.
        """
        if role.choices:
            return list(role.choices)
        feed = hub.feed
        if feed.subscribe_ctrl is not None:
            ctrl = feed.ctrl(role.name, axis)
            if (
                ctrl is None or ctrl.choices is None
            ):  # unread choices refuse: fail closed
                raise Refused(f"{role.name}: its choices cannot be read (not yet read)")
            return list(ctrl.choices)
        # A feed without the DBR_CTRL transport: the write record's names, read now.
        pv = hub.device.prefix + axis + (role.write_suffix or role.suffix)
        try:
            return await self.transport.choices(pv)
        except Exception as exc:  # unread choices refuse: fail closed
            raise Refused(f"{role.name}: its choices cannot be read ({exc})") from exc

    # ---- the IOC ---------------------------------------------------------------------

    async def _write(
        self, hub: DeviceHub, role: Role, axis: str, value: object
    ) -> _Outcome:
        device = hub.device
        suffix = role.write_suffix or role.suffix
        pv = device.prefix + axis + suffix
        settle = role.settled_by
        try:
            if settle is None:
                await self.transport.put(pv, value, wait=True)
            else:
                ticks = _counter_ticks(hub, axis)
                await self.transport.put(pv, value, wait=False)
        except Exception as exc:  # access security, a timeout: the IOC refused
            raise Refused(f"{pv}: {exc}", 502) from exc
        if settle is None:
            return await self._alarm(
                record_of(device.prefix + axis, suffix), role, value
            )
        if settle != role.name:
            return _Outcome(
                "accepted", 200, value, message=f"dispatched; done when {settle} says"
            )
        await self._settle(hub, role, axis, value, ticks)
        return _Outcome("accepted", 200, value, message="settled")

    async def _alarm(self, record: str, role: Role, value: object) -> _Outcome:
        """The written record's alarm (rule 11): INVALID is a refusal; unread, the put still took."""
        try:
            sevr, stat = await self._read_alarm(record)
        except _AlarmUnread as exc:  # the value was written: never journal it refused
            return _Outcome(
                "accepted",
                200,
                value,
                message=f"written; its alarm could not be read ({exc})",
            )
        if str(sevr) == "INVALID":
            raise Refused(f"{role.name}: the IOC did not take it (INVALID {stat})", 502)
        severity = str(sevr) if str(sevr) in SEVERITIES else None
        return _Outcome("accepted", 200, value, severity)  # type: ignore[arg-type]

    async def _read_alarm(self, record: str) -> tuple[object, object]:
        try:
            return (
                await self.transport.get(record + ".SEVR", as_string=True),
                await self.transport.get(record + ".STAT", as_string=True),
            )
        except Exception as exc:  # any CA failure: the alarm is unread
            raise _AlarmUnread(str(exc)) from exc

    async def _settle(
        self,
        hub: DeviceHub,
        role: Role,
        axis: str,
        value: object,
        ticks: dict[str, Optional[float]],
    ) -> None:
        """Until the readback reads *value*, or (a start) a frame counter advanced since the put."""
        pv = hub.device.prefix + axis + role.suffix
        loop = asyncio.get_running_loop()
        # The loop's own clock: a wait is real time, whatever clock judges staleness.
        deadline = loop.time() + self.settle_s
        got: object = None
        while True:
            try:
                got = await self.transport.get(pv)
            except Exception as exc:
                raise Refused(f"{pv}: {exc}", 502) from exc
            if got == value or (value and _counter_ticks(hub, axis) != ticks):
                return
            if loop.time() >= deadline:
                raise Refused(f"{role.name} stayed {got!r} (wanted {value!r})", 502)
            await asyncio.sleep(self.poll_s)

    def _journal(self, request: PutRequest, done: _Outcome) -> PutResult:
        entry = self.journal.record(
            operator=request.operator,
            device=request.device,
            role=request.role,
            axis=request.axis,
            asked=request.value,
            outcome=done.outcome,
            severity=done.severity,
            message=done.message,
        )
        return PutResult(
            outcome=done.outcome,
            value=done.value,
            severity=done.severity,
            message=done.message,
            journal_id=entry.id,
        )


def _counter_ticks(hub: DeviceHub, axis: str) -> dict[str, Optional[float]]:
    """When each of the device's frame counters last advanced (a capture starting moves one)."""
    feed = hub.feed
    return {
        r.name: feed.ticked(r.name, axis)
        for r in feed.device.roles
        if r.type == "counter"
    }


def coerce(role: Role, value: Value, choices: list[str]) -> object:
    """The family's own bounds on a write: the role's type, ``integer``, ``limits``, ``choices``.

    Raises
    ------
    Refused
        Naming the role and what it takes.
    """
    name = role.name
    if role.type == "command":
        if value is True or (type(value) is int and value == 1):
            return 1
        raise Refused(f"{name} takes 1 (a press)")
    if role.type == "bit":
        if isinstance(value, bool) or (type(value) is int and value in (0, 1)):
            return int(value)
        raise Refused(f"{name} takes 0 or 1, not {value!r}")
    if role.type == "enum" and not role.integer:
        if not isinstance(value, str) or value not in choices:
            raise Refused(f"{name} takes one of {', '.join(choices) or '(none read)'}")
        return value
    if role.text is not None:
        if not isinstance(value, str) or len(value) > MAX_TEXT:
            raise Refused(f"{name} takes text of at most {MAX_TEXT} characters")
        return value
    return _number(role, value)


def _number(role: Role, value: Value) -> float | int:
    """A finite number, whole where the role says, inside the role's limits."""
    name = role.name
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{name} takes a number, not {value!r}")
    if not math.isfinite(value):
        raise Refused(f"{name} takes a finite number")
    number: float | int = value
    if role.integer:
        if not float(value).is_integer():
            raise Refused(f"{name} takes a whole number, not {value!r}")
        number = int(value)
    if role.limits is not None:
        low, high = role.limits
        if not low <= number <= high:
            raise Refused(f"{name} takes {low:g} to {high:g}, not {value!r}")
    return number
