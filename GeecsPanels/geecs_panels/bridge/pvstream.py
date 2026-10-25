"""The raw PV stream: bare PVs for the inspector and a composed panel's ``pv`` widget.

``POST /api/subscriptions`` names a set of PVs and gets a handle
(:class:`~geecs_panels.contracts.SubscriptionAck`); ``GET /api/stream/{handle}``
streams it (``event: pv``, :class:`~geecs_panels.contracts.PvFrame`): every
PV on connect, then once per push period the PVs that changed. It is read
only: a write goes to a device's role, through its allow-list and hazard
confirm (``geecs_schemas.epics.panels.PvType``).

- **One monitor per PV**, shared by every handle naming it and ref-counted
  (:class:`PvMonitors`): a hundred inspectors on one PV cost the IOC one
  subscription, closed when the last handle naming it goes.
- **A handle lives while it is read**: it is released (its monitors with
  it) :data:`GRACE_S` after its last listener left, or after it was made if
  nobody ever streamed it.
- **One frame per tick, coalesced**: a PV that posts a hundred times
  between two ticks is sent once, with its latest value; one that did not
  change is not sent. A value is fresh while its channel is connected
  (a bare PV names no producer to judge it by) and ``unknown`` when not;
  ``advanced_at`` is its last post.

A bare PV is monitored one element deep: a waveform is sent as its size,
never its data (arrays are read on request, rule 3). The CA library keeps
every channel it was ever asked for (aioca's channel cache never evicts):
a subscription's monitor closes with its last handle, but its channel,
like a name typed into the inspector, stays searched for until the
process restarts.

``GET /api/pv/{name}`` (:class:`PvInfo`) is the inspector's one-shot read:
the value with its alarm and timestamp, and the record's ``DBR_CTRL``
fields (units, precision, limits, an enum's choices).
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from geecs_panels.bridge.feed import Sample, SubscribeReading, severity_name
from geecs_panels.bridge.stream import GRACE_S, KEEPALIVE_S, PUSH_S
from geecs_panels.contracts import (
    PvFrame,
    Reading,
    Severity,
    SubscriptionAck,
    Value,
    _finite,
)
from geecs_panels.feed import Subscription

#: The most handles held at once: a page leaking handles cannot exhaust the IOCs.
#: The push period, the grace period and the keepalive are the device streams'.
MAX_HANDLES = 200


class TooManyHandles(RuntimeError):
    """The bridge holds :data:`MAX_HANDLES` handles already."""


class UnknownHandle(KeyError):
    """No handle of that id (never made, or released after its grace period)."""


@dataclass
class _Monitor:
    """One PV's monitor and its latest post."""

    refs: int = 0
    sub: Optional[Subscription] = None
    connected: bool = False
    value: object = None
    severity: Optional[int] = None
    posted: Optional[float] = None


@dataclass
class PvMonitors:
    """One CA monitor per PV, shared by every handle naming it, closed with the last one."""

    subscribe: SubscribeReading
    wall: Callable[[], float] = time.time
    monitors: dict[str, _Monitor] = field(default_factory=dict)

    def acquire(self, pv: str) -> None:
        """Count one more user of *pv*; the first subscribes, and is kept only if that worked."""
        monitor = self.monitors.get(pv)
        if monitor is None:
            monitor = _Monitor()
            # Registered before subscribing so the first post finds it, and
            # dropped if the subscribe raises: a monitor without a
            # subscription would leave the PV silently dead.
            self.monitors[pv] = monitor
            try:
                monitor.sub = self.subscribe(pv, "auto", lambda s: self._update(pv, s))
            except BaseException:
                del self.monitors[pv]
                raise
        monitor.refs += 1

    def release(self, pv: str) -> None:
        """Count one user of *pv* fewer; the last closes its monitor."""
        monitor = self.monitors.get(pv)
        if monitor is None:
            return
        monitor.refs -= 1
        if monitor.refs <= 0:
            del self.monitors[pv]
            if monitor.sub is not None:
                monitor.sub.close()

    def _update(self, pv: str, sample: Optional[Sample]) -> None:
        monitor = self.monitors.get(pv)
        if monitor is None:  # a post racing the release
            return
        if sample is None:
            monitor.connected = False
            return
        monitor.connected = True
        monitor.value, monitor.severity = sample.value, sample.severity
        monitor.posted = self.wall()

    def reading(self, pv: str) -> Reading:
        """The PV's reading now: its value while connected, ``unknown`` when not."""
        monitor = self.monitors.get(pv)
        if monitor is None or not monitor.connected:
            return Reading(freshness="unknown")
        return Reading(
            value=_scalar(monitor.value),
            severity=severity_name(monitor.severity),
            advanced_at=monitor.posted,
            freshness="fresh",
        )


def _scalar(value: object) -> object:
    """A value as the wire carries it; an array as its size in words (the inspector shows it, never plots it)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return f"[{len(value)} values]"  # type: ignore[arg-type]  # an array
    except TypeError:
        return str(value)


@dataclass
class Handle:
    """One subscription: its PVs and who is reading it."""

    id: str
    pvs: list[str]
    listeners: int = 0
    #: Monotonic time it was last left alone (made, or its last listener gone).
    idle_since: Optional[float] = None


@dataclass
class PvStreams:
    """The handles, their shared monitors, and each handle's stream."""

    subscribe: SubscribeReading
    push_s: float = PUSH_S
    grace_s: float = GRACE_S
    max_handles: int = MAX_HANDLES
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    handles: dict[str, Handle] = field(default_factory=dict)
    monitors: PvMonitors = field(init=False)
    _sweeper: Optional[asyncio.Task[None]] = None

    def __post_init__(self) -> None:
        """Share one set of monitors among every handle."""
        self.monitors = PvMonitors(self.subscribe, self.wall)

    def open(self, pvs: list[str]) -> SubscriptionAck:
        """A new handle on *pvs*, its monitors acquired; :class:`TooManyHandles` past the cap."""
        self.sweep()
        if len(self.handles) >= self.max_handles:
            raise TooManyHandles(f"the bridge holds {self.max_handles} handles already")
        handle = Handle(f"h-{secrets.token_hex(6)}", list(pvs), idle_since=self.clock())
        acquired: list[str] = []
        try:
            for pv in handle.pvs:
                self.monitors.acquire(pv)
                acquired.append(pv)
        except BaseException:
            for pv in acquired:  # none of a failed handle's monitors outlives it
                self.monitors.release(pv)
            raise
        self.handles[handle.id] = handle
        self._start_sweeper()
        return SubscriptionAck(
            handle=handle.id, pvs=handle.pvs, grace_seconds=self.grace_s
        )

    def sweep(self) -> list[str]:
        """Release every handle alone for the grace period; the ids released."""
        now = self.clock()
        # idle_since is None while anyone listens.
        gone = [
            h.id
            for h in self.handles.values()
            if h.idle_since is not None and now - h.idle_since >= self.grace_s
        ]
        for hid in gone:
            for pv in self.handles.pop(hid).pvs:
                self.monitors.release(pv)
        return gone

    def _start_sweeper(self) -> None:
        if self._sweeper is not None and not self._sweeper.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop (a test calling open directly): swept on use
            return
        self._sweeper = loop.create_task(self._sweep_while_held())

    async def _sweep_while_held(self) -> None:
        while self.handles:
            await asyncio.sleep(max(self.grace_s / 4, 0.05))
            self.sweep()

    def frame(
        self, handle: Handle, seq: int, full: bool, pvs: dict[str, Reading]
    ) -> PvFrame:
        """One ``event: pv`` frame."""
        return PvFrame(
            handle=handle.id, seq=seq, server_time=self.wall(), full=full, pvs=pvs
        )

    async def listen(self, hid: str) -> AsyncIterator[Optional[PvFrame]]:
        """The handle's frames for one listener: every PV, then the changed ones once per tick.

        ``seq`` starts at the wall clock in milliseconds (as a device
        stream's), so an old ``Last-Event-ID`` is never mistaken for one
        of this stream's; a reconnect gets a full frame. Yields ``None``
        after :data:`KEEPALIVE_S` without a frame.
        """
        self.sweep()
        handle = self.handles.get(hid)
        if handle is None:
            raise UnknownHandle(hid)
        handle.listeners += 1
        handle.idle_since = None
        try:
            seq = int(self.wall() * 1000)
            sent = {pv: self.monitors.reading(pv) for pv in handle.pvs}
            yield self.frame(handle, seq, True, dict(sent))
            quiet = 0.0
            while True:
                await asyncio.sleep(self.push_s)
                now = {pv: self.monitors.reading(pv) for pv in handle.pvs}
                changed = {pv: r for pv, r in now.items() if sent.get(pv) != r}
                if changed:
                    seq += 1
                    sent.update(changed)
                    quiet = 0.0
                    yield self.frame(handle, seq, False, changed)
                    continue
                quiet += self.push_s
                if quiet >= KEEPALIVE_S:
                    quiet = 0.0
                    yield None
        finally:
            handle.listeners -= 1
            if handle.listeners == 0:
                handle.idle_since = self.clock()

    async def aclose(self) -> None:
        """Release every handle and stop the sweeper."""
        if self._sweeper is not None:
            self._sweeper.cancel()
        for hid in list(self.handles):
            for pv in self.handles.pop(hid).pvs:
                self.monitors.release(pv)


def pv_sse(frame: PvFrame) -> str:
    """One frame as an SSE event: ``id:`` is its ``seq``, ``event: pv``; NaN as null."""
    body = json.dumps(
        frame.model_dump(mode="json"), allow_nan=False, separators=(",", ":")
    )
    return f"id: {frame.seq}\nevent: pv\ndata: {body}\n\n"


# ---- the inspector's read ------------------------------------------------------------


class AlarmLimits(BaseModel):
    """A record's alarm limits (``LOLO``, ``LOW``, ``HIGH``, ``HIHI``); null where it sets none."""

    model_config = ConfigDict(extra="forbid")

    lolo: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    hihi: Optional[float] = None

    @field_validator("lolo", "low", "high", "hihi", mode="before")
    @classmethod
    def _unset_is_null(cls, value: object) -> object:
        # A record without alarm limits reports NaN: unset, and JSON has no NaN.
        return _finite(value)


class PvInfo(BaseModel):
    """``GET /api/pv/{name}``: one PV read now, with its record's ``DBR_CTRL`` fields."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The PV.")
    connected: bool = Field(description="Whether it answered.")
    value: Value = Field(
        None, description="Its value (an enum by its state name; an array as its size)."
    )
    severity: Optional[Severity] = Field(None, description="Its alarm severity.")
    timestamp: Optional[float] = Field(
        None, description="The record's own timestamp, Unix seconds."
    )
    units: Optional[str] = Field(None, description="EGU.")
    precision: Optional[int] = Field(None, description="PREC.")
    display_limits: Optional[tuple[float, float]] = Field(
        None, description="LOPR, HOPR."
    )
    control_limits: Optional[tuple[float, float]] = Field(
        None, description="DRVL, DRVH."
    )
    alarm_limits: Optional[AlarmLimits] = Field(None, description="LOLO … HIHI.")
    choices: Optional[list[str]] = Field(None, description="An enum's state names.")
    count: Optional[int] = Field(None, description="Its element count (1: a scalar).")
    error: Optional[str] = Field(None, description="Why it could not be read.")

    @field_validator("value", mode="before")
    @classmethod
    def _finite_value(cls, value: object) -> object:
        return _finite(value)

    @field_validator("display_limits", "control_limits", mode="before")
    @classmethod
    def _finite_limits(cls, limits: object) -> object:
        if isinstance(limits, (tuple, list)) and any(
            _finite(v) is None for v in limits
        ):
            return None
        return limits


#: pv -> the fields :class:`PvInfo` carries (``name`` and ``connected`` aside); raises when it does not answer.
ReadInfo = Callable[[str], Awaitable[dict[str, object]]]


class _Unanswered(Exception):
    """A PV the inspector asked for did not answer."""


async def _ask(name: str, read: ReadInfo) -> dict[str, object]:
    try:
        return await read(name)
    except Exception as exc:  # any CA failure: said, never a 500
        raise _Unanswered(str(exc)) from exc


async def pv_info(name: str, read: ReadInfo) -> PvInfo:
    """Read *name* once; a PV that does not answer is ``connected: false`` with the reason."""
    try:
        fields = await _ask(name, read)
    except _Unanswered as exc:
        return PvInfo(name=name, connected=False, error=str(exc))
    fields = {**fields, "value": _scalar(fields.get("value"))}
    return PvInfo(name=name, connected=True, **fields)  # type: ignore[arg-type]
