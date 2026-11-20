"""The monitor: one loop that evaluates every interlock and publishes a snapshot.

Each tick reads the source's latest readings, runs
:func:`~geecs_interlocks.conditions.evaluate` for every interlock, steps its
state, and swaps in one immutable :class:`Snapshot`. Every reader — the
Master Control stream, the server window, the web client — reads that same
snapshot, so they can never disagree.

An interlock's state:

``waiting``
    Armed, nothing received yet. Unsafe: a server that has not heard from a
    device never reports it safe.
``ok``
    The condition is safe.
``tripped``
    The condition is unsafe.
``settling``
    Safe again after a trip, waiting out ``settle_s``. Still unsafe.
``off``
    ``enabled: false`` in the file. Not counted in the verdict.

The overall verdict is safe only when at least one interlock is armed and
every armed one is ``ok``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from geecs_schemas import (
    AlignmentCondition,
    Interlock,
    InterlockCondition,
    InterlockSet,
    StaleCondition,
    ThresholdCondition,
)

from geecs_interlocks.conditions import (
    Inputs,
    Verdict,
    describe,
    evaluate,
    number,
)
from geecs_interlocks.sources import Source

logger = logging.getLogger(__name__)

State = Literal["waiting", "ok", "tripped", "settling", "off"]
#: Every state, in the order a table sorts them (worst first).
STATES: tuple[State, ...] = ("tripped", "waiting", "settling", "ok", "off")
_UNSAFE_STATES = frozenset({"waiting", "tripped", "settling"})

#: Seconds between two evaluations.
TICK_S = 0.1
#: Seconds of trend kept per interlock.
TREND_S = 300.0
#: Verdict changes kept for the event log.
EVENTS_KEPT = 500


@dataclass(frozen=True)
class InterlockStatus:
    """One interlock as every reader sees it, at one tick.

    Attributes
    ----------
    name, description :
        From the file.
    condition :
        One line saying what it watches (:func:`describe`).
    state :
        See the module docstring.
    diagnostic :
        Why it is not ``ok`` (empty when it is).
    value :
        The number the condition compares: the reading (threshold), the
        distance to the target (alignment), the age of the last update
        (stale); ``None`` when there is none.
    limit :
        What *value* is compared against (``None`` for a disconnect-only stale).
    since :
        Wall-clock seconds (``time.time()``) when it entered *state*.
    """

    name: str
    description: str
    condition: str
    state: State
    diagnostic: str = ""
    value: float | None = None
    limit: float | None = None
    since: float = 0.0

    @property
    def unsafe(self) -> bool:
        """Whether it counts against the verdict (waiting, tripped or settling)."""
        return self.state in _UNSAFE_STATES

    @property
    def armed(self) -> bool:
        """Whether it is watched at all (not ``off``)."""
        return self.state != "off"


@dataclass(frozen=True)
class Snapshot:
    """Every interlock at one tick, and the verdict over them."""

    statuses: tuple[InterlockStatus, ...] = ()
    #: Wall-clock seconds of the tick.
    taken: float = 0.0
    #: Where the interlocks came from (a file path, or a note).
    source: str = ""

    @property
    def safe(self) -> bool:
        """At least one interlock armed, and every armed one ``ok``."""
        armed = [s for s in self.statuses if s.armed]
        return bool(armed) and not any(s.unsafe for s in armed)

    @property
    def armed_count(self) -> int:
        """How many interlocks are watched."""
        return sum(s.armed for s in self.statuses)


@dataclass(frozen=True)
class Event:
    """One state change, for the event log."""

    at: float
    name: str
    old: State
    new: State
    diagnostic: str = ""


@dataclass
class _Tracker:
    """One interlock's mutable state between ticks (the loop thread's alone)."""

    interlock: Interlock
    state: State = "waiting"
    since: float = field(default_factory=time.time)
    settle_from: float | None = None
    trend: deque[tuple[float, float]] = field(default_factory=deque)


def _measured(
    condition: InterlockCondition, inputs: Inputs
) -> tuple[float | None, float | None]:
    """The number *condition* compares, and its limit (for the trend)."""
    if isinstance(condition, ThresholdCondition):
        return number(inputs.value), condition.limit
    if isinstance(condition, AlignmentCondition):
        value, target = number(inputs.value), number(inputs.target)
        distance = None if value is None or target is None else abs(value - target)
        return distance, condition.tolerance
    if isinstance(condition, StaleCondition):
        return inputs.age_s, condition.max_age_s
    return None, None


def _combine(verdicts: list[Verdict]) -> Verdict:
    """One verdict over a group: unsafe if any is, the reason naming every unsafe one.

    The group only counts as ``waiting`` when every unsafe condition is
    merely waiting — a real trip next to a silent device is a trip.
    """
    unsafe = [v for v in verdicts if v.unsafe]
    if not unsafe:
        return Verdict(False)
    return Verdict(
        True, "; ".join(v.diagnostic for v in unsafe), all(v.waiting for v in unsafe)
    )


class Monitor:
    """Evaluate *interlocks* over *source* every *tick* seconds.

    Parameters
    ----------
    interlocks :
        The document to watch.
    source :
        Where the readings come from; started and stopped with the monitor.
    origin :
        Where *interlocks* came from, shown to operators.
    tick :
        Seconds between evaluations.
    clock :
        ``time.monotonic`` stand-in for tests; ages are measured on it.
    """

    def __init__(
        self,
        interlocks: InterlockSet,
        source: Source,
        *,
        origin: str = "",
        tick: float = TICK_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._origin = origin
        self._tick = tick
        self._clock = clock
        self._trend_len = max(1, int(TREND_S / tick))
        self._trackers = [
            _Tracker(i, state="waiting" if i.enabled else "off")
            for i in interlocks.interlocks
        ]
        self._lock = threading.Lock()
        self._snapshot = Snapshot(source=origin)
        self._events: deque[Event] = deque(maxlen=EVENTS_KEPT)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.step()  # a snapshot exists before the first tick: every armed one waiting

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start the source and the evaluation loop."""
        self._source.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="interlock-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop, then the source."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self._tick * 10 + 1)
            self._thread = None
        self._source.stop()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                logger.exception("interlock evaluation failed")
                self._fail_all("evaluation error (see the server log)")
            self._stop.wait(self._tick)

    # -- reads ----------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        """The latest snapshot (immutable; any thread)."""
        with self._lock:
            return self._snapshot

    def events(self) -> list[Event]:
        """The state changes kept, oldest first."""
        with self._lock:
            return list(self._events)

    def trend(self, name: str) -> list[tuple[float, float]]:
        """``(wall-clock, value)`` points of *name* over the last :data:`TREND_S`."""
        with self._lock:
            for tracker in self._trackers:
                if tracker.interlock.name == name:
                    return list(tracker.trend)
        return []

    # -- one tick -------------------------------------------------------------

    def step(self) -> Snapshot:
        """Evaluate every interlock once and publish the snapshot."""
        now_wall, now = time.time(), self._clock()
        statuses: list[InterlockStatus] = []
        events: list[Event] = []
        for tracker in self._trackers:
            status = self._step_one(tracker, now_wall, now, events)
            statuses.append(status)
        snapshot = Snapshot(tuple(statuses), now_wall, self._origin)
        with self._lock:
            self._snapshot = snapshot
            self._events.extend(events)
        for event in events:
            log = logger.warning if event.new == "tripped" else logger.info
            log(
                "interlock %r: %s → %s %s",
                event.name,
                event.old,
                event.new,
                event.diagnostic,
            )
        return snapshot

    def _step_one(
        self, tracker: _Tracker, now_wall: float, now: float, events: list[Event]
    ) -> InterlockStatus:
        interlock = tracker.interlock
        line = "; ".join(describe(c) for c in interlock.conditions)
        if tracker.state == "off":
            return InterlockStatus(
                interlock.name, interlock.description, line, "off", since=tracker.since
            )
        tripped = tracker.state in ("tripped", "settling")
        inputs = [self._inputs(c, now) for c in interlock.conditions]
        verdict = _combine(
            [
                evaluate(c, i, tripped=tripped)
                for c, i in zip(interlock.conditions, inputs, strict=True)
            ]
        )
        new_state, diagnostic = self._next_state(tracker, verdict, interlock, now)
        if new_state != tracker.state:
            events.append(
                Event(now_wall, interlock.name, tracker.state, new_state, diagnostic)
            )
            tracker.state, tracker.since = new_state, now_wall
        # A trend needs one number: a group's lives on its rows (per condition).
        value, limit = (
            _measured(interlock.conditions[0], inputs[0])
            if len(interlock.conditions) == 1
            else (None, None)
        )
        if value is not None:
            with self._lock:
                tracker.trend.append((now_wall, value))
                while len(tracker.trend) > self._trend_len:
                    tracker.trend.popleft()
        return InterlockStatus(
            interlock.name,
            interlock.description,
            line,
            new_state,
            diagnostic,
            value,
            limit,
            tracker.since,
        )

    def _next_state(
        self, tracker: _Tracker, verdict: Verdict, interlock: Interlock, now: float
    ) -> tuple[State, str]:
        if verdict.unsafe:
            tracker.settle_from = None
            if verdict.waiting and tracker.state == "waiting":
                return "waiting", verdict.diagnostic
            return "tripped", verdict.diagnostic
        if tracker.state in ("waiting", "ok") or interlock.settle_s <= 0:
            tracker.settle_from = None
            return "ok", ""
        # Safe after a trip: hold for settle_s before reporting OK.
        if tracker.settle_from is None:
            tracker.settle_from = now
        left = interlock.settle_s - (now - tracker.settle_from)
        if left <= 0:
            tracker.settle_from = None
            return "ok", ""
        return "settling", f"safe again; OK in {left:.1f} s"

    def _inputs(self, condition: InterlockCondition, now: float) -> Inputs:
        if not isinstance(
            condition, ThresholdCondition | AlignmentCondition | StaleCondition
        ):
            return Inputs()
        if isinstance(condition, StaleCondition):
            reading = self._source.reading(condition.device)
            age = None if reading.updated is None else now - reading.updated
            down = (
                ()
                if reading.connected or reading.updated is None
                else (condition.device,)
            )
            return Inputs(disconnected=down, age_s=age)
        devices = [condition.device]
        if isinstance(condition, AlignmentCondition):
            devices.append(condition.target_device or condition.device)
        readings = {d: self._source.reading(d) for d in devices}
        down = tuple(
            d for d, r in readings.items() if not r.connected and r.updated is not None
        )
        value = readings[condition.device].values.get(condition.variable)
        target = None
        if isinstance(condition, AlignmentCondition):
            target_device = condition.target_device or condition.device
            target = readings[target_device].values.get(condition.target_variable)
        return Inputs(value=value, target=target, disconnected=down)

    def _fail_all(self, why: str) -> None:
        """Publish every armed interlock tripped (the loop hit an error)."""
        now = time.time()
        with self._lock:
            statuses = tuple(
                s
                if not s.armed
                else InterlockStatus(
                    s.name, s.description, s.condition, "tripped", why, since=now
                )
                for s in self._snapshot.statuses
            )
            self._snapshot = Snapshot(statuses, now, self._origin)


__all__ = [
    "EVENTS_KEPT",
    "STATES",
    "TICK_S",
    "TREND_S",
    "Event",
    "InterlockStatus",
    "Monitor",
    "Snapshot",
    "State",
]
