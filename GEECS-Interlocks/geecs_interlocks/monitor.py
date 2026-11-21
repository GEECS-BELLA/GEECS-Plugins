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

A ``config_change`` condition holds settings at the values recorded when an
operator arms it (:meth:`Monitor.arm`). The baselines live in the monitor,
so a reload, which builds a new monitor, needs arming again.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from geecs_schemas import (
    AlignmentCondition,
    ConfigChangeCondition,
    ExternalCondition,
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
from geecs_interlocks.external import SignalSource
from geecs_interlocks.sources import DeviceReading, Source

logger = logging.getLogger(__name__)

State = Literal["waiting", "ok", "tripped", "settling", "off"]
#: Every state, in the order a table sorts them (worst first).
STATES: tuple[State, ...] = ("tripped", "waiting", "settling", "ok", "off")
_UNSAFE_STATES = frozenset({"waiting", "tripped", "settling"})

#: Seconds between two evaluations.
TICK_S = 0.1
#: Seconds of trend kept per check.
TREND_S = 300.0
#: Verdict changes kept for the event log.
EVENTS_KEPT = 500


@dataclass(frozen=True)
class CheckStatus:
    """One condition of an interlock, at one tick (a group's rows in a table).

    Attributes
    ----------
    condition :
        One line saying what it watches (:func:`describe`).
    unsafe :
        Whether this condition is unsafe on its own.
    diagnostic :
        Why it is unsafe (empty when it is not).
    value, limit :
        As :class:`InterlockStatus`'s, for this condition alone.
    """

    condition: str
    unsafe: bool
    diagnostic: str = ""
    value: float | None = None
    limit: float | None = None


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
    armable :
        It holds settings an operator arms (a ``config_change`` condition).
    checks :
        Each condition on its own, in file order (empty while ``off``).
    """

    name: str
    description: str
    condition: str
    state: State
    diagnostic: str = ""
    value: float | None = None
    limit: float | None = None
    since: float = 0.0
    armable: bool = False
    checks: tuple[CheckStatus, ...] = ()

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


@dataclass(frozen=True)
class TrendSeries:
    """One check's recent values, for a plot against its limit."""

    condition: str
    limit: float | None
    points: tuple[tuple[float, float], ...]


@dataclass
class _Tracker:
    """One interlock's mutable state between ticks (the loop thread's alone)."""

    interlock: Interlock
    state: State = "waiting"
    since: float = field(default_factory=time.time)
    settle_from: float | None = None
    #: One ``(wall-clock, value)`` series per condition, in file order.
    trends: list[deque[tuple[float, float]]] = field(default_factory=list)


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
    if isinstance(condition, StaleCondition | ExternalCondition):
        return inputs.age_s, condition.max_age_s
    return None, None


def _armable(interlock: Interlock) -> bool:
    return any(isinstance(c, ConfigChangeCondition) for c in interlock.conditions)


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
    signals :
        Where external (UDP) signals come from; owned by the caller, never
        started or stopped here (``None``: every external condition waits).
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
        signals: SignalSource | None = None,
        origin: str = "",
        tick: float = TICK_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._signals = signals
        self._stale_after = interlocks.stale_after_s
        #: (interlock name, condition index) → the settings recorded at arming.
        self._baselines: dict[tuple[str, int], dict[str, Any]] = {}
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

    def reading(self, device: str) -> DeviceReading:
        """The source's latest reading of *device* (any thread)."""
        return self._source.reading(device)

    def trend(self, name: str) -> list[TrendSeries]:
        """Each check of *name*: its ``(wall-clock, value)`` points over the last :data:`TREND_S`.

        Empty for an unknown name; a check with no number (a settings check)
        has no points.
        """
        with self._lock:
            for tracker in self._trackers:
                if tracker.interlock.name == name:
                    statuses = {s.name: s for s in self._snapshot.statuses}
                    checks = statuses[name].checks if name in statuses else ()
                    return [
                        TrendSeries(
                            describe(condition),
                            checks[i].limit if i < len(checks) else None,
                            tuple(tracker.trends[i]) if i < len(tracker.trends) else (),
                        )
                        for i, condition in enumerate(tracker.interlock.conditions)
                    ]
        return []

    # -- arming ---------------------------------------------------------------

    def arm(self, name: str | None = None) -> list[str]:
        """Record the current settings of every ``config_change`` condition (of *name* only, if given).

        All or nothing: when any setting has no value yet, or its device is
        disconnected, nothing is armed. Interlocks that are ``off`` are
        skipped.

        Returns
        -------
        list of str
            The interlocks armed.

        Raises
        ------
        ValueError
            Nothing to arm, or a setting has no value to record.
        """
        pending: dict[tuple[str, int], dict[str, Any]] = {}
        missing: list[str] = []
        for tracker in self._trackers:
            interlock = tracker.interlock
            if tracker.state == "off" or name not in (None, interlock.name):
                continue
            for index, condition in enumerate(interlock.conditions):
                if not isinstance(condition, ConfigChangeCondition):
                    continue
                reading = self._source.reading(condition.device)
                values = {v: reading.values.get(v) for v in condition.variables}
                absent = [v for v, x in values.items() if x is None]
                if not reading.connected:
                    missing.append(f"{condition.device} (disconnected)")
                elif absent:
                    missing.append(f"{condition.device} {', '.join(absent)}")
                pending[(interlock.name, index)] = values
        if not pending:
            raise ValueError(
                f"{name!r} holds no settings to arm"
                if name
                else "no armed interlock holds settings to arm"
            )
        if missing:
            raise ValueError(f"cannot arm, no value yet for: {'; '.join(missing)}")
        with self._lock:
            self._baselines.update(pending)
        armed = sorted({key[0] for key in pending})
        logger.info("armed %s", ", ".join(armed))
        return armed

    def baseline(self, name: str, index: int) -> Mapping[str, Any] | None:
        """The settings recorded for condition *index* of *name* (``None``: not armed)."""
        with self._lock:
            return self._baselines.get((name, index))

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
        armable = _armable(interlock)
        if tracker.state == "off":
            return InterlockStatus(
                interlock.name,
                interlock.description,
                line,
                "off",
                since=tracker.since,
                armable=armable,
            )
        tripped = tracker.state in ("tripped", "settling")
        inputs = [
            self._inputs(interlock.name, i, c, now)
            for i, c in enumerate(interlock.conditions)
        ]
        verdicts = [
            evaluate(c, i, tripped=tripped)
            for c, i in zip(interlock.conditions, inputs, strict=True)
        ]
        verdict = _combine(verdicts)
        checks = tuple(
            CheckStatus(describe(c), v.unsafe, v.diagnostic, *_measured(c, i))
            for c, i, v in zip(interlock.conditions, inputs, verdicts, strict=True)
        )
        new_state, diagnostic = self._next_state(tracker, verdict, interlock, now)
        if new_state != tracker.state:
            events.append(
                Event(now_wall, interlock.name, tracker.state, new_state, diagnostic)
            )
            tracker.state, tracker.since = new_state, now_wall
        with self._lock:
            while len(tracker.trends) < len(checks):
                tracker.trends.append(deque(maxlen=self._trend_len))
            for series, check in zip(tracker.trends, checks, strict=False):
                if check.value is not None:
                    series.append((now_wall, check.value))
        # The interlock's own number: its check's, when it has only one.
        value, limit = (
            _measured(interlock.conditions[0], inputs[0])
            if len(interlock.conditions) == 1
            else (None, None)
        )
        return InterlockStatus(
            interlock.name,
            interlock.description,
            line,
            new_state,
            diagnostic,
            value,
            limit,
            tracker.since,
            armable,
            checks,
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
        return "settling", f"clear again; OK in {left:.1f} s"

    def _inputs(
        self, name: str, index: int, condition: InterlockCondition, now: float
    ) -> Inputs:
        if isinstance(condition, ExternalCondition):
            signal = (
                None if self._signals is None else self._signals.signal(condition.port)
            )
            age = (
                None
                if signal is None or signal.received is None
                else now - signal.received
            )
            return Inputs(signal=signal, age_s=age)
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
        down, stale = self._health(readings, now)
        values = readings[condition.device].values
        if isinstance(condition, ConfigChangeCondition):
            return Inputs(
                disconnected=down,
                stale=stale,
                settings={v: values.get(v) for v in condition.variables},
                baseline=self.baseline(name, index),
            )
        target = None
        if isinstance(condition, AlignmentCondition):
            target_device = condition.target_device or condition.device
            target = readings[target_device].values.get(condition.target_variable)
        return Inputs(
            value=values.get(condition.variable),
            target=target,
            disconnected=down,
            stale=stale,
        )

    def _health(
        self, readings: Mapping[str, DeviceReading], now: float
    ) -> tuple[tuple[str, ...], tuple[tuple[str, float], ...]]:
        """The devices among *readings* that are down, and those silent past ``stale_after_s``.

        A device not heard from yet is neither: its missing values make the
        condition wait instead.
        """
        down = tuple(
            d for d, r in readings.items() if not r.connected and r.updated is not None
        )
        if self._stale_after is None:
            return down, ()
        stale = tuple(
            (d, now - r.updated)
            for d, r in readings.items()
            if r.connected
            and r.updated is not None
            and now - r.updated > self._stale_after
        )
        return down, stale

    def _fail_all(self, why: str) -> None:
        """Publish every armed interlock tripped (the loop hit an error)."""
        now = time.time()
        with self._lock:
            statuses = tuple(
                s
                if not s.armed
                else InterlockStatus(
                    s.name,
                    s.description,
                    s.condition,
                    "tripped",
                    why,
                    since=now,
                    armable=s.armable,
                )
                for s in self._snapshot.statuses
            )
            self._snapshot = Snapshot(statuses, now, self._origin)


__all__ = [
    "EVENTS_KEPT",
    "CheckStatus",
    "STATES",
    "TICK_S",
    "TREND_S",
    "Event",
    "InterlockStatus",
    "Monitor",
    "Snapshot",
    "TrendSeries",
    "State",
]
