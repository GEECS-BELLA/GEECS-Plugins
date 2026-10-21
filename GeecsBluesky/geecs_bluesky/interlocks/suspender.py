"""One interlock as a Bluesky suspender over ophyd-async signals.

A stock suspender watches one ophyd (v1) signal: it calls
``signal.subscribe(self, event_type=..., run=True)`` and expects to be
called back with a raw ``value``.  An ophyd-async signal's ``subscribe``
takes one callback and calls it with ``{name: Reading}``, and an interlock
watches several signals at once (the variable, an alignment's target, the
device's ``CONNECTED`` PV, a stale condition's ``acq_timestamp``).
:class:`WatchedSignals` is the adapter between the two: the ``signal`` the
stock :class:`~bluesky.suspenders.SuspenderBase` subscribes to, fanning one
subscription out to every watched signal and calling the suspender back
with no value after each update.  The suspender keeps the latest reading
of each, and :func:`~geecs_bluesky.interlocks.conditions.evaluate` decides.

Everything else is the stock suspender: a trip during a run suspends it
(the RunEngine pauses every Pausable the run has touched, so the trigger
box goes quiet), runs ``pre_plan`` (the interlock's ``on_trip`` action
plan), waits until the condition is safe, sleeps ``settle_s``, runs
``post_plan`` (``before_resume``) and resumes.  A trip while the
RunEngine is idle makes the next plan wait at its start until the
condition clears; no ``on_trip`` plan runs then, because nothing was
running.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from bluesky.suspenders import SuspenderBase
from geecs_schemas import AlignmentCondition, Interlock, StaleCondition

from geecs_bluesky.devices.ca._pv import GATEWAY_DISCONNECTED
from geecs_bluesky.devices.ca.liveness import reading_invalid
from geecs_bluesky.interlocks.conditions import Inputs, Sample, Verdict, evaluate

logger = logging.getLogger(__name__)

#: How late past ``max_age_s`` the stale re-check fires, so the age it
#: reads is past the limit rather than equal to it.
STALE_SLACK_S = 0.05

#: A reading as ophyd-async delivers it to a subscriber.
ReadingCallback = Callable[[Mapping[str, Mapping[str, Any]]], None]


@dataclass(frozen=True)
class Watched:
    """The signals one interlock reads (ophyd-async ``SignalR`` objects)."""

    value: Any | None = None
    target: Any | None = None
    stamp: Any | None = None
    #: GEECS device name → its ``connected_status`` signal.
    connected: Mapping[str, Any] = field(default_factory=dict)


class WatchedSignals:
    """The stock suspender's ``signal``: one subscription over many signals.

    Parameters
    ----------
    name :
        What the suspender's repr and justification call the signal.
    handlers :
        ``(signal, handler)`` pairs: on every update of *signal* its
        handler gets the ophyd-async reading first, then the subscriber is
        called with ``None``.
    """

    def __init__(self, name: str, handlers: list[tuple[Any, ReadingCallback]]):
        self.name = name
        self._handlers = handlers
        self._subs: dict[Any, list[tuple[Any, ReadingCallback]]] = {}

    def subscribe(
        self, callback: Callable[..., Any], event_type: Any = None, run: bool = True
    ) -> None:
        """Subscribe *callback* to every watched signal (the ophyd v1 signature)."""
        subs = []
        for signal, handler in self._handlers:

            def on_update(reading, handler=handler):
                handler(reading)
                callback(None)

            signal.subscribe(on_update)
            subs.append((signal, on_update))
        self._subs[callback] = subs

    def clear_sub(self, callback: Callable[..., Any]) -> None:
        """Remove every subscription :meth:`subscribe` made for *callback*."""
        for signal, on_update in self._subs.pop(callback, []):
            signal.clear_sub(on_update)

    def __repr__(self) -> str:
        """Name the interlock, as the suspender's own repr shows it."""
        return f"WatchedSignals({self.name!r})"


def _only_reading(reading: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """The one ``Reading`` in an ophyd-async ``{name: Reading}`` callback."""
    return next(iter(reading.values()))


class GeecsInterlockSuspender(SuspenderBase):
    """Suspend a run while one interlock's condition is unsafe.

    Parameters
    ----------
    interlock :
        The interlock document entry.
    watched :
        The signals its condition reads.
    pre_plan, post_plan :
        Callables returning the ``on_trip`` / ``before_resume`` plans (or
        ``None``): run when the suspension starts and just before resume.
    on_change :
        Called with the suspender whenever its verdict changes (the
        registry's log line and the status stream).
    clock :
        Monotonic seconds (tests).
    """

    def __init__(
        self,
        interlock: Interlock,
        watched: Watched,
        *,
        pre_plan: Callable[[], Any] | None = None,
        post_plan: Callable[[], Any] | None = None,
        on_change: Callable[[GeecsInterlockSuspender], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interlock = interlock
        self._condition = interlock.condition
        self._clock = clock
        self._on_change = on_change
        self._cache_lock = threading.Lock()
        self._samples = {"value": Sample(), "target": Sample()}
        self._connected: dict[str, Any] = {}
        self._stamp: Any = None
        self._stamp_at: float | None = None
        self._timer: Any = None
        self._verdict: Verdict | None = None
        handlers: list[tuple[Any, ReadingCallback]] = []
        if watched.value is not None:
            handlers.append((watched.value, self._sample_handler("value")))
        if watched.target is not None:
            handlers.append((watched.target, self._sample_handler("target")))
        if watched.stamp is not None:
            handlers.append((watched.stamp, self._on_stamp))
        for device, signal in watched.connected.items():
            handlers.append((signal, self._connected_handler(device)))
        super().__init__(
            WatchedSignals(interlock.name, handlers),
            sleep=interlock.settle_s,
            pre_plan=pre_plan,
            post_plan=post_plan,
        )

    @property
    def name(self) -> str:
        """The interlock's name."""
        return self.interlock.name

    @property
    def verdict(self) -> Verdict | None:
        """The latest verdict (``None`` before the first evaluation)."""
        return self._verdict

    # -- the cached readings ------------------------------------------------

    def _sample_handler(self, role: str) -> ReadingCallback:
        def handle(reading):
            r = _only_reading(reading)
            with self._cache_lock:
                self._samples[role] = Sample(r.get("value"), reading_invalid(r))

        return handle

    def _connected_handler(self, device: str) -> ReadingCallback:
        def handle(reading):
            with self._cache_lock:
                self._connected[device] = _only_reading(reading).get("value")

        return handle

    def _on_stamp(self, reading) -> None:
        value = _only_reading(reading).get("value")
        with self._cache_lock:
            if value == self._stamp:
                return
            self._stamp = value
            self._stamp_at = self._clock()
        self._arm_stale_timer()

    def _inputs(self) -> Inputs:
        with self._cache_lock:
            age = None
            if self._stamp_at is not None:
                age = self._clock() - self._stamp_at
            return Inputs(
                value=self._samples["value"],
                target=self._samples["target"],
                disconnected=tuple(
                    device
                    for device, value in self._connected.items()
                    if value == GATEWAY_DISCONNECTED
                ),
                stamp_age_s=age,
            )

    # -- the stale re-check -------------------------------------------------

    def _arm_stale_timer(self) -> None:
        """(Re)schedule the stale check ``max_age_s`` after the last stamp.

        The gateway posts a stamp only when it changes, so a frozen device
        sends nothing at all: this timer is what notices.
        """
        condition = self._condition
        if not isinstance(condition, StaleCondition) or condition.max_age_s is None:
            return
        run_engine = self.RE
        if run_engine is None:
            return
        loop = run_engine._loop
        delay = condition.max_age_s + STALE_SLACK_S

        def arm() -> None:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = loop.call_later(delay, self._recheck)

        loop.call_soon_threadsafe(arm)

    def _recheck(self) -> None:
        self._timer = None
        self(None)

    # -- the stock suspender hooks -----------------------------------------

    def install(self, RE: Any, *, event_type: Any = None) -> None:
        """Install on *RE*; a stale condition's age counts from now."""
        with self._cache_lock:
            self._stamp_at = self._clock()
        super().install(RE, event_type=event_type)
        self._arm_stale_timer()
        self(None)  # a condition with nothing cached yet is unsafe at once

    def remove(self) -> None:
        """Remove from the RunEngine and cancel the stale re-check."""
        run_engine = self.RE
        timer, self._timer = self._timer, None
        if timer is not None and run_engine is not None:
            run_engine._loop.call_soon_threadsafe(timer.cancel)
        super().remove()

    def _should_suspend(self, value: Any) -> bool:
        verdict = evaluate(self._condition, self._inputs(), tripped=self._tripped)
        previous, self._verdict = self._verdict, verdict
        if previous != verdict and self._on_change is not None:
            try:
                self._on_change(self)
            except Exception:
                logger.exception("interlock %s: change callback failed", self.name)
        return verdict.unsafe

    def _should_resume(self, value: Any) -> bool:
        # The stock __call__ asks _should_suspend first, every time.
        return self._verdict is not None and not self._verdict.unsafe

    def _get_justification(self) -> str:
        if not self.tripped:
            return ""
        reason = self._verdict.diagnostic if self._verdict else "unsafe"
        return f"interlock {self.name}: {reason}"


def watched_devices(interlock: Interlock) -> list[str]:
    """The GEECS devices an interlock's condition reads, in order, once each."""
    condition = interlock.condition
    devices = [condition.device]
    if isinstance(condition, AlignmentCondition) and condition.target_device:
        devices.append(condition.target_device)
    return list(dict.fromkeys(devices))


__all__ = [
    "STALE_SLACK_S",
    "GeecsInterlockSuspender",
    "Watched",
    "WatchedSignals",
    "watched_devices",
]
