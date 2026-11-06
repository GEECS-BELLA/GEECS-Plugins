"""Fly mode: a software tick is the shot clock; no timing in the loop.

``acquisition="fly"`` on a bound ``count`` / ``sweep``.  The run is the
shot-table layout (:mod:`~geecs_bluesky.plans.on_demand`) clocked by a
**soft-tick witness** (:class:`SoftTickClock`): a timer in the worker that
publishes the witness contract — a count, a stamp per count, an epoch (the
run's uid) — at ``shot_period`` (default 1 s).  One ``primary`` row per tick;
the start document says ``clock: "soft-tick"``, so no reader mistakes a
tick for a counted shot.

**The trigger box is never driven**: no state, no fire, no read of it.  A
Software-source native camera is pressed at each tick and acquires on the
row's clock (``synchronous``); every other per-shot device free-runs into
its stream (``asynchronous``: the join samples it last-before, with its age,
*stale* past a limit) — so a device on a hardware trigger line records only
while the box already free-runs.

**Completeness.**  A pressed device is complete when its counter moved
between the read before the tick and the read after the wait (the wait ends
at the next tick).  A free-running one is complete when it recorded in the
window ending at the tick, (t_{k-1}, t_k] — what the join's
last-before rule assigns.  So a required free-running device slower than
the tick rate holds ticks.  A short tick holds one more tick (a new row,
``attempt`` + 1) up to the budget, then is flagged; two exhausted in a row
pause the run with the reason.  Nothing fires, so nothing refires.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Sequence
from typing import Any, Callable

import bluesky.plan_stubs as bps
from bluesky.preprocessors import plan_mutator
from bluesky.utils import Msg
from geecs_data_utils.shot_table import (
    ASYNCHRONOUS,
    SOFT_TICK,
    STREAM_META_KEY,
    SYNCHRONOUS,
)

from geecs_bluesky.devices.witness import COUNT_KEY, EPOCH_KEY, STAMP_KEY
from geecs_bluesky.plans.on_demand import (
    POLL_S,
    OnDemand,
    SlotColumns,
    _until_moved,
    on_demand_per_shot,
    on_demand_per_step,
    start_keys,
)
from geecs_bluesky.plans.streams import DeviceStreams, stream_name

logger = logging.getLogger(__name__)

#: The tick period when the request names no ``shot_period``, seconds.
DEFAULT_TICK_PERIOD_S = 1.0
#: The run's log line: what a fly run does not do.
BOX_NOT_DRIVEN = (
    "the box is not driven in fly mode — a device on a hardware trigger line "
    "records only while the box already free-runs"
)


class SoftTickClock:
    """The soft-tick witness: count, stamp (Unix seconds) and epoch (the run uid) per row.

    A plain Readable read in every row like the hardware witness, under the
    same keys.  It is also the slot's pause source: the RunEngine pauses
    every object it has seen, and the stock verb stages it before
    ``open_run``, so ``pause_count`` sees a pause in any slot.
    """

    name = "soft_tick"
    parent = None
    #: The slot reads it as a shot control (:class:`OnDemand`): no box at all.
    boxless = False

    def __init__(self) -> None:
        self.count = 0
        self.stamp = math.nan
        self.epoch = ""
        self.pause_count = 0
        self._column_headers = {
            COUNT_KEY: "soft-tick count",
            STAMP_KEY: "soft-tick stamp",
            EPOCH_KEY: "soft-tick epoch",
        }

    def tick(self) -> None:
        """The next shot: count + 1, stamped now."""
        self.count += 1
        self.stamp = time.time()

    def read(self) -> dict[str, Any]:
        """The row's three witness columns."""
        at = time.time() if math.isnan(self.stamp) else self.stamp
        data = {COUNT_KEY: self.count, STAMP_KEY: self.stamp, EPOCH_KEY: self.epoch}
        return {k: {"value": v, "timestamp": at} for k, v in data.items()}

    def describe(self) -> dict[str, Any]:
        """The three data keys."""
        dtypes = {COUNT_KEY: "integer", STAMP_KEY: "number", EPOCH_KEY: "string"}
        return {
            k: {"source": f"soft-tick:{k}", "dtype": d, "shape": []}
            for k, d in dtypes.items()
        }

    def pause(self) -> None:
        """A RunEngine pause: counted, for the slot's interruption guard."""
        self.pause_count += 1

    def resume(self) -> None:
        """Nothing to restore."""

    def wrap(self, plan: Any) -> Any:
        """*plan* with the run's uid taken as the epoch at ``open_run`` (the count from 0)."""

        def capture(msg: Msg):
            if msg.command != "open_run":
                return None, None

            def opened():
                uid = yield msg
                self.count, self.stamp, self.epoch = 0, math.nan, str(uid)
                return uid

            return opened(), None

        return (yield from plan_mutator(plan, capture))


class FlyRun(OnDemand):
    """The on-demand slot with the tick for its fire (module docstring).

    Parameters
    ----------
    clock :
        The run's :class:`SoftTickClock`.
    streams, required :
        As :class:`OnDemand`'s.
    period :
        Seconds between ticks.
    max_refires :
        Extra ticks a step may hold after an incomplete one.
    """

    AGAIN = "holding another tick"
    EXHAUSTED_HINT = f" — {BOX_NOT_DRIVEN}"

    def __init__(
        self,
        clock: SoftTickClock,
        streams: DeviceStreams,
        required: Sequence[Any],
        *,
        period: float = DEFAULT_TICK_PERIOD_S,
        max_refires: int = 2,
    ) -> None:
        super().__init__(
            clock, streams, required, max_refires=max_refires, shot_period=period
        )
        self.clock = clock
        self.period = float(period)
        self.columns = SlotColumns(
            [p.name for p in self.required], scan_mode=False, fly=True
        )
        self.pressed = {self.progress[id(d)].name for d in self.presses}
        self._next: float | None = None
        self._prev: dict[str, Any] = {}

    def _throttle(self):
        if self._next is None:
            # The first tick waits one period, so row 1 has a window too.
            self._prev = yield from self._counters()
            self._next = time.monotonic() + self.period
        remaining = self._next - time.monotonic()
        if remaining > 0:
            yield from bps.sleep(remaining)

    def _fire(self):
        # An absolute schedule; more than a period late (a pause, an
        # overrun) re-anchors at now, so no burst of catch-up ticks.
        now = time.monotonic()
        following = (self._next or now) + self.period
        self._next = following if following > now else now + self.period
        self.clock.tick()
        if self.presses:
            yield from bps.wait_for([self._press])

    def _wait(self, base: dict[str, Any]):
        pressed = [p for p in self.required if p.name in self.pressed]
        if not pressed:
            return
        until_next = (self._next or time.monotonic()) - time.monotonic() - POLL_S
        budget = max(min(max(p.timeout for p in pressed), until_next), 0.0)

        async def wait() -> None:
            deadline = asyncio.get_running_loop().time() + budget
            await asyncio.gather(
                *(_until_moved(p, base[p.name], deadline) for p in pressed)
            )

        yield from bps.wait_for([wait])

    def _moved(self, p: Any, base: dict[str, Any], now: dict[str, Any]) -> bool:
        if p.name in self.pressed:
            return super()._moved(p, base, now)
        return p.moved(self._prev.get(p.name, ()), base[p.name])

    def _settled(self, base: dict[str, Any]) -> None:
        self._prev = base

    def _verdict(self, base: dict[str, Any], now: dict[str, Any]):
        values = self.columns.values
        missing = [p for p in self.required if not self._moved(p, base, now)]
        values["missing"] = ",".join(p.name for p in missing)
        values["complete"] = not missing
        return missing, True  # a tick always happened


def fly_start_keys(
    streams: DeviceStreams, required: Sequence[Any], run: FlyRun
) -> dict[str, Any]:
    """The start-document keys of a fly run: the shot-table keys, the soft-tick clock."""
    keys = start_keys(streams, required, witnessed=True)
    pressed = {stream_name(d) for d in run.presses}
    keys[STREAM_META_KEY] = {
        name: {
            "sync_class": SYNCHRONOUS if name in pressed else ASYNCHRONOUS,
            "stale_after": None,
        }
        for name in streams.names
    }
    keys["clock"] = SOFT_TICK
    keys["shot_period"] = run.period
    return keys


def bind_fly(
    streams: DeviceStreams,
    required: Sequence[Any],
    *,
    hook: str,
    shots_per_step: int,
    period: float,
) -> tuple[SoftTickClock, dict[str, Any], Callable]:
    """The clock, the start keys and the stock hook of one fly run."""
    clock = SoftTickClock()
    run = FlyRun(clock, streams, required, period=period)
    per = (
        on_demand_per_step(run, shots_per_step=shots_per_step)
        if hook == "per_step"
        else on_demand_per_shot(run)
    )
    return clock, fly_start_keys(streams, required, run), per


__all__ = [
    "BOX_NOT_DRIVEN",
    "DEFAULT_TICK_PERIOD_S",
    "FlyRun",
    "SoftTickClock",
    "bind_fly",
    "fly_start_keys",
]
