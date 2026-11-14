"""Fly mode: a software tick is the shot clock; no timing in the loop.

``acquisition="fly"`` on a bound ``count`` / ``sweep``: the shot-table layout
clocked by a **soft-tick witness** (:class:`SoftTickClock`: a count, a stamp
per count, the run's uid as epoch) every ``shot_period`` (default 1 s); the
start document says ``clock: "soft-tick"``.  **The trigger box is never
driven** (no state, no fire, no read), and no default profile is resolved: a
named scan-mode profile is refused, any other ignored.

A Software-source native is pressed at each tick (``synchronous``); every
other per-shot device free-runs (``asynchronous``).  Completeness and the
join's assignment are one interval rule: a pressed frame in
[t_k - m, t_k+1 - m) (*m* the stamp budget), a free-runner's record in
(t_k-1, t_k], stale past a period — so a required free-runner slower than
the tick holds ticks.  That window is re-opened (counters read again, then
a full period) at the run's first tick, each step's first and after a
pause, so it never spans a move or a pause.  A short tick holds another (a
new row, ``attempt`` + 1) up to the budget, then is flagged; two exhausted
in a row pause the run, the reason saying the box is not driven.
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
    STAMP_BUDGET_S,
    STREAM_META_KEY,
    SYNCHRONOUS,
)

from geecs_bluesky.devices.witness import COUNT_KEY, EPOCH_KEY, STAMP_KEY
from geecs_bluesky.plans.on_demand import (
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
    #: A free-running camera is legitimate in fly mode.
    FRAME_CHECK = False

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
        self._prev_pauses = self._base_pauses = 0

    def restart(self) -> None:
        """A new step (the motors just moved): the next tick opens a fresh window."""
        self._next = None

    def _throttle(self):
        # The window (t_k-1, t_k] of a free-running device must not span a
        # move or a pause: at the run's first tick, a step's first tick and
        # the first tick after a pause the window is re-opened — the
        # counters read again, then one full period before the tick.
        while True:
            if self._next is None or self.clock.pause_count != self._prev_pauses:
                self._prev_pauses = self.clock.pause_count
                self._prev = yield from self._counters()
                self._next = time.monotonic() + self.period
            remaining = self._next - time.monotonic()
            if remaining > 0:
                yield from bps.sleep(remaining)
            if self.clock.pause_count == self._prev_pauses:
                break
        self._base_pauses = self.clock.pause_count

    def _fire(self):
        # An absolute schedule; a tick that would follow this one by under
        # half a period (this one late: an overrun, a pause) re-anchors at
        # now, so no catch-up tick comes right after a late one.
        now = time.monotonic()
        following = (self._next or now) + self.period
        if following - now < self.period / 2:
            following = now + self.period
        self._next = following
        self.clock.tick()
        if self.presses:
            yield from bps.wait_for([self._press])

    def _wait(self, base: dict[str, Any]):
        pressed = [p for p in self.required if p.name in self.pressed]
        if not pressed:
            return
        # The tick's interval [t_k - m, t_k+1 - m), the join's (m the stamp
        # budget): a frame stamped later is the next row's.
        until_next = (
            (self._next or time.monotonic()) - time.monotonic() - STAMP_BUDGET_S
        )
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
        self._prev, self._prev_pauses = base, self._base_pauses

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
            # One period: the join's freshness is the plan's window.
            "stale_after": None if name in pressed else run.period,
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
