"""ShotSampler — one event per shot for every device that has no plugin.

The gated batch (``Planning/native_bluesky/08_gated_batch.md`` §4.7, Sam's
answer to §6 Q1) has no per-shot ``create/read/save``: the box free-runs
and the plugin-backed cameras count their own frames.  Every *other*
device of the run — the scalar-only devices (magnets, gauges, stages at
their own ~5 Hz cadence), the triggered scalar devices without a plugin
(ICTs, energy meters), a camera's ``.scalars`` view, the scanned motors'
readbacks and the ``bin_number`` counter — is recorded by this small
software device, driven by the stock ``prepare → kickoff → complete →
collect`` verbs (``Flyable`` + ``EventCollectable`` + ``Preparable``):

- **Clock.**  An essential *triggered* device's ``acq_timestamp`` — the
  signal strict waits on.  Every advance of it past the value read at
  ``kickoff`` is one shot; ticks after the quota are ignored.
- **Row.**  On each tick, the latest cached reading of every member (the
  strict row's rule: *the latest value of every subscribed non-plugin
  signal, into a row the trigger generated*), plus the clock's own stamp
  column so the row joins to the cameras' frames by stamp (§4.5).
- **Quota.**  ``prepare(N)`` sets the step's shot count; ``complete`` is
  done after *N* ticks, or fails with
  :exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError` when the clock
  stops for ``shot_timeout`` — the sampler *counts*, so a gated run needs at
  least one essential triggered device even without a camera.
- **Stream.**  ``collect`` yields the rows as events of the ``shots``
  stream (one event per shot; at 1 Hz, no flood).  ``describe_collect`` is
  the union of the members' descriptions.

The members are read through their own ``read`` / ``describe`` (a
``GeecsDetector`` or its ``.scalars`` view through the detector's scalar
signals — the detector's own ``read`` needs a prepare it never gets here);
staged by the stock plan, they read from their monitor caches.  Nothing
here touches Channel Access directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from bluesky.protocols import Reading
from bluesky.utils import maybe_await
from event_model import DataKey
from event_model.documents.event import PartialEvent
from ophyd_async.core import AsyncStatus, SignalR, merge_gathered_dicts

from geecs_bluesky.devices.ca._view import ScalarsView
from geecs_bluesky.devices.detector import DEFAULT_SHOT_TIMEOUT, GeecsDetector
from geecs_bluesky.exceptions import GeecsTriggerTimeoutError

logger = logging.getLogger(__name__)

_Describe = Callable[[], Awaitable[dict[str, DataKey]]]
_Read = Callable[[], Awaitable[dict[str, Reading]]]


def _readers_for(obj: Any) -> list[tuple[Any, _Describe, _Read]]:
    """``(key object, describe, read)`` per column source of *obj*.

    A :class:`GeecsDetector` (a triggered scalar device without a plugin)
    and a :class:`ScalarsView` of one contribute the detector's scalar
    signals individually — the detector's ``read`` needs a prepare context
    the sampler never gives it, and the view masks a missed shot the
    sampler never fires.  Anything else with ``read`` / ``describe`` (a
    scalar-only device, a motor, a signal, the bin counter) is one source.
    """
    owner = obj._owner if isinstance(obj, ScalarsView) else obj
    if isinstance(owner, GeecsDetector):
        return [
            (sig, sig.describe, sig.read)  # type: ignore[list-item]
            for sig in owner._scalar_signals()
        ]
    if isinstance(obj, ScalarsView):
        return [(owner, owner.describe, owner.read)]

    async def describe() -> dict[str, DataKey]:
        return await maybe_await(obj.describe())

    async def read() -> dict[str, Reading]:
        return await maybe_await(obj.read())

    return [(obj, describe, read)]


class ShotSampler:
    """One event per shot of the clock, over every non-plugin member.

    Parameters
    ----------
    members :
        The devices / signals to record per shot (the strict row's
        non-plugin part: scalar-only devices, triggered scalar devices,
        ``.scalars`` views, motors, the bin counter).
    clock :
        The ``acq_timestamp`` signal of an essential triggered device; its
        advance is the shot.  Its column is always in the row.
    clock_name :
        The GEECS name of the clock's device (messages).
    shot_timeout :
        Seconds the clock may stay silent before ``complete`` fails — one
        trigger period plus the device's exposure and drain, the strict
        budget.
    name :
        The Bluesky object name (the ``shots`` stream's collect object).
    """

    parent = None

    def __init__(
        self,
        members: Sequence[Any],
        clock: SignalR[float],
        *,
        clock_name: str,
        shot_timeout: float = DEFAULT_SHOT_TIMEOUT,
        name: str = "shot_sampler",
    ) -> None:
        self.name = name
        self._members = list(members)
        self._clock = clock
        self._clock_name = clock_name
        self.shot_timeout = shot_timeout
        readers: dict[int, tuple[_Describe, _Read]] = {}
        for member in self._members:
            for key, describe, read in _readers_for(member):
                readers.setdefault(id(key), (describe, read))
        readers.setdefault(id(clock), (clock.describe, clock.read))
        self._readers = list(readers.values())
        self._quota = 0
        self._rows: list[PartialEvent] = []
        self._ticks: asyncio.Queue[float] = asyncio.Queue()
        self._baseline: float | None = None
        self._last: float | None = None
        self._task: asyncio.Task[None] | None = None
        self._cancelled = asyncio.Event()
        self._subscribed = False
        #: Shots sampled in the current step (tests, logs).
        self.sampled = 0

    # ----------------------------------------------------------- protocols
    @property
    def members(self) -> list[Any]:
        """The recorded members (the clock's own signal aside)."""
        return list(self._members)

    @property
    def clock_name(self) -> str:
        """The GEECS device whose stamp is the shot clock."""
        return self._clock_name

    def prepare(self, value: int) -> AsyncStatus:
        """Arm the step: *value* shots to sample, the rows cleared."""

        async def do() -> None:
            quota = int(value)
            if quota < 1:
                raise ValueError(f"the sampler needs at least one shot, got {quota}")
            await self._stop_task()
            self._quota = quota
            self._rows = []
            self.sampled = 0
            self._ticks = asyncio.Queue()
            self._cancelled = asyncio.Event()

        return AsyncStatus(do())

    def kickoff(self) -> AsyncStatus:
        """Read the clock's current stamp as the baseline and start sampling."""

        async def do() -> None:
            if self._quota < 1:
                raise RuntimeError("ShotSampler.kickoff before prepare")
            await self._stop_task()
            self._baseline = await self._clock.get_value()
            self._last = self._baseline
            self._ticks = asyncio.Queue()
            self._cancelled = asyncio.Event()
            self._clock.subscribe_reading(self._on_clock)
            self._subscribed = True
            self._task = asyncio.create_task(self._sample())

        return AsyncStatus(do())

    def complete(self) -> AsyncStatus:
        """Done when the quota is sampled; fails when the clock stops."""

        async def do() -> None:
            if self._task is None:
                raise RuntimeError("ShotSampler.complete before kickoff")
            await self._task

        return AsyncStatus(do())

    async def describe_collect(self) -> dict[str, DataKey]:
        """The row's data keys: every member's, plus the clock's stamp."""
        return await merge_gathered_dicts(describe() for describe, _ in self._readers)

    async def collect(self):
        """Yield the sampled rows (one event per shot) and forget them."""
        rows, self._rows = self._rows, []
        for row in rows:
            yield row

    async def cancel_step(self) -> None:
        """Abandon the step: stop sampling, drop the rows, settle the task."""
        self._cancelled.set()
        await self._stop_task()
        self._rows = []

    # ------------------------------------------------------------ internals
    def _on_clock(self, reading: dict[str, Any]) -> None:
        value = reading[self._clock.name]["value"]
        if value is None or value <= 0 or value == self._last:
            return  # the subscribe echo, the gateway's placeholder, or a repeat
        self._last = value
        self._ticks.put_nowait(float(value))

    async def _sample(self) -> None:
        try:
            while self.sampled < self._quota:
                tick = asyncio.ensure_future(self._ticks.get())
                cancel = asyncio.ensure_future(self._cancelled.wait())
                done, _ = await asyncio.wait(
                    {tick, cancel},
                    timeout=self.shot_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if tick not in done:
                    tick.cancel()
                    cancel.cancel()
                    if cancel in done:
                        return
                    raise GeecsTriggerTimeoutError(
                        self._clock_name,
                        self.shot_timeout,
                        f"shot sampler: no shot from {self._clock_name} (the shot "
                        f"clock) within {self.shot_timeout:.1f}s while the box ran",
                    )
                cancel.cancel()
                stamp = tick.result()
                readings = await merge_gathered_dicts(
                    read() for _, read in self._readers
                )
                data = {k: r["value"] for k, r in readings.items()}
                # The row is the tick's shot: its stamp is the shot id even
                # when the next shot's stamp landed in the cache meanwhile.
                data[self._clock.name] = stamp
                self._rows.append(
                    PartialEvent(
                        time=time.time(),
                        data=data,
                        timestamps={k: r["timestamp"] for k, r in readings.items()},
                    )
                )
                self.sampled += 1
                logger.debug(
                    "%s: shot %d/%d (stamp %s)",
                    self.name,
                    self.sampled,
                    self._quota,
                    stamp,
                )
        finally:
            self._unsubscribe()

    def _unsubscribe(self) -> None:
        if self._subscribed:
            try:
                self._clock.clear_sub(self._on_clock)
            except Exception:  # noqa: BLE001 - best effort on a torn-down signal
                logger.debug("%s: clear_sub failed", self.name, exc_info=True)
            self._subscribed = False

    async def _stop_task(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            self._cancelled.set()
            try:
                await task
            except Exception:  # noqa: BLE001 - settled here on purpose
                logger.debug(
                    "%s: sampling task ended with an error", self.name, exc_info=True
                )
        self._unsubscribe()


__all__ = ["ShotSampler"]
