"""ShotSampler — one event per shot for every device that has no plugin.

The gated batch has no per-shot ``create/read/save``: the box free-runs
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
  column so the row joins to the cameras' frames by stamp — and, for a
  native-saving essential (a device without a plugin whose LabVIEW files
  are its record), its ``-nonscalar_save_path`` column, the run-long
  constant its files are found under; they join by stamp too.
- **Settle.**  A member with a stamp of its own (a triggered device
  without a plugin, or its view) is not read at the tick: its stamp lands
  after the clock's whenever its device is slower (the HASO: ~40 ms, Scan015
  of 26_0925), and a reading taken at the tick is then the *previous*
  shot's.  Each such member is given :data:`SETTLE_TIMEOUT_S` for its
  cached stamp to fall within :data:`SHOT_WINDOW_S` of the clock's; then
  its scalars, its stamp and its save-path column are read.  One that does
  not make it missed the shot: its numeric columns read ``NaN`` (a string
  column, the save path, stays), the miss is counted in :attr:`missed` and
  logged once per member per step, and its file for that row is simply
  absent — never the previous shot's.
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

#: How long the sampler waits, after the clock's stamp, for each other
#: stamped member's stamp to be this shot's: half the 1 Hz period.  A
#: device that has not stamped by then missed the shot — its columns read
#: NaN, never the previous shot's values.  Found on Scan015 of 26_0925:
#: the HASO's stamp lands ~40 ms after the clock's, and sampling at the
#: tick recorded every row one frame late, so its files joined one row
#: late too.
SETTLE_TIMEOUT_S = 0.5
#: A member's stamp is this shot's when it lies within this many seconds
#: of the clock's stamp (the devices stamp one edge within ~0.3 s of each
#: other; the period is 1 s).  A cached stamp further away is a previous
#: shot's (or a self-triggered frame's) and is waited past, not recorded.
SHOT_WINDOW_S = 0.5


def _blank(readings: dict[str, Reading]) -> dict[str, Reading]:
    """*readings* with every numeric value ``NaN``: the member missed the shot.

    Strings stay (a native saver's ``-nonscalar_save_path`` is a run-long
    constant, not a per-shot reading); so do booleans.
    """
    out: dict[str, Reading] = {}
    for key, reading in readings.items():
        value = reading["value"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            reading = {**reading, "value": float("nan")}
        out[key] = reading
    return out


def _readers_for(obj: Any) -> list[tuple[Any, _Describe, _Read]]:
    """``(key object, describe, read)`` per column source of *obj*.

    A :class:`GeecsDetector` (a triggered device without a plugin) and a
    :class:`ScalarsView` of one contribute the detector's scalar signals
    individually — the detector's ``read`` is its prepared readables, and
    the view masks a missed shot the sampler never fires.  A
    **native-saving** detector listed itself (not through its view) also
    contributes those prepared readables: the gated plan prepares it once,
    unbounded, before the ``shots`` stream is declared, and in that fly
    prepare its whole per-event reading is the ``-nonscalar_save_path``
    column — the run-long constant that says where its files landed.
    Anything else with ``read`` / ``describe`` (a scalar-only device, a
    motor, a signal, the bin counter) is one source.
    """
    owner = obj._owner if isinstance(obj, ScalarsView) else obj
    if isinstance(owner, GeecsDetector):
        readers: list[tuple[Any, _Describe, _Read]] = [
            (sig, sig.describe, sig.read)  # type: ignore[list-item]
            for sig in owner._scalar_signals()
        ]
        if obj is owner and owner.native_save and not owner.plugin_backed:
            readers.append((owner, owner.describe, owner.read))
        return readers
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
    settle_timeout, shot_window :
        The other stamped members' budget and window
        (:data:`SETTLE_TIMEOUT_S`, :data:`SHOT_WINDOW_S`): after the tick,
        each triggered member without a plugin is given *settle_timeout*
        for its own stamp to land within *shot_window* of the clock's;
        one that does not is recorded as ``NaN`` for that shot.
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
        settle_timeout: float = SETTLE_TIMEOUT_S,
        shot_window: float = SHOT_WINDOW_S,
        name: str = "shot_sampler",
    ) -> None:
        self.name = name
        self._members = list(members)
        self._clock = clock
        self._clock_name = clock_name
        self.shot_timeout = shot_timeout
        self.settle_timeout = float(settle_timeout)
        self.shot_window = float(shot_window)
        # The plain sources — the clock's own signal, the clock device's
        # scalars, every unstamped member — are read at the tick.  A member
        # with a stamp of its own (a triggered device without a plugin, or
        # its view) is read once its stamp says the shot is this one, or
        # blanked (``_settle``): sampled at the tick it would carry the
        # previous shot's values whenever its stamp lands after the clock's.
        plain: dict[int, tuple[_Describe, _Read]] = {}
        stamped: dict[
            int, tuple[str, SignalR[float], dict[int, tuple[_Describe, _Read]]]
        ] = {}
        for member in self._members:
            owner = member._owner if isinstance(member, ScalarsView) else member
            acq = getattr(owner, "acq_timestamp", None)
            if (
                isinstance(owner, GeecsDetector)
                and acq is not None
                and acq is not clock
            ):
                _name, _acq, sources = stamped.setdefault(
                    id(owner), (owner._geecs_device_name, acq, {})
                )
                for key, describe, read in _readers_for(member):
                    sources.setdefault(id(key), (describe, read))
            else:
                for key, describe, read in _readers_for(member):
                    plain.setdefault(id(key), (describe, read))
        plain.setdefault(id(clock), (clock.describe, clock.read))
        self._readers = list(plain.values())
        self._stamped: list[
            tuple[str, SignalR[float], list[tuple[_Describe, _Read]]]
        ] = [
            (name_, acq, list(sources.values()))
            for name_, acq, sources in stamped.values()
        ]
        #: Shots each stamped member missed in the current step (name → count).
        self.missed: dict[str, int] = {}
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
            self.missed = {}
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
        describes = [describe for describe, _ in self._readers]
        describes += [
            describe for _, _, sources in self._stamped for describe, _ in sources
        ]
        return await merge_gathered_dicts(describe() for describe in describes)

    async def collect(self):
        """Yield the sampled rows (one event per shot) and forget them."""
        rows, self._rows = self._rows, []
        for row in rows:
            yield row

    def mark_cancelled(self) -> None:
        """Synchronously: the step is over; the sampling task ends quietly."""
        self._cancelled.set()

    async def cancel_step(self) -> None:
        """Abandon the step: stop sampling, drop the rows, settle the task."""
        self.mark_cancelled()
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
                # The other stamped members: this shot's stamp, or NaN.
                deadline = time.monotonic() + self.settle_timeout
                settled = await asyncio.gather(
                    *(self._settle(acq, stamp, deadline) for _, acq, _ in self._stamped)
                )
                readings = await merge_gathered_dicts(
                    read() for _, read in self._readers
                )
                for (member_name, _acq, sources), ok in zip(self._stamped, settled):
                    member = await merge_gathered_dicts(read() for _, read in sources)
                    if not ok:
                        member = _blank(member)
                        self.missed[member_name] = self.missed.get(member_name, 0) + 1
                        if self.missed[member_name] == 1:
                            logger.info(
                                "%s: %s stamped nothing within %.2f s of shot %s — "
                                "its columns read NaN (a missing frame; counted at "
                                "the step's end)",
                                self.name,
                                member_name,
                                self.settle_timeout,
                                stamp,
                            )
                    readings.update(member)
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
            if self.missed:
                logger.info(
                    "%s: shots missed this step — %s",
                    self.name,
                    ", ".join(
                        f"{name_} {count} of {self.sampled}"
                        for name_, count in self.missed.items()
                    ),
                )
        finally:
            self._unsubscribe()

    async def _settle(self, acq: SignalR[float], tick: float, deadline: float) -> bool:
        """Whether *acq*'s cached stamp becomes this shot's before *deadline*.

        This shot's: within :attr:`shot_window` of the clock's *tick*.  The
        signal is staged by the plan, so each look is the monitor cache,
        not a Channel Access get.
        """
        while True:
            value = await acq.get_value()
            if value is not None and abs(float(value) - tick) <= self.shot_window:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.01)

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
