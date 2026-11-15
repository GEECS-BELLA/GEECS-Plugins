"""StampStream — one event per stamp a streamed triggered device publishes.

A per-shot device without a file plugin (a LabVIEW-native saver, a scalar
device with a stamp, a detector's ``.scalars`` view) is recorded for the
run in its own ``<name>_stream``: one event per stamp, joined to the rows
by stamp afterwards.  Members are read from monitor caches; nothing here
touches Channel Access.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from bluesky.protocols import Reading
from bluesky.utils import maybe_await
from event_model import DataKey
from event_model.documents.event import PartialEvent
from ophyd_async.core import AsyncStatus, SignalR, merge_gathered_dicts

from geecs_bluesky.devices.ca._view import ScalarsView, owner_of
from geecs_bluesky.devices.detector import GeecsDetector

logger = logging.getLogger(__name__)

_Describe = Callable[[], Awaitable[dict[str, DataKey]]]
_Read = Callable[[], Awaitable[dict[str, Reading]]]


def _readers_for(obj: Any) -> list[tuple[Any, _Describe, _Read]]:
    """``(key object, describe, read)`` per column source of *obj*.

    A :class:`GeecsDetector` (a triggered device without a plugin) and a
    :class:`ScalarsView` of one contribute the detector's scalar signals
    individually — the detector's ``read`` is its prepared readables, and
    the view masks a missed shot no one fires here.  A **native-saving**
    detector listed itself (not through its view) also contributes those
    prepared readables: the streams prepare it once, unbounded, and in that
    fly prepare its whole per-event reading is the ``-nonscalar_save_path``
    column — the run-long constant that says where its files landed.
    Anything else with ``read`` / ``describe`` is one source.
    """
    owner = owner_of(obj)
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


class StampStream:
    """One event per stamp a streamed device publishes, for the whole run.

    The **stream** of a device with no file plugin: a *triggered* device —
    a camera or a wavefront sensor saving its own LabVIEW files, a scalar
    device with a stamp (a power supply, a gauge) — in a shot-table run
    (every per-shot device).
    Sampling it at the row would either wait for it (its write time back in
    the rep rate) or record the previous shot.  So it gets its own event
    stream instead, the shape a plugin camera's ``<name>_stream`` has: this
    small software device (``Flyable`` + ``EventCollectable``) subscribes
    to the device's ``acq_timestamp`` at ``kickoff`` and, **the moment**
    each new stamp arrives, records one event — the stamp itself, the
    device's cached scalars and (a native saver, prepared unbounded by the
    plan) its ``-nonscalar_save_path``.  Nothing waits on it and nothing
    attributes at read time: its events join the rows by stamp afterwards
    (``geecs_data_utils.shot_join``), a device slower than the rep rate
    leaving ``NaN`` on the rows it missed.

    Rule 2 holds: the device's own lifecycle switches its saving on (the
    plan's unbounded prepare) and off (its ``unstage``); this object only
    reads.  ``complete`` is immediate — the run's close is the stream's
    end — and ``collect`` yields what was recorded.

    Parameters
    ----------
    device :
        The streamed device: a :class:`GeecsDetector` without a file
        plugin, or a detector's ``.scalars`` view (scalars and stamp only,
        no files) — or a plugin camera's gateway stamp, for a *companion*.
    name :
        The stream object's name; the device's own by default.  A plugin
        camera's **companion** (optimize's per-shot camera stamp, beside the
        camera's own datum-only stream) takes another, so the two streams
        never share a name.
    """

    parent = None

    def __init__(self, device: Any, *, name: str | None = None) -> None:
        owner = owner_of(device)
        if not isinstance(owner, GeecsDetector):
            raise TypeError(
                f"{getattr(device, 'name', device)!r} has no shot stamp: a "
                "device stream records a triggered device"
            )
        self._device = device
        self._owner = owner
        # The DEVICE's name: the descriptor's configuration and object_keys
        # are keyed by the collect object's name, and the join looks the
        # drain offset up under the device's (``<name>-drain_offset``).
        self.name = owner.name if name is None else name
        self._stamp: SignalR[float] = owner.acq_timestamp
        self._readers = [(d, r) for _, d, r in _readers_for(device)]
        self._events: list[PartialEvent] = []
        self._queue: asyncio.Queue[float | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._last: float | None = None
        self._subscribed = False
        #: Events recorded this run (the close's WARNING when it stays 0).
        self.published = 0
        #: Stamps queued and not yet recorded.
        self.pending = 0
        #: Stamps that arrived but could not be recorded (a failed read):
        #: settled, never coming — a waiter counts them with ``published``.
        self.failed = 0

    @property
    def device(self) -> Any:
        """The device this stream records (as listed)."""
        return self._device

    @property
    def device_name(self) -> str:
        """The GEECS device name (messages)."""
        return self._owner._geecs_device_name

    async def describe_collect(self) -> dict[str, DataKey]:
        """The event's data keys: the device's scalars, its stamp, its save path."""
        return await merge_gathered_dicts(describe() for describe, _ in self._readers)

    async def read_configuration(self) -> dict[str, Reading]:
        """The device's drain offset — what the join corrects its stamps by."""
        return await self._owner.drain_offset.read()

    async def describe_configuration(self) -> dict[str, DataKey]:
        """Data key of the drain offset."""
        return await self._owner.drain_offset.describe()

    def kickoff(self) -> AsyncStatus:
        """Take the current stamp as already seen and record every later one."""

        async def do() -> None:
            await self._stop()
            self._events = []
            self.published = self.pending = self.failed = 0
            self._last = await self._stamp.get_value()
            self._queue = asyncio.Queue()
            self._stamp.subscribe_reading(self._on_stamp)
            self._subscribed = True
            self._task = asyncio.create_task(self._record())

        return AsyncStatus(do())

    def complete(self) -> AsyncStatus:
        """Immediately: stop listening, record what already arrived, done."""
        return AsyncStatus(self._stop())

    async def collect(self):
        """Yield the recorded events (one per stamp) and forget them."""
        events, self._events = self._events, []
        for event in events:
            yield event

    # ------------------------------------------------------------ internals
    def _on_stamp(self, reading: dict[str, Any]) -> None:
        value = reading[self._stamp.name]["value"]
        if value is None or value <= 0 or value == self._last:
            return  # the subscribe echo, the gateway's placeholder, or a repeat
        self._last = value
        self.pending += 1
        self._queue.put_nowait(float(value))

    async def _record(self) -> None:
        while True:
            stamp = await self._queue.get()
            if stamp is None:
                return
            try:
                readings = await merge_gathered_dicts(
                    read() for _, read in self._readers
                )
            except Exception as exc:  # noqa: BLE001 - a stream never fails
                self.pending -= 1
                self.failed += 1
                logger.warning(
                    "%s: stamp %s not recorded (%s: %s)",
                    self.name,
                    stamp,
                    type(exc).__name__,
                    exc,
                )
                continue
            data = {k: r["value"] for k, r in readings.items()}
            # The event is the stamp's shot even if a newer stamp has landed
            # in the cache meanwhile.
            data[self._stamp.name] = stamp
            self._events.append(
                PartialEvent(
                    time=time.time(),
                    data=data,
                    timestamps={k: r["timestamp"] for k, r in readings.items()},
                )
            )
            self.published += 1
            self.pending -= 1

    async def _stop(self) -> None:
        if self._subscribed:
            try:
                self._stamp.clear_sub(self._on_stamp)
            except Exception:  # noqa: BLE001 - best effort on a torn-down signal
                logger.debug("%s: clear_sub failed", self.name, exc_info=True)
            self._subscribed = False
        task, self._task = self._task, None
        if task is not None and not task.done():
            self._queue.put_nowait(None)  # after every stamp already queued
            await task


__all__ = ["StampStream"]
