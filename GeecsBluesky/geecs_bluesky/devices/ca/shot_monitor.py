"""The GEECS shot signal as reusable mixins: acq_timestamp monitor + trigger.

The GEECS shot signal is a device's ``acq_timestamp`` advancing once per
shot.  Two mixins carry the machinery so any ophyd-async device that owns an
``acq_timestamp`` child can be shot-aware without re-implementing it:

* :class:`AcqTimestampMonitorMixin` — a **persistent monitor** on the
  ``acq_timestamp`` signal (started at ``connect()``, stopped by
  ``disconnect()``) feeding a latest-value cache (``_last_acq``) and a
  bounded drop-oldest update queue (``_shot_queue``).
* :class:`ShotTriggerMixin` — adds ``trigger()``, which completes once the
  queue delivers a value different from the baseline captured
  synchronously inside ``trigger()``.

Hosts must provide ``self.acq_timestamp`` (a ``SignalR[float]``) and
``self._geecs_device_name`` before calling :meth:`_init_shot_monitor`, and
must call ``super().connect(...)`` through the MRO (the mixin's ``connect``
starts the monitor after the device's signals are connected).

The historical hosts are :class:`~geecs_bluesky.devices.ca.triggerable.CaAcqTimestampReadable`
and :class:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable`; the
namespace device (:mod:`geecs_bluesky.devices.geecs_device`) is the second
host — the reason the logic lives here rather than in one class hierarchy.
Design rationale: ``GeecsBluesky/CLAUDE.md`` (Device Layer).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ophyd_async.core import AsyncStatus

from geecs_bluesky.exceptions import GeecsTriggerTimeoutError

logger = logging.getLogger(__name__)


class AcqTimestampMonitorMixin:
    """Persistent ``acq_timestamp`` monitor: latest-value cache + update queue.

    Class attributes hosts may override
    -----------------------------------
    _acq_timestamp_variable : str
        GEECS variable that advances per shot.  Default ``"acq_timestamp"``.
    _shot_queue_maxsize : int
        Bound on the shot-update queue (default 128 — the worst case is
        rep-rate × trigger-timeout updates between baseline and the awaited
        get, i.e. 15 at the 5 Hz system limit, with a wide margin so the
        bound never needs revisiting below ~40 Hz).  Only ``trigger()``
        drains the queue, so an unbounded queue would grow one float per
        machine shot on idle devices; overflow is drop-oldest and
        correctness-preserving (any surviving post-baseline update passes
        the ``!= t0`` shot test).
    """

    _acq_timestamp_variable: str = "acq_timestamp"
    _shot_queue_maxsize: int = 128

    # Provided by the host device.
    acq_timestamp: Any
    _geecs_device_name: str

    def _init_shot_monitor(self) -> None:
        """Initialise the monitor state; call once, after the host's ``__init__``."""
        self._last_acq: float | None = None
        self._shot_queue: asyncio.Queue[float] = asyncio.Queue(
            maxsize=self._shot_queue_maxsize
        )
        self._monitoring = False

    async def connect(
        self,
        mock: Any = False,
        timeout: float = 10.0,
        force_reconnect: bool = False,
    ) -> None:
        """Connect all signals, then start the persistent acq_timestamp monitor."""
        await super().connect(  # type: ignore[misc]  # the host is a Device
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        if not self._monitoring:
            self.acq_timestamp.subscribe_reading(self._on_acq_timestamp)
            self._monitoring = True

    async def disconnect(self) -> None:
        """Stop the persistent ``acq_timestamp`` monitor and drop shot state.

        Per-scan teardown hook (the runner's ``session.disconnect`` cleanup).
        Unsubscribing removes the signal cache's reference to this instance's
        bound callback — without it every per-scan device object stays alive
        and keeps enqueuing monitor updates for the rest of the process.

        Idempotent; ``connect()`` may be called again to resubscribe.
        """
        if self._monitoring:
            # SignalR.clear_sub removes the subscribe_reading callback; when
            # no listeners remain it also drops the signal cache, closing the
            # underlying CA monitor.
            self.acq_timestamp.clear_sub(self._on_acq_timestamp)
            self._monitoring = False
        while not self._shot_queue.empty():
            self._shot_queue.get_nowait()
        self._last_acq = None

    def _on_acq_timestamp(self, reading: dict[str, Any]) -> None:
        """Monitor callback: cache the latest value and queue the update.

        Non-positive values are ignored — ``0.0`` is the gateway channel's
        pre-acquisition placeholder, so "never acquired" reads as ``None``.
        The queue is a drop-oldest ring; this preserves ``trigger()``'s
        no-blind-window guarantee: the queue is drained empty at baseline
        capture, so anything dropped afterwards is older than a surviving
        update, and every post-baseline update passes the ``!= t0`` shot
        test.  Callback and consumers share the RE event loop, so the
        two-step replace is race-free.
        """
        value = reading[self.acq_timestamp.name]["value"]
        if value is None or value <= 0:
            return
        self._last_acq = value
        try:
            self._shot_queue.put_nowait(value)
        except asyncio.QueueFull:
            self._shot_queue.get_nowait()  # drop the oldest update
            self._shot_queue.put_nowait(value)


class ShotTriggerMixin(AcqTimestampMonitorMixin):
    """``trigger()`` that completes once ``acq_timestamp`` has advanced.

    Class attributes hosts may override
    -----------------------------------
    _trigger_timeout : float
        Seconds to wait for the next shot before raising
        :exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError`.  Default 3.0.
    """

    _trigger_timeout: float = 3.0

    def trigger(self) -> AsyncStatus:
        """Return a status that completes once ``acq_timestamp`` has advanced.

        The stale-update drain and baseline capture happen synchronously
        *here*, not in the returned coroutine — so a shot fired immediately
        after this call (the strict single-shot pattern) can never land in a
        blind window and be missed (pinned by a mock race test).  The drain
        can never discard a real requested shot: nothing fires before
        ``trigger()`` returns.
        """
        t0 = self._last_acq
        while not self._shot_queue.empty():
            self._shot_queue.get_nowait()
        return AsyncStatus(self._wait_for_shot(t0))

    async def _wait_for_shot(self, t0: float | None) -> None:
        """Wait for the next monitor update carrying a new ``acq_timestamp``.

        Cold-cache path (``t0 is None``): deliberately **no CA-get baseline**
        — a baseline get raced the shot itself (a first acquisition landing
        inside the get's round-trip became the baseline and the strict single
        shot timed out).  ``trigger()`` already drained anything older, so on
        a cold cache the first positive arrival *is* the shot.
        """
        logger.debug(
            "%s: waiting for %s to advance past %s (timeout=%.1fs)",
            self._geecs_device_name,
            self._acq_timestamp_variable,
            t0,
            self._trigger_timeout,
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._trigger_timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise GeecsTriggerTimeoutError(
                    self._geecs_device_name, self._trigger_timeout
                )
            try:
                value = await asyncio.wait_for(
                    self._shot_queue.get(), timeout=remaining
                )
            except asyncio.TimeoutError:
                raise GeecsTriggerTimeoutError(
                    self._geecs_device_name, self._trigger_timeout
                ) from None
            if value != t0:
                logger.debug(
                    "%s: shot detected (%s → %s)",
                    self._geecs_device_name,
                    t0,
                    value,
                )
                return
