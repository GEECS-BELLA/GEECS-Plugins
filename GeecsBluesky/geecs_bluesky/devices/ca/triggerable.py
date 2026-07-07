"""CA acq_timestamp-monitored readables: the shot-aware CA device bases.

The GEECS shot signal is the device's ``acq_timestamp`` advancing once per
shot.  Over the gateway that is a readback PV; a **persistent CA monitor** on
it (started at ``connect()``) feeds a local cache and an event queue:

* :class:`CaAcqTimestampReadable` — readable signals plus the monitor/cache.
  ``_last_acq`` is the CA-side analogue of the direct backend's TCP shot
  cache; free-run *contributors* build on this (no blocking trigger).
* :class:`CaTriggerable` — adds ``trigger()``, which blocks until the queue
  delivers a value different from the baseline — the CA-sourced analogue of
  :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable`.

Like ``GeecsTriggerable``, the stale-frame drain and baseline capture happen
**synchronously inside** ``trigger()``, before the status is returned — so a
shot fired immediately after ``bps.trigger`` (the strict single-shot pattern:
trigger → fire → wait) can never land in a blind window between baselining and
waiting and be missed.  Free-run behavior is hardware-verified (one Bluesky row
per real shot, no coalescing).

The event queue is **bounded** (drop-oldest ring, ``_shot_queue_maxsize``):
only ``trigger()`` ever drains it, so contributors and idle detectors would
otherwise accumulate one float per machine shot for as long as the persistent
monitor lives.  Dropping the oldest entry preserves the no-blind-window
guarantee — see :meth:`CaAcqTimestampReadable._on_acq_timestamp`.

Instances are torn down with :meth:`CaAcqTimestampReadable.disconnect`, which
unsubscribes the persistent monitor so per-scan device objects do not stay
reachable (and firing) through the signal cache's callback reference.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ophyd_async.core import AsyncStatus, StandardReadable
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.exceptions import GeecsTriggerTimeoutError
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class CaAcqTimestampReadable(StandardReadable):
    """Readable GEECS device over gateway PVs with a persistent shot monitor.

    One ``epics_signal_r`` child is created per data variable, plus an
    ``acq_timestamp`` child that carries the shot stamp.  A monitor
    subscription (started at ``connect()``, stopped by ``disconnect()``) keeps
    ``_last_acq`` (latest value) and ``_shot_queue`` (bounded update stream)
    current.

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"UC_Amp2_IR_input"``).
    variables : str or list of str
        GEECS scalar variable name(s) to read (e.g. ``"centroidx"``).  The
        acquisition timestamp variable is filtered out if listed — it is
        always created as the dedicated ``acq_timestamp`` child.
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    datatype : type
        Scalar CA datatype for the data variables (default ``float``).

    Class attributes subclasses may override
    ----------------------------------------
    _acq_timestamp_variable : str
        GEECS variable that advances per shot.  Default ``"acq_timestamp"``.
    _shot_queue_maxsize : int
        Bound on the shot-update queue (default 32).  Only ``trigger()``
        drains the queue, so without a bound an idle device (a free-run
        contributor, or a detector between scans) accumulates one float per
        machine shot for as long as the monitor lives — tens of thousands
        per hour at typical rep rates.  32 comfortably covers the real need
        (updates arriving between ``trigger()``'s baseline capture and the
        awaited get: at most rep-rate × trigger-timeout, ~15 at 5 Hz / 3 s).
    """

    _acq_timestamp_variable: str = "acq_timestamp"
    _shot_queue_maxsize: int = 32

    def __init__(
        self,
        device: str,
        variables: str | list[str],
        *,
        experiment: str | None = None,
        name: str = "",
        datatype: type = float,
    ) -> None:
        if isinstance(variables, str):
            variables = [variables]
        self._geecs_device_name = device
        with self.add_children_as_readables():
            for var in variables:
                if var == self._acq_timestamp_variable:
                    continue  # created below as the dedicated timestamp child
                setattr(
                    self,
                    safe_name(var),
                    epics_signal_r(datatype, ca_pv(experiment, device, var)),
                )
            self.acq_timestamp = epics_signal_r(
                float, ca_pv(experiment, device, self._acq_timestamp_variable)
            )
        super().__init__(name=name)
        # Persistent-monitor state (populated by _on_acq_timestamp): the latest
        # seen value, and a bounded queue of updates trigger() waits on.
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
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        if not self._monitoring:
            self.acq_timestamp.subscribe_reading(self._on_acq_timestamp)
            self._monitoring = True

    async def disconnect(self) -> None:
        """Stop the persistent ``acq_timestamp`` monitor and drop shot state.

        The per-scan teardown hook: the scanner bridge's
        ``_disconnect_devices_sync`` runs ``device.disconnect()`` as a
        coroutine on the RunEngine loop after every scan.  Unsubscribing
        removes the signal cache's reference back to this instance (its
        ``_on_acq_timestamp`` bound method) — without it, every per-scan
        device object stays alive and keeps enqueuing monitor updates for
        the rest of the process.

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

        Non-positive values are ignored: a real GEECS ``acq_timestamp`` is a
        LabVIEW-epoch time (~3.9e9), while ``0.0`` is the gateway channel's
        initial placeholder before the device's first acquisition — treating it
        as data would fake a t0 and make the placeholder→first-frame jump look
        like a shot.  "Never acquired" therefore reads as ``_last_acq is None``
        on both backends.

        The queue is a **drop-oldest ring** bounded at ``_shot_queue_maxsize``:
        when full, the oldest entry is discarded to admit the newest, so an
        idle device (nothing draining the queue) holds a fixed handful of
        floats instead of one per machine shot.  This preserves ``trigger()``'s
        no-blind-window guarantee: the queue is drained empty at baseline
        capture, so anything dropped afterwards is *older* than an update
        still enqueued — and every post-baseline update satisfies the
        ``!= t0`` shot test, so a newer survivor detects the shot just as
        well.  Callback and consumers share the RE event loop (no awaits
        between the full-check and the put), so the two-step replace is
        race-free.
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


class CaTriggerable(CaAcqTimestampReadable):
    """A triggered GEECS detector whose ``trigger()`` waits for one real shot.

    Parameters are those of :class:`CaAcqTimestampReadable`.

    Class attributes subclasses may override
    ----------------------------------------
    _trigger_timeout : float
        Seconds to wait for the next shot before raising
        :exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError`.  Default 3.0.
    """

    _trigger_timeout: float = 3.0

    def trigger(self) -> AsyncStatus:
        """Return a status that completes once ``acq_timestamp`` has advanced.

        When the monitor cache is already populated, the stale-update drain and
        baseline capture happen synchronously *here*, not in the returned
        coroutine — so a shot fired immediately after this call (the strict
        single-shot pattern) can never be missed.  On a cold cache
        (``_last_acq is None``) nothing is drained: no real acquisition has
        ever reached the monitor, so anything that lands in the queue after
        this call *is* the shot — see :meth:`_wait_for_shot`.
        """
        t0 = self._last_acq
        if t0 is not None:
            while not self._shot_queue.empty():
                self._shot_queue.get_nowait()
        return AsyncStatus(self._wait_for_shot(t0))

    async def _wait_for_shot(self, t0: float | None) -> None:
        """Wait for the next monitor update carrying a new ``acq_timestamp``.

        Cold-cache path (``t0 is None``): the monitor callback filters
        non-positive values and caches everything else, so a cold cache means
        the monitor has delivered *no* real acquisition since subscribe.  Any
        positive value in the queue at this point is therefore a genuinely new
        acquisition — the shot — never a stale replay.  That invariant lets
        this path (1) treat an already-queued value as the shot immediately,
        (2) take a CA-get baseline only when the queue is empty, normalizing
        the gateway's ``0.0`` pre-acquisition placeholder to ``None`` so every
        future positive arrival passes the shot test, and (3) never drain the
        queue here — this coroutine runs *after* ``trigger()`` returned, so a
        drain could discard the strict-mode shot fired in between.

        Residual race: a shot arriving between the queue check in (1) and
        ``get_value()`` returning in (2) can hand back that same shot's
        timestamp as the baseline, making the queued equal value fail the
        ``!= t0`` test.  It requires the very first acquisition since
        subscribe to land inside a single CA round-trip and is the only
        remaining window; the persistent monitor normally warms the cache
        long before a scan, so the cold path itself is not expected mid-scan.
        """
        if t0 is None:
            if not self._shot_queue.empty():
                value = self._shot_queue.get_nowait()
                logger.debug(
                    "%s: shot detected on cold cache (queued %s)",
                    self._geecs_device_name,
                    value,
                )
                return
            v = await self.acq_timestamp.get_value()
            t0 = v if v and v > 0 else None

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
