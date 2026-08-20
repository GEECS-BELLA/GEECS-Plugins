"""CA acq_timestamp-monitored readables: the shot-aware CA device bases.

The GEECS shot signal is the device's ``acq_timestamp`` advancing once per
shot.  A **persistent CA monitor** on that PV (started at ``connect()``,
stopped by ``disconnect()``) feeds a local cache and a bounded drop-oldest
event queue:

* :class:`CaAcqTimestampReadable` — readable signals plus the monitor/cache
  (``_last_acq``); free-run *contributors* build on this (no blocking trigger).
* :class:`CaTriggerable` — adds ``trigger()``, which blocks until the queue
  delivers a value different from the baseline.

The stale-frame drain and baseline capture happen **synchronously inside**
``trigger()`` so an immediately-fired shot cannot be missed — see
:meth:`CaTriggerable.trigger`.  Design rationale: ``GeecsBluesky/CLAUDE.md``
(Device Layer).
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

    A non-readable ``connected_status`` child reads the gateway's per-device
    liveness PV (``[Experiment:]Device:CONNECTED``): the authoritative
    mode-independent "is this device's TCP stream up" signal.  It is a child
    (so it connects/mocks with the device) but never part of ``read()`` /
    ``describe()``.  Consumers: the scanner's pre-flight liveness check and
    the strict single-shot refire gate.

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
        # Liveness signal — the gateway's per-device ``CONNECTED`` status PV
        # (PV_CONTRACT.md §1).  Created OUTSIDE add_children_as_readables()
        # so it never appears in event rows or describe(): liveness is
        # pre-flight / plan metadata, not shot data.  Read as str; only the
        # exact "Disconnected" value means down (fail-open — a mock backend's
        # "" default and a status-PV-less old gateway both read as live).
        self.connected_status = epics_signal_r(
            str, ca_pv(experiment, device, "CONNECTED")
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
