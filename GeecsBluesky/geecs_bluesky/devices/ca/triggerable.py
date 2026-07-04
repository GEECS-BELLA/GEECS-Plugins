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
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ophyd_async.core import AsyncStatus, StandardReadable
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.exceptions import GeecsTriggerTimeoutError
from geecs_ca_gateway.pv_naming import pv_name
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)


class CaAcqTimestampReadable(StandardReadable):
    """Readable GEECS device over gateway PVs with a persistent shot monitor.

    One ``epics_signal_r`` child is created per data variable, plus an
    ``acq_timestamp`` child that carries the shot stamp.  A monitor
    subscription (started at ``connect()``) keeps ``_last_acq`` (latest value)
    and ``_shot_queue`` (update stream) current.

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
    """

    _acq_timestamp_variable: str = "acq_timestamp"

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
                    epics_signal_r(datatype, pv_name(experiment, device, var)),
                )
            self.acq_timestamp = epics_signal_r(
                float, pv_name(experiment, device, self._acq_timestamp_variable)
            )
        super().__init__(name=name)
        # Persistent-monitor state (populated by _on_acq_timestamp): the latest
        # seen value, and a queue of updates trigger() waits on.
        self._last_acq: float | None = None
        self._shot_queue: asyncio.Queue[float] = asyncio.Queue()
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

    def _on_acq_timestamp(self, reading: dict[str, Any]) -> None:
        """Monitor callback: cache the latest value and queue the update.

        Non-positive values are ignored: a real GEECS ``acq_timestamp`` is a
        LabVIEW-epoch time (~3.9e9), while ``0.0`` is the gateway channel's
        initial placeholder before the device's first acquisition — treating it
        as data would fake a t0 and make the placeholder→first-frame jump look
        like a shot.  "Never acquired" therefore reads as ``_last_acq is None``
        on both backends.
        """
        value = reading[self.acq_timestamp.name]["value"]
        if value is None or value <= 0:
            return
        self._last_acq = value
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
        single-shot pattern) can never be missed.
        """
        t0 = self._last_acq
        if t0 is not None:
            while not self._shot_queue.empty():
                self._shot_queue.get_nowait()
        return AsyncStatus(self._wait_for_shot(t0))

    async def _wait_for_shot(self, t0: float | None) -> None:
        """Wait for the next monitor update carrying a new ``acq_timestamp``."""
        if t0 is None:
            # No monitor update yet at trigger() time (e.g. device INVALID or
            # just connected) — fall back to an async CA get for the baseline.
            # Best-effort: the persistent monitor normally populates the cache
            # long before a scan, so this path is not expected mid-scan.
            t0 = await self.acq_timestamp.get_value()
            while not self._shot_queue.empty():
                self._shot_queue.get_nowait()

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
