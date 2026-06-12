"""GeecsTriggerable — mixin adding the Bluesky Triggerable protocol to GEECS devices.

The GEECS hardware at BELLA runs at 1 Hz driven by a DG645 delay generator.
Every triggerable device exposes an ``acq_timestamp`` variable that advances
each time the device captures a shot.  This mixin waits for that change via
an event queue populated by the shared TCP subscriber on every push frame —
fully event-driven, no UDP polling.

Usage::

    from geecs_bluesky.devices.triggerable import GeecsTriggerable
    from geecs_bluesky.devices.geecs_device import GeecsDevice
    from geecs_bluesky.signals import geecs_signal_r
    from geecs_bluesky.transport.udp_client import GeecsUdpClient

    class MyCameraDevice(GeecsTriggerable, GeecsDevice):
        _acq_timestamp_variable = "acq_timestamp"   # override if device differs

        def __init__(self, device_name, host, port, name=""):
            udp = GeecsUdpClient(host, port)
            with self.add_children_as_readables():
                self.filepath = geecs_signal_r(str, device_name,
                                               "SavedFile", host, port,
                                               shared_udp=udp)
            super().__init__(name=name, shared_udp=udp)

Note on MRO
-----------
List ``GeecsTriggerable`` *before* ``GeecsDevice`` in the class bases so that
its ``trigger()`` is not shadowed.  ``GeecsTriggerable`` has no ``__init__`` —
all keyword arguments flow through to ``GeecsDevice`` via ``super()``.
"""

from __future__ import annotations

import asyncio
import logging

from ophyd_async.core import AsyncStatus

from geecs_bluesky.exceptions import GeecsTriggerTimeoutError

logger = logging.getLogger(__name__)


class GeecsTriggerable:
    """Mixin: adds ``trigger() → AsyncStatus`` via ``acq_timestamp`` event queue.

    Relies on the cooperating :class:`~geecs_bluesky.devices.geecs_device.GeecsDevice`
    to set ``self._shot_cache`` and ``self._shot_queue`` during ``connect()``.
    The ``acq_timestamp`` variable is automatically added to the TCP subscription
    by :meth:`GeecsDevice.connect` when this mixin is present.

    Class-level attributes that subclasses may override
    ----------------------------------------------------
    _acq_timestamp_variable : str
        GEECS variable name for the acquisition timestamp.  Default:
        ``"acq_timestamp"``.
    _trigger_timeout : float
        Maximum seconds to wait for a new shot before raising
        :exc:`TimeoutError`.  Default: ``3.0`` (three 1-Hz shots).
    """

    _subscribe_acq_timestamp: bool = True
    _acq_timestamp_variable: str = "acq_timestamp"
    _trigger_timeout: float = 3.0

    def trigger(self) -> AsyncStatus:
        """Return a status that completes once ``acq_timestamp`` has advanced.

        When the TCP cache is already populated, the stale-frame drain and
        baseline capture happen synchronously *here*, not in the returned
        coroutine — so a shot fired immediately after this call (the strict
        single-shot pattern: ``bps.trigger`` → fire → ``bps.wait``) can never
        land in the window between baselining and waiting and be missed.
        """
        shot_queue: asyncio.Queue | None = getattr(self, "_shot_queue", None)
        shot_cache: dict | None = getattr(self, "_shot_cache", None)
        var = self._acq_timestamp_variable
        t0 = None
        if shot_queue is not None and shot_cache is not None and var in shot_cache:
            while not shot_queue.empty():
                shot_queue.get_nowait()
            t0 = shot_cache[var]
        return AsyncStatus(self._wait_for_shot(t0))

    async def _wait_for_shot(self, t0: object | None = None) -> None:
        """Wait for the next TCP push frame that carries a new ``acq_timestamp``."""
        shot_queue: asyncio.Queue | None = getattr(self, "_shot_queue", None)
        udp = getattr(self, "_shared_udp", None)
        var = self._acq_timestamp_variable

        if shot_queue is None:
            raise RuntimeError(
                "GeecsTriggerable: _shot_queue not set; "
                "ensure GeecsDevice.connect() completed successfully"
            )

        if t0 is None:
            # Cache was empty at trigger() time — fall back to a UDP GET for
            # the baseline.  Best-effort: connect() pre-populates the cache,
            # so this path is not expected during scans.
            if udp is None:
                raise RuntimeError(
                    "GeecsTriggerable: no shot_cache or UDP client available; "
                    "ensure device is connected"
                )
            t0 = await udp.get(var)
            while not shot_queue.empty():
                shot_queue.get_nowait()

        logger.debug(
            "trigger: waiting for %s to advance past %s (timeout=%.1fs)",
            var,
            t0,
            self._trigger_timeout,
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._trigger_timeout

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise GeecsTriggerTimeoutError(
                    getattr(self, "_geecs_device_name", self.name),
                    self._trigger_timeout,
                )
            try:
                frame = await asyncio.wait_for(shot_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise GeecsTriggerTimeoutError(
                    getattr(self, "_geecs_device_name", self.name),
                    self._trigger_timeout,
                )
            if frame.get(var, t0) != t0:
                logger.debug("trigger: shot detected (%s → %s)", t0, frame.get(var))
                return
