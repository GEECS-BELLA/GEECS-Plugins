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

    _acq_timestamp_variable: str = "acq_timestamp"
    _trigger_timeout: float = 3.0

    def trigger(self) -> AsyncStatus:
        """Return a status that completes once ``acq_timestamp`` has advanced.

        The returned :class:`~ophyd_async.core.AsyncStatus` is called by
        :func:`bluesky.plan_stubs.trigger_and_read` for each shot.
        """
        return AsyncStatus(self._wait_for_shot())

    async def _wait_for_shot(self) -> None:
        """Wait for the next TCP push frame that carries a new ``acq_timestamp``."""
        shot_queue: asyncio.Queue | None = getattr(self, "_shot_queue", None)
        shot_cache: dict | None = getattr(self, "_shot_cache", None)
        udp = getattr(self, "_shared_udp", None)
        var = self._acq_timestamp_variable

        # Determine baseline timestamp
        if shot_cache is not None and var in shot_cache:
            t0 = shot_cache[var]
        elif udp is not None:
            t0 = await udp.get(var)
        else:
            raise RuntimeError(
                "GeecsTriggerable: no shot_cache or UDP client available; "
                "ensure device is connected"
            )

        if shot_queue is None:
            raise RuntimeError(
                "GeecsTriggerable: _shot_queue not set; "
                "ensure GeecsDevice.connect() completed successfully"
            )

        # Drain stale frames accumulated before this trigger call
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
