"""GeecsTriggerable — mixin adding the Bluesky Triggerable protocol to GEECS devices.

The GEECS hardware at BELLA runs at 1 Hz driven by a DG645 delay generator.
Every triggerable device exposes an ``acq_timestamp`` variable (hardware-sourced,
transmitted via the TCP push stream) that advances each time the device captures
a shot.  Waiting for that timestamp to change is the correct completion signal
for a trigger — more reliable than shot numbers, which can be out-of-sync
across devices powered on at different times.

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

logger = logging.getLogger(__name__)


class GeecsTriggerable:
    """Mixin: adds ``trigger() → AsyncStatus`` via ``acq_timestamp`` polling.

    Relies on ``self._shared_udp`` being set by the cooperating
    :class:`~geecs_bluesky.devices.geecs_device.GeecsDevice` initialiser.

    Class-level attributes that subclasses may override
    ----------------------------------------------------
    _acq_timestamp_variable : str
        GEECS variable name for the acquisition timestamp.  Default:
        ``"acq_timestamp"``.
    _trigger_timeout : float
        Maximum seconds to wait for a new shot before raising
        :exc:`TimeoutError`.  Default: ``3.0`` (three 1-Hz shots).
    _trigger_poll_interval : float
        Polling cadence in seconds.  Default: ``0.05`` (50 ms).
    """

    _acq_timestamp_variable: str = "acq_timestamp"
    _trigger_timeout: float = 3.0
    _trigger_poll_interval: float = 0.05

    def trigger(self) -> AsyncStatus:
        """Return a status that completes once ``acq_timestamp`` has advanced.

        The returned :class:`~ophyd_async.core.AsyncStatus` is called by
        :func:`bluesky.plan_stubs.trigger_and_read` for each shot.
        """
        return AsyncStatus(self._wait_for_shot())

    async def _wait_for_shot(self) -> None:
        """Poll ``acq_timestamp`` until it changes from its current value."""
        # Retrieve the UDP client set by GeecsDevice.__init__
        from geecs_bluesky.transport.udp_client import GeecsUdpClient

        udp: GeecsUdpClient | None = getattr(self, "_shared_udp", None)
        if udp is None:
            raise RuntimeError(
                "GeecsTriggerable requires _shared_udp to be set; "
                "ensure GeecsDevice.__init__ was called with shared_udp=..."
            )

        var = self._acq_timestamp_variable
        t0 = await udp.get(var)
        logger.debug(
            "trigger: waiting for %s to advance past %s (timeout=%.1fs)",
            var,
            t0,
            self._trigger_timeout,
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._trigger_timeout

        while True:
            await asyncio.sleep(self._trigger_poll_interval)
            t1 = await udp.get(var)
            if t1 != t0:
                logger.debug("trigger: shot detected (%s → %s)", t0, t1)
                return
            if loop.time() > deadline:
                raise TimeoutError(
                    f"No new shot: {var!r} unchanged ({t0!r}) after "
                    f"{self._trigger_timeout:.1f}s"
                )
