"""ModeImager — concrete camera device for the UC_ModeImager GEECS camera.

Combines :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable` with
:class:`~geecs_bluesky.devices.geecs_device.GeecsDevice` to produce a device
that:

* Exposes ``hardware_ts`` — the camera's hardware clock timestamp (float).
* Responds to ``trigger()`` by waiting for ``acq_timestamp`` to advance,
  i.e., the DG645 fires and the camera captures the next shot.

Verified on real hardware 2026-04-19 at BELLA.

Typical usage::

    from geecs_bluesky.devices.mode_imager import ModeImager
    from bluesky import RunEngine
    import asyncio

    cam = ModeImager.from_db(name="mode_imager")

    RE = RunEngine()
    asyncio.run_coroutine_threadsafe(cam.connect(), RE._loop).result(timeout=10)

    # In a step scan:
    from geecs_bluesky.plans.step_scan import geecs_step_scan
    RE(geecs_step_scan(motor=motor, positions=[4.0, 5.0, 6.0],
                       detectors=[cam], shots_per_step=3))
"""

from __future__ import annotations

import logging
from typing import Any

from geecs_bluesky.devices.geecs_device import GeecsDevice
from geecs_bluesky.devices.triggerable import GeecsTriggerable
from geecs_bluesky.signals import geecs_signal_r
from geecs_bluesky.transport.udp_client import GeecsUdpClient

logger = logging.getLogger(__name__)

_DEVICE_NAME = "UC_ModeImager"


class ModeImager(GeecsTriggerable, GeecsDevice):
    """Bluesky device for the UC_ModeImager camera at BELLA.

    Signals (readable)
    ------------------
    hardware_ts : SignalR[float]
        Camera hardware clock timestamp (seconds since some epoch).
        Advances by ~1 s on each 1-Hz DG645 shot.

    Trigger completion
    ------------------
    Polls ``acq_timestamp`` via UDP — the shot acquisition timestamp that
    advances with every camera frame.  :attr:`_trigger_timeout` is 5 s to
    tolerate occasional missed shots at 1 Hz.

    Notes
    -----
    Image analysis scalars (``MeanCounts``, ``FWHMx``, etc.) are not available
    via UDP get on this device; they are pushed via the TCP subscription stream.
    Future work: add TCP-backed signals for those variables.
    """

    _acq_timestamp_variable: str = "acq_timestamp"
    _trigger_timeout: float = 5.0

    def __init__(
        self,
        host: str,
        port: int,
        name: str = "mode_imager",
    ) -> None:
        udp = GeecsUdpClient(host, port)
        with self.add_children_as_readables():
            self.hardware_ts = geecs_signal_r(
                float,
                _DEVICE_NAME,
                "hardware_timestamp",
                host,
                port,
                shared_udp=udp,
            )
        super().__init__(name=name, shared_udp=udp)

    @classmethod
    def from_db(cls, name: str = "mode_imager", **kwargs: Any) -> "ModeImager":
        """Construct by resolving ``UC_ModeImager`` from the GEECS database."""
        from geecs_bluesky.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(_DEVICE_NAME)
        logger.info("DB resolved %s → %s:%s", _DEVICE_NAME, host, port)
        return cls(host, port, name=name, **kwargs)
