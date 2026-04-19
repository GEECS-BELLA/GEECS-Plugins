"""ophyd-async Device base class for GEECS hardware.

Usage pattern::

    from geecs_bluesky.devices.geecs_device import GeecsDevice
    from geecs_bluesky.signals import geecs_signal_rw, geecs_signal_r
    from geecs_bluesky.transport.udp_client import GeecsUdpClient

    class JetStage(GeecsDevice):
        def __init__(self, host: str, port: int, name: str = ""):
            dev = "U_ESP_JetXYZ"
            udp = GeecsUdpClient(host, port)   # shared; serialises parallel reads
            with self.add_children_as_readables():
                self.x = geecs_signal_rw(float, dev, "Position.Axis 1", host, port,
                                         units="mm", shared_udp=udp)
                self.y = geecs_signal_rw(float, dev, "Position.Axis 2", host, port,
                                         units="mm", shared_udp=udp)
            super().__init__(name=name, shared_udp=udp)

    motor = JetStage("192.168.8.198", 65158, name="jet")
    await motor.connect()   # connects the shared UDP client, then all signals

DB-resolved construction (requires ``geecs-pythonapi``)::

    motor = JetStage.from_db("U_ESP_JetXYZ")
"""

from __future__ import annotations

import logging
from typing import Any

from ophyd_async.core import StandardReadable

from geecs_bluesky.transport.udp_client import GeecsUdpClient

logger = logging.getLogger(__name__)


class GeecsDevice(StandardReadable):
    """Thin ``StandardReadable`` subclass for GEECS devices.

    Provides:

    * :meth:`from_db` — construct from GEECS MySQL database lookup.
    * Shared-UDP-client lifecycle — pass ``shared_udp`` to ``super().__init__``
      and the base class will :meth:`connect` / close it around the signal
      lifecycle so all signals use one serialised socket.

    For signal creation, use the standalone factories in
    :mod:`geecs_bluesky.signals` (``geecs_signal_rw``, ``geecs_signal_r``,
    ``geecs_signal_w``), passing the same ``shared_udp`` instance.
    """

    def __init__(
        self,
        *args: Any,
        shared_udp: GeecsUdpClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise device, optionally storing a shared UDP client."""
        super().__init__(*args, **kwargs)
        self._shared_udp = shared_udp

    async def connect(
        self,
        mock: bool = False,
        timeout: float = 10.0,
        force_reconnect: bool = False,
    ) -> None:
        """Connect the shared UDP client first, then all child signals."""
        if self._shared_udp is not None and self._shared_udp._cmd_transport is None:
            await self._shared_udp.connect()
            logger.debug("GeecsDevice shared UDP connected")
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )

    @classmethod
    def from_db(
        cls,
        device_name: str,
        name: str = "",
        **kwargs: Any,
    ) -> "GeecsDevice":
        """Construct a device by looking up ``(host, port)`` from the GEECS database.

        Requires ``geecs-pythonapi`` to be installed::

            pip install -e path/to/GEECS-PythonAPI

        Parameters
        ----------
        device_name:
            Device name to resolve via ``GeecsDatabase.find_device()``.
        name:
            ophyd-async device name.
        **kwargs:
            Additional keyword arguments forwarded to ``cls.__init__``.
        """
        try:
            from geecs_python_api.controls.interface.geecs_database import (
                GeecsDatabase,
            )
        except ImportError as exc:
            raise ImportError(
                "DB lookup requires geecs-pythonapi. "
                "Install with: pip install -e path/to/GEECS-PythonAPI"
            ) from exc

        host, port = GeecsDatabase.find_device(device_name)
        if not host or port < 0:
            raise RuntimeError(
                f"Device '{device_name}' not found in GEECS database "
                f"(got host={host!r}, port={port})"
            )
        logger.info("DB resolved %s → %s:%s", device_name, host, port)
        return cls(host=host, port=port, name=name, **kwargs)
