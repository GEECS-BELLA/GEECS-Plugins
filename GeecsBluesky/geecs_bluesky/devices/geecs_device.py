"""ophyd-async Device base class for GEECS hardware.

Usage pattern::

    from ophyd_async.core import StandardReadable
    from geecs_bluesky.signals import geecs_signal_rw, geecs_signal_r

    class JetStage(GeecsDevice):
        def __init__(self, host: str, port: int, name: str = ""):
            dev = "U_ESP_JetXYZ"
            with self.add_children_as_readables():
                self.position_x = geecs_signal_rw(float, dev, "Jet_X (mm)", host, port, units="mm")
                self.position_y = geecs_signal_rw(float, dev, "Jet_Y (mm)", host, port, units="mm")
            super().__init__(name=name)

    motor = JetStage("192.168.1.10", 9000, name="jet")
    await motor.connect()

DB-resolved construction (requires ``geecs-pythonapi``)::

    motor = JetStage.from_db("U_ESP_JetXYZ")
"""

from __future__ import annotations

import logging
from typing import Any

from ophyd_async.core import StandardReadable

logger = logging.getLogger(__name__)


class GeecsDevice(StandardReadable):
    """Thin ``StandardReadable`` subclass for GEECS devices.

    Provides a :meth:`from_db` class method for constructing devices whose
    ``(host, port)`` is resolved from the GEECS MySQL database.

    For signal creation, use the standalone factories in
    :mod:`geecs_bluesky.signals` (``geecs_signal_rw``, ``geecs_signal_r``,
    ``geecs_signal_w``).
    """

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
