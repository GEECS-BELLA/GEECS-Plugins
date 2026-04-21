"""GeecsSettable — generic Movable for any writable GEECS variable.

Wraps a single read/write GEECS variable (motor, power supply, DAC, analog
output, …) as an ophyd-async ``Movable``.  ``set()`` sends the value via UDP
and resolves immediately after the ACK (plus an optional ``settle_time``).

For position-feedback motors that need polling until arrival, use
:class:`~geecs_bluesky.devices.motor.GeecsMotor` which overrides
``_set_and_wait`` with a tolerance-based polling loop.

Typical usage::

    from geecs_bluesky.devices.settable import GeecsSettable

    # Explicit host/port:
    laser_power = GeecsSettable(
        "UC_LaserPower", "Power",
        "192.168.8.10", 65100,
        name="laser_power", units="W",
        settle_time=0.5,
    )

    # Via database:
    laser_power = GeecsSettable.from_db_var(
        "UC_LaserPower", "Power", name="laser_power", units="W",
    )

    await laser_power.connect()
    await laser_power.set(50.0)   # sends UDP, waits settle_time
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ophyd_async.core import AsyncStatus

from geecs_bluesky.devices.geecs_device import GeecsDevice
from geecs_bluesky.signals import geecs_signal_rw
from geecs_bluesky.transport.udp_client import GeecsUdpClient

logger = logging.getLogger(__name__)


class GeecsSettable(GeecsDevice):
    """Generic GEECS variable wrapped as a Bluesky ``Movable``.

    Parameters
    ----------
    device_name:
        GEECS device name (e.g. ``"UC_LaserPower"``).
    variable:
        Variable name on that device (e.g. ``"Power"``).
    host:
        Device IP address.
    port:
        Device UDP/TCP port.
    name:
        ophyd-async device name (namespaces signal keys in events).
    units:
        Physical units string included in event DataKey.
    limits:
        ``(low, high)`` control limits, or ``None`` to omit.
    settle_time:
        Seconds to wait after the UDP ACK before resolving the status.
    _readback_attr:
        Name to use for the readback signal attribute on the device instance.
        Subclasses can override (e.g. ``"position"`` for motors).
    """

    def __init__(
        self,
        device_name: str,
        variable: str,
        host: str,
        port: int,
        name: str = "settable",
        units: str = "",
        limits: tuple[float, float] | None = None,
        settle_time: float = 0.0,
        _readback_attr: str = "readback",
    ) -> None:
        udp = GeecsUdpClient(host, port)
        _sig = geecs_signal_rw(
            float,
            device_name,
            variable,
            host,
            port,
            units=units,
            limits=limits,
            shared_udp=udp,
        )
        with self.add_children_as_readables():
            setattr(self, _readback_attr, _sig)
        super().__init__(name=name, shared_udp=udp)
        # Store the attr name (a str) so _set_and_wait can retrieve the signal
        # without creating a second Device attribute that would trigger a rename.
        self._readback_attr_name = _readback_attr
        self._geecs_device_name = device_name
        self._variable = variable
        self._settle_time = settle_time

    # ------------------------------------------------------------------
    # Movable protocol
    # ------------------------------------------------------------------

    def set(self, value: float) -> AsyncStatus:
        """Send *value* to the device and return a Status that resolves on ACK.

        Implements :class:`bluesky.protocols.Movable`.
        """
        logger.info("%s: setting %s → %s", self.name, self._variable, value)
        return AsyncStatus(self._set_and_wait(value))

    async def _set_and_wait(self, value: float) -> None:
        """Write the setpoint and wait for settle_time.

        Subclasses (e.g. GeecsMotor) override this to add readback polling.
        """
        sig = getattr(self, self._readback_attr_name)
        await sig.set(value)
        if self._settle_time > 0:
            await asyncio.sleep(self._settle_time)

    # ------------------------------------------------------------------
    # DB constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_db_var(
        cls,
        device_name: str,
        variable: str,
        name: str = "settable",
        **kwargs: Any,
    ) -> "GeecsSettable":
        """Construct from a GEECS database lookup.

        Parameters
        ----------
        device_name:
            GEECS device name to resolve.
        variable:
            Variable name on that device.
        name:
            ophyd-async device name.
        **kwargs:
            Forwarded to the constructor (units, limits, settle_time, …).
        """
        from geecs_bluesky.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(device_name)
        logger.info("DB resolved %s → %s:%s", device_name, host, port)
        return cls(device_name, variable, host, port, name=name, **kwargs)
