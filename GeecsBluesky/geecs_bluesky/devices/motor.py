"""GeecsMotor — single-axis motor implementing the Bluesky Movable protocol.

Maps one GEECS position variable (read/write via UDP) into an ophyd-async
device whose ``set(value)`` method moves the axis and returns an
:class:`~ophyd_async.core.AsyncStatus` that completes when the readback
is within ``tolerance`` of the requested position.

Typical usage::

    from geecs_bluesky.devices.motor import GeecsMotor

    # Explicit host/port:
    motor = GeecsMotor("U_ESP_JetXYZ", "Position.Axis 1",
                       "192.168.8.198", 65158,
                       name="jet_x", units="mm",
                       limits=(-150.0, 150.0))

    # Via database:
    motor = GeecsMotor.from_db_axis("U_ESP_JetXYZ", "Position.Axis 1",
                                    name="jet_x", units="mm")

    await motor.connect()
    status = motor.set(5.0)   # non-blocking, returns Status
    await status              # wait for motion to complete
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

_DEFAULT_MOVE_TIMEOUT = 30.0  # seconds


class GeecsMotor(GeecsDevice):
    """Single-axis GEECS motor implementing :class:`bluesky.protocols.Movable`.

    Parameters
    ----------
    device_name:
        GEECS device name (e.g. ``"U_ESP_JetXYZ"``).
    variable:
        Position variable name (e.g. ``"Position.Axis 1"``).
    host:
        Device IP address.
    port:
        Device UDP/TCP port.
    name:
        ophyd-async device name (used to namespace signal keys in events).
    units:
        Physical units string (used in :class:`~event_model.DataKey`).
    limits:
        ``(low, high)`` control limits in the same units as the variable,
        or ``None`` to omit from the DataKey.
    tolerance:
        Move completion tolerance in ``units``.  ``set()`` resolves when
        ``|readback − setpoint| ≤ tolerance``.  Default: ``0.005``.
    settle_time:
        Extra seconds to wait after position is within tolerance before
        marking the status complete.  Default: ``0.0``.
    move_timeout:
        Maximum seconds to wait for motion to complete.  Default: ``30.0``.
    """

    def __init__(
        self,
        device_name: str,
        variable: str,
        host: str,
        port: int,
        name: str = "motor",
        units: str = "",
        limits: tuple[float, float] | None = None,
        tolerance: float = 0.005,
        settle_time: float = 0.0,
        move_timeout: float = _DEFAULT_MOVE_TIMEOUT,
    ) -> None:
        udp = GeecsUdpClient(host, port)
        with self.add_children_as_readables():
            self.position = geecs_signal_rw(
                float,
                device_name,
                variable,
                host,
                port,
                units=units,
                limits=limits,
                shared_udp=udp,
            )
        super().__init__(name=name, shared_udp=udp)
        self._tolerance = tolerance
        self._settle_time = settle_time
        self._move_timeout = move_timeout
        self._geecs_device_name = device_name
        self._variable = variable

    # ------------------------------------------------------------------
    # Movable protocol
    # ------------------------------------------------------------------

    def set(self, value: float) -> AsyncStatus:
        """Move to *value* and return a Status that completes on arrival.

        Implements :class:`bluesky.protocols.Movable`.  The RunEngine
        (and :func:`bluesky.plan_stubs.mv`) await the returned status
        before proceeding.
        """
        logger.info(
            "%s: moving %s → %s (tol=%.4g)",
            self.name,
            self._variable,
            value,
            self._tolerance,
        )
        return AsyncStatus(self._set_and_wait(value))

    async def _set_and_wait(self, value: float) -> None:
        """Write the setpoint then poll via UDP until position is within tolerance.

        Both UDP GET and TCP push are gated by the same 5-Hz device loop, so
        neither is faster than the other.  Polling via UDP keeps the position
        check independent of the shared TCP subscriber and avoids filling the
        shot queue with intermediate motor positions.
        The cache is updated once on completion so subsequent read() is correct.
        """
        await self.position.set(value)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._move_timeout

        while True:
            raw = await self._shared_udp.get(self._variable)
            current = float(raw)
            if abs(current - value) <= self._tolerance:
                logger.debug(
                    "%s: arrived at %.6g (target=%.6g, tol=%.4g)",
                    self.name,
                    current,
                    value,
                    self._tolerance,
                )
                break
            if loop.time() > deadline:
                raise TimeoutError(
                    f"{self.name}: timed out waiting to reach {value} "
                    f"(current={current:.6g}, timeout={self._move_timeout}s)"
                )
            await asyncio.sleep(0.1)

        # Write confirmed position back to cache so read() returns the correct value
        self._shot_cache[self._variable] = current

        if self._settle_time > 0:
            await asyncio.sleep(self._settle_time)

    # ------------------------------------------------------------------
    # DB constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_db_axis(
        cls,
        device_name: str,
        variable: str,
        name: str = "motor",
        **kwargs: Any,
    ) -> "GeecsMotor":
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
            Forwarded to :class:`GeecsMotor` (units, limits, tolerance, …).
        """
        from geecs_bluesky.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(device_name)
        logger.info("DB resolved %s → %s:%s", device_name, host, port)
        return cls(device_name, variable, host, port, name=name, **kwargs)
