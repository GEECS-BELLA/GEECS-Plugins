"""GeecsMotor — position-feedback motor implementing the Bluesky Movable protocol.

Extends :class:`~geecs_bluesky.devices.settable.GeecsSettable` with a
tolerance-based readback polling loop so that ``set()`` only resolves once
the axis has physically arrived at the target position.

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

from geecs_bluesky.devices.settable import GeecsSettable
from geecs_bluesky.exceptions import GeecsMotorTimeoutError

logger = logging.getLogger(__name__)

_DEFAULT_MOVE_TIMEOUT = 30.0  # seconds


class GeecsMotor(GeecsSettable):
    """GEECS motor with position-feedback polling.

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
        super().__init__(
            device_name,
            variable,
            host,
            port,
            name=name,
            units=units,
            limits=limits,
            settle_time=settle_time,
            _readback_attr="position",
        )
        self._tolerance = tolerance
        self._move_timeout = move_timeout

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

        UDP polling keeps the position check independent of the shared TCP
        subscriber and avoids filling the shot queue with intermediate motor
        positions.  The cache is updated once on completion so subsequent
        ``read()`` returns the correct value.
        """
        sig = getattr(self, self._readback_attr_name)
        await sig.set(value)

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
                raise GeecsMotorTimeoutError(
                    self._geecs_device_name,
                    self._variable,
                    target=value,
                    current=current,
                    timeout=self._move_timeout,
                )
            await asyncio.sleep(0.1)

        # Write confirmed position back to cache so read() returns correct value
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
        return cls.from_db_var(device_name, variable, name=name, **kwargs)
