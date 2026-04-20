"""ophyd-async 0.16 SignalBackend implementation backed by the GEECS UDP protocol.

Reads are served from a device-level shot cache populated by the shared TCP
subscriber (see :class:`~geecs_bluesky.devices.geecs_device.GeecsDevice`).
Writes (``put``) go via UDP and immediately update the local cache.
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Callable, Type

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import SignalBackend, SignalDatatypeT

from geecs_bluesky.transport.udp_client import GeecsUdpClient

logger = logging.getLogger(__name__)

ReadingCallback = Callable[[Reading], None]


class GeecsSignalBackend(SignalBackend[SignalDatatypeT]):
    """ophyd-async SignalBackend for a single GEECS device variable.

    Each instance represents one variable (e.g. ``"Position (mm)"``) on one
    GEECS device.

    Parameters
    ----------
    datatype:
        Python type for this variable (``float``, ``int``, ``str``, ``bool``).
    device_name:
        GEECS device name as it appears in protocol messages.
    variable:
        Variable name exactly as declared in the device's variable list.
    host:
        Device IP / hostname.
    port:
        Device UDP+TCP port number.
    units:
        Optional units string (used in DataKey metadata).
    limits:
        Optional ``(low, high)`` tuple for control limits.
    shared_udp:
        Shared UDP client managed by the parent :class:`GeecsDevice`.
    """

    def __init__(
        self,
        datatype: Type[SignalDatatypeT] | None,
        device_name: str,
        variable: str,
        host: str,
        port: int,
        units: str = "",
        limits: tuple[float, float] | None = None,
        shared_udp: GeecsUdpClient | None = None,
    ) -> None:
        super().__init__(datatype)
        self._device_name = device_name
        self._variable = variable
        self._host = host
        self._port = port
        self._units = units
        self._limits = limits
        self._shared_udp = shared_udp

        self._udp: GeecsUdpClient | None = None
        self._callback: ReadingCallback | None = None
        self._setpoint: Any = None
        # Injected by GeecsDevice.connect() — shared across all backends on this device
        self._shot_cache: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # SignalBackend interface
    # ------------------------------------------------------------------

    def source(self, name: str, read: bool) -> str:
        """Return the URI identifying this signal's hardware source."""
        return f"geecs://{self._host}:{self._port}/{self._device_name}/{self._variable}"

    async def connect(self, timeout: float = 5.0) -> None:
        """Wire up the UDP client (cache injection happens via GeecsDevice.connect)."""
        if self._shared_udp is not None:
            self._udp = self._shared_udp
        else:
            self._udp = GeecsUdpClient(self._host, self._port, exe_timeout=timeout)
            await asyncio.wait_for(self._udp.connect(), timeout=timeout)
        logger.debug(
            "GeecsSignalBackend connected: %s/%s", self._device_name, self._variable
        )

    async def get_reading(self) -> Reading:
        """Return the current value plus timestamp and alarm severity."""
        value = await self.get_value()
        return Reading(value=value, timestamp=time.monotonic(), alarm_severity=0)

    async def get_value(self) -> SignalDatatypeT:
        """Return the current value of this signal.

        Reads from the device-level shot cache when available (populated by the
        shared TCP subscriber — all variables from one shot, atomically).
        Falls back to a direct UDP GET if the cache has no value yet.
        """
        if self._shot_cache is not None and self._variable in self._shot_cache:
            return self._cast(self._shot_cache[self._variable])
        if self._udp is None:
            raise RuntimeError(
                f"Backend not connected: {self._device_name}/{self._variable}"
            )
        raw = await self._udp.get(self._variable)
        return self._cast(raw)

    async def get_setpoint(self) -> SignalDatatypeT:
        """Return the last value written via :meth:`put`."""
        if self._setpoint is None:
            return await self.get_value()
        return self._setpoint

    async def put(self, value: SignalDatatypeT | None) -> None:
        """Write ``value`` to the device variable via UDP and update local cache."""
        if self._udp is None:
            raise RuntimeError(
                f"Backend not connected: {self._device_name}/{self._variable}"
            )
        if value is None:
            return
        self._setpoint = value
        await self._udp.set(self._variable, value)
        # Update the cache immediately so subsequent reads reflect the write
        if self._shot_cache is not None:
            self._shot_cache[self._variable] = value

    async def get_datakey(self, source: str) -> DataKey:
        """Return metadata (dtype, shape, units, limits) for this signal."""
        dtype = _python_type_to_dtype(self.datatype)
        key: dict[str, Any] = {
            "source": source,
            "dtype": dtype,
            "shape": [],
        }
        if self._units:
            key["units"] = self._units
        if self._limits is not None:
            key["lower_ctrl_limit"] = self._limits[0]
            key["upper_ctrl_limit"] = self._limits[1]
        return key  # type: ignore[return-value]

    def set_callback(self, callback: ReadingCallback | None) -> None:
        """Register (or clear) a monitoring callback.

        When a callback is registered (e.g. during bluesky staging), an initial
        reading is delivered immediately from the shot cache so that ophyd-async's
        internal ``_valid`` event is set without waiting for the next TCP push.
        Subsequent readings are forwarded by the device-level TCP push handler.
        """
        self._callback = callback
        if callback is None:
            return

        # Deliver initial reading immediately if cache is populated
        if self._shot_cache is not None and self._variable in self._shot_cache:
            raw = self._shot_cache[self._variable]
            value = self._cast(raw)
            reading = Reading(value=value, timestamp=time.monotonic(), alarm_severity=0)
            try:
                callback(reading)
            except Exception:
                logger.exception(
                    "initial callback delivery raised for %s/%s",
                    self._device_name,
                    self._variable,
                )
        elif self._udp is not None:
            # Cache not yet populated — schedule a background UDP GET to unblock _valid
            asyncio.create_task(
                self._deliver_initial_reading(callback),
                name=f"initial-read[{self._device_name}/{self._variable}]",
            )

    async def _deliver_initial_reading(self, callback: ReadingCallback) -> None:
        """Fallback: UDP GET to deliver the first reading when cache is empty."""
        if self._udp is None:
            return
        try:
            raw = await self._udp.get(self._variable)
            if self._shot_cache is not None:
                self._shot_cache[self._variable] = raw
            value = self._cast(raw)
            reading = Reading(value=value, timestamp=time.monotonic(), alarm_severity=0)
            if self._callback is callback:  # still registered?
                callback(reading)
        except Exception:
            logger.exception(
                "initial reading delivery failed for %s/%s",
                self._device_name,
                self._variable,
            )

    def _cast(self, raw: Any) -> SignalDatatypeT:
        if self.datatype is None:
            return raw  # type: ignore[return-value]
        try:
            return self.datatype(raw)  # type: ignore[call-arg]
        except (TypeError, ValueError):
            return raw  # type: ignore[return-value]

    async def disconnect(self) -> None:
        """Release resources."""
        self._callback = None
        self._shot_cache = None
        if self._udp is not None and self._shared_udp is None:
            await self._udp.close()
        self._udp = None


def _python_type_to_dtype(t: type | None) -> str:
    if t in (int, bool):
        return "integer"
    if t is float:
        return "number"
    return "string"
