"""ophyd-async 0.16 SignalBackend implementation backed by the GEECS UDP/TCP protocol."""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Callable, Type

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import SignalBackend, SignalDatatypeT

from geecs_bluesky.transport.udp_client import GeecsUdpClient
from geecs_bluesky.transport.tcp_subscriber import GeecsTcpSubscriber

logger = logging.getLogger(__name__)

# ophyd-async 0.16 callback signature: callback(reading) -> None
ReadingCallback = Callable[[Reading], None]


class GeecsSignalBackend(SignalBackend[SignalDatatypeT]):
    """ophyd-async SignalBackend for a single GEECS device variable.

    Each instance represents one variable (e.g. ``"Jet_X (mm)"``) on one
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
        # If a shared client is provided it must already be connected;
        # this backend will NOT close it on disconnect.
        self._shared_udp = shared_udp

        self._udp: GeecsUdpClient | None = None
        self._callback: ReadingCallback | None = None
        self._setpoint: Any = None
        self._monitor_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # SignalBackend interface
    # ------------------------------------------------------------------

    def source(self, name: str, read: bool) -> str:
        """Return the URI identifying this signal's hardware source."""
        return f"geecs://{self._host}:{self._port}/{self._device_name}/{self._variable}"

    async def connect(self, timeout: float = 5.0) -> None:
        """Open the UDP client (TCP opened lazily on first subscription)."""
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
        """Return the current value of this signal."""
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
        """Write ``value`` to the device variable."""
        if self._udp is None:
            raise RuntimeError(
                f"Backend not connected: {self._device_name}/{self._variable}"
            )
        if value is None:
            return
        self._setpoint = value
        await self._udp.set(self._variable, value)

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
        """Register (or clear) a callback invoked on each 5-Hz TCP update."""
        self._callback = callback
        if callback is not None:
            if self._monitor_task is None or self._monitor_task.done():
                self._monitor_task = asyncio.create_task(
                    self._run_subscription(),
                    name=f"monitor[{self._device_name}/{self._variable}]",
                )
        else:
            if self._monitor_task is not None and not self._monitor_task.done():
                self._monitor_task.cancel()
            self._monitor_task = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_subscription(self) -> None:
        try:
            sub = GeecsTcpSubscriber(self._host, self._port)
            async with sub:
                await sub.subscribe([self._variable], self._on_update)
                while self._callback is not None:
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "TCP subscription failed for %s/%s", self._device_name, self._variable
            )

    def _on_update(self, update: dict[str, Any]) -> None:
        if self._callback is None:
            return
        raw = update.get(self._variable)
        if raw is None:
            return
        value = self._cast(raw)
        reading = Reading(value=value, timestamp=time.monotonic(), alarm_severity=0)
        try:
            self._callback(reading)
        except Exception:
            logger.exception("signal callback raised")

    def _cast(self, raw: Any) -> SignalDatatypeT:
        if self.datatype is None:
            return raw  # type: ignore[return-value]
        try:
            return self.datatype(raw)  # type: ignore[call-arg]
        except (TypeError, ValueError):
            return raw  # type: ignore[return-value]

    async def disconnect(self) -> None:
        """Release resources."""
        self.set_callback(None)
        if self._udp is not None and self._shared_udp is None:
            await self._udp.close()
        self._udp = None


def _python_type_to_dtype(t: type | None) -> str:
    if t in (int, bool):
        return "integer"
    if t is float:
        return "number"
    return "string"
