"""GeecsCaGateway — serve GEECS devices as EPICS Channel Access PVs.

Architecture (see the package README for the full rationale)::

    GEECS device  --TCP stream-->  readback PV   (caget / camonitor)
    GEECS device  <--UDP set-----  setpoint PV   (caput)

One asyncio event loop runs both the caproto CA server and the per-device
subscription tasks.  Reads are served from a cache the subscription keeps warm;
puts are forwarded to the device over UDP.  This is a *façade* over GEECS, which
remains the authoritative control system.
"""

from __future__ import annotations

import logging
from typing import Any

from caproto import ChannelData
from caproto.asyncio.server import start_server

from geecs_bluesky.transport.tcp_subscriber import GeecsTcpSubscriber
from geecs_bluesky.transport.udp_client import GeecsUdpClient

from .channels import cast_value, make_readback_channel, make_setpoint_channel
from .config import DType, GatewayConfig
from .naming import pv_name

logger = logging.getLogger(__name__)


class GeecsCaGateway:
    """A single-process CA soft-IOC fronting one or more GEECS devices.

    Parameters
    ----------
    config : GatewayConfig
        Declarative description of the devices and variables to serve.

    Notes
    -----
    This proof of concept intentionally omits an automatic reconnect supervisor:
    :meth:`GeecsTcpSubscriber._listen_loop` exits on a dropped connection.
    Surviving a device power-cycle (the robustness of the legacy SVE tool) means
    wrapping the subscription in a supervising retry loop — the next hardening
    step, deliberately left out of the first cut.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.pvdb: dict[str, ChannelData] = {}
        self._udp: dict[str, GeecsUdpClient] = {}
        self._subs: dict[str, GeecsTcpSubscriber] = {}
        # device name -> {geecs_var -> (readback channel, dtype)}
        self._readbacks: dict[str, dict[str, tuple[ChannelData, DType]]] = {}
        self._build_pvdb()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_pvdb(self) -> None:
        """Populate ``self.pvdb`` and the readback routing map from config."""
        for dev in self.config.devices:
            readback_map: dict[str, tuple[ChannelData, DType]] = {}
            for var in dev.variables:
                full = pv_name(dev.pv_prefix, var.pv_suffix)
                readback = make_readback_channel(
                    var.dtype,
                    egu=var.egu,
                    precision=var.precision,
                    lo=var.lo,
                    hi=var.hi,
                )
                self.pvdb[full] = readback
                readback_map[var.geecs_var] = (readback, var.dtype)

                if var.settable:
                    setter = self._make_setter(dev.name, var.geecs_var)
                    self.pvdb[f"{full}:SP"] = make_setpoint_channel(
                        var.dtype,
                        setter,
                        egu=var.egu,
                        precision=var.precision,
                        lo=var.lo,
                        hi=var.hi,
                    )
            self._readbacks[dev.name] = readback_map

    def _make_setter(self, device_name: str, geecs_var: str):
        """Return an async setter closure that forwards a value over UDP."""

        async def setter(value: Any) -> Any:
            return await self._udp[device_name].set(geecs_var, value)

        return setter

    def _make_callback(self, device_name: str):
        """Return the subscription callback that fans a push frame into PVs."""
        readback_map = self._readbacks[device_name]

        async def callback(update: dict[str, Any]) -> None:
            for var, raw in update.items():
                entry = readback_map.get(var)
                if entry is None:
                    continue
                channel, dtype = entry
                try:
                    await channel.write(cast_value(dtype, raw))
                except Exception:
                    logger.warning(
                        "%s: failed to update PV for %s=%r",
                        device_name,
                        var,
                        raw,
                        exc_info=True,
                    )

        return callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open UDP and TCP connections to every configured device."""
        for dev in self.config.devices:
            udp = GeecsUdpClient(dev.host, dev.port, device_name=dev.name)
            await udp.connect()
            self._udp[dev.name] = udp

            sub = GeecsTcpSubscriber(dev.host, dev.port)
            await sub.connect()
            self._subs[dev.name] = sub
        logger.info("connected to %d device(s)", len(self.config.devices))

    async def subscribe(self) -> None:
        """Start the background subscription that keeps readback PVs warm."""
        for dev in self.config.devices:
            variables = [v.geecs_var for v in dev.variables]
            await self._subs[dev.name].subscribe(
                variables, self._make_callback(dev.name)
            )
        logger.info("subscribed to device streams")

    async def serve(self) -> None:
        """Run the CA server until cancelled (serves ``self.pvdb``)."""
        await start_server(self.pvdb, log_pv_names=True)

    async def run(self) -> None:
        """Connect, subscribe, and serve — the full gateway lifecycle."""
        await self.connect()
        await self.subscribe()
        try:
            await self.serve()
        finally:
            await self.close()

    async def close(self) -> None:
        """Tear down all subscriptions and UDP clients."""
        for sub in self._subs.values():
            await sub.close()
        for udp in self._udp.values():
            await udp.close()
        self._subs.clear()
        self._udp.clear()
