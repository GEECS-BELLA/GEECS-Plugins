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

import asyncio
import logging
from typing import Any

from caproto import AlarmSeverity, AlarmStatus, ChannelData
from caproto.asyncio.server import start_server

from geecs_bluesky.transport.tcp_subscriber import GeecsTcpSubscriber
from geecs_bluesky.transport.udp_client import GeecsUdpClient

from .channels import cast_value, make_readback_channel, make_setpoint_channel
from .config import DType, GatewayConfig

logger = logging.getLogger(__name__)


class GeecsCaGateway:
    """A single-process CA soft-IOC fronting one or more GEECS devices.

    Each device's TCP subscription runs under a supervising task that reconnects
    with exponential backoff on a dropped connection, and marks that device's
    readback PVs ``INVALID`` (alarm severity) while it is down so clients can tell
    live data from stale.  Live frames clear the alarm automatically.

    Parameters
    ----------
    config : GatewayConfig
        Declarative description of the devices and variables to serve.
    reconnect_min_s, reconnect_max_s : float
        Backoff bounds for the subscription reconnect loop.
    """

    def __init__(
        self,
        config: GatewayConfig,
        *,
        reconnect_min_s: float = 0.5,
        reconnect_max_s: float = 30.0,
    ) -> None:
        self.config = config
        self._reconnect_min = reconnect_min_s
        self._reconnect_max = reconnect_max_s
        self._supervisors: list[asyncio.Task] = []
        self._closing = False
        self.pvdb: dict[str, ChannelData] = {}
        # PV name -> (device name, geecs_var, "readback"|"setpoint"). The
        # authoritative bidirectional map (PV mapping is lossy at the string
        # level); doubles as the collision guard at build time.
        self.manifest: dict[str, tuple[str, str, str]] = {}
        self._udp: dict[str, GeecsUdpClient] = {}
        self._subs: dict[str, GeecsTcpSubscriber] = {}
        # device name -> {geecs_var -> (readback channel, dtype)}
        self._readbacks: dict[str, dict[str, tuple[ChannelData, DType]]] = {}
        self._build_pvdb()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_pvdb(self) -> None:
        """Populate ``self.pvdb``, the readback routing map, and the manifest."""
        for dev in self.config.devices:
            readback_map: dict[str, tuple[ChannelData, DType]] = {}
            for var in dev.variables:
                full = dev.pv_name_for(var)
                self._register(full, dev.name, var.geecs_var, "readback")
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
                    sp_name = f"{full}:SP"
                    self._register(sp_name, dev.name, var.geecs_var, "setpoint")
                    setter = self._make_setter(dev.name, var.geecs_var)
                    self.pvdb[sp_name] = make_setpoint_channel(
                        var.dtype,
                        setter,
                        egu=var.egu,
                        precision=var.precision,
                        lo=var.lo,
                        hi=var.hi,
                    )
            self._readbacks[dev.name] = readback_map

    def _register(self, pv: str, device: str, geecs_var: str, kind: str) -> None:
        """Record ``pv`` in the manifest, erroring on a name collision."""
        if pv in self.manifest:
            other_dev, other_var, _ = self.manifest[pv]
            raise ValueError(
                f"PV name collision: {pv!r} maps to both "
                f"{other_dev}/{other_var} and {device}/{geecs_var}"
            )
        self.manifest[pv] = (device, geecs_var, kind)

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
        """Open UDP clients (used for sets) to every configured device.

        The TCP subscription connection is opened and re-opened by the per-device
        supervisor started in :meth:`subscribe`.
        """
        for dev in self.config.devices:
            udp = GeecsUdpClient(dev.host, dev.port, device_name=dev.name)
            await udp.connect()
            self._udp[dev.name] = udp
        logger.info("opened UDP to %d device(s)", len(self.config.devices))

    async def subscribe(self) -> None:
        """Launch a supervised, auto-reconnecting TCP subscription per device."""
        for dev in self.config.devices:
            task = asyncio.create_task(
                self._supervise(dev), name=f"supervise[{dev.name}]"
            )
            self._supervisors.append(task)
        logger.info("started %d subscription supervisor(s)", len(self._supervisors))

    async def _supervise(self, dev) -> None:
        """Keep one device's subscription alive; reconnect with backoff on drop.

        The dropped-connection wait polls ``_listen_task.done()`` via a cancellable
        sleep rather than awaiting the subscriber's internal task directly — the
        latter has awkward cancellation semantics (its loop swallows
        ``CancelledError``), which can wedge :meth:`close`.
        """
        variables = [v.geecs_var for v in dev.variables]
        callback = self._make_callback(dev.name)
        backoff = self._reconnect_min
        while not self._closing:
            sub = GeecsTcpSubscriber(dev.host, dev.port)
            try:
                await sub.connect()
                self._subs[dev.name] = sub
                await sub.subscribe(variables, callback)
                logger.info("%s: subscription live", dev.name)
                backoff = self._reconnect_min
                # Wait for the listener to finish (connection drop), staying
                # cleanly cancellable at the sleep.
                while (
                    not self._closing
                    and sub._listen_task is not None
                    and not sub._listen_task.done()
                ):
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                await self._safe_close(sub)
                raise
            except Exception:
                logger.warning(
                    "%s: subscription error; will retry", dev.name, exc_info=True
                )
            await self._safe_close(sub)
            if self._closing:
                break
            # Connection dropped or failed to (re)establish.
            await self._mark_device_invalid(dev.name)
            logger.warning(
                "%s: subscription down; reconnecting in %.1fs", dev.name, backoff
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._reconnect_max)

    @staticmethod
    async def _safe_close(sub: GeecsTcpSubscriber) -> None:
        """Close a subscriber, swallowing teardown errors."""
        try:
            await sub.close()
        except Exception:
            logger.debug("error closing subscriber", exc_info=True)

    async def _mark_device_invalid(self, device: str) -> None:
        """Set a device's readback PVs to INVALID severity (data no longer live).

        Live frames issue plain value writes, which reset severity to NO_ALARM, so
        recovery is automatic — only the disconnect transition is set here.
        """
        for pv, (dev, _var, kind) in self.manifest.items():
            if dev != device or kind != "readback":
                continue
            channel = self.pvdb[pv]
            try:
                await channel.write(
                    channel.value,
                    severity=AlarmSeverity.INVALID_ALARM,
                    status=AlarmStatus.COMM,
                )
            except Exception:
                logger.debug("failed to mark %s INVALID", pv, exc_info=True)

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
        """Cancel supervisors and tear down all subscriptions and UDP clients."""
        self._closing = True
        for task in self._supervisors:
            task.cancel()
        if self._supervisors:
            await asyncio.gather(*self._supervisors, return_exceptions=True)
        self._supervisors.clear()
        for sub in list(self._subs.values()):
            await self._safe_close(sub)
        for udp in self._udp.values():
            await udp.close()
        self._udp.clear()
        self._subs.clear()
