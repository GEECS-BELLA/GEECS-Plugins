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
import time
from typing import Any

from caproto import AlarmSeverity, AlarmStatus, ChannelData
from caproto.asyncio.server import start_server

from geecs_bluesky.transport.tcp_subscriber import GeecsTcpSubscriber
from geecs_bluesky.transport.udp_client import GeecsUdpClient

from .channels import (
    cast_value,
    enum_index,
    make_readback_channel,
    make_setpoint_channel,
)
from .config import GatewayConfig, VariableSpec

logger = logging.getLogger(__name__)

# Seconds between the LabVIEW epoch (1904-01-01) and the Unix epoch (1970-01-01).
# GEECS timestamps (e.g. `systimestamp`) are LabVIEW-epoch; subtract to get Unix.
_LABVIEW_EPOCH_OFFSET = 2_082_844_800

# Sentinel for "no value written yet" in the deadband cache.
_UNSET = object()


def _extract_timestamp(update: dict[str, Any], ts_vars: list[str]) -> float | None:
    """Return a Unix-epoch timestamp from a frame per the ladder, or ``None``.

    Tries each variable in ``ts_vars`` in order, converts the LabVIEW-epoch value
    to Unix, and returns the first plausible (positive) result.  ``None`` means
    "let caproto default to receive-time".
    """
    for name in ts_vars:
        if name not in update:
            continue
        try:
            unix = float(update[name]) - _LABVIEW_EPOCH_OFFSET
        except (TypeError, ValueError):
            continue
        if unix > 0:
            return unix
    return None


class GeecsCaGateway:
    """A single-process CA soft-IOC fronting one or more GEECS devices.

    Each device's TCP subscription runs under a supervising task that reconnects
    with exponential backoff on a dropped connection, and marks that device's
    readback PVs ``INVALID`` (alarm severity) while it is down so clients can tell
    live data from stale.  Live frames clear the alarm automatically.

    A drop is detected two ways: the socket closing (the listener task ends), and
    a stall watchdog — GEECS pushes at ~5 Hz, so no frame for ``stall_timeout_s``
    means the device went silent (e.g. powered off with no TCP FIN), which the
    socket-close signal alone would miss.

    Parameters
    ----------
    config : GatewayConfig
        Declarative description of the devices and variables to serve.
    reconnect_min_s, reconnect_max_s : float
        Backoff bounds for the subscription reconnect loop.
    stall_timeout_s : float
        Treat the stream as dropped if no frame arrives for this long.
    """

    def __init__(
        self,
        config: GatewayConfig,
        *,
        reconnect_min_s: float = 0.5,
        reconnect_max_s: float = 30.0,
        stall_timeout_s: float = 2.0,
    ) -> None:
        self.config = config
        self._reconnect_min = reconnect_min_s
        self._reconnect_max = reconnect_max_s
        self._stall_timeout = stall_timeout_s
        self._supervisors: list[asyncio.Task] = []
        self._last_frame: dict[str, float] = {}
        # device name -> {geecs_var -> last value written} for deadband suppression
        self._last_written: dict[str, dict[str, Any]] = {}
        # (device, geecs_var) that have already logged a coercion warning, so a
        # mistyped variable warns once instead of every ~5 Hz frame.
        self._coerce_warned: set[tuple[str, str]] = set()
        self._closing = False
        self.pvdb: dict[str, ChannelData] = {}
        # PV name -> (device name, geecs_var, "readback"|"setpoint"). The
        # authoritative bidirectional map (PV mapping is lossy at the string
        # level); doubles as the collision guard at build time.
        self.manifest: dict[str, tuple[str, str, str]] = {}
        self._udp: dict[str, GeecsUdpClient] = {}
        self._subs: dict[str, GeecsTcpSubscriber] = {}
        # device name -> {geecs_var -> (readback channel, variable spec)}
        self._readbacks: dict[str, dict[str, tuple[ChannelData, VariableSpec]]] = {}
        self._build_pvdb()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_pvdb(self) -> None:
        """Populate ``self.pvdb``, the readback routing map, and the manifest."""
        for dev in self.config.devices:
            readback_map: dict[str, tuple[ChannelData, VariableSpec]] = {}
            for var in dev.variables:
                full = dev.pv_name_for(var)
                if not self._register(full, dev.name, var.geecs_var, "readback"):
                    continue  # exact duplicate — already built
                readback = make_readback_channel(var)
                self.pvdb[full] = readback
                readback_map[var.geecs_var] = (readback, var)

                if var.settable:
                    sp_name = f"{full}:SP"
                    if self._register(sp_name, dev.name, var.geecs_var, "setpoint"):
                        setter = self._make_setter(dev.name, var.geecs_var)
                        self.pvdb[sp_name] = make_setpoint_channel(var, setter)
            self._readbacks[dev.name] = readback_map

    def _register(self, pv: str, device: str, geecs_var: str, kind: str) -> bool:
        """Record ``pv`` in the manifest; return whether it was newly added.

        An exact duplicate — the same ``(device, geecs_var, kind)`` (the GEECS DB
        can list a variable more than once) — is tolerated and returns ``False``.
        A genuine collision — a *different* source mapping to the same PV — raises.
        """
        existing = self.manifest.get(pv)
        if existing is not None:
            if existing == (device, geecs_var, kind):
                return False
            other_dev, other_var, _ = existing
            raise ValueError(
                f"PV name collision: {pv!r} maps to both "
                f"{other_dev}/{other_var} and {device}/{geecs_var}"
            )
        self.manifest[pv] = (device, geecs_var, kind)
        return True

    def _make_setter(self, device_name: str, geecs_var: str):
        """Return an async setter closure that forwards a value over UDP."""

        async def setter(value: Any) -> Any:
            return await self._udp[device_name].set(geecs_var, value)

        return setter

    def _warn_once(self, device: str, geecs_var: str, message: str) -> None:
        """Log a per-(device, variable) warning once, to avoid ~5 Hz spam."""
        key = (device, geecs_var)
        if key not in self._coerce_warned:
            self._coerce_warned.add(key)
            logger.warning(message)

    def _make_callback(self, dev):
        """Return the subscription callback that fans a push frame into PVs.

        Each frame is stamped with a wall-clock timestamp from the device's
        timestamp ladder (``dev.timestamp_vars``) so the PV carries the GEECS
        acquisition time rather than gateway-receive time.
        """
        device_name = dev.name
        readback_map = self._readbacks[device_name]
        ts_vars = dev.timestamp_vars

        async def callback(update: dict[str, Any]) -> None:
            self._last_frame[device_name] = time.monotonic()
            timestamp = _extract_timestamp(update, ts_vars)
            extra = {"timestamp": timestamp} if timestamp is not None else {}
            last_map = self._last_written.setdefault(device_name, {})
            for var, raw in update.items():
                entry = readback_map.get(var)
                if entry is None:
                    continue
                channel, spec = entry
                try:
                    if spec.dtype == "enum":
                        value = enum_index(spec.choices, raw)
                        if value is None:
                            self._warn_once(
                                device_name,
                                var,
                                f"{device_name}: {var}={raw!r} not in enum choices "
                                f"{spec.choices}; ignoring (DB choice mismatch?)",
                            )
                            continue
                    else:
                        value = cast_value(spec.dtype, raw)
                except (ValueError, TypeError):
                    self._warn_once(
                        device_name,
                        var,
                        f"{device_name}: {var}={raw!r} is not a {spec.dtype} "
                        f"(likely a DB variabletype mismatch); ignoring this variable",
                    )
                    continue

                # Deadband / change suppression: don't re-post an unchanged value
                # (floats within the deadband). Keeps CA + archiver traffic to
                # real changes; a static device costs nothing.
                prev = last_map.get(var, _UNSET)
                if prev is not _UNSET:
                    if spec.dtype == "float":
                        if abs(value - prev) <= spec.deadband:
                            continue
                    elif value == prev:
                        continue
                last_map[var] = value
                try:
                    await channel.write(value, **extra)
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
        # Subscribe to data variables plus the timestamp ladder (deduped, order
        # preserved) so the device pushes the timestamp alongside the data.
        data_vars = [v.geecs_var for v in dev.variables]
        variables = list(dict.fromkeys(data_vars + dev.timestamp_vars))
        callback = self._make_callback(dev)
        backoff = self._reconnect_min
        while not self._closing:
            sub = GeecsTcpSubscriber(dev.host, dev.port)
            try:
                await sub.connect()
                self._subs[dev.name] = sub
                await sub.subscribe(variables, callback)
                # Seed so the watchdog doesn't trip before the first frame lands.
                self._last_frame[dev.name] = time.monotonic()
                # Clear the deadband cache so the first frame always posts (and
                # clears any INVALID left from the drop), even if unchanged.
                self._last_written[dev.name] = {}
                logger.info("%s: subscription live", dev.name)
                backoff = self._reconnect_min
                # Wait for a drop, staying cleanly cancellable at the sleep. A
                # drop is either the socket closing (listener task ends) or the
                # stream going silent (stall watchdog).
                while (
                    not self._closing
                    and sub._listen_task is not None
                    and not sub._listen_task.done()
                ):
                    if (
                        time.monotonic() - self._last_frame[dev.name]
                        > self._stall_timeout
                    ):
                        logger.warning(
                            "%s: no frames for >%.1fs; treating as stalled",
                            dev.name,
                            self._stall_timeout,
                        )
                        break
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
