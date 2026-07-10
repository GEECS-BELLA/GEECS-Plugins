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
import math
from dataclasses import dataclass
from typing import Any

from caproto import (
    AlarmSeverity,
    AlarmStatus,
    ChannelData,
    ChannelDouble,
    ChannelEnum,
    ChannelInteger,
    ChannelString,
)
from caproto.asyncio.server import start_server

from geecs_ca_gateway.exceptions import GeecsConnectionError
from geecs_ca_gateway.transport.tcp_subscriber import GeecsTcpSubscriber
from geecs_ca_gateway.transport.udp_client import GeecsUdpClient

from .alarms import AlarmLevel, AlarmSeverityName
from .channels import (
    cast_value,
    enum_index,
    make_description_channel,
    make_readback_channel,
    make_restart_channel,
    make_setpoint_channel,
    read_only,
)
from .config import GatewayConfig, VariableSpec
from .derived import DerivedChannelSpec, ExpressionEvaluator, derived_pv_name
from .pv_naming import pv_name

logger = logging.getLogger(__name__)

# Seconds between the LabVIEW epoch (1904-01-01) and the Unix epoch (1970-01-01).
# GEECS timestamps (e.g. `systimestamp`) are LabVIEW-epoch; subtract to get Unix.
_LABVIEW_EPOCH_OFFSET = 2_082_844_800

# Sentinel for "no value written yet" in the deadband cache.
_UNSET = object()

# Default budget (seconds) for a *setpoint* UDP exchange. GEECS sets are
# blocking — the exe response arrives only once the device reports the set
# converged (or failed) — so a slow-but-healthy axis can legitimately take tens
# of seconds. This must not undercut the motor-side move contract: CaMotor in
# GeecsBluesky (geecs_bluesky/devices/ca/motor.py, `_DEFAULT_MOVE_TIMEOUT`)
# gives the CA put its full 30 s move budget, on the principle that "a slow
# axis is not a dead one". Gets keep the shorter GeecsUdpClient default — a
# read that takes 10 s *is* a dead device.
_SET_EXE_TIMEOUT = 30.0

_ALARM_SEVERITY = {
    AlarmSeverityName.MINOR: AlarmSeverity.MINOR_ALARM,
    AlarmSeverityName.MAJOR: AlarmSeverity.MAJOR_ALARM,
    AlarmSeverityName.INVALID: AlarmSeverity.INVALID_ALARM,
}

_ALARM_STATUS = {
    AlarmLevel.LOLO: AlarmStatus.LOLO,
    AlarmLevel.LOW: AlarmStatus.LOW,
    AlarmLevel.HIGH: AlarmStatus.HIGH,
    AlarmLevel.HIHI: AlarmStatus.HIHI,
}

_DerivedInputKey = tuple[str, str]


@dataclass(frozen=True)
class _DerivedRuntime:
    """Runtime state for one derived PV."""

    spec: DerivedChannelSpec
    channel: ChannelData
    evaluator: ExpressionEvaluator
    pv: str


@dataclass(frozen=True)
class _DerivedInputValue:
    """Latest numeric value observed for one derived input."""

    value: float
    received_at: float


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
    with exponential backoff when the connection actually drops (the socket
    closes), and marks that device's readback PVs ``INVALID`` (alarm severity)
    while it is disconnected.

    A device merely going *quiet* is not treated as a drop: GEECS devices are
    legitimately silent for seconds (waiting on triggers, slow online analysis,
    toggled), so silence just ages the PV timestamp rather than forcing a
    (pointless) reconnect.  The remaining gap — a device hard-powered-off with the
    socket left open (no TCP FIN) — is best handled by TCP keepalive, a future
    refinement; app-level silence-guessing conflates "slow" with "dead".

    Parameters
    ----------
    config : GatewayConfig
        Declarative description of the devices and variables to serve.
    reconnect_min_s, reconnect_max_s : float
        Backoff bounds for the subscription reconnect loop.
    set_timeout_s : float
        Budget (seconds) for the UDP exe response of a *setpoint* write. GEECS
        sets block until convergence, so this must cover the slowest legitimate
        move; the default matches CaMotor's 30 s move budget in GeecsBluesky
        (see ``_SET_EXE_TIMEOUT``). Gets are unaffected.
    """

    def __init__(
        self,
        config: GatewayConfig,
        *,
        reconnect_min_s: float = 0.5,
        reconnect_max_s: float = 30.0,
        set_timeout_s: float = _SET_EXE_TIMEOUT,
    ) -> None:
        self.config = config
        self._reconnect_min = reconnect_min_s
        self._reconnect_max = reconnect_max_s
        self._set_timeout = set_timeout_s
        self._supervisors: list[asyncio.Task] = []
        # device name -> {geecs_var -> last value written} for deadband suppression
        self._last_written: dict[str, dict[str, Any]] = {}
        # (device, geecs_var) that have already logged a coercion warning, so a
        # mistyped variable warns once instead of every ~5 Hz frame.
        self._coerce_warned: set[tuple[str, str]] = set()
        # Last active value-alarm threshold, for per-PV hysteresis handling.
        self._value_alarm_levels: dict[tuple[str, str], AlarmLevel | None] = {}
        # devices currently logged as down, for one-line state-change logging
        # (a warning when it goes down, an info when it reconnects).
        self._down_logged: set[str] = set()
        self._closing = False
        self.pvdb: dict[str, ChannelData] = {}
        # PV name -> (device name, geecs_var, "readback"|"setpoint"). The
        # authoritative bidirectional map (PV mapping is lossy at the string
        # level); doubles as the collision guard at build time.
        self.manifest: dict[str, tuple[str, str, str]] = {}
        self._udp: dict[str, GeecsUdpClient] = {}
        self._subs: dict[str, GeecsTcpSubscriber] = {}
        # Self-diagnostics (devIocStats-style): per-device connection-state
        # PVs plus gateway uptime/heartbeat, so Phoebus/alarm layers see
        # liveness explicitly instead of inferring it from INVALID severity.
        self._connected: dict[str, ChannelData] = {}
        self._gateway_status: dict[str, ChannelData] = {}
        self._status_task: asyncio.Task | None = None
        self._started_at: float | None = None
        # Set by a CA put to CAGateway:RESTART; run() then shuts down cleanly
        # and reports it, so the entrypoint can exit with the restart code.
        self._restart_requested = asyncio.Event()
        # device name -> {geecs_var -> (readback channel, variable spec)}
        self._readbacks: dict[str, dict[str, tuple[ChannelData, VariableSpec]]] = {}
        # source device -> derived channel runtimes that depend on that source.
        self._derived_by_source: dict[str, list[_DerivedRuntime]] = {}
        self._derived_runtimes: list[_DerivedRuntime] = []
        # (source device, source variable) -> latest numeric value + receive time.
        self._derived_input_cache: dict[_DerivedInputKey, _DerivedInputValue] = {}
        # full derived PV name -> last value written, for deadband suppression
        self._derived_last_written: dict[str, Any] = {}
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
                if var.description:
                    # Serve the description as the PV's .DESC field. A plain
                    # '<pv>.DESC' string entry is resolved by the CA server's
                    # first-line name lookup — no record machinery needed.
                    self.pvdb[f"{full}.DESC"] = make_description_channel(
                        var.description
                    )

                if var.settable:
                    sp_name = f"{full}:SP"
                    if self._register(sp_name, dev.name, var.geecs_var, "setpoint"):
                        setter = self._make_setter(dev.name, var.geecs_var)
                        self.pvdb[sp_name] = make_setpoint_channel(var, setter)

            # Intrinsic timestamp variables (systimestamp/acq_timestamp) aren't in
            # the DB, but expose them as float readback PVs too — carrying the RAW
            # LabVIEW value, which is what's stamped on saved external assets
            # (images), so it's a per-device acquisition/synchronicity signal.
            data_var_names = {v.geecs_var for v in dev.variables}
            for ts_name in dev.timestamp_vars:
                if ts_name in data_var_names or ts_name in readback_map:
                    continue
                ts_spec = VariableSpec(geecs_var=ts_name, dtype="float")
                full = dev.pv_name_for(ts_spec)
                if self._register(full, dev.name, ts_name, "readback"):
                    channel = make_readback_channel(ts_spec)
                    self.pvdb[full] = channel
                    readback_map[ts_name] = (channel, ts_spec)
            self._readbacks[dev.name] = readback_map

            # Per-device connection-state PV (severity MAJOR while down).
            conn_pv = dev.pv_name_for(VariableSpec(geecs_var="CONNECTED"))
            if self._register(conn_pv, dev.name, "CONNECTED", "status"):
                conn = read_only(ChannelEnum)(
                    value="Disconnected",
                    enum_strings=["Disconnected", "Connected"],
                )
                self.pvdb[conn_pv] = conn
                self._connected[dev.name] = conn

        self._build_derived_pvs()
        self._build_gateway_status_pvs()

    def _default_experiment(self) -> str | None:
        """Return the first configured experiment prefix, if any."""
        return next((d.experiment for d in self.config.devices if d.experiment), None)

    def _build_derived_pvs(self) -> None:
        """Build read-only PVs for configured derived channels."""
        if not self.config.derived_channels:
            return
        source_devices = {dev.name for dev in self.config.devices}
        default_experiment = self._default_experiment()
        for spec in self.config.derived_channels:
            unknown_sources = sorted(spec.source_devices - source_devices)
            if unknown_sources:
                raise ValueError(
                    f"Derived channel {spec.device}:{spec.variable} references "
                    f"unknown source device(s) {unknown_sources!r}"
                )
            full = derived_pv_name(spec, default_experiment)
            if not self._register(full, spec.source_device, spec.variable, "derived"):
                continue
            var_spec = VariableSpec(
                geecs_var=spec.variable,
                pv=spec.pv,
                dtype="float",
                egu=spec.egu,
                precision=spec.precision,
                lo=spec.lo,
                hi=spec.hi,
                deadband=spec.deadband,
                description=spec.description,
            )
            channel = make_readback_channel(
                var_spec,
                initial_severity=AlarmSeverity.INVALID_ALARM,
                initial_status=AlarmStatus.UDF,
            )
            self.pvdb[full] = channel
            if var_spec.description:
                self.pvdb[f"{full}.DESC"] = make_description_channel(
                    var_spec.description
                )
            evaluator = ExpressionEvaluator(
                spec.expression, {inp.symbol for inp in spec.inputs}
            )
            runtime = _DerivedRuntime(spec, channel, evaluator, full)
            self._derived_runtimes.append(runtime)
            for source in sorted(spec.source_devices):
                self._derived_by_source.setdefault(source, []).append(runtime)

    def _build_gateway_status_pvs(self) -> None:
        """Expose gateway self-diagnostics under ``[Experiment:]CAGateway:*``."""
        experiment = next(
            (d.experiment for d in self.config.devices if d.experiment), None
        )
        try:
            from importlib.metadata import version

            pkg_version = version("geecs-ca-gateway")
        except Exception:
            pkg_version = "unknown"
        channels: dict[str, ChannelData] = {
            "UPTIME": read_only(ChannelDouble)(value=0.0, units="s", precision=0),
            "HEARTBEAT": read_only(ChannelInteger)(value=0),
            "DEVICES_CONNECTED": read_only(ChannelInteger)(value=0),
            "VERSION": read_only(ChannelString)(value=pkg_version),
            # Client-writable remote-restart control (devIocStats SYSRESET
            # pattern) — also the DB-resync mechanism after a device-set edit.
            "RESTART": make_restart_channel(self._request_restart),
        }
        for suffix, channel in channels.items():
            pv = pv_name(experiment, "CAGateway", suffix)
            if self._register(pv, "CAGateway", suffix, "status"):
                self.pvdb[pv] = channel
                self._gateway_status[suffix] = channel

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
        """Return an async setter closure that forwards a value over UDP.

        The set exchange gets the (longer) ``set_timeout_s`` budget rather than
        the client's default exe timeout: GEECS sets block until the device
        reports convergence, and a legitimate slow move (~10-30 s stage travel)
        must not be failed as a dead connection mid-flight.
        """

        async def setter(value: Any) -> Any:
            udp = self._udp.get(device_name)
            if udp is None:
                # UDP bind failed at startup (see connect()) — fail the caput
                # with a clear cause rather than a KeyError.
                raise GeecsConnectionError(
                    f"{device_name}: no UDP client (bind failed at startup); "
                    f"cannot forward set of {geecs_var!r}"
                )
            return await udp.set(geecs_var, value, timeout=self._set_timeout)

        return setter

    def _warn_once(self, device: str, geecs_var: str, message: str) -> None:
        """Log a per-(device, variable) warning once, to avoid ~5 Hz spam."""
        key = (device, geecs_var)
        if key not in self._coerce_warned:
            self._coerce_warned.add(key)
            logger.warning(message)

    def _alarm_write_kwargs(
        self, device: str, geecs_var: str, spec: VariableSpec, value: Any
    ) -> dict[str, Any]:
        """Return caproto alarm kwargs for a live readback write.

        Disconnect/stale INVALID is handled separately by
        :meth:`_mark_device_invalid`.  This method only evaluates live values
        against curated scalar alarm limits.
        """
        limits = spec.alarm_limits
        if limits is None:
            return {}
        key = (device, geecs_var)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            self._value_alarm_levels[key] = None
            return {"severity": AlarmSeverity.NO_ALARM, "status": AlarmStatus.NO_ALARM}
        evaluation = limits.evaluate(numeric, self._value_alarm_levels.get(key))
        self._value_alarm_levels[key] = None if evaluation is None else evaluation.level
        if evaluation is None:
            return {"severity": AlarmSeverity.NO_ALARM, "status": AlarmStatus.NO_ALARM}
        return {
            "severity": _ALARM_SEVERITY[evaluation.severity],
            "status": _ALARM_STATUS[evaluation.level],
        }

    async def _write_stream_value(
        self,
        device_name: str,
        var: str,
        raw: Any,
        channel: ChannelData,
        spec: VariableSpec,
        extra: dict[str, Any],
    ) -> None:
        """Cast and write one raw stream value into its readback PV."""
        if (
            spec.dtype in ("float", "int", "enum")
            and isinstance(raw, str)
            and not raw.strip()
        ):
            # Devices push '' for numeric/enum values they haven't computed yet
            # (camera analysis fields before the first acquisition, idle devices'
            # whole frames). Not a type mismatch — skip quietly and leave the PV
            # at its previous/placeholder value until real data arrives.
            # (string/path dtypes are exempt: '' is a legitimate value there,
            # e.g. a cleared save path.)
            logger.debug("%s: %s is empty (no value yet); skipping", device_name, var)
            return
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
                    return
            else:
                value = cast_value(spec.dtype, raw)
        except (ValueError, TypeError):
            self._warn_once(
                device_name,
                var,
                f"{device_name}: {var}={raw!r} is not a {spec.dtype} "
                f"(likely a DB variabletype mismatch); ignoring this variable",
            )
            return

        # Deadband / change suppression: don't re-post an unchanged value
        # (floats within the deadband). Keeps CA + archiver traffic to real
        # changes; a static device costs nothing.
        last_map = self._last_written.setdefault(device_name, {})
        prev = last_map.get(var, _UNSET)
        if prev is not _UNSET:
            if spec.dtype == "float":
                both_nan = math.isnan(value) and math.isnan(prev)
                if both_nan or abs(value - prev) <= spec.deadband:
                    return
            elif value == prev:
                return
        last_map[var] = value
        try:
            # Curated value-alarm severity rides the live write (device
            # disconnect/stale INVALID is handled separately by
            # _mark_device_invalid). verify_value is disabled only when we
            # supply alarm state, so caproto's native limit evaluation never
            # fights the overlay (the single-evaluator contract).
            alarm_kwargs = self._alarm_write_kwargs(device_name, var, spec, value)
            await channel.write(
                value,
                **extra,
                **alarm_kwargs,
                verify_value=not bool(alarm_kwargs),
            )
        except Exception:
            self._warn_once(
                device_name,
                var,
                f"{device_name}: failed to write PV {var}={value!r} "
                f"(skipping this variable)",
            )

    async def _write_derived_channels(
        self, device_name: str, update: dict[str, Any], extra: dict[str, Any]
    ) -> None:
        """Evaluate derived PVs affected by this source frame."""
        runtimes = self._derived_by_source.get(device_name, [])
        if not runtimes:
            return
        now = asyncio.get_running_loop().time()
        self._update_derived_input_cache(device_name, update, runtimes, now)
        for runtime in runtimes:
            values = self._derived_values(runtime, now)
            if values is None:
                await self._mark_derived_invalid(
                    runtime.channel, runtime.pv, AlarmStatus.UDF
                )
                continue
            try:
                value = runtime.evaluator.evaluate(values)
            except Exception:
                self._warn_once(
                    device_name,
                    runtime.pv,
                    f"{device_name}: derived PV {runtime.pv} expression failed; "
                    "marking INVALID",
                )
                await self._mark_derived_invalid(
                    runtime.channel, runtime.pv, AlarmStatus.CALC
                )
                continue
            output_extra = extra if not runtime.spec.is_cross_device else {}
            await self._write_derived_value(runtime, value, output_extra)

    async def _sweep_stale_derived_channels(self, now: float) -> None:
        """Mark derived PVs invalid when cached inputs age out with no frames."""
        for runtime in self._derived_runtimes:
            if runtime.spec.stale_after is None:
                continue
            if self._derived_values(runtime, now) is None:
                await self._mark_derived_invalid(
                    runtime.channel, runtime.pv, AlarmStatus.UDF
                )

    def _update_derived_input_cache(
        self,
        device_name: str,
        update: dict[str, Any],
        runtimes: list[_DerivedRuntime],
        received_at: float,
    ) -> None:
        """Cache numeric inputs from one source frame for derived channels."""
        seen_inputs = {
            inp.variable
            for runtime in runtimes
            for inp in runtime.spec.inputs
            if inp.device == device_name
        }
        for variable in seen_inputs:
            key = (device_name, variable)
            raw = update.get(variable, _UNSET)
            if raw is _UNSET or (isinstance(raw, str) and not raw.strip()):
                self._derived_input_cache.pop(key, None)
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                self._derived_input_cache.pop(key, None)
                for runtime in runtimes:
                    if any(
                        inp.device == device_name and inp.variable == variable
                        for inp in runtime.spec.inputs
                    ):
                        self._warn_once(
                            device_name,
                            runtime.pv,
                            f"{device_name}: derived PV {runtime.pv} input "
                            f"{variable}={raw!r} is not numeric; marking INVALID",
                        )
                continue
            self._derived_input_cache[key] = _DerivedInputValue(value, received_at)

    def _derived_values(
        self, runtime: _DerivedRuntime, now: float
    ) -> dict[str, float] | None:
        """Return expression values for a runtime, or None when invalid/stale."""
        values: dict[str, float] = {}
        for inp in runtime.spec.inputs:
            cached = self._derived_input_cache.get((inp.device, inp.variable))
            if cached is None:
                return None
            if (
                runtime.spec.stale_after is not None
                and now - cached.received_at > runtime.spec.stale_after
            ):
                return None
            values[inp.symbol] = cached.value
        return values

    async def _write_derived_value(
        self, runtime: _DerivedRuntime, value: float, extra: dict[str, Any]
    ) -> None:
        """Write one computed derived value, applying deadband suppression."""
        prev = self._derived_last_written.get(runtime.pv, _UNSET)
        if prev is not _UNSET:
            both_nan = math.isnan(value) and math.isnan(prev)
            if both_nan or abs(value - prev) <= runtime.spec.deadband:
                return
        self._derived_last_written[runtime.pv] = value
        try:
            await runtime.channel.write(
                value,
                **extra,
                severity=AlarmSeverity.NO_ALARM,
                status=AlarmStatus.NO_ALARM,
                verify_value=False,
            )
        except Exception:
            self._warn_once(
                runtime.spec.source_device,
                runtime.pv,
                f"{runtime.spec.source_device}: failed to write derived PV "
                f"{runtime.pv}={value!r}",
            )

    async def _mark_derived_invalid(
        self, channel: ChannelData, pv: str, status: AlarmStatus
    ) -> None:
        """Mark a derived PV invalid and clear its deadband cache."""
        self._derived_last_written.pop(pv, None)
        if channel.severity == AlarmSeverity.INVALID_ALARM and channel.status == status:
            return
        try:
            await channel.write(
                channel.value,
                severity=AlarmSeverity.INVALID_ALARM,
                status=status,
                verify_value=False,
            )
        except Exception:
            logger.debug("failed to mark derived PV %s INVALID", pv, exc_info=True)

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
            """Fan one push frame into readback PVs, timestamp variables last.

            Ordering guarantee: all data variables of a frame are posted before
            its timestamp variable(s), so a client triggering on
            ``acq_timestamp`` observes the completed frame.  Each PV write is
            an ``await``, so posting the shot id first would let a strict-mode
            Bluesky ``CaTriggerable`` complete its ``trigger()`` on the new
            shot while the data PVs still hold the previous frame's values. The
            stable split preserves device payload order among data variables;
            derived channels post after raw data and before timestamps.
            """
            timestamp = _extract_timestamp(update, ts_vars)
            extra = {"timestamp": timestamp} if timestamp is not None else {}
            data_items = [
                (var, raw) for var, raw in update.items() if var not in ts_vars
            ]
            timestamp_items = [
                (var, raw) for var, raw in update.items() if var in ts_vars
            ]
            for var, raw in data_items:
                entry = readback_map.get(var)
                if entry is None:
                    continue
                channel, spec = entry
                await self._write_stream_value(
                    device_name, var, raw, channel, spec, extra
                )
            await self._write_derived_channels(device_name, update, extra)
            for var, raw in timestamp_items:
                entry = readback_map.get(var)
                if entry is None:
                    continue
                channel, spec = entry
                await self._write_stream_value(
                    device_name, var, raw, channel, spec, extra
                )

        return callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open UDP clients (used for sets) to every configured device.

        Per-device fault tolerance, matching the subscription supervisor's
        philosophy: one device failing to bind (e.g. an unroutable IP making
        local-IP detection fall back to ``""``, which raises ``EADDRNOTAVAIL``
        on PPP/VPN links — see the ``udp_client`` module docstring) is logged
        loudly and skipped, and the rest of the gateway starts normally. The
        skipped device's readbacks still stream over TCP; only its setpoint
        writes fail (with :class:`GeecsConnectionError`) until a restart.

        The TCP subscription connection is opened and re-opened by the per-device
        supervisor started in :meth:`subscribe`.
        """
        for dev in self.config.devices:
            udp = GeecsUdpClient(dev.host, dev.port, device_name=dev.name)
            try:
                await udp.connect()
            except OSError:
                logger.error(
                    "%s: UDP bind/connect to %s:%s failed — starting without it "
                    "(setpoint writes for this device will fail until the "
                    "gateway is restarted)",
                    dev.name,
                    dev.host,
                    dev.port,
                    exc_info=True,
                )
                # Release any partially-created transports for this device.
                await udp.close()
                continue
            self._udp[dev.name] = udp
        logger.info(
            "opened UDP to %d/%d device(s)",
            len(self._udp),
            len(self.config.devices),
        )

    async def subscribe(self) -> None:
        """Launch a supervised, auto-reconnecting TCP subscription per device."""
        for dev in self.config.devices:
            task = asyncio.create_task(
                self._supervise(dev), name=f"supervise[{dev.name}]"
            )
            self._supervisors.append(task)
        logger.info("started %d subscription supervisor(s)", len(self._supervisors))
        self._status_task = asyncio.create_task(
            self._status_loop(), name="gateway-status"
        )

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
        derived_inputs = [
            inp.variable
            for runtime in self._derived_by_source.get(dev.name, [])
            for inp in runtime.spec.inputs
            if inp.device == dev.name
        ]
        variables = list(dict.fromkeys(data_vars + derived_inputs + dev.timestamp_vars))
        # Text-typed variables must reach cast_value as the exact wire text —
        # numeric coercion round-trips '007' → 7 → "7", mangling string PVs.
        # Enums are included too: choice labels are strings, and a numeric-
        # looking label (e.g. "1.0") must survive verbatim for enum_index.
        text_variables = {
            v.geecs_var for v in dev.variables if v.dtype in ("string", "path", "enum")
        }
        callback = self._make_callback(dev)
        backoff = self._reconnect_min
        while not self._closing:
            sub = GeecsTcpSubscriber(dev.host, dev.port)
            try:
                await sub.connect()
                self._subs[dev.name] = sub
                await sub.subscribe(variables, callback, text_variables=text_variables)
                # Clear the deadband cache so the first frame always posts (and
                # clears any INVALID left from the drop), even if unchanged.
                self._last_written[dev.name] = {}
                for runtime in self._derived_by_source.get(dev.name, []):
                    self._derived_last_written.pop(runtime.pv, None)
                for key in list(self._derived_input_cache):
                    if key[0] == dev.name:
                        self._derived_input_cache.pop(key, None)
                if dev.name in self._down_logged:
                    self._down_logged.discard(dev.name)
                    logger.info("%s: reconnected", dev.name)
                else:
                    logger.info("%s: subscription live", dev.name)
                await self._set_connected(dev.name, True)
                backoff = self._reconnect_min
                # Wait for an actual disconnect — the listener task ends when the
                # socket closes. A device merely going quiet is NOT a drop; poll
                # so we stay cleanly cancellable at the sleep.
                while (
                    not self._closing
                    and sub._listen_task is not None
                    and not sub._listen_task.done()
                ):
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                await self._safe_close(sub)
                raise
            except (OSError, asyncio.TimeoutError):
                pass  # device off/unreachable — logged once below, no traceback
            except Exception:
                logger.warning(
                    "%s: unexpected subscription error", dev.name, exc_info=True
                )
            await self._safe_close(sub)
            if self._closing:
                break
            # Dropped or failed to (re)establish: mark stale and log once per down
            # episode (recovery is logged as "reconnected" above).
            await self._mark_device_invalid(dev.name)
            await self._set_connected(dev.name, False)
            if dev.name not in self._down_logged:
                self._down_logged.add(dev.name)
                logger.warning(
                    "%s: unreachable/dropped; retrying (up to every %.0fs)",
                    dev.name,
                    self._reconnect_max,
                )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._reconnect_max)

    async def _set_connected(self, device: str, connected: bool) -> None:
        """Update the device's CONNECTED PV and the gateway connected-count."""
        channel = self._connected.get(device)
        if channel is None:
            return
        try:
            await channel.write(
                1 if connected else 0,
                severity=(
                    AlarmSeverity.NO_ALARM if connected else AlarmSeverity.MAJOR_ALARM
                ),
                status=AlarmStatus.NO_ALARM if connected else AlarmStatus.COMM,
            )
            count = self._gateway_status.get("DEVICES_CONNECTED")
            if count is not None:
                total = sum(
                    1 for ch in self._connected.values() if str(ch.value) == "Connected"
                )
                await count.write(total)
        except Exception:
            logger.debug("failed to update CONNECTED for %s", device, exc_info=True)

    async def _status_loop(self, period_s: float = 5.0) -> None:
        """Tick the gateway UPTIME/HEARTBEAT PVs (devIocStats-style)."""
        uptime = self._gateway_status.get("UPTIME")
        heartbeat = self._gateway_status.get("HEARTBEAT")
        beats = 0
        loop = asyncio.get_running_loop()
        self._started_at = loop.time()
        while True:
            try:
                if uptime is not None:
                    await uptime.write(loop.time() - self._started_at)
                if heartbeat is not None:
                    beats += 1
                    await heartbeat.write(beats)
                await self._sweep_stale_derived_channels(loop.time())
            except Exception:
                logger.debug("status loop write failed", exc_info=True)
            await asyncio.sleep(period_s)

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
                    verify_value=False,
                )
            except Exception:
                logger.debug("failed to mark %s INVALID", pv, exc_info=True)
        for key in list(self._derived_input_cache):
            if key[0] == device:
                self._derived_input_cache.pop(key, None)
        for runtime in self._derived_by_source.get(device, []):
            await self._mark_derived_invalid(
                runtime.channel, runtime.pv, AlarmStatus.COMM
            )

    def _request_restart(self) -> None:
        """Handle a CA put to ``CAGateway:RESTART``: initiate clean shutdown."""
        if not self._restart_requested.is_set():
            logger.warning(
                "restart requested via CAGateway:RESTART — shutting down "
                "(systemd relaunches the service; config rebuilds from the DB)"
            )
            self._restart_requested.set()

    async def serve(self) -> None:
        """Run the CA server until cancelled (serves ``self.pvdb``)."""
        await start_server(self.pvdb, log_pv_names=True)

    async def run(self) -> bool:
        """Connect, subscribe, and serve — the full gateway lifecycle.

        Returns
        -------
        bool
            True if shutdown was requested via the ``CAGateway:RESTART`` PV
            (the entrypoint turns this into the restart exit code), False if
            the server ended any other way.
        """
        await self.connect()
        await self.subscribe()
        serve_task = asyncio.create_task(self.serve(), name="ca-server")
        restart_task = asyncio.create_task(
            self._restart_requested.wait(), name="restart-wait"
        )
        try:
            await asyncio.wait(
                {serve_task, restart_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if serve_task.done():
                serve_task.result()  # propagate a server crash
        finally:
            for task in (serve_task, restart_task):
                task.cancel()
            await asyncio.gather(serve_task, restart_task, return_exceptions=True)
            await self.close()
        return self._restart_requested.is_set()

    async def close(self) -> None:
        """Cancel supervisors and tear down all subscriptions and UDP clients."""
        self._closing = True
        if self._status_task is not None:
            self._status_task.cancel()
            self._status_task = None
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
