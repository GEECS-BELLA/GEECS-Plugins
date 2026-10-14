r"""ScanModeControl — the worker's side of GEECS scan mode, as one Movable.

Under a trigger profile with ``use_scan_mode`` the bound plans drive this
device at three points of a strict run (the rules: CLAUDE.md, "Scan mode"):

- ``mv(control, EnterScanMode(devices, parameter))`` right after ARMED,
  before the stage and the claim: ``ScanStart`` on the experiment's
  multicast group, then every run device's acknowledgement (one retry),
  every run host's path-channel connection (port 6342) and one drain.  A
  silent or refusing device refuses the run by name
  (:exc:`~geecs_bluesky.exceptions.GeecsScanModeError`): nothing claimed.
- ``mv(control, ANNOUNCE)`` on every shot, between the detectors' trigger
  and the fire: the run-wide counter advances (never re-sent) and
  ``Path>>…\ScanNNN_xxx_SSS.fileextension`` — the claimed folder on the
  device-side root — goes out on every connection; the row reads the
  counter back as :data:`ANNOUNCED_SHOT`, the files' join key.
- ``mv(control, EXIT)`` in the bracket's finalizer, before STANDBY: the
  ``ScanEnd`` multicast, the ``ScanEnd`` frame on every connection (the
  exit the devices act on), the acknowledgements (logged), the server
  closed.  Idempotent.

The announced address must be this host's (a locked device takes commands
only from it; ``client_ip`` names it under another name).  A state file
records the run; :meth:`ScanModeControl.release_stale` clears a dead
worker's at environment open.  Not an ophyd device: nothing to connect.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import socket
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ophyd_async.core import AsyncStatus

from geecs_bluesky.data_paths import device_server_save_path
from geecs_bluesky.exceptions import GeecsConfigurationError, GeecsScanModeError
from geecs_core.transport.scan_mode import (
    SCAN_END,
    SCAN_MODE_TCP_PORT,
    SCAN_START,
    ScanModeAck,
    ScanModeAckListener,
    ScanModeMulticaster,
    ScanModeServer,
    device_shot_path,
)
from geecs_core.transport.udp_client import detect_local_ip

logger = logging.getLogger(__name__)

#: ``mv(control, ANNOUNCE)``: announce the next shot (the path push).
ANNOUNCE = "announce"
#: ``mv(control, EXIT)``: end scan mode on the run's devices.
EXIT = "exit"
#: The row column: the shot number announced for the row's shot.
ANNOUNCED_SHOT = "announced_shot"
#: Master Control's spacing between the multicast ``ScanEnd`` and the frame.
EXIT_FRAME_DELAY_S = 0.04
#: The drain after entry: one trigger period plus the gated plan's margin —
#: a frame an edge put in flight before the devices paused lands within it.
DEFAULT_DRAIN_S = 1.25
#: The state file's environment override and per-user default.
ENV_STATE_FILE = "GEECS_SCAN_MODE_STATE"
DEFAULT_STATE_FILE = Path("~/.local/state/geecs-bluesky/scan_mode.json")

_SCAN_FOLDER = re.compile(r"Scan(\d+)$")


@dataclass(frozen=True)
class EnterScanMode:
    """``mv(control, EnterScanMode(...))``: ``ScanStart``, then wait for *devices*.

    Parameters
    ----------
    devices :
        The GEECS names of the run's devices: every one must acknowledge
        ``ScanStart`` and its host connect to the path channel before the
        first shot.
    scan_parameter :
        The second ``ScanStart`` argument — Master Control's scan-tab
        selection, which the device stores and reads nothing from.
    """

    devices: tuple[str, ...]
    scan_parameter: str = "Shot #"


def default_state_file(env: Mapping[str, str] | None = None) -> Path:
    """The state file: ``$GEECS_SCAN_MODE_STATE``, else the per-user default."""
    source = os.environ if env is None else env
    raw = source.get(ENV_STATE_FILE)
    return Path(raw).expanduser() if raw else DEFAULT_STATE_FILE.expanduser()


def scan_number_of(folder: Path | str) -> int:
    """The scan number a ``scans/ScanNNN`` folder's name carries."""
    match = _SCAN_FOLDER.search(Path(folder).name)
    if match is None:
        raise GeecsConfigurationError(
            f"scan mode: {folder} is not a ScanNNN folder — no shot path can "
            "be built from it"
        )
    return int(match.group(1))


class ScanModeControl:
    """The scan-mode announce: Movable over the three commands, Readable per row.

    Parameters
    ----------
    experiment :
        The GEECS experiment, the first ``ScanStart`` argument.
    endpoint :
        ``(group, port)`` — ``expt.MultiCastIP`` / ``MultiCastRemotePort``;
        ``None`` disables the control (every entry is refused by name).
    path_provider :
        The shared run-folder provider (``GeecsScanPathProvider``): its
        ``folder`` is the claimed run folder, read at every announce.
    device_endpoints :
        Device → ``(ip, command port)`` for every enabled device of the
        experiment (the roster's ``device_endpoints``): the hosts that must
        connect and the ports the acknowledgements arrive on.
    local_ip :
        The address to bind and announce; ``None`` resolves it per run with
        the gateway's rule (``detect_local_ip`` against the first run
        device's host).
    client_ip :
        The string announced in ``ScanStart``/``ScanEnd`` when the devices
        know this host by another name; must resolve to the local IP.
    tcp_port :
        The path channel's port (the framework constant; ``0`` lets the OS
        choose, for tests that set their fake devices' port afterwards).
    ack_timeout_s, connect_timeout_s, end_timeout_s :
        Seconds to wait for the ``ScanStart`` acknowledgements (per
        attempt), the hosts' connections, and the ``ScanEnd``
        acknowledgements plus the devices' own closes.
    drain_s :
        Seconds to wait after a successful entry before the first shot.
    retries :
        How many times ``ScanStart`` is repeated for devices still silent.
    state_file :
        Where the run is recorded between entry and exit (default
        :func:`default_state_file`).
    device_path :
        Local run folder → the device-side folder (the data-paths mapping).
    boundary_writes, db_root_path :
        The DB's ``set='yes'`` rows per device and its ``RootPath`` —
        carried for the preflight (``plans/scan_mode.py``), unused here.
    """

    name = "scan_mode"
    parent = None

    def __init__(
        self,
        experiment: str,
        *,
        endpoint: tuple[str, int] | None,
        path_provider: Any,
        device_endpoints: Mapping[str, tuple[str, int]],
        local_ip: str | None = None,
        client_ip: str | None = None,
        tcp_port: int = SCAN_MODE_TCP_PORT,
        ack_timeout_s: float = 5.0,
        connect_timeout_s: float = 5.0,
        end_timeout_s: float = 5.0,
        drain_s: float = DEFAULT_DRAIN_S,
        retries: int = 1,
        state_file: Path | None = None,
        device_path: Callable[[str], str] = device_server_save_path,
        boundary_writes: Mapping[str, list[Mapping[str, Any]]] | None = None,
        db_root_path: str | None = None,
    ) -> None:
        self._experiment = experiment
        self._endpoint = (str(endpoint[0]), int(endpoint[1])) if endpoint else None
        self._path_provider = path_provider
        self._device_endpoints = {
            device: (str(ip), int(port))
            for device, (ip, port) in device_endpoints.items()
        }
        self._fixed_local_ip = local_ip
        self._client_ip = client_ip
        self._tcp_port = int(tcp_port)
        self._ack_timeout_s = float(ack_timeout_s)
        self._connect_timeout_s = float(connect_timeout_s)
        self._end_timeout_s = float(end_timeout_s)
        self._drain_s = float(drain_s)
        self._retries = max(0, int(retries))
        self._state_file = state_file or default_state_file()
        self._device_path = device_path
        self.boundary_writes: dict[str, list[Mapping[str, Any]]] = {
            device: list(rows) for device, rows in (boundary_writes or {}).items()
        }
        self.db_root_path = db_root_path

        self._server: ScanModeServer | None = None
        self._listener: ScanModeAckListener | None = None
        self._sender: ScanModeMulticaster | None = None
        self._run_devices: tuple[str, ...] = ()
        self._scan_parameter = "Shot #"
        self._local_ip: str | None = None
        self._announced_as: str | None = None
        self._shot = 0
        self._last_path: str | None = None

    # ------------------------------------------------------------ building

    @classmethod
    def from_experiment(
        cls,
        experiment: str,
        *,
        path_provider: Any,
        device_endpoints: Mapping[str, tuple[str, int]],
        geecs_db: Any | None = None,
        **kwargs: Any,
    ) -> ScanModeControl:
        """Build from the GEECS DB's experiment row, fail-soft.

        Reads the multicast endpoint, the scan-boundary rows and the
        device-side root once (at environment open, like the roster).  A
        DB failure or a blank endpoint leaves the control disabled with a
        WARNING: the worker still opens, and a run under a scan-mode
        profile is refused by name.
        """
        if geecs_db is None:
            from geecs_core.db.geecs_db import GeecsDb as geecs_db
        endpoint: tuple[str, int] | None = None
        boundary: Mapping[str, list[Mapping[str, Any]]] = {}
        root: str | None = None
        try:
            found = geecs_db.get_scan_mode_endpoint(experiment)
            endpoint = (found.group, int(found.port)) if found is not None else None
            boundary = geecs_db.get_scan_boundary_writes(experiment)
            root = geecs_db.get_root_path(experiment)
        except Exception as exc:  # noqa: BLE001 — worker must still start
            logger.warning(
                "scan mode: the experiment's scan-mode facts could not be read "
                "from the GEECS DB (%s: %s) — runs under a scan-mode trigger "
                "profile are refused until the environment is reopened",
                type(exc).__name__,
                exc,
            )
            endpoint = None
        if endpoint is None:
            logger.info(
                "scan mode: %r has no multicast endpoint (expt.MultiCastIP / "
                "MultiCastRemotePort) — scan-mode trigger profiles are refused "
                "on this worker",
                experiment,
            )
        return cls(
            experiment,
            endpoint=endpoint,
            path_provider=path_provider,
            device_endpoints=device_endpoints,
            boundary_writes=boundary,
            db_root_path=root,
            **kwargs,
        )

    # ------------------------------------------------------------- facts

    @property
    def experiment(self) -> str:
        """The experiment announced in ``ScanStart``."""
        return self._experiment

    @property
    def enabled(self) -> bool:
        """Whether the experiment has a multicast endpoint to announce on."""
        return self._endpoint is not None

    @property
    def endpoint(self) -> tuple[str, int] | None:
        """``(group, port)``, or ``None`` when disabled."""
        return self._endpoint

    @property
    def device_endpoints(self) -> dict[str, tuple[str, int]]:
        """Device → ``(ip, command port)`` for the experiment's devices."""
        return dict(self._device_endpoints)

    @property
    def entered(self) -> bool:
        """Whether a run is in scan mode right now (entered, not yet exited)."""
        return self._server is not None

    @property
    def run_devices(self) -> tuple[str, ...]:
        """The devices of the run in scan mode (empty between runs)."""
        return self._run_devices

    @property
    def shot_number(self) -> int:
        """The last shot number announced this run (``0`` before the first)."""
        return self._shot

    @property
    def last_path(self) -> str | None:
        """The last path pushed, ``None`` before the first shot."""
        return self._last_path

    @property
    def local_ip(self) -> str | None:
        """The address bound and announced for the current run."""
        return self._local_ip

    @property
    def state_file(self) -> Path:
        """Where the run is recorded between entry and exit."""
        return self._state_file

    @property
    def device_path(self) -> Callable[[str], str]:
        """The local-to-device-side folder mapping the shot paths use."""
        return self._device_path

    def resolve_local_ip(self, device_host: str) -> str:
        """The address to bind and announce: the fixed one, else the gateway's rule.

        ``detect_local_ip`` against *device_host* is the interface the
        gateway's own UDP sets to that device leave from — the address a
        device in scan mode will take commands from.
        """
        if self._fixed_local_ip:
            return self._fixed_local_ip
        local_ip = detect_local_ip(device_host)
        if not local_ip:
            raise GeecsConfigurationError(
                f"scan mode: no local interface routes to the device host "
                f"{device_host} — nothing to announce"
            )
        return local_ip

    def announce_address(self, local_ip: str) -> str:
        """The string named in ``ScanStart``: ``client_ip`` when given, else *local_ip*."""
        return self._client_ip or local_ip

    @staticmethod
    def check_announce_address(announced: str, local_ip: str) -> None:
        """Refuse an announced address the devices would not resolve to this host.

        A device locks its command port to the announced address and
        connects to it; one that resolves elsewhere locks every device to
        a host the fire never comes from and leaves them connected nowhere.
        """
        try:
            resolved = {
                info[4][0]
                for info in socket.getaddrinfo(announced, None, socket.AF_INET)
            }
        except OSError as exc:
            raise GeecsConfigurationError(
                f"scan mode: the announced address {announced!r} does not "
                f"resolve ({exc}) — the devices could not connect back to it"
            ) from exc
        if local_ip not in resolved:
            raise GeecsConfigurationError(
                f"scan mode: the announced address {announced!r} resolves to "
                f"{', '.join(sorted(resolved)) or 'nothing'}, not to this "
                f"worker's {local_ip} — a device locked to it would refuse "
                "every command from here (config.ini [scan_mode] client_ip)"
            )

    # ------------------------------------------------- the Bluesky protocols

    def set(self, value: Any) -> AsyncStatus:
        """``EnterScanMode(...)`` enters, :data:`ANNOUNCE` announces a shot, :data:`EXIT` ends."""
        return AsyncStatus(self._apply(value))

    async def _apply(self, value: Any) -> None:
        if isinstance(value, EnterScanMode):
            await self._enter(value)
        elif value == ANNOUNCE:
            await self._announce()
        elif value == EXIT:
            await self._exit()
        else:
            raise GeecsConfigurationError(
                f"scan mode: {value!r} is not a command (EnterScanMode(...), "
                f"{ANNOUNCE!r} or {EXIT!r})"
            )

    def read(self) -> dict[str, Any]:
        """The announced shot number of the row's shot."""
        return {ANNOUNCED_SHOT: {"value": self._shot, "timestamp": time.time()}}

    def describe(self) -> dict[str, Any]:
        """The data key: a scalar integer sourced from the plan."""
        return {
            ANNOUNCED_SHOT: {
                "source": "plan:scan_mode",
                "dtype": "integer",
                "shape": [],
            }
        }

    # ------------------------------------------------------------- entry

    async def _enter(self, entry: EnterScanMode) -> None:
        if self._endpoint is None:
            raise GeecsConfigurationError(
                f"scan mode: experiment {self._experiment!r} has no multicast "
                "endpoint in the GEECS DB (expt.MultiCastIP / "
                "MultiCastRemotePort) — scan mode cannot be driven; drop "
                "use_scan_mode from the trigger profile or fill the row"
            )
        devices = tuple(dict.fromkeys(entry.devices))
        if not devices:
            raise GeecsConfigurationError(
                "scan mode: the run names no device to announce to"
            )
        unknown = [d for d in devices if d not in self._device_endpoints]
        if unknown:
            raise GeecsConfigurationError(
                f"scan mode: {', '.join(unknown)} not in the experiment's "
                f"device roster — a run device must be an enabled device of "
                f"{self._experiment!r}"
            )
        if self._server is not None:
            logger.warning(
                "scan mode: the previous run (%d device(s): %s) never ended "
                "scan mode — a halt skips the finalizers; ending it before "
                "this run",
                len(self._run_devices),
                ", ".join(self._run_devices),
            )
            await self._exit()
        hosts = {self._device_endpoints[d][0] for d in devices}
        ports = {self._device_endpoints[d][1] for d in devices}
        local_ip = self.resolve_local_ip(self._device_endpoints[devices[0]][0])
        announced = self.announce_address(local_ip)
        self.check_announce_address(announced, local_ip)
        group, port = self._endpoint
        server = ScanModeServer(local_ip, self._tcp_port)
        listener = ScanModeAckListener(local_ip, ports)
        sender = ScanModeMulticaster(
            group, port, local_ip=local_ip, client_ip=announced
        )
        try:
            await server.start()
            await listener.open()
            await sender.open()
        except BaseException:
            await self._release(server, listener, sender)
            raise
        self._server, self._listener, self._sender = server, listener, sender
        self._run_devices = devices
        self._scan_parameter = entry.scan_parameter
        self._local_ip, self._announced_as = local_ip, announced
        self._shot = 0
        self._last_path = None
        self._write_state()
        started = time.monotonic()
        try:
            acks = await self._scan_start(devices)
            await self._wait_connected(devices, hosts)
        except BaseException:
            # Ends it for the devices that did enter; the server's close
            # ends it for any that connected without acknowledging.
            await self._exit()
            raise
        lags = sorted(ack.received_at for ack in acks.values())
        logger.info(
            "scan mode: %d device(s) on %d host(s) entered scan mode for %r "
            "as %s (%d connection(s) in %.1f s); draining %.2f s before the "
            "first shot",
            len(devices),
            len(hosts),
            self._experiment,
            announced,
            server.connection_count,
            time.monotonic() - started,
            self._drain_s,
        )
        del lags
        if self._drain_s > 0:
            await asyncio.sleep(self._drain_s)

    async def _scan_start(self, devices: tuple[str, ...]) -> dict[str, ScanModeAck]:
        assert self._sender is not None and self._listener is not None
        loop = asyncio.get_running_loop()
        since = loop.time()  # before the announce: a repeat of an old ack never counts
        attempts = 1 + self._retries
        acks: dict[str, ScanModeAck] = {}
        silent: list[str] = list(devices)
        for attempt in range(1, attempts + 1):
            await self._sender.announce_scan_start(
                self._experiment, self._scan_parameter
            )
            acks = await self._listener.wait_for(
                SCAN_START, devices, self._ack_timeout_s, since=since
            )
            silent = [d for d in devices if d not in acks]
            if not silent:
                break
            if attempt < attempts:
                logger.warning(
                    "scan mode: %d of %d device(s) silent %.1f s after ScanStart "
                    "(%s) — sending ScanStart once more",
                    len(silent),
                    len(devices),
                    self._ack_timeout_s,
                    ", ".join(silent),
                )
        errors = {d: ack.status for d, ack in acks.items() if not ack.ok}
        if silent or errors:
            parts: list[str] = []
            if silent:
                parts.append(
                    f"no acknowledgement from {', '.join(silent)} after "
                    f"{attempts} ScanStart(s)"
                )
            if errors:
                parts.append(
                    "refused by "
                    + ", ".join(f"{d} ({status})" for d, status in errors.items())
                )
            raise GeecsScanModeError(
                "scan mode: "
                + "; ".join(parts)
                + " — the run was refused before any shot (a device that did "
                "not process ScanStart never enters scan mode and its start "
                "values stay unapplied). Check the device, then resubmit.",
                devices=[*silent, *errors],
            )
        return acks

    async def _wait_connected(
        self, devices: tuple[str, ...], hosts: Collection[str]
    ) -> None:
        assert self._server is not None
        missing = await self._server.wait_for_hosts(hosts, self._connect_timeout_s)
        if missing:
            names = [d for d in devices if self._device_endpoints[d][0] in missing]
            raise GeecsScanModeError(
                f"scan mode: no path-channel connection from "
                f"{', '.join(sorted(missing))} ({', '.join(names)}) within "
                f"{self._connect_timeout_s:.1f} s of ScanStart — the devices "
                f"acknowledged but never connected to "
                f"{self._local_ip}:{self._tcp_port}; check that the port is "
                "reachable from the device hosts (firewall)",
                devices=names,
            )

    # ---------------------------------------------------------- announce

    async def _announce(self) -> None:
        if self._server is None:
            raise GeecsConfigurationError(
                "scan mode: a shot was announced before ScanStart — the run "
                "bracket enters scan mode first (plans/scan_mode.py)"
            )
        folder = getattr(self._path_provider, "folder", None)
        if folder is None:
            raise GeecsConfigurationError(
                "scan mode: no scan claimed — the shot path needs the run "
                "folder (is the claim_scan preprocessor installed?)"
            )
        number = scan_number_of(folder)
        self._shot += 1
        path = device_shot_path(self._device_path(str(folder)), number, self._shot)
        count = await self._server.push_path(path)
        self._last_path = path
        if count == 0:
            logger.warning(
                "scan mode: shot %d announced to no device — every path-channel "
                "connection is gone; the shot will have no file",
                self._shot,
            )
        else:
            logger.debug(
                "scan mode: shot %d announced on %d connection(s): %s",
                self._shot,
                count,
                path,
            )

    # -------------------------------------------------------------- exit

    async def _exit(self) -> None:
        server, listener, sender = self._server, self._listener, self._sender
        if server is None or listener is None or sender is None:
            logger.info("scan mode: nothing to release — no run is in scan mode")
            return
        devices = self._run_devices
        try:
            loop = asyncio.get_running_loop()
            since = loop.time()
            await sender.announce_scan_end(self._experiment, self._scan_parameter)
            await asyncio.sleep(EXIT_FRAME_DELAY_S)
            pushed = await server.push_scan_end(
                self._experiment, self._scan_parameter, self._announced_as or ""
            )
            acks = (
                await listener.wait_for(
                    SCAN_END, devices, self._end_timeout_s, since=since
                )
                if devices
                else {}
            )
            silent = [d for d in devices if d not in acks]
            deadline = loop.time() + self._end_timeout_s
            while server.connection_count and loop.time() < deadline:
                await asyncio.sleep(0.02)
            still_open = server.connection_count
            if silent or still_open:
                logger.warning(
                    "scan mode: ScanEnd pushed on %d connection(s); %d of %d "
                    "device(s) did not acknowledge it (%s) and %d connection(s) "
                    "stayed open — closing the channel; a device still holding "
                    "it ends the scan by itself",
                    pushed,
                    len(silent),
                    len(devices),
                    ", ".join(silent) or "-",
                    still_open,
                )
            else:
                lags = [(ack.received_at - since) * 1e3 for ack in acks.values()] or [
                    0.0
                ]
                logger.info(
                    "scan mode: %d device(s) left scan mode (ScanEnd frame on %d "
                    "connection(s), acknowledged within %.0f ms)",
                    len(devices),
                    pushed,
                    max(lags),
                )
        finally:
            await self._release(server, listener, sender)
            self._server = self._listener = self._sender = None
            self._run_devices = ()
            self._clear_state()

    async def _release(
        self,
        server: ScanModeServer | None,
        listener: ScanModeAckListener | None,
        sender: ScanModeMulticaster | None,
    ) -> None:
        for piece, close in (
            (server, server.stop if server else None),
            (listener, listener.close if listener else None),
            (sender, sender.close if sender else None),
        ):
            if close is None:
                continue
            try:
                await close()
            except Exception:  # releasing never raises over a peer
                logger.exception("scan mode: %s did not close cleanly", piece)

    # --------------------------------------------------------- the record

    def release_stale(self) -> bool:
        """Clear a record a dead worker left; ``True`` when there was one.

        Called at environment open.  The connections of the worker that
        wrote the record died with it, and a device ends the scan by itself
        when its path channel closes (seen live on both experiments), so
        nothing is sent: the fact is logged and the record removed.
        """
        try:
            text = self._state_file.read_text()
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("scan mode: cannot read %s: %s", self._state_file, exc)
            return False
        try:
            record = json.loads(text)
        except ValueError:
            record = {}
        devices = record.get("devices") or []
        logger.warning(
            "scan mode: the previous worker left %r in scan mode (entered %s, "
            "%d device(s): %s) and never sent ScanEnd — its path-channel "
            "connections died with it, and a device ends the scan by itself "
            "when the channel closes; clearing the record. A device still "
            "holding its start values needs a look.",
            record.get("experiment", self._experiment),
            record.get("entered_at", "?"),
            len(devices),
            ", ".join(devices) or "-",
        )
        self._clear_state()
        return True

    def _write_state(self) -> None:
        record = {
            "experiment": self._experiment,
            "scan_parameter": self._scan_parameter,
            "entered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "local_ip": self._local_ip,
            "announced_as": self._announced_as,
            "devices": list(self._run_devices),
        }
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(record, indent=1))
        except OSError as exc:
            logger.warning(
                "scan mode: cannot record the run in %s (%s) — a crash would "
                "leave no trace of the devices in scan mode",
                self._state_file,
                exc,
            )

    def _clear_state(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._state_file.unlink()


__all__ = [
    "ANNOUNCE",
    "ANNOUNCED_SHOT",
    "DEFAULT_DRAIN_S",
    "ENV_STATE_FILE",
    "EXIT",
    "EnterScanMode",
    "ScanModeControl",
    "default_state_file",
    "scan_number_of",
]
