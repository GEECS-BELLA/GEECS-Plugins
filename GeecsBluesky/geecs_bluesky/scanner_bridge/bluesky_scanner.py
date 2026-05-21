"""BlueskyScanner — drop-in RunEngine backend for GEECS Scanner GUI.

Provides the same interface as ``ScanManager`` so that ``RunControl`` can
route scan requests to either the legacy ScanManager or this Bluesky backend,
controlled by a single flag.

Supported scan modes (MVP):
    STANDARD — step scan via :func:`~geecs_bluesky.plans.step_scan.geecs_step_scan`
    NOSCAN   — fixed-position data collection via ``bluesky.plans.count``

The RunEngine is created once on ``__init__`` and its internal event loop
persists for the lifetime of this object.  Devices are created from the GEECS
database and connected in the RE's loop at the start of each scan, then
disconnected when the scan finishes or is aborted.

Usage (standalone)::

    from geecs_bluesky.scanner_bridge import BlueskyScanner
    from geecs_scanner.engine.models.scan_execution_config import ScanExecutionConfig
    from geecs_data_utils import ScanConfig, ScanMode

    scanner = BlueskyScanner()
    exec_config = ScanExecutionConfig(scan_config=ScanConfig(
        scan_mode=ScanMode.STANDARD,
        device_var="U_ESP_JetXYZ:Position.Axis 1",
        start=4.0, end=6.0, step=0.5, wait_time=5.0,
    ))
    scanner.reinitialize(exec_config=exec_config)
    scanner.start_scan_thread()
    while scanner.is_scanning_active():
        time.sleep(0.5)
        print(f"{scanner.estimate_current_completion()*100:.0f}%")
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
from typing import Any

import numpy as np
from bluesky import RunEngine
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from ophyd_async.core import AsyncStatus

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.transport.udp_client import GeecsUdpClient
from geecs_bluesky.utils import safe_name

# ScanConfig / ScanMode are only imported for type hints; duck-typing is used at
# runtime so this module can be imported without geecs_data_utils installed.
try:
    from geecs_data_utils import ScanConfig as _ScanConfig  # noqa: F401
except Exception:
    _ScanConfig = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 20.0
_DISCONNECT_TIMEOUT = 10.0


class _UdpSetter:
    """Minimal Bluesky Movable: sets one GEECS variable via a shared UDP client.

    Values are sent as strings, matching the GEECS wire protocol and the
    shot-control YAML format (which may contain numeric strings or words
    like ``"on"``/``"off"``).
    """

    def __init__(self, udp: GeecsUdpClient, variable: str) -> None:
        self._udp = udp
        self._variable = variable

    def set(self, value: Any) -> AsyncStatus:
        """Send *value* to the device; resolves when the UDP ACK is received."""

        async def _do() -> None:
            await self._udp.set(self._variable, str(value))

        return AsyncStatus(_do())


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------


def _save_cleanup_plan(saving_detectors: list[tuple]):
    """Turn saving off for all saving detectors (runs even on abort)."""
    if not saving_detectors:
        return
    mv_args: list = []
    for det, _path in saving_detectors:
        mv_args.extend([det.save, "off"])
        logger.debug("Saving disabled for %s", det.name)
    yield from bps.mv(*mv_args)


def _scan_with_saving(inner_plan, saving_detectors: list[tuple]):
    """Wrap *inner_plan* with per-detector save enable/disable.

    For each detector in *saving_detectors* (list of ``(det, path)`` tuples):
    - Creates the save directory
    - Sets ``localsavingpath`` and ``save = "on"`` before the scan
    - Sets ``save = "off"`` in a finalise wrapper (runs even on abort)
    """
    if saving_detectors:
        mv_args: list = []
        for det, path in saving_detectors:
            os.makedirs(path, exist_ok=True)
            logger.info("Save path for %s: %s", det.name, path)
            mv_args.extend([det.localsavingpath, path, det.save, "on"])
        # Single bps.mv fans out via asyncio.gather — all devices set concurrently
        yield from bps.mv(*mv_args)

    yield from bpp.finalize_wrapper(
        inner_plan,
        _save_cleanup_plan(saving_detectors),
    )


def _build_positions(scan_config: Any) -> list[float]:
    """Convert ScanConfig start/end/step to an explicit list of positions."""
    start = float(scan_config.start)
    end = float(scan_config.end)
    step = float(scan_config.step)
    if step == 0 or start == end:
        return [start]
    n = max(2, round(abs(end - start) / abs(step)) + 1)
    return list(np.linspace(start, end, n))


class BlueskyScanner:
    """RunEngine-backed scan executor compatible with the ``ScanManager`` API.

    Parameters
    ----------
    experiment_dir:
        Experiment name (currently unused; reserved for future path integration).
    shot_control_information:
        Shot-controller YAML config dict with keys ``"device"`` (GEECS device
        name) and ``"variables"`` (mapping of variable name → state → value,
        where states are ``"OFF"``, ``"SCAN"``, ``"STANDBY"``,
        ``"SINGLESHOT"``).  When provided, the DG645 is armed before each
        acquisition step and disarmed after.
    tiled_uri:
        URI of the Tiled catalog server (e.g. ``"http://192.168.6.14:8000"``).
        When provided, a :class:`~bluesky.callbacks.tiled_writer.TiledWriter`
        is subscribed to the RunEngine so every scan is persisted automatically.
        Requires ``tiled[client]`` to be installed; silently skipped if absent.
    tiled_api_key:
        API key for the Tiled server, if authentication is enabled.
    """

    def __init__(
        self,
        experiment_dir: str = "",
        shot_control_information: dict | None = None,
        tiled_uri: str | None = None,
        tiled_api_key: str | None = None,
    ) -> None:
        self._experiment_dir = experiment_dir
        # context_managers=[] disables SIGINT handling, which fails when the
        # RunEngine is called from a background thread (not the main thread).
        self._RE = RunEngine(context_managers=[])
        self._RE.subscribe(self._on_document)

        # GUI compatibility shims
        self.dialog_queue: queue.Queue = (
            queue.Queue()
        )  # always empty; RE uses bps.pause()
        self.restore_failures: list = []  # no legacy device restore needed

        self._tiled_token: int | None = None
        if tiled_uri is None:
            tiled_uri, tiled_api_key = self._read_tiled_config()
        if tiled_uri:
            self._subscribe_tiled(tiled_uri, tiled_api_key)

        self._scan_thread: threading.Thread | None = None
        self._scan_config: Any = None
        self._shots_per_step: int = 10
        self._devices_config: dict[str, Any] = {}

        self._total_shots: int = 0
        self._completed_shots: int = 0

        # Devices held open between reinitialize() and scan completion
        self._motor: GeecsMotor | None = None
        self._detectors: list = []
        # Subset of detectors that save per-shot files: list of (detector, path)
        self._saving_detectors: list[tuple] = []

        # Shot control — parsed from shot_control_information YAML
        self._shot_control_device_name: str | None = None
        self._shot_control_variables: dict[str, dict] = {}
        self._shot_control_udp: GeecsUdpClient | None = None
        self._shot_control_setters: dict[str, _UdpSetter] = {}
        if shot_control_information:
            self._shot_control_device_name = shot_control_information.get("device")
            self._shot_control_variables = shot_control_information.get("variables", {})

        # Lock for serialising _motor create/destroy across threads
        self._device_lock = threading.Lock()

        logger.info("BlueskyScanner initialised (RunEngine ready)")

    # ------------------------------------------------------------------
    # ScanManager-compatible public API
    # ------------------------------------------------------------------

    def reinitialize(self, exec_config: Any, **_kwargs: Any) -> bool:
        """Store the scan configuration for the next run.

        Parameters
        ----------
        exec_config : ScanExecutionConfig
            Validated scan configuration produced by the GUI.  Provides device
            list, scan parameters, and execution options.

        Returns
        -------
        bool
            Always ``True`` (no hardware initialisation done here).
        """
        self._scan_config = exec_config.scan_config

        # Derive shots from rep_rate × wait_time (ScanOptions has no shots_per_step field)
        rep_rate = getattr(exec_config.options, "rep_rate_hz", 1.0)
        wait_time = getattr(self._scan_config, "wait_time", 10.0)
        self._shots_per_step = max(1, round(rep_rate * wait_time))

        # Build a plain-dict device map compatible with _build_detectors
        save_config = exec_config.save_config
        devices_pydantic = getattr(save_config, "Devices", None) or {}
        self._devices_config = {
            name: (dev.model_dump() if hasattr(dev, "model_dump") else dict(dev))
            for name, dev in devices_pydantic.items()
        }

        self._completed_shots = 0
        self._total_shots = 0
        self._saving_detectors = []

        # Disconnect any leftover devices from a previous scan
        self._disconnect_devices_sync()

        logger.info(
            "BlueskyScanner reinitialised — shots_per_step=%d, devices=%s",
            self._shots_per_step,
            list(self._devices_config),
        )
        return True

    def start_scan_thread(self) -> None:
        """Launch the scan stored by :meth:`reinitialize` in a background thread."""
        if self.is_scanning_active():
            logger.warning(
                "start_scan_thread called while scan already active; ignoring"
            )
            return

        self._completed_shots = 0
        self._scan_thread = threading.Thread(
            target=self._run_scan,
            args=(self._scan_config,),
            daemon=True,
            name="bluesky-scan",
        )
        self._scan_thread.start()
        logger.info("BlueskyScanner scan thread started")

    def stop_scanning_thread(self) -> None:
        """Abort the running scan and wait for the thread to finish."""
        logger.info("BlueskyScanner: abort requested")
        try:
            self._RE.abort(reason="stop_scanning_thread called")
        except Exception:
            logger.debug("RE.abort() raised (may not be running)", exc_info=True)
        if self._scan_thread is not None:
            self._scan_thread.join(timeout=15)
            self._scan_thread = None

    def pause_scan(self) -> None:
        """Request the RunEngine to pause between plans steps."""
        logger.info("BlueskyScanner: pause requested")
        try:
            self._RE.request_pause()
        except Exception:
            logger.debug("RE.request_pause() raised", exc_info=True)

    def resume_scan(self) -> None:
        """Resume a paused RunEngine."""
        logger.info("BlueskyScanner: resume requested")
        try:
            self._RE.resume()
        except Exception:
            logger.debug("RE.resume() raised", exc_info=True)

    def is_scanning_active(self) -> bool:
        """Return ``True`` if a scan thread is currently running."""
        return bool(self._scan_thread and self._scan_thread.is_alive())

    def estimate_current_completion(self) -> float:
        """Return fraction complete (0.0–1.0) based on shots emitted."""
        if self._total_shots == 0:
            return 0.0
        return min(self._completed_shots / self._total_shots, 1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_tiled_config() -> tuple[str | None, str | None]:
        """Read Tiled URI and API key from ~/.config/geecs_python_api/config.ini.

        Returns ``(uri, api_key)``, either of which may be ``None`` if absent.
        """
        import configparser
        from pathlib import Path

        config_path = Path.home() / ".config" / "geecs_python_api" / "config.ini"
        if not config_path.exists():
            return None, None

        cfg = configparser.ConfigParser()
        cfg.read(config_path)

        if "tiled" not in cfg:
            return None, None

        uri = cfg["tiled"].get("uri") or None
        api_key = cfg["tiled"].get("api_key") or None
        logger.debug("Tiled config loaded from %s — uri=%s", config_path, uri)
        return uri, api_key

    def _subscribe_tiled(self, tiled_uri: str, api_key: str | None = None) -> None:
        """Subscribe a TiledWriter to the RunEngine.

        Silently skips if ``tiled[client]`` is not installed or the server is
        unreachable, so the scanner remains functional without Tiled.
        """
        try:
            from bluesky.callbacks.tiled_writer import TiledWriter
            from tiled.client import from_uri
        except ImportError:
            logger.warning(
                "tiled not installed — Tiled storage disabled. "
                "Enable with: pip install 'tiled[client]'"
            )
            return

        try:
            client = from_uri(tiled_uri, api_key=api_key)
            writer = TiledWriter(client)
            self._tiled_token = self._RE.subscribe(writer)
            logger.info("TiledWriter subscribed — catalog at %s", tiled_uri)
        except Exception:
            logger.warning(
                "Could not connect TiledWriter to %s — Tiled storage disabled",
                tiled_uri,
                exc_info=True,
            )

    def _on_document(self, name: str, doc: dict) -> None:
        if name == "event":
            self._completed_shots += 1

    def _connect_device(self, device) -> None:
        """Connect an ophyd-async device in the RE's persistent event loop."""
        asyncio.run_coroutine_threadsafe(device.connect(), self._RE._loop).result(
            timeout=_CONNECT_TIMEOUT
        )

    def _disconnect_device(self, device) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                device.disconnect(), self._RE._loop
            ).result(timeout=_DISCONNECT_TIMEOUT)
        except Exception:
            logger.debug("disconnect raised for %s", device, exc_info=True)

    def _disconnect_devices_sync(self) -> None:
        """Disconnect motor, detectors, and shot controller (called from non-scan threads)."""
        with self._device_lock:
            if self._motor is not None:
                self._disconnect_device(self._motor)
                self._motor = None
            for det in self._detectors:
                self._disconnect_device(det)
            self._detectors = []
            self._saving_detectors = []

        if self._shot_control_udp is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._shot_control_udp.close(), self._RE._loop
                ).result(timeout=_DISCONNECT_TIMEOUT)
            except Exception:
                logger.debug("Shot control UDP close raised", exc_info=True)
            self._shot_control_udp = None
            self._shot_control_setters = {}

    def _build_shot_controller(self) -> None:
        """Resolve the shot control device and create one _UdpSetter per variable.

        Silently skips if no shot control device is configured or the DB
        lookup fails, so scans without trigger control still run (e.g. in
        internal-trigger test mode).
        """
        if not self._shot_control_device_name or not self._shot_control_variables:
            logger.debug("No shot control device configured — trigger control disabled")
            return
        try:
            from geecs_bluesky.db.geecs_db import GeecsDb

            host, port = GeecsDb.find_device(self._shot_control_device_name)
            udp = GeecsUdpClient(host, port, device_name=self._shot_control_device_name)
            asyncio.run_coroutine_threadsafe(udp.connect(), self._RE._loop).result(
                timeout=_CONNECT_TIMEOUT
            )
            self._shot_control_udp = udp
            self._shot_control_setters = {
                var: _UdpSetter(udp, var) for var in self._shot_control_variables
            }
            logger.info(
                "Shot controller ready: %s (%d variables)",
                self._shot_control_device_name,
                len(self._shot_control_setters),
            )
        except Exception:
            logger.warning(
                "Could not build shot controller — trigger control disabled",
                exc_info=True,
            )

    def _arm_trigger(self):
        """Bluesky plan stub: set all shot control variables to SCAN state."""
        yield from self._set_trigger_state("SCAN")

    def _disarm_trigger(self):
        """Bluesky plan stub: set all shot control variables to STANDBY state."""
        yield from self._set_trigger_state("STANDBY")

    def _set_trigger_state(self, state: str):
        """Bluesky plan stub: drive all shot control variables to *state*.

        Uses ``bps.abs_set`` + ``bps.wait`` rather than ``bps.mv`` because
        ``bps.mv`` inspects ``.parent`` for coupled-device handling — an
        ophyd-specific attribute that ``_UdpSetter`` intentionally omits.
        All active variables are set concurrently then waited on as a group.
        """
        if not self._shot_control_setters:
            return
        group = f"shot_ctrl_{state}"
        n_set = 0
        for var_name, setter in self._shot_control_setters.items():
            val = self._shot_control_variables[var_name].get(state)
            if val:  # skip None and empty string (matches TriggerController)
                yield from bps.abs_set(setter, val, group=group)
                n_set += 1
        if n_set:
            yield from bps.wait(group)
            logger.info("Shot controller → %s", state)

    def _run_scan(self, scan_config: Any) -> None:
        """Scan thread body: create devices, run plan, clean up."""
        try:
            mode = scan_config.scan_mode
            # Support both ScanMode enum and plain strings
            mode_val = mode.value if hasattr(mode, "value") else str(mode).lower()
            logger.info("BlueskyScanner: starting %s scan", mode_val)

            if mode_val == "standard":
                self._run_standard_scan(scan_config)
            elif mode_val == "noscan":
                self._run_noscan(scan_config)
            else:
                logger.warning(
                    "BlueskyScanner: scan mode %r not yet supported; skipping", mode_val
                )
        except Exception:
            logger.exception("BlueskyScanner scan thread raised an exception")
        finally:
            self._disconnect_devices_sync()
            logger.info("BlueskyScanner: scan thread finished")

    def _claim_scan_number(self) -> tuple[int | None, str | None]:
        """Claim the next scan number from the filesystem via ScanPaths.

        Returns ``(scan_number, scan_folder_str)`` on success, or
        ``(None, None)`` if ``geecs_data_utils`` is not installed, the NetApp
        is unreachable, or the call fails for any other reason.
        """
        try:
            from geecs_data_utils import ScanPaths
        except Exception:
            logger.debug("geecs_data_utils not available; scan numbering disabled")
            return None, None

        try:
            if ScanPaths.paths_config is None:
                ScanPaths.reload_paths_config(
                    default_experiment=self._experiment_dir or None
                )
            tag = ScanPaths.get_next_scan_tag(experiment=self._experiment_dir or None)
            scan_data = ScanPaths(tag=tag, read_mode=False)
            folder = scan_data.get_folder()
            logger.info("Claimed scan number %d → %s", tag.number, folder)
            return tag.number, str(folder) if folder else None
        except Exception:
            logger.warning("Could not claim scan number", exc_info=True)
            return None, None

    def _build_detectors(self, scan_folder: str | None = None) -> None:
        """Create and connect detector devices from ``self._devices_config``.

        Populates ``self._detectors`` (and ``self._saving_detectors`` for
        devices with ``save_nonscalar_data=True``) in place.  Devices that
        fail to resolve or connect are logged and skipped.
        """
        for device_name, dev_cfg in self._devices_config.items():
            if isinstance(dev_cfg, dict):
                variable_list = dev_cfg.get("variable_list", [])
                save_nonscalar = bool(dev_cfg.get("save_nonscalar_data", False))
            else:
                variable_list = getattr(dev_cfg, "variable_list", [])
                save_nonscalar = bool(getattr(dev_cfg, "save_nonscalar_data", False))

            if not variable_list:
                logger.debug("Skipping %s: empty variable_list", device_name)
                continue
            ophyd_name = safe_name(device_name)
            try:
                det = GeecsGenericDetector.from_db(
                    device_name,
                    list(variable_list),
                    name=ophyd_name,
                    save_nonscalar_data=save_nonscalar,
                )
                self._connect_device(det)
                with self._device_lock:
                    self._detectors.append(det)
                    if save_nonscalar and scan_folder is not None:
                        save_path = os.path.join(scan_folder, device_name)
                        self._saving_detectors.append((det, save_path))
                logger.info(
                    "Detector ready: %s (%d variables, save_nonscalar=%s)",
                    device_name,
                    len(variable_list),
                    save_nonscalar,
                )
            except Exception:
                logger.warning(
                    "Failed to create/connect detector %s — skipping",
                    device_name,
                    exc_info=True,
                )

    def _run_standard_scan(self, scan_config: Any) -> None:
        """Step scan: move motor through positions, collect shots at each step."""
        if not scan_config.device_var:
            logger.error("STANDARD scan requires device_var; got None")
            return

        if ":" not in scan_config.device_var:
            logger.error(
                "device_var must be 'DeviceName:VariableName', got: %r",
                scan_config.device_var,
            )
            return

        device_name, variable = scan_config.device_var.split(":", 1)
        ophyd_name = safe_name(f"{device_name}_{variable}")

        logger.info(
            "Creating motor: %s / %s → name=%r", device_name, variable, ophyd_name
        )
        motor = GeecsMotor.from_db_axis(device_name, variable, name=ophyd_name)

        logger.info("Connecting motor …")
        self._connect_device(motor)
        with self._device_lock:
            self._motor = motor

        # Claim scan number before building detectors so save paths are known
        scan_number, scan_folder = self._claim_scan_number()

        self._build_detectors(scan_folder=scan_folder)

        positions = _build_positions(scan_config)
        n_steps = len(positions)
        n_shots = n_steps * self._shots_per_step
        self._total_shots = n_shots

        logger.info(
            "Step scan: %d steps × %d shots/step = %d total events",
            n_steps,
            self._shots_per_step,
            n_shots,
        )
        logger.info("Positions: %s", positions)

        md: dict[str, Any] = {
            "operator": "",
            "scan_mode": getattr(
                scan_config.scan_mode, "value", str(scan_config.scan_mode)
            ),
            "device_var": scan_config.device_var,
            "wait_time": scan_config.wait_time,
            "bluesky_backend": True,
        }
        if scan_number is not None:
            md["scan_number"] = scan_number
        if scan_folder is not None:
            md["scan_folder"] = scan_folder
        if scan_config.additional_description:
            md["description"] = scan_config.additional_description

        self._build_shot_controller()
        arm = self._arm_trigger if self._shot_control_setters else None
        disarm = self._disarm_trigger if self._shot_control_setters else None

        plan = _scan_with_saving(
            geecs_step_scan(
                motor=motor,
                positions=positions,
                detectors=list(self._detectors),
                shots_per_step=self._shots_per_step,
                arm_trigger=arm,
                disarm_trigger=disarm,
                md=md,
            ),
            saving_detectors=list(self._saving_detectors),
        )
        # Outer finalize ensures disarm even on mid-step abort
        if disarm is not None:
            plan = bpp.finalize_wrapper(plan, self._disarm_trigger())
        self._RE(plan)

    def _run_noscan(self, scan_config: Any) -> None:
        """Fixed-position data collection — count plan if detectors present."""
        scan_number, scan_folder = self._claim_scan_number()

        self._build_detectors(scan_folder=scan_folder)

        n_shots = self._shots_per_step
        self._total_shots = n_shots

        if not self._detectors:
            logger.info(
                "NOSCAN with no detectors: nothing to collect (%d shots requested). "
                "Add detector devices via the device configuration to enable data collection.",
                n_shots,
            )
            return

        logger.info(
            "NOSCAN: collecting %d shots from %d detectors",
            n_shots,
            len(self._detectors),
        )

        md: dict[str, Any] = {
            "scan_mode": getattr(
                scan_config.scan_mode, "value", str(scan_config.scan_mode)
            ),
            "bluesky_backend": True,
        }
        if scan_number is not None:
            md["scan_number"] = scan_number
        if scan_folder is not None:
            md["scan_folder"] = scan_folder
        self._build_shot_controller()
        arm = self._arm_trigger if self._shot_control_setters else None
        disarm = self._disarm_trigger if self._shot_control_setters else None

        @bpp.run_decorator(md=md)
        def _noscan_plan():
            if arm is not None:
                yield from arm()
            for _ in range(n_shots):
                yield from bps.trigger_and_read(self._detectors)
            if disarm is not None:
                yield from disarm()

        plan = _scan_with_saving(
            _noscan_plan(),
            saving_detectors=list(self._saving_detectors),
        )
        if disarm is not None:
            plan = bpp.finalize_wrapper(plan, self._disarm_trigger())
        self._RE(plan)
