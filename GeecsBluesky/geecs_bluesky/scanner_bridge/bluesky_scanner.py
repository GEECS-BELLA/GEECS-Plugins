"""BlueskyScanner — drop-in RunEngine backend for GEECS Scanner GUI.

Provides the same interface as ``ScanManager`` so that ``RunControl`` can
route scan requests to either the legacy ScanManager or this Bluesky backend,
controlled by a single flag.

Supported scan modes (MVP):
    STANDARD — step scan, dispatched by ``acquisition_mode``:
        ``strict_shot_control`` → :func:`~geecs_bluesky.plans.step_scan.geecs_step_scan`
        ``free_run_time_sync``  → :func:`~geecs_bluesky.plans.free_run_step_scan.geecs_free_run_step_scan`
    NOSCAN   — statistics collection: the same step-scan plan with no scan
               variable moved (``motor=None``, one no-move bin), so it honours
               the same ``acquisition_mode`` dispatch

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
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
from bluesky import RunEngine


from geecs_bluesky.assets import AssetDefinition, get_asset_definitions
from geecs_bluesky.data_paths import (
    asset_resource_root_paths,
    device_server_save_path,
)
from geecs_bluesky.db.geecs_db import GeecsDb
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.plans.orchestration import build_step_scan_plan
from geecs_bluesky.plans.run_wrapper import claim_scan_number
from geecs_bluesky.shot_controller import ShotController
from geecs_bluesky.tiled_integration import subscribe_tiled
from geecs_bluesky.utils import safe_name

# ScanConfig / ScanMode are only imported for type hints; duck-typing is used at
# runtime so this module can be imported without geecs_data_utils installed.
try:
    from geecs_data_utils import ScanConfig as _ScanConfig  # noqa: F401
except Exception:
    _ScanConfig = None  # type: ignore[assignment,misc]

try:
    from geecs_scanner.engine.scan_events import (
        ScanEvent,
        ScanLifecycleEvent,
        ScanState,
    )
except Exception:
    ScanEvent = Any  # type: ignore[misc,assignment]
    ScanLifecycleEvent = None  # type: ignore[assignment]
    ScanState = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 20.0
_DISCONNECT_TIMEOUT = 10.0

_STRICT_MODE = "strict_shot_control"
_FREE_RUN_MODE = "free_run_time_sync"
_VALID_ACQUISITION_MODES = (_STRICT_MODE, _FREE_RUN_MODE)


def _cfg_field(cfg: Any, key: str, default: Any) -> Any:
    """Read *key* from a device config that may be a dict or a Pydantic model."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _ensure_timestamp_variable(
    variable_list: list[str],
    *,
    timestamp_variable: str,
) -> list[str]:
    """Return variables with the hardware timestamp variable present."""
    if timestamp_variable in variable_list:
        return list(variable_list)
    if timestamp_variable != "acq_timestamp" and "acq_timestamp" in variable_list:
        return [
            timestamp_variable if item == "acq_timestamp" else item
            for item in variable_list
        ]
    return [*variable_list, timestamp_variable]


class _ScanLogContextFilter(logging.Filter):
    """Add scan id context to records written to one scan log."""

    def __init__(self, scan_id: str) -> None:
        super().__init__()
        self._scan_id = scan_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.scan_id = self._scan_id
        return True


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------


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
        ``"SINGLESHOT"``, ``"ARMED"``).  Required for
        ``strict_shot_control`` acquisition, which uses ``ARMED`` +
        ``SINGLESHOT`` for plan-owned shots.
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
        on_event: Callable[[ScanEvent], None] | None = None,
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
        self.last_reinit_error: str | None = None
        self._on_event = on_event
        self._current_state = self._scan_state("IDLE")
        self._abort_requested = False

        self._tiled_token: int | None = subscribe_tiled(
            self._RE, tiled_uri, tiled_api_key
        )

        self._scan_thread: threading.Thread | None = None
        self._scan_config: Any = None
        self._rep_rate_hz: float = 1.0
        self._shots_per_step: int = 10
        self._acquisition_mode: str = _STRICT_MODE
        self._devices_config: dict[str, Any] = {}

        # Devices are CA-backed only (the direct UDP/TCP backend was removed
        # once the CA backend reached verified parity; the GeecsCAGateway is
        # the one component that speaks GEECS wire protocol). Catch a stale
        # environment loudly rather than silently changing behavior.
        legacy_backend = os.environ.get("GEECS_BLUESKY_DEVICE_BACKEND")
        if legacy_backend and legacy_backend.strip().lower() != "ca":
            raise ValueError(
                f"GEECS_BLUESKY_DEVICE_BACKEND={legacy_backend!r} is set, but "
                "the direct device backend was removed — devices are CA-backed "
                "via the GeecsCAGateway. Unset the variable (or set it to 'ca')."
            )

        self._total_shots: int = 0
        self._completed_shots: int = 0
        # uid of the most recent run's start document (for the Tiled exporter)
        self._last_run_uid: str | None = None

        # Devices held open between reinitialize() and scan completion
        self._motor: Any | None = None
        self._detectors: list = []
        # Free-run pacemaker (first synchronous device); None in strict mode
        self._reference_detector: Any | None = None
        # Subset of detectors that save per-shot files: list of (detector, path)
        self._saving_detectors: list[tuple] = []
        self._nonscalar_save_paths: dict[str, str] = {}

        # Shot control — validated from the shot_control_information YAML/dict
        self._shot_control: ShotControlConfig | None = (
            ShotControlConfig.from_information(shot_control_information)
        )
        self._shot_controller: ShotController | None = None

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
        self._rep_rate_hz = float(rep_rate or 1.0)
        wait_time = getattr(self._scan_config, "wait_time", 10.0)
        self._shots_per_step = max(1, round(self._rep_rate_hz * wait_time))

        self._acquisition_mode = self._resolve_acquisition_mode(exec_config.options)

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
        self._nonscalar_save_paths = {}

        # Disconnect any leftover devices from a previous scan
        self._disconnect_devices_sync()

        logger.info(
            "BlueskyScanner reinitialised — mode=%s, shots_per_step=%d, devices=%s",
            self._acquisition_mode,
            self._shots_per_step,
            list(self._devices_config),
        )
        return True

    @staticmethod
    def _resolve_acquisition_mode(options: Any, env: dict | None = None) -> str:
        """Resolve the acquisition mode from options, with an env override.

        Precedence: ``GEECS_BLUESKY_ACQUISITION_MODE`` env var (quick
        switching) > ``options.acquisition_mode`` > default
        ``strict_shot_control``.  An unrecognised value raises instead of
        silently changing scan semantics.  *env* is injectable for testing.
        """
        env = os.environ if env is None else env
        raw = (
            env.get("GEECS_BLUESKY_ACQUISITION_MODE")
            or getattr(options, "acquisition_mode", None)
            or _STRICT_MODE
        )
        mode = str(raw).strip().lower()
        if mode not in _VALID_ACQUISITION_MODES:
            raise GeecsConfigurationError(
                f"Unknown acquisition_mode {raw!r}; expected one of "
                f"{', '.join(_VALID_ACQUISITION_MODES)}"
            )
        return mode

    @staticmethod
    def _classify_device_roles(
        devices_config: dict[str, Any], mode: str
    ) -> list[tuple[str, str]]:
        """Assign each configured device a role from the acquisition mode.

        Roles: ``"snapshot"`` (asynchronous), ``"triggered"`` (synchronous in
        strict mode), ``"reference"`` (first synchronous device in free-run
        mode — the pacemaker), ``"contributor"`` (later synchronous devices in
        free-run mode).  Returns ``(device_name, role)`` pairs in config order.
        """
        free_run = mode == _FREE_RUN_MODE
        roles: list[tuple[str, str]] = []
        reference_assigned = False
        for name, cfg in devices_config.items():
            synchronous = bool(_cfg_field(cfg, "synchronous", False))
            if not synchronous:
                role = "snapshot"
            elif not free_run:
                role = "triggered"
            elif not reference_assigned:
                role = "reference"
                reference_assigned = True
            else:
                role = "contributor"
            roles.append((name, role))
        return roles

    @property
    def current_state(self):
        """Current scan lifecycle state for GUI compatibility."""
        return self._current_state

    def start_scan_thread(self) -> None:
        """Launch the scan stored by :meth:`reinitialize` in a background thread."""
        if self.is_scanning_active():
            logger.warning(
                "start_scan_thread called while scan already active; ignoring"
            )
            return

        self._completed_shots = 0
        self._abort_requested = False
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
        self._abort_requested = True
        self._set_state("STOPPING")
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

    def _on_document(self, name: str, doc: dict) -> None:
        if name == "start":
            self._last_run_uid = doc.get("uid")
        elif name == "event":
            self._completed_shots += 1

    @staticmethod
    def _scan_state(state_name: str):
        """Return a ScanState enum member when geecs_scanner is importable."""
        if ScanState is None:
            return state_name.lower()
        return getattr(ScanState, state_name)

    def _set_state(self, state_name: str, total_shots: int = 0) -> None:
        """Update lifecycle state and emit a GUI scan lifecycle event if possible."""
        state = self._scan_state(state_name)
        self._current_state = state
        if self._on_event is None or ScanLifecycleEvent is None:
            return
        try:
            self._on_event(ScanLifecycleEvent(state=state, total_shots=total_shots))
        except Exception:
            logger.debug("on_event callback raised; ignoring", exc_info=True)

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

        self._shot_controller = None

    def _build_shot_controller(self) -> None:
        """Build the ShotController: UDP setters (direct) or gateway :SP (ca).

        Silently skips if no shot control device is configured or the setup
        fails, so scans without trigger control still run (e.g. in
        internal-trigger test mode).
        """
        if self._shot_control is None or not self._shot_control.variables:
            logger.debug("No shot control device configured — trigger control disabled")
            return
        try:
            self._shot_controller = ShotController.over_ca(
                self._shot_control,
                experiment=self._experiment_dir,
                rep_rate_hz=self._rep_rate_hz,
            )
            logger.info(
                "Shot controller ready: %s (%d variables, via gateway :SP)",
                self._shot_control.device,
                len(self._shot_control.variables),
            )
        except Exception:
            logger.warning(
                "Could not build shot controller — trigger control disabled",
                exc_info=True,
            )

    def _require_strict_single_shot(self) -> None:
        """Raise unless strict mode can fire plan-owned single shots."""
        guidance = (
            "Use acquisition_mode='free_run_time_sync' for free-running "
            "trigger acquisition."
        )
        if self._shot_control is None:
            raise GeecsConfigurationError(
                "strict_shot_control requires shot_control_information with a "
                f"non-empty ARMED state. {guidance}"
            )
        if self._shot_controller is None:
            raise GeecsConfigurationError(
                "strict_shot_control requires a reachable shot-control device "
                f"with configured setters before acquisition can start. {guidance}"
            )
        self._shot_controller.require_strict_single_shot()

    def _run_scan(self, scan_config: Any) -> None:
        """Scan thread body: create devices, run plan, clean up."""
        failed = False
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
            failed = True
            logger.exception("BlueskyScanner scan thread raised an exception")
        finally:
            self._disconnect_devices_sync()
            if self._abort_requested or failed:
                self._set_state("ABORTED")
            else:
                self._set_state("DONE")
            logger.info("BlueskyScanner: scan thread finished")

    def _claim_scan_number(self) -> tuple[int | None, str | None]:
        """Claim the next day-scoped scan number/folder for this experiment.

        Thin wrapper over :func:`~geecs_bluesky.plans.run_wrapper.claim_scan_number`
        (the shared scanner-side claim used by both the scanner and notebooks).
        """
        return claim_scan_number(self._experiment_dir)

    def _write_scan_info_ini(
        self, scan_config: Any, scan_number: int, scan_folder: str
    ) -> None:
        """Write ``ScanInfoScanNNN.ini`` into the claimed scan folder.

        Replicates the legacy ``ScanDataManager.write_scan_info_ini`` ``[Scan
        Info]`` format so existing analysis tooling parses Bluesky scans
        unchanged.  Writes only into the already-claimed ``scans/ScanNNN/``
        folder — it never creates the scan folder (cross-package invariant).
        """
        from pathlib import Path

        folder = Path(scan_folder)
        if not folder.is_dir():
            logger.warning(
                "Scan folder %s does not exist; skipping ScanInfo write", folder
            )
            return

        scan_var = scan_config.device_var or "Shotnumber"
        description = scan_config.additional_description or ""
        scan_mode = getattr(scan_config.scan_mode, "value", str(scan_config.scan_mode))
        background = bool(getattr(scan_config, "background", False))
        scan_info = f"Bluesky scan. scanning {scan_var}. {description}".strip()

        lines = [
            "[Scan Info]\n",
            f"Scan No = {scan_number}\n",
            f'ScanStartInfo = "{scan_info}"\n',
            f'Scan Parameter = "{scan_var}"\n',
            f"Start = {scan_config.start}\n",
            f"End = {scan_config.end}\n",
            f"Step size = {scan_config.step}\n",
            f"Shots per step = {scan_config.wait_time}\n",
            'ScanEndInfo = ""\n',
            f"Background = {str(background).lower()}\n",
            f'ScanMode = "{scan_mode}"\n',
        ]
        path = folder / f"ScanInfo{folder.name}.ini"
        try:
            with path.open("w") as f:
                f.writelines(lines)
            logger.info("Scan info written to %s", path)
        except OSError:
            logger.warning("Could not write ScanInfo to %s", path, exc_info=True)

    @contextmanager
    def _scan_log(self, scan_number: int | None, scan_folder: str | None):
        """Attach a per-scan log file for Bluesky scanner runs."""
        if scan_number is None or scan_folder is None:
            yield
            return

        folder = Path(scan_folder)
        if not folder.is_dir():
            logger.warning("Scan folder %s does not exist; skipping scan.log", folder)
            yield
            return

        scan_id = f"Scan{scan_number:03d}"
        handler = logging.FileHandler(folder / "scan.log", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d %(levelname)s %(name)s "
                "[%(threadName)s] scan=%(scan_id)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler.addFilter(_ScanLogContextFilter(scan_id))

        package_logger = logging.getLogger("geecs_bluesky")
        old_level = package_logger.level
        if old_level == logging.NOTSET or old_level > logging.INFO:
            package_logger.setLevel(logging.INFO)
        package_logger.addHandler(handler)
        try:
            logger.info("scan %s: starting (dir=%s)", scan_id, scan_folder)
            yield
            logger.info("scan %s: finished", scan_id)
        finally:
            package_logger.removeHandler(handler)
            handler.close()
            package_logger.setLevel(old_level)

    def _export_scalar_files(self, scan_number: int) -> None:
        """Write legacy ScanData/s-file from the just-recorded Tiled run.

        Best-effort: any failure (Tiled unreachable, missing run, export bug)
        is logged and swallowed so it never crashes the scan thread.
        """
        uid = self._last_run_uid
        if uid is None:
            logger.warning(
                "No run uid captured; skipping scalar-file export for scan %s",
                scan_number,
            )
            return
        try:
            from geecs_data_utils import write_scalar_files_from_tiled

            result = write_scalar_files_from_tiled(uid)
            if result is not None:
                logger.info("Wrote legacy scalar files: %s, %s", *result)
        except Exception:
            logger.warning(
                "Could not export legacy scalar files for scan %s (uid=%s)",
                scan_number,
                uid,
                exc_info=True,
            )

    def _build_detectors(
        self, scan_folder: str | None = None, scan_number: int | None = None
    ) -> None:
        """Create and connect detector devices from ``self._devices_config``.

        Each device's role is decided by :meth:`_classify_device_roles` from
        the acquisition mode.  In free-run mode the first synchronous device
        becomes the reference (pacemaker) and later synchronous devices become
        non-blocking contributors anchored to it; in strict mode every
        synchronous device is a triggered detector.  All devices are CA-backed
        (:mod:`geecs_bluesky.devices.ca`), consuming the gateway PVs.
        Populates ``self._detectors``, ``self._reference_detector``, and
        ``self._saving_detectors`` in place.  Devices that fail to resolve or
        connect are logged and skipped.
        """
        self._reference_detector = None
        roles = dict(
            self._classify_device_roles(self._devices_config, self._acquisition_mode)
        )
        for device_name, dev_cfg in self._devices_config.items():
            variable_list = list(_cfg_field(dev_cfg, "variable_list", []) or [])
            synchronous = bool(_cfg_field(dev_cfg, "synchronous", False))
            save_nonscalar = bool(_cfg_field(dev_cfg, "save_nonscalar_data", False))
            role = roles[device_name]
            device_type = self._device_type_for_device(device_name)
            timestamp_variable = self._timestamp_variable_for_device_type(device_type)

            if synchronous:
                variable_list = _ensure_timestamp_variable(
                    variable_list,
                    timestamp_variable=timestamp_variable,
                )

            if not variable_list:
                logger.debug("Skipping %s: empty variable_list", device_name)
                continue
            ophyd_name = safe_name(device_name)

            try:
                det = self._build_ca_detector(
                    role,
                    device_name,
                    variable_list,
                    ophyd_name,
                    save_nonscalar,
                    timestamp_variable,
                )
                with self._device_lock:
                    self._detectors.append(det)
                    saves_files = role in ("reference", "triggered", "contributor")
                    if saves_files and save_nonscalar and scan_folder is not None:
                        save_path = os.path.join(scan_folder, device_name)
                        device_save_path = device_server_save_path(save_path)
                        det.configure_nonscalar_file_logging(save_path)
                        if scan_number is not None:
                            asset_definitions = self._asset_definitions_for_device_type(
                                device_name,
                                device_type,
                            )
                            if asset_definitions:
                                asset_root, asset_local_root = (
                                    asset_resource_root_paths()
                                )
                                det.configure_external_asset_logging(
                                    scan_number=scan_number,
                                    asset_definitions=asset_definitions,
                                    root_path=asset_root or scan_folder,
                                    local_root_path=asset_local_root or scan_folder,
                                )
                        self._saving_detectors.append(
                            (det, save_path, device_save_path)
                        )
                        self._nonscalar_save_paths[device_name] = save_path
                logger.info(
                    "Detector ready: %s (role=%s, %d variables, save_nonscalar=%s)",
                    device_name,
                    role,
                    len(variable_list),
                    save_nonscalar,
                )
            except Exception:
                logger.warning(
                    "Failed to create/connect detector %s — skipping",
                    device_name,
                    exc_info=True,
                )

    def _build_ca_detector(
        self,
        role: str,
        device_name: str,
        variable_list: list[str],
        ophyd_name: str,
        save_nonscalar: bool,
        timestamp_variable: str,
    ):
        """Construct and connect one CA-backed detector for *role*.

        Mirrors the direct-backend role dispatch in :meth:`_build_detectors`;
        the CA classes compose the same shot-id/contributor/save mixins, so
        behavior past construction is shared.
        """
        # Deferred import: needs the `ca` extra (aioca).
        from geecs_bluesky.devices.ca import (
            CaGenericDetector,
            CaSnapshotReadable,
            CaTimestampedReadable,
        )

        if role == "contributor":
            det = CaTimestampedReadable(
                device_name,
                variable_list,
                experiment=self._experiment_dir,
                name=ophyd_name,
                save_nonscalar_data=save_nonscalar,
                acq_timestamp_variable=timestamp_variable,
            )
            self._connect_device(det)
            det.configure_shot_id(self._rep_rate_hz)
            if self._reference_detector is not None:
                det.set_reference(self._reference_detector)
        elif role == "snapshot":
            if save_nonscalar:
                logger.warning(
                    "Ignoring save_nonscalar_data for asynchronous snapshot device %s",
                    device_name,
                )
            det = CaSnapshotReadable(
                device_name,
                variable_list,
                experiment=self._experiment_dir,
                name=ophyd_name,
            )
            self._connect_device(det)
        else:  # "reference" or "triggered"
            det = CaGenericDetector(
                device_name,
                variable_list,
                experiment=self._experiment_dir,
                name=ophyd_name,
                save_nonscalar_data=save_nonscalar,
                acq_timestamp_variable=timestamp_variable,
            )
            self._connect_device(det)
            det.configure_shot_id(self._rep_rate_hz)
            if role == "reference":
                self._reference_detector = det
        return det

    @staticmethod
    def _timestamp_variable_for_device_type(device_type: str | None) -> str:
        """Return the hardware timestamp variable for a GEECS device type."""
        _ = device_type
        return "acq_timestamp"

    def _device_type_for_device(self, device_name: str) -> str | None:
        """Return the GEECS device type, logging and continuing on failure."""
        try:
            return GeecsDb.get_device_type(device_name)
        except Exception:
            logger.warning(
                "Could not resolve device type for %s",
                device_name,
                exc_info=True,
            )
            return None

    def _asset_definitions_for_device_type(
        self,
        device_name: str,
        device_type: str | None,
    ) -> tuple[AssetDefinition, ...]:
        """Return registered external asset definitions for *device_name*."""
        if device_type is None:
            logger.warning(
                "Could not resolve device type for %s; external asset docs disabled",
                device_name,
            )
            return ()
        definitions = get_asset_definitions(device_type)
        if not definitions:
            logger.debug(
                "No external asset registry entry for %s device type %r",
                device_name,
                device_type,
            )
        return definitions

    def _run_standard_scan(self, scan_config: Any) -> None:
        """Step scan: move a scan device through positions, collect shots each step."""
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
        from geecs_bluesky.devices.ca import CaMotor

        motor = CaMotor(
            device_name,
            variable,
            experiment=self._experiment_dir,
            name=ophyd_name,
        )

        logger.info("Connecting motor …")
        self._connect_device(motor)
        with self._device_lock:
            self._motor = motor

        positions = _build_positions(scan_config)
        extra_md: dict[str, Any] = {"device_var": scan_config.device_var}
        if scan_config.additional_description:
            extra_md["description"] = scan_config.additional_description
        self._run_step_scan(scan_config, motor, positions, extra_md)

    def _run_noscan(self, scan_config: Any) -> None:
        """Statistics collection: N shots at fixed settings, no scan variable moved.

        Routed through the same plan as a motor scan with ``motor=None`` and a
        single no-move bin, so it works identically in both acquisition modes
        (strict and free-run) instead of being a separate code path.
        """
        extra_md: dict[str, Any] = {}
        if getattr(scan_config, "additional_description", None):
            extra_md["description"] = scan_config.additional_description
        self._run_step_scan(
            scan_config, motor=None, positions=[None], extra_md=extra_md
        )

    def _run_step_scan(
        self,
        scan_config: Any,
        motor: Any | None,
        positions: list[float | None],
        extra_md: dict[str, Any] | None = None,
    ) -> None:
        """Shared body for motor scans and statistics collection.

        ``motor=None`` with ``positions=[None]`` is statistics collection (one
        no-move bin); a real motor with explicit positions is a step scan.
        Either way the acquisition mode picks the plan: free-run
        (reference-paced) or strict plan-owned single-shot.
        """
        # Claim scan number before building detectors so save paths are known
        scan_number, scan_folder = self._claim_scan_number()
        if scan_number is not None and scan_folder is not None:
            self._write_scan_info_ini(scan_config, scan_number, scan_folder)
        self._build_detectors(scan_folder=scan_folder, scan_number=scan_number)

        n_steps = len(positions)
        n_shots = n_steps * self._shots_per_step
        self._total_shots = n_shots
        self._set_state("INITIALIZING", total_shots=n_shots)

        # A motorless run with no detectors records nothing meaningful.
        if not self._detectors and motor is None:
            logger.info(
                "Statistics collection with no detectors: nothing to collect "
                "(%d shots requested). Add detector devices to enable collection.",
                n_shots,
            )
            return

        logger.info(
            "%s: %d step(s) × %d shots/step = %d total events",
            "Statistics collection" if motor is None else "Step scan",
            n_steps,
            self._shots_per_step,
            n_shots,
        )

        # Scanner-contributed run metadata (the plan owns its own intrinsic md:
        # plan_name, acquisition_mode, geecs_event_schema, positions, …).
        run_md: dict[str, Any] = {
            "operator": "",
            "scan_mode": getattr(
                scan_config.scan_mode, "value", str(scan_config.scan_mode)
            ),
            "wait_time": getattr(scan_config, "wait_time", None),
            **(extra_md or {}),
        }

        self._build_shot_controller()
        strict = self._acquisition_mode != _FREE_RUN_MODE
        if strict:
            self._require_strict_single_shot()
        elif self._reference_detector is None:
            logger.error(
                "free-run scan requires at least one synchronous device as "
                "reference; none found — aborting"
            )
            return

        # The one shared orchestration recipe (also used by GeecsSession):
        # mode dispatch + run wrapper + guaranteed finalize disarm.
        plan = build_step_scan_plan(
            strict=strict,
            motor=motor,
            positions=positions,
            reference=self._reference_detector,
            detectors=list(self._detectors),
            shots_per_step=self._shots_per_step,
            controller=self._shot_controller,
            experiment=self._experiment_dir,
            scan_number=scan_number,
            scan_folder=scan_folder,
            saving_detectors=list(self._saving_detectors),
            extra_md=run_md,
        )
        log_context = (
            self._scan_log(scan_number, scan_folder)
            if scan_number is not None and scan_folder is not None
            else nullcontext()
        )
        with log_context:
            self._set_state("RUNNING")
            self._last_run_uid = None
            self._RE(plan)

            # After the run completes, export legacy scalar files from Tiled.
            if scan_number is not None and scan_folder is not None:
                self._export_scalar_files(scan_number)
