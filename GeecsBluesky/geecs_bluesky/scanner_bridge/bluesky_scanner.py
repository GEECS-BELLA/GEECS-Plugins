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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np


from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.plans.run_wrapper import claim_scan, claim_scan_number
from geecs_bluesky.session import GeecsSession
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
    optimization_loader:
        Callable injected by the GUI layer for OPTIMIZATION scans:
        ``optimization_loader(optimizer_config_path)`` returns a bridge object
        exposing ``variable_names`` (``"Device:Variable"`` VOCS keys) and
        ``bind(devices=..., scan_tag=...) -> (objective, suggester)`` for
        :meth:`GeecsSession.optimize`.  Lives on the GUI side because the
        config-driven optimizer stack (Xopt, evaluators, ScanAnalysis
        analyzers) belongs to ``geecs_scanner.optimization`` — this package
        cannot import it (dependency direction).  Without a loader,
        optimization-mode requests are logged and skipped.
    """

    def __init__(
        self,
        experiment_dir: str = "",
        shot_control_information: dict | None = None,
        tiled_uri: str | None = None,
        tiled_api_key: str | None = None,
        on_event: Callable[[ScanEvent], None] | None = None,
        optimization_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self._experiment_dir = experiment_dir
        # The session owns the RunEngine (created with context_managers=[] so
        # it works off the main thread), the Tiled subscription, the device
        # factories, and the scan discipline; this class is the GUI adapter.
        self._session = GeecsSession(
            experiment_dir,
            tiled_uri=tiled_uri,
            tiled_api_key=tiled_api_key,
        )
        self._RE = self._session.RE
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

        self._scan_thread: threading.Thread | None = None
        self._scan_config: Any = None
        self._rep_rate_hz: float = 1.0
        self._shots_per_step: int = 10
        self._acquisition_mode: str = _STRICT_MODE
        self._devices_config: dict[str, Any] = {}
        self._optimization_loader = optimization_loader

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

        # Shot control — validated from the shot_control_information YAML/dict
        self._shot_control: ShotControlConfig | None = (
            ShotControlConfig.from_information(shot_control_information)
        )

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
        self._session.rep_rate_hz = self._rep_rate_hz

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
            elif mode_val == "optimization":
                self._run_optimization(scan_config)
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
        motor = self._session.motor(device_name, variable, name=ophyd_name)
        with self._device_lock:
            self._motor = motor

        positions = _build_positions(scan_config)
        extra_md: dict[str, Any] = {"device_var": scan_config.device_var}
        if scan_config.additional_description:
            extra_md["description"] = scan_config.additional_description
        self._execute_scan(scan_config, motor, positions, extra_md)

    def _run_noscan(self, scan_config: Any) -> None:
        """Statistics collection: N shots at fixed settings, no scan variable moved.

        Routed through the same plan as a motor scan with ``motor=None`` and a
        single no-move bin, so it works identically in both acquisition modes
        (strict and free-run) instead of being a separate code path.
        """
        extra_md: dict[str, Any] = {}
        if getattr(scan_config, "additional_description", None):
            extra_md["description"] = scan_config.additional_description
        self._execute_scan(scan_config, motor=None, positions=[None], extra_md=extra_md)

    def _run_optimization(self, scan_config: Any) -> None:
        """Optimization as a scan, driven by the GUI's config-based optimizer stack.

        The GUI-injected ``optimization_loader`` owns everything Xopt/evaluator
        (see the constructor docstring); this method maps the scan request onto
        :meth:`GeecsSession.optimize`: VOCS variables become session settables,
        save devices become the detectors, iterations come from the configured
        step count and shots-per-iteration from ``rep_rate × wait_time`` —
        exactly the legacy optimization-scan shape (iteration = bin).
        """
        if self._optimization_loader is None:
            logger.warning(
                "BlueskyScanner: optimization scan requested but no "
                "optimization_loader was injected; skipping"
            )
            return
        config_path = getattr(scan_config, "optimizer_config_path", None)
        if not config_path:
            logger.error("optimization scan requires optimizer_config_path; got None")
            return

        bridge = self._optimization_loader(str(config_path))

        strict = self._acquisition_mode != _FREE_RUN_MODE
        if strict and self._shot_control is None:
            raise GeecsConfigurationError(
                "strict_shot_control requires shot_control_information with a "
                "non-empty ARMED state. Use acquisition_mode="
                "'free_run_time_sync' for free-running trigger acquisition."
            )
        self._session.shot_control(self._shot_control)

        # Claim first so the scan log wraps the whole run and the bridge's
        # analyzers get the real ScanTag for native-file loading.
        scan_tag, scan_folder = claim_scan(self._experiment_dir)
        scan_number = scan_tag.number if scan_tag is not None else None

        variables: dict[str, Any] = {}
        for key in bridge.variable_names:
            device_name, variable = key.split(":", 1)
            variables[key] = self._session.settable(
                device_name, variable, name=safe_name(f"{device_name}_{variable}")
            )
        detectors = self._build_session_devices()
        if not detectors:
            logger.error("optimization scan has no detector devices; aborting")
            return
        with self._device_lock:
            self._detectors = list(self._detectors) + list(variables.values())

        objective, suggester = bridge.bind(
            devices=list(variables.values()) + detectors, scan_tag=scan_tag
        )

        max_iterations = len(_build_positions(scan_config))
        n_shots = max_iterations * self._shots_per_step
        self._total_shots = n_shots
        self._set_state("INITIALIZING", total_shots=n_shots)
        logger.info(
            "Optimization: up to %d iteration(s) × %d shots = %d events, variables=%s",
            max_iterations,
            self._shots_per_step,
            n_shots,
            list(variables),
        )

        run_md: dict[str, Any] = {
            "operator": "",
            "scan_mode": "optimization",
            "wait_time": getattr(scan_config, "wait_time", None),
            "optimizer_config_path": str(config_path),
        }
        with self._scan_log(scan_number, scan_folder):
            self._set_state("RUNNING")
            self._session.optimize(
                variables=variables,
                detectors=detectors,
                objective=objective,
                suggester=suggester,
                shots_per_iteration=self._shots_per_step,
                max_iterations=max_iterations,
                mode="strict" if strict else "free_run",
                description=getattr(scan_config, "additional_description", "") or "",
                md=run_md,
                scan_number=scan_number,
                scan_folder=scan_folder,
                # Bridges map optimizer-config end-of-run policy (e.g. the
                # legacy move_to_best_on_finish flag) onto the session's.
                on_finish=getattr(bridge, "on_finish", "hold"),
            )

    def _execute_scan(
        self,
        scan_config: Any,
        motor: Any | None,
        positions: list[float | None],
        extra_md: dict[str, Any] | None = None,
    ) -> None:
        """Translate the GUI scan request into a session scan (the thin adapter).

        The session owns the discipline (claiming is done here first so the
        per-scan log can wrap the whole run); this method only maps
        ``exec_config`` shapes onto :meth:`GeecsSession.scan` arguments.
        """
        scan_number, scan_folder = claim_scan_number(self._experiment_dir)

        detectors = self._build_session_devices()
        if not detectors and motor is None:
            logger.info(
                "Statistics collection with no detectors: nothing to collect. "
                "Add detector devices to enable collection."
            )
            return

        strict = self._acquisition_mode != _FREE_RUN_MODE
        if strict and self._shot_control is None:
            raise GeecsConfigurationError(
                "strict_shot_control requires shot_control_information with a "
                "non-empty ARMED state. Use acquisition_mode="
                "'free_run_time_sync' for free-running trigger acquisition."
            )
        self._session.shot_control(self._shot_control)

        n_shots = len(positions) * self._shots_per_step
        self._total_shots = n_shots
        self._set_state("INITIALIZING", total_shots=n_shots)
        logger.info(
            "%s: %d step(s) × %d shots/step = %d total events",
            "Statistics collection" if motor is None else "Step scan",
            len(positions),
            self._shots_per_step,
            n_shots,
        )

        scan_mode = getattr(scan_config.scan_mode, "value", str(scan_config.scan_mode))
        run_md: dict[str, Any] = {
            "operator": "",
            "scan_mode": scan_mode,
            "wait_time": getattr(scan_config, "wait_time", None),
            **(extra_md or {}),
        }
        scan_info = {
            "scan_parameter": scan_config.device_var or "Shotnumber",
            "start": scan_config.start,
            "end": scan_config.end,
            "step": scan_config.step,
            # Legacy-format quirk preserved: the GUI ini records wait_time here.
            "shots": scan_config.wait_time,
            "background": bool(getattr(scan_config, "background", False)),
            "scan_mode": scan_mode,
        }

        with self._scan_log(scan_number, scan_folder):
            self._set_state("RUNNING")
            self._session.scan(
                detectors=detectors,
                motor=motor,
                positions=positions,
                shots_per_step=self._shots_per_step,
                mode="strict" if strict else "free_run",
                description=getattr(scan_config, "additional_description", "") or "",
                md=run_md,
                scan_number=scan_number,
                scan_folder=scan_folder,
                scan_info=scan_info,
            )

    def _build_session_devices(self) -> list:
        """Create CA devices from the GUI device table via the session factories.

        Role classification is unchanged; the returned list is ordered with
        the free-run reference first (``session.scan`` anchors contributors to
        ``detectors[0]``).  Devices that fail to construct/connect are logged
        and skipped.
        """
        roles = self._classify_device_roles(
            self._devices_config, self._acquisition_mode
        )
        reference: list = []
        others: list = []
        for device_name, role in roles:
            cfg = self._devices_config[device_name]
            variables = list(_cfg_field(cfg, "variable_list", []) or [])
            save = bool(_cfg_field(cfg, "save_nonscalar_data", False))
            if not variables:
                logger.debug("Skipping %s: empty variable_list", device_name)
                continue
            name = safe_name(device_name)
            try:
                if role == "snapshot":
                    if save:
                        logger.warning(
                            "Ignoring save_nonscalar_data for asynchronous "
                            "snapshot device %s",
                            device_name,
                        )
                    det = self._session.snapshot(device_name, variables, name=name)
                elif role == "contributor":
                    det = self._session.contributor(
                        device_name, variables, save_images=save, name=name
                    )
                else:  # "reference" or "triggered"
                    det = self._session.detector(
                        device_name, variables, save_images=save, name=name
                    )
                (reference if role == "reference" else others).append(det)
                logger.info(
                    "Detector ready: %s (role=%s, %d variables, save_nonscalar=%s)",
                    device_name,
                    role,
                    len(variables),
                    save,
                )
            except Exception:
                logger.warning(
                    "Failed to create/connect detector %s — skipping",
                    device_name,
                    exc_info=True,
                )
        detectors = reference + others
        with self._device_lock:
            self._detectors = list(detectors)
        return detectors
