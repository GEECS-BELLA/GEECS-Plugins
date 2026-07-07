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
_THREAD_JOIN_TIMEOUT = 15.0

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
        ``bind(devices=..., scan_tag=..., scan_folder=...) -> (objective,
        suggester)`` for
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
        """Abort the running scan and wait for the thread to finish.

        The abort flag is honoured even before the plan reaches the
        RunEngine (``RE.abort()`` on an idle engine is a no-op, so a stop
        clicked during device connect relies on the flag — see
        :meth:`_abort_before_acquisition`).  If the scan thread does not
        exit within the join timeout, the thread handle is *kept* so
        :meth:`is_scanning_active` keeps reporting ``True``: clearing it
        would let a second ``start_scan_thread`` run the still-busy
        RunEngine and let ``reinitialize`` disconnect devices under a live
        scan.
        """
        logger.info("BlueskyScanner: abort requested")
        self._abort_requested = True
        self._set_state("STOPPING")
        try:
            # RE.abort() on an idle engine raises a TransitionError; the
            # pre-RE abort path is covered by the _abort_requested flag.
            if self._RE.state != "idle":
                self._RE.abort(reason="stop_scanning_thread called")
        except Exception:
            logger.debug("RE.abort() raised (may not be running)", exc_info=True)
        thread = self._scan_thread
        if thread is not None:
            thread.join(timeout=_THREAD_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.error(
                    "BlueskyScanner: scan thread did not stop within %.0f s — "
                    "keeping the thread handle; the scanner still reports "
                    "active and must not be restarted or reinitialized until "
                    "the thread exits",
                    _THREAD_JOIN_TIMEOUT,
                )
            else:
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

    def _abort_before_acquisition(self) -> bool:
        """Return ``True`` (logging loudly) if a stop arrived before the plan ran.

        ``RE.abort()`` cannot stop a scan that has not reached the RunEngine
        yet, so the scan thread checks this flag between thread start and
        the ``RE(plan)`` invocation (after device connect, before the scan
        folder is claimed).
        """
        if not self._abort_requested:
            return False
        logger.warning(
            "BlueskyScanner: stop requested before acquisition started; "
            "aborting the scan before it reaches the RunEngine"
        )
        return True

    @staticmethod
    def _log_claimed_scan_failure(
        scan_number: int | None, scan_folder: str | None
    ) -> None:
        """Log loudly that a claimed scan folder was left behind by a failure.

        The folder is never deleted (scan-folder lifecycle invariant: once a
        ``scans/ScanNNN/`` folder exists it must not be removed or
        recreated), so the claimed-but-failed state is surfaced here instead
        of being silent.
        """
        if scan_number is None and scan_folder is None:
            return
        logger.error(
            "Scan %s failed or aborted after its folder was claimed at %s; "
            "the folder is left in place (never deleted) and may be missing "
            "ScanInfo or data",
            scan_number,
            scan_folder,
        )

    def _run_scan(self, scan_config: Any) -> None:
        """Scan thread body: create devices, run plan, clean up."""
        failed = False
        try:
            if self._abort_before_acquisition():
                return
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

        # Capture the whole scan story, not just this package: during
        # optimization scans the evaluator (geecs_scanner.optimization) and
        # its analyzers (scan_analysis, image_analysis) do the per-bin work,
        # and their file-mapping / objective lines belong in scan.log too.
        capture_loggers = [
            logging.getLogger(name)
            for name in (
                "geecs_bluesky",
                "geecs_scanner.optimization",
                "scan_analysis",
                "image_analysis",
            )
        ]
        old_levels = [lg.level for lg in capture_loggers]
        for lg in capture_loggers:
            if lg.level == logging.NOTSET or lg.level > logging.INFO:
                lg.setLevel(logging.INFO)
            lg.addHandler(handler)
        try:
            logger.info("scan %s: starting (dir=%s)", scan_id, scan_folder)
            yield
            logger.info("scan %s: finished", scan_id)
        finally:
            for lg, old_level in zip(capture_loggers, old_levels):
                lg.removeHandler(handler)
                lg.setLevel(old_level)
            handler.close()

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

    def _merge_optimization_device_requirements(self, requirements: Any) -> None:
        """Merge optimizer ``device_requirements`` into the save-device set.

        Legacy parity: ``ScanManager`` feeds ``optimizer.device_requirements``
        (shape ``{"Devices": {name: cfg}}``, auto-generated by the optimizer
        config from the evaluator's analyzers with a synchronous +
        ``save_nonscalar_data=True`` + ``acq_timestamp`` template) through
        ``device_manager.load_from_dictionary``, so every device the
        objective needs is acquired and natively saved without being on the
        GUI save list.  Mirrored here on ``self._devices_config`` before
        :meth:`_build_session_devices` runs:

        - a required device absent from the GUI list is added with the
          requirement's own config (flags default to the legacy
          ``DeviceConfig`` defaults when the requirement omits them); it is
          appended after the GUI devices, so the free-run reference
          (pacemaker) choice is unchanged;
        - a device already on the GUI list keeps its GUI config and only
          gains required variables it was missing (the legacy
          ``subscribe_var_values`` extension).

        Device names are matched **case-insensitively** (``str.casefold``):
        GEECS itself is case-inconsistent about device-name spelling (the DB
        may say ``UC_Amp4_IR_input`` while native files say
        ``UC_Amp4_IR_Input``), so optimizer configs drift.  On a
        case-insensitive hit the requirement merges into the *existing*
        entry under the GUI's spelling — the GUI/DB spelling is the one
        whose CA PVs actually connect (CA names are case-sensitive) — and
        the difference is logged at INFO.  Live-observed 2026-07-06: a
        wrong-case duplicate entry NotConnectedError'd on every PV.

        *requirements* is duck-typed off the bridge object (``Any``): a
        ``{"Devices": ...}`` mapping, or ``None``/empty when the bridge
        exposes no requirements (unchanged behavior).
        """
        devices = (
            requirements.get("Devices") if isinstance(requirements, dict) else None
        )
        if not devices:
            return
        for device_name, req in devices.items():
            req_vars = list(_cfg_field(req, "variable_list", []) or [])
            configured_name = device_name
            existing = self._devices_config.get(device_name)
            if existing is None:
                folded = device_name.casefold()
                match = next(
                    (
                        name
                        for name in self._devices_config
                        if name.casefold() == folded
                    ),
                    None,
                )
                if match is not None:
                    configured_name = match
                    existing = self._devices_config[match]
                    logger.info(
                        "Optimization: required device %s differs only in "
                        "case from configured device %s; merging under the "
                        "GUI spelling %s (CA PV names are case-sensitive)",
                        device_name,
                        match,
                        match,
                    )
            if existing is None:
                entry = {
                    "synchronous": bool(_cfg_field(req, "synchronous", False)),
                    "save_nonscalar_data": bool(
                        _cfg_field(req, "save_nonscalar_data", False)
                    ),
                    "variable_list": req_vars,
                }
                self._devices_config[device_name] = entry
                logger.info(
                    "Optimization: auto-provisioned required device %s "
                    "(synchronous=%s, save_nonscalar=%s, variables=%s) — "
                    "verify the spelling matches the GEECS database; "
                    "CA PV names are case-sensitive",
                    device_name,
                    entry["synchronous"],
                    entry["save_nonscalar_data"],
                    req_vars,
                )
            else:
                current = list(_cfg_field(existing, "variable_list", []) or [])
                missing = [v for v in req_vars if v not in current]
                if not missing:
                    continue
                merged = current + missing
                if isinstance(existing, dict):
                    existing["variable_list"] = merged
                else:
                    existing.variable_list = merged
                logger.info(
                    "Optimization: added required variable(s) %s to "
                    "configured device %s (GUI settings preserved)",
                    missing,
                    configured_name,
                )

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

        # Legacy parity: merge the optimizer config's device requirements
        # into the save-device set before detectors are built, so the
        # objective's diagnostics acquire and save even when they are not on
        # the GUI save list.  Duck-typed off the bridge (like ``on_finish``/
        # ``finish``) — geecs_bluesky never imports geecs_scanner.
        self._merge_optimization_device_requirements(
            getattr(bridge, "device_requirements", None)
        )

        strict = self._acquisition_mode != _FREE_RUN_MODE
        if strict and self._shot_control is None:
            raise GeecsConfigurationError(
                "strict_shot_control requires shot_control_information with a "
                "non-empty ARMED state. Use acquisition_mode="
                "'free_run_time_sync' for free-running trigger acquisition."
            )
        self._session.shot_control(self._shot_control)

        detectors = self._build_session_devices()
        if not detectors:
            logger.error(
                "optimization scan has no detector devices; aborting before "
                "claiming a scan folder"
            )
            return

        # Connect the VOCS settables only after the detector check, so an
        # early exit cannot leak their persistent CA monitors.  Each settable
        # joins self._detectors as soon as it is connected — the scan
        # thread's cleanup then disconnects it even when a later step fails.
        variables: dict[str, Any] = {}
        for key in bridge.variable_names:
            device_name, variable = key.split(":", 1)
            settable = self._session.settable(
                device_name, variable, name=safe_name(f"{device_name}_{variable}")
            )
            variables[key] = settable
            with self._device_lock:
                self._detectors.append(settable)

        if self._abort_before_acquisition():
            return

        # Claim only after everything above that can fail has succeeded.  The
        # claim still precedes the bridge bind (its analyzers get the real
        # ScanTag for native-file loading) and the scan log (so the log wraps
        # the whole run).
        scan_tag, scan_folder = claim_scan(self._experiment_dir)
        scan_number = scan_tag.number if scan_tag is not None else None

        try:
            objective, suggester = bridge.bind(
                devices=list(variables.values()) + detectors,
                scan_tag=scan_tag,
                scan_folder=scan_folder,
            )

            max_iterations = len(_build_positions(scan_config))
            n_shots = max_iterations * self._shots_per_step
            self._total_shots = n_shots
            self._set_state("INITIALIZING", total_shots=n_shots)
            logger.info(
                "Optimization: up to %d iteration(s) × %d shots = %d events, "
                "variables=%s",
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
                    description=getattr(scan_config, "additional_description", "")
                    or "",
                    md=run_md,
                    scan_number=scan_number,
                    scan_folder=scan_folder,
                    # Bridges map optimizer-config end-of-run policy (e.g. the
                    # legacy move_to_best_on_finish flag) onto the session's.
                    on_finish=getattr(bridge, "on_finish", "hold"),
                )
                # Post-run bookkeeping owned by the bridge (e.g. the legacy
                # xopt_dump.yaml written into the scan folder).
                finish = getattr(bridge, "finish", None)
                if callable(finish):
                    finish()
        except BaseException:
            self._log_claimed_scan_failure(scan_number, scan_folder)
            raise

    def _execute_scan(
        self,
        scan_config: Any,
        motor: Any | None,
        positions: list[float | None],
        extra_md: dict[str, Any] | None = None,
    ) -> None:
        """Translate the GUI scan request into a session scan (the thin adapter).

        The session owns the discipline; this method only maps
        ``exec_config`` shapes onto :meth:`GeecsSession.scan` arguments.
        Everything that can fail (device connect, mode validation, shot
        control) runs *before* the scan folder is claimed, so an early exit
        never leaves an empty claimed ``ScanNNN/`` folder behind; the claim
        still precedes the per-scan log so the log wraps the whole run.
        """
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

        if self._abort_before_acquisition():
            return

        scan_number, scan_folder = claim_scan_number(self._experiment_dir)

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

        try:
            with self._scan_log(scan_number, scan_folder):
                self._set_state("RUNNING")
                self._session.scan(
                    detectors=detectors,
                    motor=motor,
                    positions=positions,
                    shots_per_step=self._shots_per_step,
                    mode="strict" if strict else "free_run",
                    description=getattr(scan_config, "additional_description", "")
                    or "",
                    md=run_md,
                    scan_number=scan_number,
                    scan_folder=scan_folder,
                    scan_info=scan_info,
                )
        except BaseException:
            self._log_claimed_scan_failure(scan_number, scan_folder)
            raise

    def _build_session_devices(self) -> list:
        """Create CA devices from the GUI device table via the session factories.

        The returned list is ordered with the free-run reference first
        (``session.scan`` anchors contributors to ``detectors[0]``).  Devices
        that fail to construct/connect are logged and skipped — except the
        free-run reference (pacemaker): silently skipping it would let a
        non-Triggerable contributor land in ``detectors[0]`` and inherit
        pacemaker duty, and ``trigger_and_read`` would then record unpaced
        duplicate rows of cached frames.  Instead, the next synchronous
        device is promoted to the reference role (built Triggerable via
        ``session.detector``); if no synchronous device connects at all,
        this raises rather than acquire garbage.

        An empty ``variable_list`` does not disqualify a synchronous device:
        ``acq_timestamp`` is always created as a dedicated child, and native
        file saving may be the element's whole purpose (image-only camera) —
        matching the legacy scanner, which force-appends ``acq_timestamp`` to
        every synchronous device.  Only an asynchronous snapshot device with
        no variables is skipped (it would record nothing).

        Raises
        ------
        GeecsConfigurationError
            In free-run mode, when synchronous devices are configured but
            none of them could be connected as the reference (pacemaker).
            Devices that did connect are left in ``self._detectors`` so the
            scan thread's cleanup disconnects them.
        """
        roles = self._classify_device_roles(
            self._devices_config, self._acquisition_mode
        )
        free_run = self._acquisition_mode == _FREE_RUN_MODE
        reference_configured = any(role == "reference" for _name, role in roles)
        reference: list = []
        others: list = []
        failures: list[str] = []
        promote_next_contributor = False
        for device_name, role in roles:
            cfg = self._devices_config[device_name]
            variables = list(_cfg_field(cfg, "variable_list", []) or [])
            save = bool(_cfg_field(cfg, "save_nonscalar_data", False))
            if promote_next_contributor and role == "contributor":
                role = "reference"
                promote_next_contributor = False
                logger.warning(
                    "Promoting %s to free-run reference (pacemaker) — the "
                    "configured reference device was unavailable",
                    device_name,
                )
            if not variables and role == "snapshot":
                # An asynchronous device with no variables records nothing at
                # all. Synchronous devices are still built: acq_timestamp is
                # always created as a dedicated child (and native saving may
                # be the whole point, e.g. an image-only camera element) —
                # matching the legacy scanner, which force-appends
                # acq_timestamp to every synchronous device.
                logger.warning(
                    "Skipping asynchronous device %s: empty variable_list "
                    "(nothing to record)",
                    device_name,
                )
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
            except Exception as exc:
                failures.append(f"{device_name}: {exc.__class__.__name__}: {exc}")
                if role == "reference":
                    # Reclassify rather than silently skip: the next
                    # contributor must take over pacemaker duty (Triggerable).
                    promote_next_contributor = True
                    logger.warning(
                        "Free-run reference %s failed to connect; the next "
                        "synchronous device will be promoted to reference",
                        device_name,
                    )
                logger.warning(
                    "Failed to create/connect detector %s — skipping",
                    device_name,
                    exc_info=True,
                )
        if free_run and reference_configured and not reference:
            # Keep whatever connected so the caller's cleanup disconnects it.
            with self._device_lock:
                self._detectors = list(others)
            detail = "; ".join(failures) if failures else "no failure recorded"
            raise GeecsConfigurationError(
                "free_run_time_sync scan: the reference (pacemaker) device "
                "failed to connect and no other synchronous device could be "
                "promoted to reference — aborting instead of acquiring "
                f"unpaced data. Device failures: {detail}"
            )
        detectors = reference + others
        with self._device_lock:
            self._detectors = list(detectors)
        return detectors
