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
import threading
from contextlib import contextmanager
from typing import Any, Callable

from geecs_bluesky.devices.ca._pv import GATEWAY_DISCONNECTED
from geecs_bluesky.exceptions import GeecsConfigurationError

# Bound at module level (not via the events module) because hermetic tests
# monkeypatch these names — the `is None` guards downstream are that seam.
from geecs_bluesky.events import (
    DialogRequest,
    ScanDialogEvent,
    ScanEvent,
    ScanLifecycleEvent,
    ScanState,
    ScanStepEvent,
)
from geecs_bluesky.models.shot_control import ShotControlConfig, ShotControlWrites
from geecs_bluesky.operator_channel import (
    EventStreamOperator,
    NullOperator,
    OperatorChannel,
    OperatorQuestion,
)
from geecs_bluesky.plans.run_wrapper import claim_scan, claim_scan_number
from geecs_bluesky.preflight import (
    LABVIEW_EPOCH_OFFSET,
    FreeRunStalenessCheck,
    GatewayLivenessCheck,
    PreflightContext,
    run_preflight,
)
from geecs_bluesky.scan_request_runner import (
    ConfigResolver,
    ConfigsRepoResolver,
    resolve_and_validate_actions,
    resolve_defaults_for,
    resolve_movable_target,
    resolve_save_sets_and_rituals,
    run_scan_request,
    trigger_writes_from_profile,
)
from geecs_bluesky.scan_log import log_claimed_scan_failure, scan_log
from geecs_bluesky.session import GeecsSession, _positions
from geecs_bluesky.utils import safe_name
from geecs_schemas import (
    AcquisitionMode,
    OptimizationSpec,
    ScanRequest,
    ScanRequestMode,
)

logger = logging.getLogger(__name__)

_DISCONNECT_TIMEOUT = 10.0
_THREAD_JOIN_TIMEOUT = 15.0

# LabVIEW→Unix epoch offset; owned by geecs_bluesky.preflight, aliased here
# because tests and external callers reference it at this path.
_LABVIEW_EPOCH_OFFSET = LABVIEW_EPOCH_OFFSET

# CONNECTED-PV read budget; on timeout/error the device counts as live
# (fail-open — an old gateway without status PVs must never block a scan).
_LIVENESS_READ_TIMEOUT_S = 2.0

# Free-run pre-flight freshness (free-run ONLY): a sync device with no cached
# frame newer than this is stale; all-CONNECTED-but-all-stale means the
# trigger is off / not free-running.
_STALE_SYNC_THRESHOLD_S = 10.0

# One stale verdict gets a second look after ~one trigger period (a fresh
# monitor may not have delivered its first frame yet).
_STALE_RECHECK_WAIT_S = 2.0

# Pre-flight dialog wait; on timeout the scan proceeds (fail-loud at t0
# sync) — an unattended scan must never hang on an unanswered dialog.
_PREFLIGHT_DIALOG_TIMEOUT_S = 30.0

_STRICT_MODE = "strict_shot_control"
_FREE_RUN_MODE = "free_run_time_sync"
_VALID_ACQUISITION_MODES = (_STRICT_MODE, _FREE_RUN_MODE)


def _cfg_field(cfg: Any, key: str, default: Any) -> Any:
    """Read *key* from a device config that may be a dict or a Pydantic model."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------


def _build_positions(scan_config: Any) -> list[float]:
    """Convert ScanConfig start/end/step to an explicit list of positions."""
    return _positions(
        float(scan_config.start), float(scan_config.end), float(scan_config.step)
    )


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
        Callable injected by the GUI layer for OPTIMIZATION scans; returns
        a bridge object exposing ``variable_names`` (``"Device:Variable"``
        VOCS keys) and ``bind(devices=..., scan_tag=..., scan_folder=...)
        -> (objective, suggester)`` for :meth:`GeecsSession.optimize`.
        Called with one argument whose type depends on the submission
        path: the legacy exec_config path passes the optimizer-config YAML
        path (``str``); the delegated ScanRequest path passes the
        request's resolved :class:`~geecs_schemas.OptimizationSpec` — a
        loader serving both paths dispatches on the argument type.  Lives
        on the GUI side because the config-driven optimizer stack (Xopt,
        evaluators, ScanAnalysis analyzers) belongs to
        ``geecs_scanner.optimization`` — this package cannot import it
        (dependency direction).  Without a loader, exec_config
        optimization scans are logged and skipped, and optimize-mode
        ScanRequests are refused at :meth:`reinitialize`.
    """

    def __init__(
        self,
        experiment_dir: str = "",
        shot_control_information: dict | None = None,
        tiled_uri: str | None = None,
        tiled_api_key: str | None = None,
        on_event: Callable[[ScanEvent], None] | None = None,
        optimization_loader: Callable[[str | OptimizationSpec], Any] | None = None,
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

        # GUI compatibility: the GUI reads this defensively for both backends
        # (never populated on the Bluesky path — reinitialize raises instead).
        self.last_reinit_error: str | None = None
        self._on_event = on_event
        self._current_state = self._scan_state("IDLE")
        self._abort_requested = False
        self._scan_finished = False

        self._scan_thread: threading.Thread | None = None
        self._scan_config: Any = None
        self._rep_rate_hz: float = 1.0
        self._shots_per_step: int = 10
        self._acquisition_mode: str = _STRICT_MODE
        self._devices_config: dict[str, Any] = {}
        self._optimization_loader = optimization_loader
        # ScanRequest entry (the acceptance seam of the schema milestone):
        # set by reinitialize(ScanRequest); None on the exec_config path.
        # The scan thread delegates a stored request to run_scan_request.
        self._scan_request: ScanRequest | None = None
        self._request_resolver: ConfigResolver | None = None

        # Catch a stale env loudly rather than silently changing behavior.
        legacy_backend = os.environ.get("GEECS_BLUESKY_DEVICE_BACKEND")
        if legacy_backend and legacy_backend.strip().lower() != "ca":
            raise ValueError(
                f"GEECS_BLUESKY_DEVICE_BACKEND={legacy_backend!r} is set, but "
                "the direct device backend was removed — devices are CA-backed "
                "via the GeecsCAGateway. Unset the variable (or set it to 'ca')."
            )

        self._total_shots: int = 0
        self._total_steps: int = 0
        self._completed_shots: int = 0
        # The claimed day-scoped scan number for the current scan (None until
        # claimed); stamped onto every lifecycle event via _set_state.
        self._scan_number: int | None = None
        # uid of the most recent run's start document (for the Tiled exporter)
        self._last_run_uid: str | None = None
        # Ownership tracking for the shared session RunEngine (issue #511):
        # the start-document uid of the run *this scanner* owns, plus the
        # descriptor uids belonging to it.  Foreign runs (e.g. a headless
        # GeecsSession scan driven directly on self._session) flow through
        # the same _on_document subscription and must not mutate GUI
        # progress; see _on_document for the claiming/matching rules.
        self._active_run_uid: str | None = None
        self._active_descriptor_uids: set[str] = set()

        # Devices held open between reinitialize() and scan completion
        self._motor: Any | None = None
        self._detectors: list = []

        # Shot control — validated from the shot_control_information YAML/dict
        # (legacy path).  reinitialize(ScanRequest) stores generalized
        # ShotControlWrites instead; GeecsSession.shot_control accepts both.
        self._shot_control: ShotControlConfig | ShotControlWrites | None = (
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

        Accepts either shape (duck-detected by type):

        - a ``ScanExecutionConfig`` (the GUI path, unchanged), or
        - a :class:`~geecs_schemas.scan_request.ScanRequest` — the one
          submission object of the target architecture; names are resolved
          through a :class:`~geecs_bluesky.scan_request_runner.ConfigResolver`
          (pass ``resolver=`` to override; defaults to
          :class:`~geecs_bluesky.scan_request_runner.ConfigsRepoResolver`).

        Parameters
        ----------
        exec_config : ScanExecutionConfig or ScanRequest
            Validated scan configuration produced by the GUI, or a schema
            scan request.

        Returns
        -------
        bool
            Always ``True`` (no hardware initialisation done here).
        """
        if isinstance(exec_config, ScanRequest):
            return self._reinitialize_from_scan_request(
                exec_config, resolver=_kwargs.get("resolver")
            )
        self._scan_request = None
        self._request_resolver = None
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
        self._total_steps = 0
        self._scan_number = None
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

    def _reinitialize_from_scan_request(
        self, request: ScanRequest, resolver: ConfigResolver | None = None
    ) -> bool:
        """Validate a ScanRequest fail-fast and store it for delegated execution.

        The scan thread hands a stored request to
        :func:`~geecs_bluesky.scan_request_runner.run_scan_request` (see
        :meth:`_run_delegated_request`), so the full schema surface —
        actions, entry rituals, multi-axis grids, db_scalars, telemetry —
        runs through the one engine definition.  This method only (a)
        resolves every referenced name so the GUI gets an immediate error,
        discarding the results, and (b) stores state.  The **original**
        pre-defaults request is stored — ``run_scan_request`` applies
        experiment defaults itself, so storing a post-defaults copy would
        apply them twice.  ``acquisition`` comes from the request —
        deliberately no env override, a request declares intent.  Optimize
        mode requires the GUI-injected ``optimization_loader`` (refused
        here otherwise — the loader cannot appear later); the scan thread
        hands the request's resolved ``OptimizationSpec`` to the loader and
        threads the returned bridge's ``bind`` into the delegated runner
        (see :meth:`_run_delegated_request`).

        Returns ``True`` (matching :meth:`reinitialize`).
        """
        if resolver is None:
            resolver = ConfigsRepoResolver(self._experiment_dir)

        if request.mode is ScanRequestMode.OPTIMIZE and (
            self._optimization_loader is None
        ):
            raise NotImplementedError(
                "optimize-mode ScanRequest execution through BlueskyScanner "
                "needs the GUI-injected optimization_loader (the config-"
                "driven Xopt/evaluator stack lives in geecs_scanner."
                "optimization, which this package cannot import); construct "
                "the scanner with optimization_loader=..., or run headless "
                "via GeecsSession.run(request, resolver, objective=..., "
                "suggester=...)"
            )

        # Fail-fast validation on a LOCAL post-defaults copy; every result
        # is discarded — run_scan_request re-resolves at execution time.
        validated, _applied = resolve_defaults_for(resolver, request)
        resolve_and_validate_actions(validated.actions, resolver)
        if not validated.save_sets:
            raise GeecsConfigurationError(
                f"a {request.mode.value!r} ScanRequest needs at least one "
                "save set in save_sets — without one the scan would record "
                "nothing"
            )
        resolve_save_sets_and_rituals(resolver, validated.save_sets)
        if validated.trigger_profile:
            # Adapt (and discard) the writes so an unknown trigger_variant
            # fails here, not in the scan thread.
            profile = resolver.resolve_trigger_profile(validated.trigger_profile)
            trigger_writes_from_profile(profile, validated.trigger_variant)
        if validated.mode is ScanRequestMode.STEP:
            for axis in validated.axes:
                # Resolve to an executable target so pseudo (composite)
                # variables are refused here, not in the scan thread.
                spec = resolver.resolve_scan_variable(axis.variable)
                resolve_movable_target(spec, axis.variable)
        if validated.mode is ScanRequestMode.OPTIMIZE and validated.optimization:
            for name in validated.optimization.variables:
                # Catalog names must resolve (and pseudo variables are
                # refused) here, not in the scan thread; 'Device:Variable'
                # strings pass through, matching the runner's dispatch.
                if ":" not in name:
                    spec = resolver.resolve_scan_variable(name)
                    resolve_movable_target(spec, name)

        # Store the ORIGINAL pre-defaults request (see docstring).
        self._scan_request = request
        self._request_resolver = resolver
        self._scan_config = None
        self._devices_config = {}
        self._acquisition_mode = (
            _FREE_RUN_MODE
            if request.acquisition is AcquisitionMode.FREE_RUN
            else _STRICT_MODE
        )
        self._shots_per_step = int(request.shots_per_step)
        self._completed_shots = 0
        self._total_shots = 0
        self._total_steps = 0
        self._scan_number = None
        self._session.rep_rate_hz = self._rep_rate_hz

        # Disconnect any leftover devices from a previous scan
        self._disconnect_devices_sync()

        logger.info(
            "BlueskyScanner reinitialised from ScanRequest — mode=%s, "
            "acquisition=%s, shots_per_step=%d, save_sets=%s",
            request.mode.value,
            self._acquisition_mode,
            self._shots_per_step,
            list(request.save_sets),
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
        self._scan_number = None
        self._abort_requested = False
        self._scan_finished = False
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
        """Return ``True`` if a scan thread is currently running.

        ``False`` as soon as the scan's cleanup has completed — *before* the
        terminal DONE/ABORTED lifecycle event is emitted — so an event-driven
        GUI that re-checks this from its terminal-state handler never races
        the scan thread's last few instructions (the thread emits the event
        and exits milliseconds later; ``is_alive()`` alone reported ``True``
        in that window, leaving Start disabled until an operator clicked
        Stop).  A thread that is genuinely stuck mid-plan still reports
        active: the flag is only set once the ``finally`` cleanup ran.
        """
        return bool(
            self._scan_thread
            and self._scan_thread.is_alive()
            # getattr: hermetic tests build bare scanners via __new__.
            and not getattr(self, "_scan_finished", False)
        )

    def estimate_current_completion(self) -> float:
        """Return fraction complete (0.0–1.0) based on shots emitted."""
        if self._total_shots == 0:
            return 0.0
        return min(self._completed_shots / self._total_shots, 1.0)

    # ------------------------------------------------------------------
    # On-demand actions (G-actions v1) — the GUI contract
    # ------------------------------------------------------------------
    # These two signatures are mirrored by the console's Submitter protocol;
    # do not change them without flagging it loudly.

    def _action_resolver(self) -> ConfigResolver:
        """The resolver on-demand actions resolve against.

        The stored ScanRequest resolver when one is active (so converter
        state, e.g. extracted element plans, is shared), else a fresh
        configs-repo resolver over this scanner's experiment.
        """
        return self._request_resolver or ConfigsRepoResolver(self._experiment_dir)

    def run_action(self, name: str) -> None:
        """Execute the named ActionPlan on demand — refused while scanning.

        Thin delegation to :meth:`GeecsSession.run_action` with this
        bridge's resolver.  V1 during-scan behavior is refusal (the
        pause/decide/resume flow is issue #552).

        Raises
        ------
        RuntimeError
            While a scan is active: ``"scan in progress — action not
            started"`` (the GUI surfaces this message verbatim).
        """
        if self.is_scanning_active():
            raise RuntimeError("scan in progress — action not started")
        self._session.run_action(name, self._action_resolver())

    def describe_action(self, name: str) -> list[dict]:
        """Dry-run the named ActionPlan (no CA, no execution) — see the session.

        Thin delegation to :meth:`GeecsSession.describe_action` with this
        bridge's resolver.  Deliberately NOT refused while scanning: the
        dry-run is pure (zero CA), and "what would this action do?" is
        exactly the question an operator asks while a scan runs.  Only
        :meth:`run_action` carries the scan-in-progress refusal.
        """
        return self._session.describe_action(name, self._action_resolver())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_document(self, name: str, doc: dict) -> None:
        """Route RunEngine documents into GUI progress — owned runs only.

        The session RunEngine is shared: a foreign run (e.g. a headless
        ``GeecsSession`` scan driven directly on ``self._session``) flows
        through this same subscription.  A start document is claimed as
        *ours* only while this scanner is RUNNING — every scan path sets
        RUNNING before handing its plan to the (serial) RunEngine, so a
        start document arriving while the scanner is idle/done is foreign.
        Event documents carry a ``descriptor`` uid rather than the run
        start uid, so descriptors of the claimed run are collected and
        events are matched through them; foreign events never touch
        ``_completed_shots`` or emit ``ScanStepEvent`` s (issue #511).

        Residual limitation: a foreign run squeezed onto the RunEngine
        after this scanner enters RUNNING but before its own plan opens
        its run would still be mis-claimed.  The RunEngine executes plans
        serially, so that window is only the bridge's pre-plan setup; a
        full fix would need the plan to hand its run uid out-of-band.
        """
        if name == "start":
            if self._current_state != self._scan_state("RUNNING"):
                return  # foreign run — this scanner has no scan in flight
            self._active_run_uid = doc.get("uid")
            self._active_descriptor_uids = set()
            self._last_run_uid = doc.get("uid")
            self._note_claimed_scan_number(doc.get("scan_number"))
        elif name == "descriptor":
            uid = doc.get("uid")
            if (
                uid is not None
                and self._active_run_uid is not None
                and doc.get("run_start") == self._active_run_uid
            ):
                self._active_descriptor_uids.add(uid)
        elif name == "event":
            if doc.get("descriptor") not in self._active_descriptor_uids:
                return  # event from a run this scanner does not own
            self._completed_shots += 1
            self._emit_step_progress(doc)
        elif name == "stop":
            if (
                self._active_run_uid is not None
                and doc.get("run_start") == self._active_run_uid
            ):
                self._active_run_uid = None
                self._active_descriptor_uids = set()

    def _note_claimed_scan_number(self, scan_number: Any) -> None:
        """Carry an engine-claimed scan number into the lifecycle stream.

        On the delegated ScanRequest path the bridge never pre-claims —
        ``session.scan`` claims the number inside the engine and stamps it
        into the run start document (``geecs_run_wrapper`` metadata,
        ``EVENT_SCHEMA.md``).  The INITIALIZING/RUNNING lifecycle events
        were already emitted (pre-claim, ``scan_number=None``) by then, so
        pick the number up here and re-emit RUNNING carrying it — consumers
        get the number while the scan runs, not only on DONE/ABORTED.

        The exec_config paths set ``_scan_number`` at their own claim sites
        before any lifecycle emission, so this is a no-op there (the start
        document repeats the same number).  The RUNNING guard keeps a
        headless run on the shared RunEngine (not this scanner's scan) from
        flipping the GUI state.
        """
        if scan_number is None or self._scan_number is not None:
            return
        if self._current_state != self._scan_state("RUNNING"):
            return
        try:
            self._scan_number = int(scan_number)
        except (TypeError, ValueError):
            logger.debug(
                "start document carried a non-integer scan_number %r; ignoring",
                scan_number,
            )
            return
        self._set_state("RUNNING")

    def _emit_step_progress(self, doc: dict) -> None:
        """Emit a :class:`ScanStepEvent` for one Bluesky event document.

        Runs on the RunEngine thread — thread-safe the same way the lifecycle
        events are: the GUI's ``on_event`` hops to the Qt main thread via the
        ``_scan_event_received`` pyqtSignal, so events may be emitted from any
        thread.

        One event document is one completed shot, so every document carries a
        shot-level progress update (``phase="completed"``); the GUI computes
        its progress fraction from ``shots_completed`` against the
        ``total_shots`` it cached from the INITIALIZING lifecycle event.  The
        step index is derived from the schema-v1 ``bin_number`` column when
        present (1-based → 0-based), else 0.

        ``shots_completed`` is clamped at ``total_shots``: free-run scans emit
        one extra tail-flush event document (known, cosmetic overcount), and
        the progress bar must not report beyond 100 %.
        """
        if self._on_event is None or ScanStepEvent is None:
            return
        shots = self._completed_shots
        if self._total_shots:
            shots = min(shots, self._total_shots)
        data = doc.get("data") or {}
        try:
            step_index = int(data.get("bin_number", 1)) - 1
        except (TypeError, ValueError):
            step_index = 0
        step_index = max(step_index, 0)
        if self._total_steps:
            step_index = min(step_index, self._total_steps - 1)
        try:
            self._on_event(
                ScanStepEvent(
                    step_index=step_index,
                    total_steps=self._total_steps,
                    shots_completed=shots,
                    phase="completed",
                )
            )
        except Exception:
            logger.debug("on_event callback raised; ignoring", exc_info=True)

    @staticmethod
    def _scan_state(state_name: str):
        """Return a ScanState enum member when geecs_scanner is importable."""
        if ScanState is None:
            return state_name.lower()
        return getattr(ScanState, state_name)

    def _set_state(self, state_name: str, total_shots: int = 0) -> None:
        """Update lifecycle state and emit a GUI scan lifecycle event if possible.

        Every emission carries the claimed scan number (``None`` until the
        scan folder is claimed), so consumers can display "Scan NNN" from
        the lifecycle stream alone.
        """
        state = self._scan_state(state_name)
        self._current_state = state
        if self._on_event is None or ScanLifecycleEvent is None:
            return
        try:
            self._on_event(
                ScanLifecycleEvent(
                    state=state,
                    total_shots=total_shots,
                    scan_number=self._scan_number,
                )
            )
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

    # ------------------------------------------------------------------
    # Pre-flight: gateway liveness (both modes) + free-run staleness
    # ------------------------------------------------------------------

    @staticmethod
    def _device_label(device: Any) -> str:
        """Return the GEECS device name for operator-facing messages."""
        return str(
            getattr(device, "_geecs_device_name", getattr(device, "name", device))
        )

    def _read_gateway_liveness(self, device: Any) -> bool:
        """Read the device's gateway ``CONNECTED`` PV; ``False`` means down.

        The gateway serves every DB device's data PVs whether or not the
        device's TCP stream is up, so CA-connect success never implied
        liveness; the per-device ``CONNECTED`` status PV (enum
        ``Disconnected``/``Connected``, MAJOR while down — PV_CONTRACT.md
        §1/§5) is the authoritative, acquisition-mode-independent signal.

        Runs on the scan thread, so the signal's coroutine is dispatched to
        the RunEngine loop via ``run_coroutine_threadsafe`` (the same pattern
        as device connect/disconnect) with a short budget.  **Fail-open**: a
        missing ``connected_status`` attribute, a read error, or a timeout
        (e.g. an old gateway without status PVs) is logged at DEBUG and
        treated as live — liveness checking must never block a scan.  Only
        the exact ``Disconnected`` choice string reads as down.
        """
        signal = getattr(device, "connected_status", None)
        if signal is None:
            return True
        try:
            value = asyncio.run_coroutine_threadsafe(
                signal.get_value(), self._RE._loop
            ).result(timeout=_LIVENESS_READ_TIMEOUT_S)
        except Exception:
            logger.debug(
                "CONNECTED read failed for %s; treating as live (fail-open)",
                self._device_label(device),
                exc_info=True,
            )
            return True
        return str(value) != GATEWAY_DISCONNECTED

    def _operator_channel(self) -> OperatorChannel:
        """Return the operator channel matching this scanner's wiring.

        With an ``on_event`` consumer: an
        :class:`~geecs_bluesky.operator_channel.EventStreamOperator`
        reproducing the legacy dialog channel end to end (DialogRequest in a
        ScanDialogEvent, GUI answers via ``request.response_event``).
        Headless (``on_event=None``): a
        :class:`~geecs_bluesky.operator_channel.NullOperator` that returns
        each question's default immediately.

        Built per call (not cached) so the module-level ``ScanDialogEvent``
        / ``DialogRequest`` / ``_PREFLIGHT_DIALOG_TIMEOUT_S`` names are read
        at ask time — the hermetic tests monkeypatch them to simulate a
        consumer-less environment and shortened timeouts.
        """
        if self._on_event is None or ScanDialogEvent is None or DialogRequest is None:
            return NullOperator()
        return EventStreamOperator(
            self._on_event,
            dialog_event_type=ScanDialogEvent,
            request_type=DialogRequest,
            default_timeout=_PREFLIGHT_DIALOG_TIMEOUT_S,
        )

    def _request_operator_decision(
        self,
        exc: Exception,
        *,
        title: str,
        continue_label: str,
        abort_label: str = "Abort Scan",
        context: str | None = None,
    ) -> str:
        """Ask the operator one question through the injected channel.

        Kept as the scanner's one-question seam (ad-hoc callers outside the
        pre-flight pipeline).  Behavior is that of
        :class:`~geecs_bluesky.operator_channel.EventStreamOperator`:

        Returns
        -------
        str
            ``"continue"`` or ``"abort"`` when the consumer answered;
            ``"default"`` when there is no consumer (headless), the event
            types are unavailable, or no answer arrived within
            :data:`_PREFLIGHT_DIALOG_TIMEOUT_S` — callers must then preserve
            today's behavior (proceed, fail loudly later).
        """
        return self._operator_channel().ask(
            OperatorQuestion(
                message=str(exc),
                exc=exc,
                context=context,
                title=title,
                continue_label=continue_label,
                abort_label=abort_label,
                timeout=_PREFLIGHT_DIALOG_TIMEOUT_S,
            )
        )

    def _drop_devices(
        self, detectors: list, drop_ids: set[int], *, disconnect: bool = True
    ) -> list:
        """Remove the given devices (operator chose drop), disconnecting by default.

        ``disconnect=False`` is the delegated-request path: the runner's
        ``finally`` owns disconnection of everything it created, so the drop
        is logged but the device is left connected for that cleanup.
        """
        for device in detectors:
            if id(device) in drop_ids:
                logger.warning(
                    "Pre-flight: dropping device %s from this scan "
                    "(operator chose drop-and-continue)%s",
                    self._device_label(device),
                    "; disconnecting it" if disconnect else "",
                )
                if disconnect:
                    self._disconnect_device(device)
        remaining = [d for d in detectors if id(d) not in drop_ids]
        with self._device_lock:
            self._detectors = [d for d in self._detectors if id(d) not in drop_ids]
        return remaining

    def _preflight_check_sync_liveness(
        self, detectors: list, *, strict: bool = False, disconnect_on_drop: bool = True
    ) -> list | None:
        """Pre-flight: catch dead sync devices before the claim (pipeline).

        Thin call into :mod:`geecs_bluesky.preflight` (liveness + free-run
        staleness), pre-claim so an abort burns no scan number.  Headless /
        no answer → proceed and fail loudly downstream.  The module-level
        knobs are read at call time — the hermetic tests monkeypatch them.
        ``disconnect_on_drop=False`` (delegated-request path) keeps dropped
        devices connected for the runner's own cleanup.

        Returns
        -------
        list or None
            The (possibly reduced) detector list, or ``None`` on abort.
        """
        ctx = PreflightContext(
            detectors=detectors,
            strict=strict,
            read_liveness=self._read_gateway_liveness,
            drop_devices=lambda dets, ids: self._drop_devices(
                dets, ids, disconnect=disconnect_on_drop
            ),
            device_label=self._device_label,
            dialog_timeout=_PREFLIGHT_DIALOG_TIMEOUT_S,
        )
        checks = [
            GatewayLivenessCheck(),
            FreeRunStalenessCheck(
                threshold_s=_STALE_SYNC_THRESHOLD_S,
                recheck_wait_s=_STALE_RECHECK_WAIT_S,
            ),
        ]
        return run_preflight(checks, ctx, self._operator_channel())

    def _run_scan(self, scan_config: Any) -> None:
        """Scan thread body: create devices, run plan, clean up."""
        failed = False
        try:
            if self._abort_before_acquisition():
                return
            # ScanRequest path: delegate to the engine's request machinery
            # (scan_config is None on this path — this guard runs first).
            if getattr(self, "_scan_request", None) is not None:
                self._run_delegated_request()
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
        except Exception as exc:
            if self._abort_requested:
                # Operator-requested abort: anything unwinding out of the
                # scan here is part of the stop, not a failure — one INFO
                # line, no traceback (ABORTED state is set below either way).
                logger.info(
                    "BlueskyScanner: scan thread exiting after operator abort (%s)",
                    exc,
                )
            else:
                failed = True
                logger.exception("BlueskyScanner scan thread raised an exception")
        finally:
            self._disconnect_devices_sync()
            # Cleanup is complete: mark inactive BEFORE emitting the terminal
            # state, so a consumer reacting to DONE/ABORTED observes
            # is_scanning_active() == False (see its docstring).
            self._scan_finished = True
            if self._abort_requested or failed:
                self._set_state("ABORTED")
            else:
                self._set_state("DONE")
            logger.info("BlueskyScanner: scan thread finished")

    @contextmanager
    def _scan_log(self, scan_number: int | None, scan_folder: str | None):
        """Attach a per-scan log file (the shared scan_log helper).

        The implementation lives in :mod:`geecs_bluesky.scan_log` so headless
        :class:`~geecs_bluesky.session.GeecsSession` scans get the identical
        ``scan.log``; the bridge keeps this thin method because its scans
        pre-claim the number and wrap device building + the session call.
        The GeecsSession side only self-attaches when *it* claimed the
        number, so bridge scans never get a second (duplicating) handler.
        """
        with scan_log(scan_number, scan_folder):
            yield

    def _run_delegated_request(self) -> None:
        """Run the stored ScanRequest through the engine's one definition.

        Delegates to
        :func:`~geecs_bluesky.scan_request_runner.run_scan_request`.
        Bridge-specific facts: the two seams are passed as runner hooks
        (:meth:`_delegated_preflight`, :meth:`_on_delegated_scan_start`);
        the bridge must NOT pre-claim — the claim happens inside the runner
        (``session.scan``, or the runner itself pre-bind on optimize),
        which also owns the ``scan.log`` attach; exceptions propagate to
        :meth:`_run_scan`'s cleanup (ABORTED state + disconnect).

        Optimize-mode requests hand the resolved ``OptimizationSpec`` to
        the GUI-injected ``optimization_loader`` (presence enforced at
        :meth:`_reinitialize_from_scan_request`); the returned bridge's
        ``bind`` becomes the runner's ``optimization_binder``, and its
        optional ``finish()`` bookkeeping (e.g. the legacy
        ``xopt_dump.yaml``) runs after a successful run.
        """
        request = self._scan_request
        resolver = self._request_resolver or ConfigsRepoResolver(self._experiment_dir)
        optimization_binder = None
        opt_bridge: Any | None = None
        if request.mode is ScanRequestMode.OPTIMIZE:
            if self._optimization_loader is None:
                # reinitialize refused already; guard direct/stale callers.
                raise GeecsConfigurationError(
                    "optimize-mode ScanRequest reached the scan thread "
                    "without an optimization_loader"
                )
            opt_bridge = self._optimization_loader(request.optimization)
            optimization_binder = opt_bridge.bind
        run_scan_request(
            self._session,
            request,
            resolver,
            preflight=self._delegated_preflight,
            on_scan_start=self._on_delegated_scan_start,
            optimization_binder=optimization_binder,
        )
        if opt_bridge is not None and not getattr(
            self._session, "last_run_aborted", False
        ):
            # Post-run bookkeeping owned by the bridge (legacy parity with
            # _run_optimization); skipped on failure (exceptions propagate)
            # and on an operator abort (the quiet aborted-outcome return).
            finish = getattr(opt_bridge, "finish", None)
            if callable(finish):
                finish()

    def _delegated_preflight(self, detectors: list, strict: bool) -> list | None:
        """Runner preflight hook: the operator-dialog pipeline, sans disconnect.

        Same pipeline as the exec_config path, but dropped devices are not
        disconnected here — on the delegated path the runner's ``finally``
        owns disconnection of everything it created.  On abort (``None``)
        the abort flag is set so the scan thread's cleanup reports ABORTED.
        """
        checked = self._preflight_check_sync_liveness(
            detectors, strict=strict, disconnect_on_drop=False
        )
        if checked is None:
            self._abort_requested = True
        return checked

    def _on_delegated_scan_start(self, total_steps: int, total_shots: int) -> None:
        """Runner totals hook: prime GUI progress and emit lifecycle states."""
        self._total_steps = total_steps
        self._total_shots = total_shots
        self._set_state("INITIALIZING", total_shots=total_shots)
        self._set_state("RUNNING")

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

        Legacy parity (``ScanManager`` → ``load_from_dictionary``): every
        device the objective needs is acquired and natively saved without
        being on the GUI save list.  A required device absent from the GUI
        list is appended after the GUI devices (the pacemaker choice is
        unchanged); one already listed keeps its GUI config and only gains
        missing variables.

        Device names match **case-insensitively** (``str.casefold``) —
        GEECS itself is case-inconsistent about device-name spelling, and a
        wrong-case duplicate entry fails every CA PV connect (CA names are
        case-sensitive; live-observed).  On a hit the requirement merges
        under the GUI's spelling, logged at INFO.

        *requirements* is duck-typed: a ``{"Devices": ...}`` mapping, or
        ``None``/empty (unchanged behavior).
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
        # Lifecycle events emitted from here on carry the claimed number.
        self._scan_number = scan_number

        try:
            objective, suggester = bridge.bind(
                devices=list(variables.values()) + detectors,
                scan_tag=scan_tag,
                scan_folder=scan_folder,
            )

            max_iterations = len(_build_positions(scan_config))
            n_shots = max_iterations * self._shots_per_step
            self._total_shots = n_shots
            self._total_steps = max_iterations
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
                if getattr(self._session, "last_run_aborted", False):
                    # Aborted outcome returned quietly: note the folder
                    # calmly and skip finish() (legacy parity — the
                    # exception path never ran it on abort either).
                    log_claimed_scan_failure(scan_number, scan_folder, aborted=True)
                else:
                    # Post-run bookkeeping owned by the bridge (e.g. the
                    # legacy xopt_dump.yaml written into the scan folder).
                    finish = getattr(bridge, "finish", None)
                    if callable(finish):
                        finish()
        except BaseException:
            log_claimed_scan_failure(
                scan_number, scan_folder, aborted=self._abort_requested
            )
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

        checked = self._preflight_check_sync_liveness(detectors, strict=strict)
        if checked is None:
            # Mirror the _abort_before_acquisition path: setting the flag
            # makes the scan thread's cleanup report ABORTED and
            # disconnect every connected device — before any claim.
            self._abort_requested = True
            return
        detectors = checked

        if self._abort_before_acquisition():
            return

        scan_number, scan_folder = claim_scan_number(self._experiment_dir)
        # Lifecycle events emitted from here on carry the claimed number.
        self._scan_number = scan_number

        n_shots = len(positions) * self._shots_per_step
        self._total_shots = n_shots
        self._total_steps = len(positions)
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
            log_claimed_scan_failure(
                scan_number, scan_folder, aborted=self._abort_requested
            )
            raise
        if getattr(self._session, "last_run_aborted", False):
            # session.scan returned the aborted outcome quietly; note the
            # claimed-but-partial folder calmly (WARNING, not ERROR).
            log_claimed_scan_failure(scan_number, scan_folder, aborted=True)

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
