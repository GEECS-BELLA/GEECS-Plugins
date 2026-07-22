"""BlueskyScanner — the GUI/console bridge onto the RunEngine scan engine.

The one submission shape is :class:`geecs_schemas.ScanRequest`:
``reinitialize(request)`` validates every referenced name fail-fast and
stores the request; ``start_scan_thread`` runs it through
:func:`~geecs_bluesky.scan_request_runner.run_scan_request` (the one engine
definition — actions, entry rituals, multi-axis grids, db_scalars,
telemetry, optimize).  The legacy ``exec_config`` duck-typed path was
deleted (G3, 2026-07-16, by owner decision — the old GUI path is
abandoned on this branch); submitting anything but a ``ScanRequest``
raises ``TypeError``.

What this class adds on top of the runner is the operator-facing seams:
lifecycle/step/dialog event emission (``on_event``), the pre-flight
operator-dialog pipeline (gateway liveness + free-run staleness), progress
totals for the GUI progress bar, thread management (start/stop/pause), and
the on-demand action-plan surface (``run_action`` / ``describe_action``).
These six methods — ``reinitialize``, ``start_scan_thread``,
``stop_scanning_thread``, ``is_scanning_active``, ``run_action``,
``describe_action`` — are the console's ``Submitter`` protocol.

The RunEngine is created once on ``__init__`` (owned by the underlying
:class:`~geecs_bluesky.session.GeecsSession`) and its internal event loop
persists for the lifetime of this object.

Usage (standalone)::

    from geecs_bluesky.scanner_bridge import BlueskyScanner
    from geecs_schemas import ScanRequest

    scanner = BlueskyScanner(experiment_dir="Undulator")
    request = ScanRequest.model_validate({
        "mode": "step",
        "shots_per_step": 5,
        "acquisition": "free_run",
        "save_sets": ["baseline"],
        "axes": [{"variable": "jet_z",
                  "positions": {"start": 4.0, "end": 6.0, "step": 0.5}}],
    })
    scanner.reinitialize(request)
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
from typing import Any, Callable

from geecs_bluesky.devices.ca._pv import GATEWAY_DISCONNECTED
from geecs_bluesky.exceptions import GeecsConfigurationError

# Bound at module level (not via the events module) because hermetic tests
# monkeypatch these names — the `is None` guards downstream are that seam.
from geecs_bluesky.events import (
    ActionDecisionRequest,
    DialogRequest,
    ScanDialogEvent,
    ScanEvent,
    ScanLifecycleEvent,
    ScanState,
    ScanStepEvent,
)
from geecs_bluesky.operator_channel import (
    ANSWER_ABORT,
    EventStreamOperator,
    NullOperator,
    OperatorChannel,
    OperatorQuestion,
)
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
    run_scan_request,
    trigger_writes_from_profile,
    validate_scan_request,
)
from geecs_bluesky.session import GeecsSession
from geecs_schemas import (
    OptimizationSpec,
    ScanRequest,
    ScanRequestMode,
)

logger = logging.getLogger(__name__)

# Bookkeeping-only join budget for stop_scanning_thread: long enough to
# reap a thread whose plan cleanup is finishing, short enough that a stop
# during initialization (the scan thread may sit in a 20 s device connect)
# never blocks the caller — completion is announced via the lifecycle
# event stream, not by this join (issue #571).
_STOP_JOIN_TIMEOUT = 2.0

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


class BlueskyScanner:
    """RunEngine-backed scan executor: the GUI/console ``Submitter`` engine.

    Parameters
    ----------
    experiment_dir:
        Experiment name (configs-repo folder / PV prefix); the default
        :class:`~geecs_bluesky.scan_request_runner.ConfigsRepoResolver`
        resolves request names against it.
    tiled_uri:
        URI of the Tiled catalog server (e.g. ``"http://192.168.6.14:8000"``).
        When provided, a :class:`~bluesky.callbacks.tiled_writer.TiledWriter`
        is subscribed to the RunEngine so every scan is persisted automatically.
        Requires ``tiled[client]`` to be installed; silently skipped if absent.
    tiled_api_key:
        API key for the Tiled server, if authentication is enabled.
    optimization_loader:
        Callable injected by the GUI layer for optimize-mode requests;
        called with the request's resolved
        :class:`~geecs_schemas.OptimizationSpec` and returns a bridge
        object exposing ``bind(devices=..., scan_tag=..., scan_folder=...)
        -> (objective, suggester)`` (plus the optional duck-typed
        ``device_requirements`` mapping and ``finish()`` bookkeeping).
        Lives on the GUI side because the config-driven optimizer stack
        (Xopt, evaluators, ScanAnalysis analyzers) belongs to
        ``geecs_scanner.optimization`` — this package cannot import it
        (dependency direction).  Without a loader, optimize-mode
        ScanRequests are refused at :meth:`reinitialize`.
    """

    def __init__(
        self,
        experiment_dir: str = "",
        tiled_uri: str | None = None,
        tiled_api_key: str | None = None,
        on_event: Callable[[ScanEvent], None] | None = None,
        optimization_loader: Callable[[OptimizationSpec], Any] | None = None,
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

        self._on_event = on_event
        self._current_state = self._scan_state("IDLE")
        self._abort_requested = False
        self._scan_finished = False

        self._scan_thread: threading.Thread | None = None
        self._optimization_loader = optimization_loader
        # The one submission shape: set by reinitialize(ScanRequest); the
        # scan thread delegates the stored request to run_scan_request.
        self._scan_request: ScanRequest | None = None
        self._request_resolver: ConfigResolver | None = None
        # The active run's pause supervisor (#552); set for the duration of
        # a delegated run, None otherwise — request_action_during_scan
        # refuses when it is None.
        self._pause_supervisor: Any | None = None

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
        # Ownership tracking for the shared session RunEngine (issue #511):
        # the start-document uid of the run *this scanner* owns, plus the
        # descriptor uids belonging to it.  Foreign runs (e.g. a headless
        # GeecsSession scan driven directly on self._session) flow through
        # the same _on_document subscription and must not mutate GUI
        # progress; see _on_document for the claiming/matching rules.
        self._active_run_uid: str | None = None
        self._active_descriptor_uids: set[str] = set()

        logger.info("BlueskyScanner initialised (RunEngine ready)")

    # ------------------------------------------------------------------
    # ScanManager-compatible public API
    # ------------------------------------------------------------------

    def reinitialize(
        self, request: ScanRequest, resolver: ConfigResolver | None = None
    ) -> bool:
        """Validate a ScanRequest fail-fast and store it for the next run.

        The one accepted submission shape is
        :class:`~geecs_schemas.scan_request.ScanRequest`; names are resolved
        through a :class:`~geecs_bluesky.scan_request_runner.ConfigResolver`
        (pass ``resolver=`` to override; defaults to
        :class:`~geecs_bluesky.scan_request_runner.ConfigsRepoResolver`).
        Anything else raises ``TypeError`` — the legacy duck-typed
        ``exec_config`` path was removed (G3).

        The scan thread hands the stored request to
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

        Returns
        -------
        bool
            Always ``True`` (no hardware initialisation done here).

        Raises
        ------
        TypeError
            When *request* is not a ``ScanRequest`` — the legacy
            exec_config path was removed; submit a
            ``geecs_schemas.ScanRequest``.
        """
        if not isinstance(request, ScanRequest):
            raise TypeError(
                "the legacy exec_config path was removed (G3); reinitialize "
                "accepts only a geecs_schemas.ScanRequest, got "
                f"{type(request).__name__} — submit a ScanRequest"
            )
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

        # Fail-fast validation through THE one definition of "what must
        # resolve" (scan_request_runner.validate_scan_request, issue #529);
        # results are discarded — run_scan_request re-resolves at execution
        # time (it runs the same function as its own first phase, so this
        # submission-time check can never drift from execution).
        validate_scan_request(request, resolver)

        # Store the ORIGINAL pre-defaults request (see docstring).
        self._scan_request = request
        self._request_resolver = resolver
        self._completed_shots = 0
        self._total_shots = 0
        self._total_steps = 0
        self._scan_number = None

        logger.info(
            "BlueskyScanner reinitialised from ScanRequest — mode=%s, "
            "acquisition=%s, shots_per_step=%d, save_sets=%s",
            request.mode.value,
            request.acquisition.value,
            int(request.shots_per_step),
            list(request.save_sets),
        )
        return True

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
            daemon=True,
            name="bluesky-scan",
        )
        self._scan_thread.start()
        logger.info("BlueskyScanner scan thread started")

    def stop_scanning_thread(self) -> None:
        """Request the running scan to stop; return promptly.

        Safe to call from any thread, including a GUI worker: this method
        never blocks on the scan winding down.  Completion is announced
        the native way — the scan thread's cleanup emits the terminal
        ABORTED/DONE :class:`ScanLifecycleEvent` through the event stream
        the GUI already consumes (a STOPPING event is emitted here first).

        How the stop lands, by phase (issue #571):

        - **Running plan**: ``RE.abort()`` directly — valid from
          ``running`` *and* ``paused``, and thread-safe (bluesky 1.15.0
          ``run_engine.py``: ``abort()`` dispatches ``_abort_coro`` onto
          the engine loop via ``__interrupter_helper``'s
          ``call_soon_threadsafe``; from ``running`` it cancels the plan
          task and returns without waiting on the plan's cleanup — no
          ``request_pause()`` needed first).
        - **Initialization** (idle engine — ``RE.abort()`` would raise
          ``TransitionError``): the ``_abort_requested`` flag is read by
          the delegated runner's init-stage checkpoints
          (``run_scan_request(should_abort=...)``, all pre-claim) and,
          for a stop landing between the last checkpoint and plan start,
          by the session's in-plan stop gate.

        If the scan thread does not exit within the short bookkeeping
        join, the handle is *kept* so :meth:`is_scanning_active` keeps
        reporting ``True``: clearing it would let a second
        ``start_scan_thread`` run the still-busy RunEngine under a live
        scan.

        With no active scan (e.g. the click raced the scan's natural
        completion) this is a logged no-op — no STOPPING event, so the
        already-emitted terminal state stays the last word.
        """
        logger.info("BlueskyScanner: abort requested")
        if not self.is_scanning_active():
            # Nothing to stop — the scan may have just finished naturally
            # (the click raced the last shot).  Emitting STOPPING here
            # would repaint the state pill *after* the terminal DONE/
            # ABORTED event and leave it stuck amber (#573 review); leave
            # the state stream alone and just reap a dead thread handle.
            thread = self._scan_thread
            if thread is not None and not thread.is_alive():
                self._scan_thread = None
            logger.info("BlueskyScanner: no active scan; nothing to stop")
            return
        self._abort_requested = True
        self._set_state("STOPPING")
        try:
            if self._RE.state == "paused":
                # A pause window (#552) is active: the scan thread is in
                # the supervisor, which reads _abort_requested and issues
                # RE.abort() from the RE's own thread context.  A second
                # abort here would race it and cancel the cleanup task
                # partway (finalize chain truncated).  Only set the flag.
                pass
            elif self._RE.state != "idle":
                # Idle engine: skip — the init-stage stop is covered by the
                # should_abort checkpoints + the session's stop gate (above).
                self._RE.abort(reason="stop_scanning_thread called")
        except Exception:
            logger.debug("RE.abort() raised (may not be running)", exc_info=True)
        thread = self._scan_thread
        if thread is not None:
            # Bookkeeping-only: never a completion wait (see docstring).
            thread.join(timeout=_STOP_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.info(
                    "BlueskyScanner: scan thread still finishing after the "
                    "%.0f s bookkeeping join (normal for a stop during "
                    "initialization); the scanner keeps reporting active "
                    "and the terminal lifecycle event will announce "
                    "completion",
                    _STOP_JOIN_TIMEOUT,
                )
            else:
                self._scan_thread = None

    # The old pause_scan/resume_scan were deleted (issue #552 PR-1): they
    # issued a HARD pause (request_pause(defer=False)) whose resume replays
    # the partial row — in strict mode refiring a physical shot.  The safe
    # operator pause is request_pause/request_resume below (deferred pause
    # at a checkpoint, driven through the pause supervisor); never
    # reintroduce a bare RE.resume() API under the old names.

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
    # On-demand actions (G-actions v1) + manual moves — the GUI contract
    # ------------------------------------------------------------------
    # These three signatures (run_action, describe_action, move_variable)
    # are mirrored by the console's Submitter protocol (GEECS-Console ≥
    # 0.19.0); do not change them without flagging it loudly.

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

    def move_variable(self, name: str, value: float, *, timeout: float = 60.0) -> dict:
        """Move a scan variable on demand — refused while scanning.

        Thin delegation to :meth:`GeecsSession.move_variable` with this
        bridge's resolver: *name* is a catalog scan-variable name (plain,
        confirm, or pseudo/composite) or a raw ``"Device:Variable"``
        string.  The move carries scan-identical completion semantics (via
        ``build_movable``), and a relative pseudo re-baselines from the
        targets' current positions on every call.

        Returns
        -------
        dict
            ``{"variable", "kind", "value", "targets"}`` — see the session.

        ``timeout`` is the overall wall-clock budget in seconds — raise
        it for legitimately slow moves (long stage travel).

        Raises
        ------
        RuntimeError
            While a scan is active (``"scan in progress — move not
            started"``) or another manual move is running (``"manual move
            in progress — move not started"``) — the GUI surfaces these
            messages verbatim.
        """
        if self.is_scanning_active():
            raise RuntimeError("scan in progress — move not started")
        return self._session.move_variable(
            name, value, self._action_resolver(), timeout=timeout
        )

    def request_pause(self) -> None:
        """Pause the running scan at its next safe point (operator Pause).

        The bare-pause counterpart of :meth:`request_action_during_scan`:
        no action is staged.  Marks the pause supervisor for a manual
        pause, emits ``PAUSING``, and asks the RunEngine to pause at its
        next checkpoint (deferred).  The supervisor drives the mode-safe
        state (free-run → ``OFF``, jet off; strict → nothing) and holds —
        non-modal, the GUI stays usable — until :meth:`request_resume` or a
        Stop.  A no-op (logged) when no scan is active.
        """
        if not self.is_scanning_active():
            logger.info("request_pause: no active scan; nothing to pause")
            return
        supervisor = self._pause_supervisor
        if supervisor is None:
            return
        supervisor.arm_manual_pause()
        logger.info("operator pause requested — pausing at next checkpoint")
        self._set_state("PAUSING")
        try:
            self._RE.request_pause(defer=True)
        except Exception:
            logger.debug("RE.request_pause() raised", exc_info=True)

    def request_resume(self) -> None:
        """Resume a scan paused by :meth:`request_pause` (operator Resume).

        Signals the parked pause supervisor (on the scan thread) to resume;
        it re-asserts the pre-pause shot-control state and the RunEngine
        rewinds to the checkpoint and continues.  A no-op when nothing is
        paused.
        """
        supervisor = self._pause_supervisor
        if supervisor is not None:
            supervisor.request_resume()
            logger.info("operator resume requested")

    def request_action_during_scan(self, name: str) -> None:
        """Request an action to run in the scan's pause window (G-actions v2).

        The during-scan counterpart of :meth:`run_action` (issue #552).
        Validated fail-fast on the calling (GUI) thread — unknown name,
        cycle, or unreachable target raises here, before anything pauses —
        then **refused if the action writes to the active scan's
        shot-control device(s)** (owner decision 11: an action must not
        perturb the trigger the scan is driving).  On success the action's
        flattened steps + a connected signal factory are staged on the
        pause supervisor, PAUSING is emitted, and the RunEngine is asked to
        pause at its next checkpoint (deferred); the supervisor then runs
        the operator's execute/ignore/abort decision on the scan thread.

        Raises
        ------
        RuntimeError
            No scan is active (``"no scan in progress — use run_action"``),
            or one is already awaiting the pause window.
        GeecsConfigurationError
            Unknown plan, unreachable target, or an action targeting the
            scan's shot-control device(s).
        ActionPlanNotFoundError, ActionPlanCycleError
            Bad nested reference / cycle.
        """
        from geecs_bluesky.pause_supervisor import PendingAction
        from geecs_bluesky.plans.action_compiler import flatten_action_steps
        from geecs_schemas.action_plan import CheckStep, SetStep

        if not self.is_scanning_active():
            raise RuntimeError("no scan in progress — use run_action")
        supervisor = self._pause_supervisor
        if supervisor is None:
            raise RuntimeError("no scan in progress — use run_action")

        resolver = self._action_resolver()
        plan, registry = self._session._resolve_action(name, resolver)
        steps = flatten_action_steps(plan, registry=registry)

        # Case-insensitive: GEECS configs disagree on device-name case, so
        # the guard folds both sides (same rule as merge_save_sets /
        # merge_optimizer_device_requirements) — a case-mismatched name
        # must not slip a shot-control write past decision 11.  CheckStep
        # (read-only) does not perturb the trigger, so only SetStep counts.
        guarded = {d.casefold() for d in self._guarded_shot_control_devices()}
        touched = {
            step.device
            for step, _ in steps
            if isinstance(step, SetStep) and step.device.casefold() in guarded
        }
        if touched:
            raise GeecsConfigurationError(
                f"action {name!r} writes to the scan's shot-control "
                f"device(s) {sorted(touched)} — refused during a scan so it "
                "cannot perturb the trigger (run it between scans instead)"
            )

        # Pre-connect every signal the steps touch (a lazy connect inside
        # the paused RE loop would deadlock), exactly as run_action does.
        factory = self._session.action_signal_factory()
        for step, _ in steps:
            if not isinstance(step, (SetStep, CheckStep)):
                continue
            try:
                if isinstance(step, SetStep):
                    factory.get_settable(step.device, step.variable)
                else:
                    factory.get_readable(step.device, step.variable)
            except Exception as exc:
                self._session.disconnect(factory)
                raise GeecsConfigurationError(
                    f"action {name!r}: cannot reach "
                    f"{step.device}:{step.variable} — {exc}"
                ) from exc

        pending = PendingAction(
            name=name,
            steps=steps,
            factory=factory,
            cleanup=lambda: self._session.disconnect(factory),
        )
        if not supervisor.set_pending(pending):
            self._session.disconnect(factory)
            raise RuntimeError("an action is already awaiting the pause window")

        logger.info("action %r requested during scan — pausing", name)
        self._set_state("PAUSING")
        try:
            self._RE.request_pause(defer=True)
        except Exception:
            # The scan may have ended between the active-check and here —
            # withdraw the staged action so nothing lingers.
            supervisor.take_unconsumed_pending()
            self._session.disconnect(factory)
            self._set_state("RUNNING")
            raise

    def _guarded_shot_control_devices(self) -> set[str]:
        """Device names the active scan's shot control drives (decision 11).

        Derived from the stored request's trigger profile (the writes the
        ShotController replays), so an action targeting any of them is
        refused during the scan.  Empty when the scan has no trigger
        profile (nothing to perturb).
        """
        request = self._scan_request
        if request is None or not getattr(request, "trigger_profile", None):
            return set()
        resolver = self._request_resolver or ConfigsRepoResolver(self._experiment_dir)
        try:
            profile = resolver.resolve_trigger_profile(request.trigger_profile)
            writes = trigger_writes_from_profile(profile, request.trigger_variant)
        except GeecsConfigurationError:
            # An unresolvable trigger profile is a scan-setup problem the
            # runner already surfaced; the guard degrades to "nothing to
            # protect" rather than masking it (narrow catch on purpose —
            # a programming error must not be swallowed here).
            logger.debug("could not resolve trigger devices for guard", exc_info=True)
            return set()
        devices: set[str] = set()
        for ordered in writes.states.values():
            devices.update(device for device, _var, _val in ordered)
        return devices

    def _make_pause_supervisor(self) -> Any:
        """Build the pause supervisor for the delegated run (#552)."""
        from geecs_bluesky.pause_supervisor import PauseSupervisor

        request = self._scan_request
        acquisition = getattr(
            getattr(request, "acquisition", None), "value", "free_run"
        )
        return PauseSupervisor(
            acquisition=acquisition,
            shot_controller=lambda: getattr(self._session, "_shot_controller", None),
            # With no event consumer, _ask_action_decision emits nowhere —
            # so hand the supervisor a None ask, which makes it default to
            # 'ignore' rather than park forever waiting for an answer that
            # can never arrive (a console-less bridge that still somehow
            # reaches a pause window).
            ask=self._ask_action_decision if self._on_event is not None else None,
            should_abort=lambda: self._abort_requested,
            on_state=lambda s: self._set_state(s.upper()),
        )

    def _ask_action_decision(self, request: "ActionDecisionRequest") -> None:
        """Deliver the three-way decision to the GUI (queued dialog event).

        Emits the request inside a ``ScanDialogEvent`` (the same transport
        as the pre-flight binary dialogs); the consumer renders the three
        choices and answers via ``request.response_event`` / ``verdict``.
        Headless / no-consumer installs never reach here — the supervisor's
        ``ask`` is ``None`` there.
        """
        if self._on_event is None or ScanDialogEvent is None:
            return
        try:
            self._on_event(ScanDialogEvent(request=request))
        except Exception:
            logger.debug("action-decision dialog emit raised; ignoring", exc_info=True)

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

        The bridge never pre-claims — ``session.scan`` claims the number
        inside the engine (the runner itself claims pre-bind on optimize)
        and stamps it into the run start document (``geecs_run_wrapper``
        metadata, ``EVENT_SCHEMA.md``).  The INITIALIZING/RUNNING lifecycle
        events were already emitted (pre-claim, ``scan_number=None``) by
        then, so pick the number up here and re-emit RUNNING carrying it —
        consumers get the number while the scan runs, not only on
        DONE/ABORTED.  The RUNNING guard keeps a headless run on the shared
        RunEngine (not this scanner's scan) from flipping the GUI state.
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

    def _abort_before_acquisition(self) -> bool:
        """Return ``True`` (logging loudly) if a stop arrived before the plan ran.

        ``RE.abort()`` cannot stop a scan that has not reached the RunEngine
        yet.  This is the scan thread's entry check (a stop that landed
        before the thread even started); the later initialization stages
        are covered by the ``should_abort`` hook the delegated runner
        consults between resolution, device connect, preflight, and the
        claim (see :meth:`_run_delegated_request`).
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

    def _drop_devices(self, detectors: list, drop_ids: set[int]) -> list:
        """Remove the given devices from the scan (operator chose drop).

        The dropped devices are left **connected**: the delegated runner's
        ``finally`` owns disconnection of everything it created, so the
        drop is logged and the device stays connected for that cleanup.
        """
        for device in detectors:
            if id(device) in drop_ids:
                logger.warning(
                    "Pre-flight: dropping device %s from this scan "
                    "(operator chose drop-and-continue)",
                    self._device_label(device),
                )
        return [d for d in detectors if id(d) not in drop_ids]

    def _preflight_check_sync_liveness(
        self, detectors: list, *, strict: bool = False
    ) -> list | None:
        """Pre-flight: catch dead sync devices before the claim (pipeline).

        Thin call into :mod:`geecs_bluesky.preflight` (liveness + free-run
        staleness), pre-claim so an abort burns no scan number.  Headless /
        no answer → proceed and fail loudly downstream.  The module-level
        knobs are read at call time — the hermetic tests monkeypatch them.
        Dropped devices stay connected (the runner's ``finally`` owns
        disconnection of everything it created).

        Returns
        -------
        list or None
            The (possibly reduced) detector list, or ``None`` on abort.
        """
        ctx = PreflightContext(
            detectors=detectors,
            strict=strict,
            read_liveness=self._read_gateway_liveness,
            drop_devices=self._drop_devices,
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

    def _run_scan(self) -> None:
        """Scan thread body: delegate the stored request, then clean up state."""
        failed = False
        try:
            if self._abort_before_acquisition():
                return
            if self._scan_request is None:
                raise GeecsConfigurationError(
                    "start_scan_thread ran with no stored ScanRequest — call "
                    "reinitialize(request) first"
                )
            self._run_delegated_request()
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
            # Cleanup is complete: mark inactive BEFORE emitting the terminal
            # state, so a consumer reacting to DONE/ABORTED observes
            # is_scanning_active() == False (see its docstring).  The runner's
            # own ``finally`` disconnected everything it created.
            self._scan_finished = True
            if self._abort_requested or failed:
                self._set_state("ABORTED")
            else:
                self._set_state("DONE")
            logger.info("BlueskyScanner: scan thread finished")

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
        :meth:`reinitialize`); the returned bridge's
        ``bind`` becomes the runner's ``optimization_binder``, its
        ``device_requirements`` (duck-typed, like ``finish``) are handed
        to the runner for auto-provisioning into the effective device set
        (the objective's diagnostics acquire and save without being named
        in the save sets), and its optional ``finish()`` bookkeeping (e.g.
        the legacy ``xopt_dump.yaml``) runs after a successful run.
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
        self._pause_supervisor = self._make_pause_supervisor()
        try:
            run_scan_request(
                self._session,
                request,
                resolver,
                preflight=self._delegated_preflight,
                on_scan_start=self._on_delegated_scan_start,
                optimization_binder=optimization_binder,
                device_requirements=(
                    getattr(opt_bridge, "device_requirements", None)
                    if opt_bridge is not None
                    else None
                ),
                operator_channel=self._delegated_operator_channel(),
                # Stop clicked during initialization: the runner consults
                # this between init stages (all pre-claim) and the session's
                # stop gate re-reads it at plan start — RE.abort() alone
                # cannot reach a scan not yet in the RunEngine (issue #571).
                should_abort=lambda: self._abort_requested,
                # The during-scan operator-action pause window (#552): a
                # deferred pause at a plan checkpoint hands the scan thread
                # to this supervisor (safe state → decide → act → restore).
                pause_supervisor=self._pause_supervisor,
            )
        finally:
            # A pause requested but never delivered (the scan ended first)
            # is the normal end-of-plan outcome recorded on #552 — withdraw
            # it, report it, release its factory.
            leftover = self._pause_supervisor.take_unconsumed_pending()
            if leftover is not None:
                logger.info(
                    "action %r was requested but the scan ended before the "
                    "pause landed — it was NOT run",
                    leftover.name,
                )
                try:
                    leftover.cleanup()
                except Exception:  # noqa: BLE001 — best-effort
                    logger.debug("leftover-action cleanup raised", exc_info=True)
            self._pause_supervisor = None
        if opt_bridge is not None and not getattr(
            self._session, "last_run_aborted", False
        ):
            # Post-run bookkeeping owned by the bridge; skipped on failure
            # (exceptions propagate) and on an operator abort (the quiet
            # aborted-outcome return).
            finish = getattr(opt_bridge, "finish", None)
            if callable(finish):
                finish()

    def _delegated_operator_channel(self) -> OperatorChannel:
        """Operator channel for the runner's own pre-flight questions.

        The delegated runner asks its config-level questions itself (today:
        the unserved-variables check, which runs before detectors exist so
        it cannot go through :meth:`_delegated_preflight`).  This wraps
        :meth:`_operator_channel` so an "abort" answer to a runner-asked
        question also sets the scanner's abort flag — the scan thread's
        cleanup then reports ABORTED instead of DONE, matching the
        detector-level pipeline's behavior.
        """
        inner = self._operator_channel()
        scanner = self

        class _AbortNotingChannel:
            def ask(self, question: OperatorQuestion) -> str:
                answer = inner.ask(question)
                if answer == ANSWER_ABORT:
                    scanner._abort_requested = True
                return answer

        return _AbortNotingChannel()

    def _delegated_preflight(self, detectors: list, strict: bool) -> list | None:
        """Runner preflight hook: the operator-dialog pipeline, sans disconnect.

        Dropped devices are not disconnected here — the runner's
        ``finally`` owns disconnection of everything it created.  On abort
        (``None``) the abort flag is set so the scan thread's cleanup
        reports ABORTED.
        """
        checked = self._preflight_check_sync_liveness(detectors, strict=strict)
        if checked is None:
            self._abort_requested = True
        return checked

    def _on_delegated_scan_start(self, total_steps: int, total_shots: int) -> None:
        """Runner totals hook: prime GUI progress and emit lifecycle states."""
        self._total_steps = total_steps
        self._total_shots = total_shots
        self._set_state("INITIALIZING", total_shots=total_shots)
        self._set_state("RUNNING")
