"""geecs_scan_request_plan — "run this ScanRequest" as one Bluesky plan.

The queueserver migration's one structural unit
(``Planning/cutover_strategy/02_queueserver_migration.md`` § The one
structural task): a plan generator whose **preamble** relocates the
:func:`~geecs_bluesky.scan_request_runner.run_scan_request` prologue to the
worker side — validate → resolve save sets/actions/trigger → construct +
connect devices → claim the scan number — and then yields the same inner
plan (:func:`~geecs_bluesky.plans.orchestration.build_step_scan_plan`) a
``run_scan_request`` scan runs today.  Because the preamble runs *inside*
the plan, every bound method and closure (ShotController stubs, per-step
action callables) is constructed worker-side from the JSON request and
never crosses a process boundary.

The prologue pieces are the runner's own module-level functions — shared,
not copied — so the legacy entry point (which the console keeps using until
Round 3) and this plan cannot drift.  Differences from the legacy path are
deliberate, per the Planning doc's decisions and 2026-08-20 amendments:

- **Validation runs here, authoritatively** (amendment 2: no separate
  validation plan stub; clients re-run :func:`validate_scan_request`
  pre-submit for immediate feedback).
- **No operator seams.**  Pre-flight questions move client-side pre-submit
  (decision 3), so the unserved-variables check runs with the headless
  default (continue-and-drop with a WARNING); the GUI ``preflight`` /
  ``on_scan_start`` / ``should_abort`` / ``pause_supervisor`` hooks have no
  plan-side equivalent (the manager's queue/status API owns stop/pause,
  decision 4; progress rides the document stream).
- **Device connects are plan messages.**  The session factories connect via
  ``run_coroutine_threadsafe`` onto the RE loop — a deadlock from inside a
  plan — so construction is deferred through a session facade and the
  batch connects ride :func:`ophyd_async.plan_stubs.ensure_connected` /
  ``bps.wait_for``, still pre-claim (a connect failure burns no number).
- **s-file export is not the plan's job**: it becomes a worker-side
  stop-document callback (a parallel task owns that seam in
  ``session.py``); the emitted documents are identical either way.
- **Optimize mode is refused loudly** (validated first).  Decision 5's
  in-worker optimization-loader invocation is a later round; step and
  noscan are this round's acceptance surface.

No queueserver import lives here — this is an ordinary plan callable; a
worker startup script registers it with the manager (and installs the
worker-wide default session via :func:`set_plan_session`) later.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from ophyd_async.plan_stubs import ensure_connected

from geecs_bluesky.config_resolver import ConfigResolver, ConfigsRepoResolver
from geecs_bluesky.db_runtime import select_telemetry_variables
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.orchestration import build_step_scan_plan
from geecs_bluesky.plans.run_wrapper import claim_scan_number
from geecs_bluesky.scan_log import (
    begin_pre_scan_capture,
    discard_pre_scan_capture,
    scan_log,
)
from geecs_bluesky.scan_request_runner import (
    _build_request_detectors,
    _defaults_flag,
    _preflight_unserved,
    assemble_action_slots,
    build_action_registry,
    build_movable,
    build_step_scan_spec,
    compile_action_slot,
    make_scalar_policy,
    prefetch_action_signals,
    resolve_movable_target,
    resolve_save_sets_and_rituals,
    save_set_to_devices_config,
    trigger_writes_from_profile,
    validate_scan_request,
    warn_if_reserved_boundary_overrides,
)
from geecs_bluesky.shot_controller import ShotController
from geecs_schemas import AcquisitionMode, ScanRequest, ScanRequestMode

logger = logging.getLogger(__name__)

__all__ = ["geecs_scan_request_plan", "set_plan_session"]

#: Per-batch device connect budget — parity with ``GeecsSession._connect``.
_CONNECT_TIMEOUT = 20.0

#: The session factory methods whose *construction* code the plan reuses
#: verbatim (only the connect step is deferred — see
#: :class:`_DeferredConnectFactories`).
_SESSION_FACTORIES = frozenset(
    {
        "detector",
        "contributor",
        "snapshot",
        "motor",
        "settable",
        "confirm_settable",
        "pseudo_movable",
        "action_signal_factory",
    }
)

#: The worker-wide default session (installed once at worker startup).
_worker_session: Any | None = None


def set_plan_session(session: Any | None) -> None:
    """Install the worker-wide default :class:`GeecsSession` for the plan.

    The queueserver registers plans by name with JSON args, so the session
    cannot travel in the call — a worker startup script constructs one
    headless session and installs it here; ``RE(geecs_scan_request_plan(
    request))`` then needs nothing else.  Pass ``None`` to clear (tests).

    Parameters
    ----------
    session :
        The session whose RunEngine will execute the plan.  The plan's
        connect messages run on that engine's loop, so running the plan on
        a *different* RunEngine than ``session.RE`` is unsupported.
    """
    global _worker_session
    _worker_session = session


class _DeferredConnectFactories:
    """A session facade: identical device construction, connect deferred.

    The session's factory methods (``detector``, ``motor``, …) are the one
    definition of how request devices are constructed — but each ends in
    ``self._connect(device)``, a blocking hop onto the RE loop that would
    deadlock inside a plan.  This facade binds those *same* class functions
    to itself (so the construction code cannot drift from the session's)
    and swaps only ``_connect``: devices are recorded on :attr:`created`
    and connected later in one in-plan batch.  Everything else (attributes
    like ``experiment`` / ``_mock`` / ``rep_rate_hz``) delegates to the
    wrapped session.

    Deliberately *not* covering ``telemetry``/``telemetry_batch``: the soft
    tier's connect-failure-drops semantics need their own in-plan gather
    (:func:`_connect_telemetry_plan`), not the strict batch connect.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self.created: list = []

    def _connect(self, device: Any) -> Any:
        self.created.append(device)
        return device

    def __getattr__(self, name: str) -> Any:
        if name in _SESSION_FACTORIES:
            func = getattr(type(self._session), name, None)
            if func is not None:
                return func.__get__(self, type(self))
        return getattr(self._session, name)


def _await_in_plan(coro_fn: Any):
    """Plan stub: await one no-arg coroutine function, propagating its error.

    ``bps.wait_for`` alone parks the plan on the future but discards its
    outcome; re-raising through ``task.result()`` keeps in-plan connects
    fail-fast (the ophyd-async ``wait_for_awaitable`` idiom, not exported
    by the pinned release).
    """
    tasks = yield from bps.wait_for([coro_fn])
    return tasks[0].result()


def _connect_in_batches(devices: list, *, mock: bool):
    """Plan stub: strict-connect *devices* via ``ensure_connected``.

    ``ensure_connected`` refuses duplicate device names within one call, and
    an action-check signal can legitimately share a name with a scan-axis
    movable on the same ``Device:Variable`` — so the list is split into
    unique-name batches instead of failing the scan on a naming accident.
    Failures propagate (strict tier fails loudly), pre-claim.
    """
    pending = list(devices)
    while pending:
        batch: list = []
        rest: list = []
        seen: set[str] = set()
        for device in pending:
            if device.name in seen:
                rest.append(device)
            else:
                seen.add(device.name)
                batch.append(device)
        yield from ensure_connected(*batch, mock=mock, timeout=_CONNECT_TIMEOUT)
        pending = rest


def _connect_telemetry_plan(session: Any, save_set: Any, scalar_policy: Any):
    """Plan stub: build + soft-connect the Tier-2 telemetry group in-plan.

    The in-plan sibling of the runner's
    :func:`~geecs_bluesky.scan_request_runner.build_telemetry_readables`
    over ``session.telemetry_batch`` (which hops onto the RE loop — a
    deadlock from inside a plan): same selection
    (:func:`~geecs_bluesky.db_runtime.select_telemetry_variables`), same
    soft-tier contract (a device unreachable at scan start is dropped with
    a warning, never an abort; only connected devices are recorded), same
    single :class:`~geecs_bluesky.devices.ca.telemetry.CaTelemetryGroup`
    per scan.  One concurrent gather, so wall time is the slowest device.

    Returns
    -------
    tuple
        ``(readables, recorded)`` as ``build_telemetry_readables``.
    """
    if scalar_policy is None:
        return [], {}
    selected = select_telemetry_variables(
        save_set, scalar_policy.subscribed_by_device()
    )
    if not selected:
        return [], {}
    # Lazy import: keeps this module importable without the `ca` extra
    # (same discipline as the runner).
    from geecs_bluesky.devices.ca.telemetry import CaTelemetryGroup, CaTelemetryReadable

    members = [
        CaTelemetryReadable(device, variables, experiment=session.experiment)
        for device, variables in selected.items()
    ]
    results: list = []

    async def _connect_all() -> None:
        results.extend(
            await asyncio.gather(
                *(m.connect(mock=session._mock) for m in members),
                return_exceptions=True,
            )
        )

    yield from bps.wait_for([_connect_all])

    connected: list = []
    recorded: dict[str, list[str]] = {}
    for member, variables, result in zip(members, selected.values(), results):
        if isinstance(result, BaseException):
            logger.warning(
                "Dropping background-telemetry device %s: unreachable at scan "
                "start (soft tier — never aborts the scan)",
                member._geecs_device_name,
                exc_info=result,
            )
        else:
            connected.append(member)
            recorded[member._geecs_device_name] = list(variables)
    if not connected:
        return [], recorded
    return [CaTelemetryGroup(connected)], recorded


def _disconnect_plan(created: list):
    """Plan stub: best-effort disconnect of everything the plan created.

    The in-plan counterpart of the runner's ``finally:
    session.disconnect(*created)`` (which hops onto the RE loop).  Runs as
    a finalize, so it executes on success, failure, and abort alike; each
    device's failure is swallowed (gather with exceptions returned) —
    cleanup never masks the plan's own outcome.
    """
    closers = [d for d in created if hasattr(d, "disconnect")]
    if not closers:
        return

    async def _disconnect_all() -> None:
        await asyncio.gather(*(d.disconnect() for d in closers), return_exceptions=True)

    yield from bps.wait_for([_disconnect_all])


def geecs_scan_request_plan(
    request: dict | ScanRequest,
    *,
    session: Any | None = None,
    resolver: ConfigResolver | None = None,
):
    """Run one ScanRequest as a single Bluesky plan (preamble + inner plan).

    ``RE(geecs_scan_request_plan(request))`` is the queueserver-shaped
    equivalent of :func:`~geecs_bluesky.scan_request_runner.run_scan_request`
    for step and noscan requests: the same documents, the same data tree,
    the same ScanInfo and per-scan ``scan.log`` — with every prologue stage
    relocated inside the plan.  Order of operations (everything before the
    claim is fail-fast and burns no scan number):

    1. validate (authoritative — the same :func:`validate_scan_request`
       clients run pre-submit) and resolve every name;
    2. construct the shot controller, action-signal stubs, detectors, and
       scan-axis movables worker-side (deferred connects);
    3. connect everything via plan messages (strict tier fails loudly,
       telemetry drops soft);
    4. claim the scan number, write ScanInfo, attach ``scan.log``;
    5. yield the same inner plan as today
       (:func:`~geecs_bluesky.plans.orchestration.build_step_scan_plan`).

    A finalize disconnects everything the plan created — success, failure,
    and abort alike.  The post-run s-file export is deliberately absent
    (worker stop-document callback; see the module docstring).

    Parameters
    ----------
    request :
        ``ScanRequest.model_dump()`` output (the queue's JSON shape) or an
        already-validated :class:`~geecs_schemas.ScanRequest`.
    session :
        The :class:`~geecs_bluesky.session.GeecsSession` whose RunEngine
        executes this plan.  Defaults to the worker-wide session installed
        by :func:`set_plan_session`.
    resolver :
        Name resolver; defaults to
        :class:`~geecs_bluesky.config_resolver.ConfigsRepoResolver` over
        the session's experiment.

    Raises
    ------
    GeecsConfigurationError
        No session configured, unresolvable names, a bad pseudo formula,
        a strict request without usable shot control, or an empty
        effective device set — all pre-claim.
    NotImplementedError
        Optimize-mode requests (validated first, refused loudly — a later
        round moves the optimization loader in-worker per decision 5).
    """
    if session is None:
        session = _worker_session
        if session is None:
            raise GeecsConfigurationError(
                "geecs_scan_request_plan has no session: install one with "
                "set_plan_session(...) at worker startup, or pass session=..."
            )
    if resolver is None:
        resolver = ConfigsRepoResolver(session.experiment)

    created: list = []
    # Pre-claim log lines (validation, connects, telemetry drops) buffer
    # into the scan.log attach at the claim, exactly as GeecsSession.run
    # does; a buffer never consumed (failure before the claim) is discarded.
    begin_pre_scan_capture()
    try:
        return (
            yield from bpp.finalize_wrapper(
                _scan_request_body(session, resolver, request, created),
                lambda: _disconnect_plan(created),
            )
        )
    finally:
        discard_pre_scan_capture()


def _scan_request_body(
    session: Any, resolver: ConfigResolver, request: dict | ScanRequest, created: list
):
    """The plan body behind :func:`geecs_scan_request_plan` (see its doc).

    *created* is the caller-owned cleanup list: everything appended here is
    disconnected by the enclosing finalize, so a failure at any stage still
    tears down whatever already existed.
    """
    # ---- phase 1: authoritative validation + pure resolution (pre-claim) --
    if not isinstance(request, ScanRequest):
        request = ScanRequest.model_validate(request)
    request, applied_defaults, defaults = validate_scan_request(request, resolver)

    if request.mode is ScanRequestMode.OPTIMIZE:
        raise NotImplementedError(
            "optimize-mode ScanRequests are not yet served by "
            "geecs_scan_request_plan (queueserver round 1 covers step and "
            "noscan; the in-worker optimization-loader invocation is a later "
            "round) — run them through GeecsSession.run / run_scan_request"
        )

    controller = None
    if request.trigger_profile:
        profile = resolver.resolve_trigger_profile(request.trigger_profile)
        writes = trigger_writes_from_profile(profile, request.trigger_variant)
        if writes.states:
            # Constructed worker-side, unconnected; the setter reachability
            # check joins the in-plan connect stage below.
            controller = ShotController.from_writes(
                writes,
                experiment=session.experiment,
                rep_rate_hz=session.rep_rate_hz,
            )
    strict = request.acquisition is AcquisitionMode.STRICT

    save_set, rituals = resolve_save_sets_and_rituals(resolver, request.save_sets)
    scalar_policy = make_scalar_policy(session)
    devices_config = save_set_to_devices_config(save_set, scalar_policy)
    # Unserved-variables check, headless by decision 3 (operator questions
    # are client-side pre-submit): continue-and-drop with a WARNING.
    checked_config, dropped_unserved, dropped_unserved_devices = _preflight_unserved(
        session, devices_config, None
    )
    if checked_config is None:  # defensive: the headless default never aborts
        raise GeecsConfigurationError(
            "unserved-variables pre-flight aborted the scan (pre-claim)"
        )
    devices_config = checked_config
    slots = assemble_action_slots(request.actions, applied_defaults, rituals)
    warn_if_reserved_boundary_overrides(save_set)
    axis_resolved = [
        resolve_movable_target(
            resolver.resolve_scan_variable(axis.variable), axis.variable
        )
        for axis in request.axes
    ]
    telemetry_enabled = (
        request.background_telemetry
        if request.background_telemetry is not None
        else _defaults_flag(defaults, "background_telemetry", True)
    )

    # ---- phase 2: worker-side construction (connects deferred) -----------
    factories = _DeferredConnectFactories(session)
    setup = per_step = closeout = None
    if any(slots.values()):
        factory = factories.action_signal_factory()
        created.append(factory)
        registry = build_action_registry(resolver)
        setup, setup_plans = compile_action_slot(
            slots["setup"], resolver, registry, factory
        )
        per_step, per_step_plans = compile_action_slot(
            slots["per_step"], resolver, registry, factory
        )
        closeout, closeout_plans = compile_action_slot(
            slots["closeout"], resolver, registry, factory
        )
        # With the facade, "prefetch" records the signals for the batch
        # connect below — same fail-fast property, now a plan message.
        prefetch_action_signals(
            setup_plans + per_step_plans + closeout_plans, registry, factory
        )

    detectors = _build_request_detectors(factories, devices_config, free_run=not strict)
    movables = [build_movable(factories, target) for target in axis_resolved]
    created.extend(factories.created)

    # ---- phase 3: in-plan connects (still pre-claim) ----------------------
    if factories.created:
        yield from _connect_in_batches(factories.created, mock=session._mock)
    telemetry_selected: dict[str, list[str]] = {}
    telemetry_readables: list = []
    if telemetry_enabled:
        telemetry_readables, telemetry_selected = yield from _connect_telemetry_plan(
            session, save_set, scalar_policy
        )
        created.extend(telemetry_readables)
    # Telemetry is soft: appended as extra snapshot columns, never the
    # reference (index 0 stays the save set's).
    all_detectors = list(detectors) + telemetry_readables
    if controller is not None and not session._mock:
        # Fail fast on an unreachable shot-control PV (the session does this
        # at attach time; in-plan it joins the pre-claim connect stage).
        yield from _await_in_plan(controller.connect_setters)

    if not all_detectors:
        raise GeecsConfigurationError(
            "ScanRequest produced no detectors (empty effective device set) — "
            "nothing would be recorded"
        )
    if strict:
        if controller is None:
            raise GeecsConfigurationError(
                "strict_shot_control requires a reachable shot-control device. "
                "Use free-run mode for free-running trigger acquisition."
            )
        controller.require_strict_single_shot()

    spec = build_step_scan_spec(
        request,
        axis_resolved,
        applied_defaults=applied_defaults,
        slots=slots,
        dropped_unserved=dropped_unserved,
        dropped_unserved_devices=dropped_unserved_devices,
        telemetry_selected=telemetry_selected if telemetry_enabled else {},
    )
    if request.mode is ScanRequestMode.NOSCAN:
        motor_arg: Any = None
    elif len(movables) == 1:
        motor_arg = movables[0]
    else:
        motor_arg = movables

    # ---- phase 4: the claim boundary --------------------------------------
    scan_number, scan_folder = claim_scan_number(session.experiment)
    if scan_number is not None:
        session._write_scan_info(
            scan_number,
            scan_folder,
            motor=motor_arg,
            positions=spec.positions,
            shots_per_step=request.shots_per_step,
            description=request.description,
            overrides=spec.scan_info,
        )

    # Role wiring + native-save configuration, exactly as GeecsSession.scan.
    reference = all_detectors[0]
    for det in all_detectors:
        if hasattr(det, "configure_shot_id"):
            det.configure_shot_id(session.rep_rate_hz)
        if hasattr(det, "set_reference") and det is not reference:
            det.set_reference(reference)
    saving_detectors = session._configure_saving(
        all_detectors, scan_number, scan_folder
    )

    # ---- phase 5: the inner plans, exactly as today ------------------------
    inner = build_step_scan_plan(
        strict=strict,
        motor=motor_arg,
        positions=spec.positions,
        reference=reference,
        detectors=all_detectors,
        shots_per_step=request.shots_per_step,
        controller=controller,
        experiment=session.experiment,
        scan_number=scan_number,
        scan_folder=scan_folder,
        saving_detectors=saving_detectors,
        extra_md={"description": request.description, **spec.md},
        setup=setup,
        per_step=per_step,
        closeout=closeout,
    )
    # The plan claimed the number, so the plan owns the per-scan scan.log
    # (the GeecsSession.scan claimed-here rule; tolerates a failed claim).
    with scan_log(scan_number, scan_folder):
        return (yield from inner)
