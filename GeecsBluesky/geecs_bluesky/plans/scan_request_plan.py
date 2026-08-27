"""geecs_scan_request_plan — "run this ScanRequest" as one Bluesky plan.

The queueserver migration's one structural unit: a plan generator whose
**preamble** relocates the
:func:`~geecs_bluesky.scan_request_runner.run_scan_request` prologue to the
worker side — validate → resolve save sets/actions/trigger → construct +
connect devices → claim the scan number — and then yields the same inner
scan recipe a ``run_scan_request`` scan runs today (step/noscan through
:func:`~geecs_bluesky.plans.orchestration.build_step_scan_plan`, optimize
through :func:`~geecs_bluesky.plans.optimize.geecs_adaptive_scan`). Because
the preamble runs *inside* the plan, every bound method and closure
(ShotController stubs, per-step action callables, optimization objective /
suggester binding) is constructed worker-side from the JSON request and
never crosses a process boundary.

The prologue pieces are the runner's own module-level functions — shared,
not copied — so the headless entry point (``run_scan_request``, via
``GeecsSession.run``) and this plan cannot drift.  Differences from that
path are deliberate, per the queueserver-migration decisions (each glossed
inline below; ``GeecsBluesky/CLAUDE.md``'s worker section carries the
operational summary):

- **Validation runs here, authoritatively** — no separate validation plan
  stub; clients re-run :func:`validate_scan_request` pre-submit for
  immediate feedback (a queue makes submission-to-execution gaps long, so
  a typo must fail at submit).
- **No operator seams.**  Pre-flight questions move client-side pre-submit
  (decision 3), so the unserved-variables check runs with the headless
  default (continue-and-drop with a WARNING); the old GUI-bridge hooks
  died with the bridge (W5, #649) — the manager's queue/status API owns
  stop/pause (decision 4) and progress rides the document stream.
- **Device connects are plan messages.**  The session factories connect via
  ``run_coroutine_threadsafe`` onto the RE loop — a deadlock from inside a
  plan — so construction is deferred through a session facade and the
  batch connects ride :func:`ophyd_async.plan_stubs.ensure_connected` /
  ``bps.wait_for``, still pre-claim (a connect failure burns no number).
- **s-file export is not the plan's job**: it is a worker-side stop-document
  callback; the emitted documents are identical either way.
- **Optimize mode runs through the same plan** when the worker startup
  registers an optimization loader; headless installs without the
  ``optimize`` extra refuse optimize requests loudly before the claim.

No queueserver import lives here — this is an ordinary plan callable; a
worker startup script registers it with the manager (and installs the
worker-wide default session via :func:`set_plan_session`) later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from ophyd_async.plan_stubs import ensure_connected

from geecs_bluesky.config_resolver import ConfigResolver, ConfigsRepoResolver
from geecs_bluesky.db_runtime import select_telemetry_variables
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.optimize import BinData
from geecs_bluesky.plans.optimize import geecs_adaptive_scan
from geecs_bluesky.plans.orchestration import _with_pause_quiescer
from geecs_bluesky.plans.pause_semantics import ShotControlPauseQuiescer
from geecs_bluesky.plans.run_wrapper import (
    claim_scan,
    claim_scan_number,
    geecs_run_wrapper,
)
from geecs_bluesky.scan_log import (
    begin_pre_scan_capture,
    discard_pre_scan_capture,
    log_claimed_scan_failure,
    scan_log,
)
from geecs_bluesky.scan_request_runner import (
    PseudoMovableTarget,
    _build_request_detectors,
    _defaults_flag,
    _preflight_connected,
    apply_native_image_save_off,
    resolve_native_image_save,
    select_capture_devices,
    _preflight_unserved,
    assemble_action_slots,
    build_action_registry,
    build_movable,
    build_step_scan_spec,
    compile_action_slot,
    make_scalar_policy,
    metadata_applied_defaults,
    metadata_submission,
    merge_optimizer_device_requirements,
    prefetch_action_signals,
    resolve_movable_target,
    resolve_save_sets_and_rituals,
    save_set_to_devices_config,
    trigger_writes_from_profile,
    validate_scan_request,
    warn_if_reserved_boundary_overrides,
)
from geecs_bluesky.session import _json_safe
from geecs_bluesky.shot_controller import ShotController
from geecs_schemas import AcquisitionMode, ScanRequest, ScanRequestMode

logger = logging.getLogger(__name__)

__all__ = [
    "geecs_scan_request_plan",
    "geecs_run_action_plan",
    "set_plan_session",
    "set_optimization_loader",
]

#: Per-connect timeout for the in-plan strict batches.  Near-parity, not
#: parity, with ``GeecsSession._connect``: there 20 s is the outer wall on
#: the loop hop while each connect runs at the ophyd-async default (10 s),
#: so an unreachable device fails at ~10 s; here the 20 s *is* the
#: per-connect timeout, so the same failure surfaces ~10 s later.
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


#: The worker-wide optimization loader (installed once at worker startup,
#: when the `optimize` extra is present).  ``None`` means optimize-mode
#: requests are refused loudly (see the OPTIMIZE branch of
#: :func:`_scan_request_body`).
_optimization_loader: Callable[[Any], Any] | None = None


def set_optimization_loader(loader: Callable[[Any], Any] | None) -> None:
    """Install the worker-wide optimization loader for the plan.

    The loader seam (its one implementation:
    ``geecs_bluesky.optimization.worker_loader``): a worker
    startup script calls this once with a callable built from
    ``geecs_bluesky.optimization`` when the ``optimize`` extra is
    installed. The plan then calls ``loader(request.optimization)`` for
    every optimize-mode request it serves. Pass ``None`` to clear (tests,
    or a headless install without the extra) — the plan refuses
    optimize-mode requests loudly rather than silently degrading.

    Parameters
    ----------
    loader :
        ``loader(spec: geecs_schemas.OptimizationSpec) -> bridge`` where
        ``bridge`` exposes ``bind(devices=..., scan_tag=..., scan_folder=...)
        -> (objective, suggester)``, an optional ``device_requirements``
        property, and an optional ``finish()`` — the same shape as
        :class:`~geecs_bluesky.optimization.session_bridge.SessionOptimizationBridge`.
    """
    global _optimization_loader
    _optimization_loader = loader


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


# Signature constraint (queueserver contract): at `queue add` the RE Manager
# re-evaluates this signature's annotation *strings* in a bare namespace —
# any non-builtin name there (ScanRequest, ConfigResolver, even typing.Any)
# makes every submission fail validation with "`Model` is not fully defined".
# `request` must stay a plain `dict` and the keyword-only injection seams must
# stay unannotated; the real types are documented in the docstring.  Pinned by
# tests/test_qserver_startup.py::test_plan_signature_passes_manager_validation.
def geecs_scan_request_plan(
    request: dict,
    *,
    session=None,
    resolver=None,
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

    The preamble's I/O is deliberately *blocking* inside the generator:
    config-repo reads, the DB policy queries, the claim's folder listing,
    and the ScanInfo write all run in the RunEngine thread, so a pause or
    stop request is not serviced until the current call returns — the
    accepted consequence of relocating the prologue into the plan.  The
    worst case is a DB or data-share stall (off the lab network, the DB
    socket timeout is ~75 s).  Executor offload of these calls is the
    filed hardening follow-up if that window bites in operation.

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
        a strict request without usable shot control, an empty effective
        device set, or (optimize mode) no optimization loader registered
        (see :func:`set_optimization_loader`) — all pre-claim.

    Notes
    -----
    This signature deliberately breaks the repo's type-hints-on-public-
    functions rule: the RE Manager re-evaluates the annotation strings in
    a bare namespace at ``queue add``, so any non-builtin name here
    (``ScanRequest``, ``ConfigResolver``, even ``typing.Any``) makes every
    submission fail validation. See the comment above the ``def`` and the
    pin test named there before adding any annotation back.
    """
    if session is None:
        session = _worker_session
        if session is None:
            raise GeecsConfigurationError(
                "geecs_scan_request_plan has no session: install one with "
                "set_plan_session(...) at worker startup, or pass session=..."
            )
    # The session's manual-move mutual exclusion (PR #597 contract) must
    # travel to the queue path: a background-executed geecs_move_variable
    # leaves the manager IDLE while a move converges, so a queue start could
    # otherwise drive hardware mid-move (#652 review finding 1).
    session._refuse_if_manual_move("scan not started")
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

    if request.mode is ScanRequestMode.OPTIMIZE:
        return (
            yield from _optimize_request_body(
                session,
                resolver,
                request,
                controller,
                strict,
                created,
                applied_defaults=applied_defaults,
            )
        )

    save_set, rituals = resolve_save_sets_and_rituals(resolver, request.save_sets)
    scalar_policy = make_scalar_policy(session)
    devices_config = save_set_to_devices_config(save_set, scalar_policy)
    # Unserved-variables check, headless by decision 3 (operator questions
    # are client-side pre-submit): continue-and-drop with a WARNING.
    checked_config, dropped_unserved, dropped_unserved_devices = _preflight_unserved(
        session, devices_config
    )
    if checked_config is None:  # defensive: the headless default never aborts
        raise GeecsConfigurationError(
            "unserved-variables pre-flight aborted the scan (pre-claim)"
        )
    devices_config = checked_config
    # CONNECTED liveness re-check (#664): the client asked pre-submit, but
    # the queue's submission-to-execution gap is long — re-check here,
    # refusing only when a row could never complete.
    disconnected_devices = _preflight_connected(session, devices_config)
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
    native_image_save = resolve_native_image_save(request, defaults)
    capture_devices = select_capture_devices(
        getattr(session, "experiment", ""), devices_config
    )
    if capture_devices and not native_image_save:
        devices_config = apply_native_image_save_off(devices_config, capture_devices)
        logger.info(
            "native_image_save=off: capture daemon owns images for %s",
            ", ".join(capture_devices),
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
        disconnected_devices=disconnected_devices,
        telemetry_selected=telemetry_selected if telemetry_enabled else {},
        capture_devices=capture_devices,
        native_image_save=native_image_save,
    )
    if request.mode is ScanRequestMode.NOSCAN:
        motor_arg: Any = None
    elif len(movables) == 1:
        motor_arg = movables[0]
    else:
        motor_arg = movables

    # ---- phase 4: the claim boundary --------------------------------------
    scan_number, scan_folder = claim_scan_number(session.experiment)

    # ---- phase 5: the shared post-claim tail + inner plans -----------------
    # ScanInfo → role wiring → native-save configuration → the inner plan:
    # THE one definition, shared with GeecsSession.scan (the PR #639
    # disposition-row-5 extraction; this used to be a ~25-line copy).
    inner = session.build_claimed_scan_plan(
        scan_number=scan_number,
        scan_folder=scan_folder,
        detectors=all_detectors,
        motor=motor_arg,
        positions=spec.positions,
        shots_per_step=request.shots_per_step,
        strict=strict,
        controller=controller,
        description=request.description,
        md=spec.md,
        scan_info_overrides=spec.scan_info,
        setup=setup,
        per_step=per_step,
        closeout=closeout,
        # Decision-4 activation (issue #641, PR #645): the queueserver path
        # opts into pause-on-failed-move — the RE Manager renders the paused
        # state and resume/stop are first-class queue verbs.  Headless
        # session.scan keeps the 'raise' default: with no operator to
        # answer, a pause would hang the scan (see the helper's docstring).
        failed_move_policy="pause",
    )
    # The plan claimed the number, so the plan owns the per-scan scan.log
    # (the GeecsSession.scan claimed-here rule; tolerates a failed claim).
    with scan_log(scan_number, scan_folder):
        return (yield from inner)


def _flatten_move_targets(
    variables: dict[str, Any], targets: dict[str, float]
) -> list[Any]:
    """``{name: value}`` -> the flat ``(movable, value, ...)`` args ``bps.mv`` wants."""
    args: list[Any] = []
    for name, value in targets.items():
        args.append(variables[name])
        args.append(value)
    return args


def _optimize_request_body(
    session: Any,
    resolver: ConfigResolver,
    request: ScanRequest,
    controller: ShotController | None,
    strict: bool,
    created: list,
    *,
    applied_defaults: dict[str, Any] | None = None,
):
    """The optimize-mode body of :func:`_scan_request_body`.

    Mirrors :func:`~geecs_bluesky.scan_request_runner._run_optimize_request`
    (the binder path) in-plan: worker-side construction with deferred
    connects, the unserved-variables pre-flight over the *effective*
    device set (save-set devices plus the loader's ``device_requirements``),
    the claim boundary (the full :class:`ScanTag`, since the loader's
    analyzers may need it), then :func:`~geecs_bluesky.plans.optimize.geecs_adaptive_scan`
    wrapped exactly as :meth:`GeecsSession.optimize` wraps it — including its
    ``on_finish`` ladder: ``spec.move_to_best_on_finish`` maps onto
    ``on_finish="best"`` (move to the best-observed inputs on success,
    restore to initial values on abort/failure — never best on abort, same
    as :meth:`GeecsSession.optimize`'s docstring contract) or
    ``on_finish="hold"`` (leave the variables wherever the suggester last
    set them, in every outcome) when unset. The schema exposes only the
    boolean, not the three-way string, so ``on_finish="initial"`` alone has
    no plan-side equivalent.

    Deliberately dropped relative to ``GeecsSession.optimize``: the
    ``should_abort`` stop probe (no plan-side equivalent — under the
    queue, stop is the manager's verb; same as the step/noscan body
    above).

    Raises
    ------
    GeecsConfigurationError
        No optimization loader registered (see :func:`set_optimization_loader`),
        an empty effective device set (before *or* after the unserved-
        variables pre-flight), or a strict request without usable shot
        control — all pre-claim.
    """
    spec = request.optimization
    assert spec is not None  # guaranteed by ScanRequest validation

    loader = _optimization_loader
    if loader is None:
        raise GeecsConfigurationError(
            "optimize-mode ScanRequest reached geecs_scan_request_plan "
            "without an optimization loader registered — call "
            "set_optimization_loader(...) at worker startup with a loader "
            "built from geecs_bluesky.optimization (needs the `optimize` "
            "extra); headless installs without one refuse optimize-mode "
            "requests here rather than silently degrading"
        )
    opt_bridge = loader(spec)
    device_requirements = getattr(opt_bridge, "device_requirements", None)

    save_set, rituals = resolve_save_sets_and_rituals(resolver, request.save_sets)
    warn_if_reserved_boundary_overrides(save_set)
    skipped: dict[str, list[str]] = {}
    ritual_names = [n for names in rituals.values() for n in names]
    if ritual_names:
        # Optimize mode does not run action plans yet (mirrors
        # _run_optimize_request) — record rather than refuse.
        skipped["save_set_rituals"] = ritual_names
    scalar_policy = make_scalar_policy(session)
    devices_config = save_set_to_devices_config(save_set, scalar_policy)
    # Auto-provision the optimizer's device requirements (issue #520's
    # reversal): the objective's diagnostics acquire and save even when the
    # request names no save sets naming them.
    provisioned = merge_optimizer_device_requirements(
        devices_config, device_requirements
    )
    if not devices_config:
        raise GeecsConfigurationError(
            "an 'optimize' ScanRequest needs at least one recording device "
            "— name save sets in save_sets, or use an optimizer whose "
            "evaluator declares device_requirements (auto-generated from "
            "its analyzers); without either the objective has nothing to "
            "read"
        )
    devices_config, dropped_unserved, dropped_unserved_devices = _preflight_unserved(
        session, devices_config
    )
    if devices_config is None:
        # No operator preflight hook on this path (same design position as
        # the step/noscan body) — _preflight_unserved never returns None
        # without one; defensive only.
        raise GeecsConfigurationError(
            "unserved-variables pre-flight aborted the optimization (pre-claim)"
        )
    if not devices_config:
        # The pre-flight check above can drop every device that WAS present
        # (unlike the emptiness check above it, which only catches an empty
        # save_sets/device_requirements union). An empty devices_config here
        # would otherwise reach `reference = detectors[0]` after the claim
        # boundary below (IndexError, burning a scan number for nothing).
        raise GeecsConfigurationError(
            "the unserved-variables pre-flight dropped every recording "
            "device for this optimization (pre-claim) — the objective "
            "would have nothing to read"
        )
    # CONNECTED liveness re-check (#664), same terms as the step/noscan path.
    disconnected_devices = _preflight_connected(session, devices_config)

    # ---- phase 2: worker-side construction (connects deferred) -----------
    factories = _DeferredConnectFactories(session)
    detectors = _build_request_detectors(factories, devices_config, free_run=not strict)
    variables: dict[str, Any] = {}
    pseudo_meta: dict[str, Any] = {}
    for name in spec.variables:
        if ":" in name:
            device, _, variable = name.partition(":")
            movable = factories.settable(device, variable)
        else:
            var_spec = resolver.resolve_scan_variable(name)
            target = resolve_movable_target(var_spec, name)
            if isinstance(target, PseudoMovableTarget):
                pseudo_meta[target.variable_name] = target.metadata()
            movable = build_movable(factories, target)
        variables[name] = movable
    created.extend(factories.created)

    # ---- phase 3: in-plan connects (still pre-claim) ----------------------
    if factories.created:
        yield from _connect_in_batches(factories.created, mock=session._mock)
    if controller is not None and not session._mock:
        yield from _await_in_plan(controller.connect_setters)

    if strict:
        if controller is None:
            raise GeecsConfigurationError(
                "strict_shot_control requires a reachable shot-control device. "
                "Use free-run mode for free-running trigger acquisition."
            )
        controller.require_strict_single_shot()

    md: dict[str, Any] = {"scan_request_mode": request.mode.value}
    if pseudo_meta:
        md["pseudo_variables"] = pseudo_meta
    if request.save_sets:
        md["save_sets"] = list(request.save_sets)
    if provisioned:
        md["provisioned_device_requirements"] = provisioned
    if dropped_unserved:
        md["dropped_unserved_variables"] = {
            dev: list(vars_) for dev, vars_ in dropped_unserved.items()
        }
    if dropped_unserved_devices:
        md["dropped_unserved_devices"] = list(dropped_unserved_devices)
    if disconnected_devices:
        md["disconnected_devices"] = list(disconnected_devices)
    if applied_defaults:
        md["applied_defaults"] = metadata_applied_defaults(applied_defaults)
    submission = metadata_submission(request)
    if submission is not None:
        md["submission"] = submission
    if skipped:
        md["skipped_action_plans"] = skipped
        logger.warning(
            "Optimize mode does not run action plans yet — skipping the "
            "following for this optimization (setup/per_step/closeout and "
            "save-set rituals do not run during optimization): %s",
            skipped,
        )
    # Telemetry is not wired into optimize yet; db_scalars above applies
    # regardless (mirrors _run_optimize_request's binder-path metadata).
    md["db_scan_runtime"] = {
        "db_scalars": "applied",
        "background_telemetry": "not_run_in_optimize",
    }

    # Initial values, read before the claim (mirrors GeecsSession.optimize):
    # the on_finish ladder below restores to these on abort/failure, and to
    # them (as a fallback) or the best-observed inputs on success.
    initial_values: dict[str, float] = {}
    for name, movable in variables.items():
        initial_values[name] = yield from bps.rd(movable)

    # ---- phase 4: the claim boundary --------------------------------------
    # The full ScanTag (not just the number): the loader's analyzers may
    # need it (mirrors _run_optimize_request's binder-path claim).
    scan_tag, scan_folder = claim_scan(session.experiment)
    scan_number = scan_tag.number if scan_tag is not None else None
    if scan_number is not None:
        session._write_scan_info(
            scan_number,
            scan_folder,
            motor=None,
            positions=[None],
            shots_per_step=request.shots_per_step,
            description=request.description,
            overrides={
                "scan_parameter": ",".join(variables),
                "scan_mode": "optimization",
                "shots": request.shots_per_step,
            },
        )

    try:
        objective, suggester = opt_bridge.bind(
            devices=list(variables.values()) + detectors,
            scan_tag=scan_tag,
            scan_folder=scan_folder,
        )

        reference = detectors[0]
        for det in detectors:
            if hasattr(det, "configure_shot_id"):
                det.configure_shot_id(session.rep_rate_hz)
        saving_detectors = session._configure_saving(
            detectors, scan_number, scan_folder
        )

        max_iterations = spec.max_iterations or 20

        # In-process row collection (plan-native equivalent of
        # GeecsSession.optimize's self.RE.subscribe(_collect)): the
        # objective evaluates between bins without a Tiled round trip.
        descriptors: dict[str, str] = {}
        rows: list[dict] = []
        history: list[dict] = []
        pending: dict[str, float] | None = None

        def _collect(name: str, doc: dict) -> None:
            if name == "descriptor":
                descriptors[doc["uid"]] = doc.get("name", "")
            elif name == "event" and descriptors.get(doc["descriptor"]) == "primary":
                rows.append(dict(doc["data"]))

        def _observe_previous(iteration: int) -> None:
            nonlocal pending
            if pending is None:
                return
            bin_rows = [r for r in rows if r.get("bin_number") == iteration]
            bin_data = BinData(iteration, bin_rows)
            try:
                value = float(objective(bin_data))
            except Exception:
                logger.warning(
                    "objective failed on iteration %d; recording NaN",
                    iteration,
                    exc_info=True,
                )
                value = float("nan")
            suggester.observe(pending, value, bin_data)
            history.append(
                {
                    "iteration": iteration,
                    "inputs": pending,
                    "objective": value,
                    "n_rows": len(bin_rows),
                }
            )
            pending = None

        def _propose(iteration: int) -> dict[str, float] | None:
            nonlocal pending
            _observe_previous(iteration - 1)
            inputs = suggester.suggest()
            if inputs is None:
                logger.info("suggester stopped at iteration %d", iteration)
                return None
            unknown = set(inputs) - set(variables)
            if unknown:
                raise KeyError(f"suggester proposed unknown variables: {unknown}")
            pending = inputs
            logger.info("iteration %d: %s", iteration, inputs)
            return inputs

        plan = geecs_adaptive_scan(
            movables=dict(variables),
            propose=_propose,
            detectors=detectors[1:] if not strict else detectors,
            reference=reference if not strict else None,
            shots_per_iteration=request.shots_per_step,
            max_iterations=max_iterations,
            fire_shot=controller.fire_shot if strict and controller else None,
            setup_trigger=(
                (lambda: controller.arm_single_shot(detectors))
                if strict and controller
                else None
            ),
            arm_trigger=controller.arm if not strict and controller else None,
            disarm_trigger=controller.disarm if not strict and controller else None,
            quiesce_trigger=controller.quiesce if not strict and controller else None,
            md={"description": request.description, **md},
        )
        run_plan = geecs_run_wrapper(
            plan,
            experiment=session.experiment,
            scan_number=scan_number,
            scan_folder=scan_folder,
            saving_detectors=saving_detectors,
            devices=detectors + list(variables.values()),
            extra_md={"description": request.description, **md},
        )
        if controller is not None:
            run_plan = bpp.finalize_wrapper(run_plan, controller.disarm())
            # Quiesce-on-pause for the optimize path (reviewer-flagged gap on
            # the 8d11fcd9 confirm): geecs_adaptive_scan never goes through
            # build_step_scan_plan, so without this an operator pause
            # mid-optimization leaves the trigger in SCAN with saving enabled
            # — the Gate-2 orphan-frame mode decision 1 exists to prevent.
            run_plan = _with_pause_quiescer(
                ShotControlPauseQuiescer(controller), run_plan
            )

        # The plan claimed the number, so the plan owns the per-scan
        # scan.log (mirrors the step/noscan body above).
        token = yield from bps.subscribe("all", _collect)
        try:
            with scan_log(scan_number, scan_folder):
                uid = yield from run_plan
        finally:
            yield from bps.unsubscribe(token)
        _observe_previous(len(history) + 1)  # final bin

        if spec.move_to_best_on_finish:
            finite = [h for h in history if math.isfinite(h["objective"])]
            if finite:
                target = max(finite, key=lambda h: h["objective"])["inputs"]
            else:
                logger.warning(
                    "move_to_best_on_finish is set but no finite objectives "
                    "were observed; restoring initial values instead"
                )
                target = initial_values
            yield from bps.mv(*_flatten_move_targets(variables, target))
            logger.info("optimize move_to_best_on_finish -> %s", target)

        if scan_folder is not None and history:
            # Same shape/guard as GeecsSession.optimize's optimization.json
            # (failed objectives are NaN in-process; allow_nan=False makes
            # any regression fail loudly instead of writing invalid JSON).
            try:
                path = Path(scan_folder) / "optimization.json"
                path.write_text(
                    json.dumps(_json_safe(history), indent=2, allow_nan=False)
                )
                logger.info("Optimization history written to %s", path)
            except Exception:
                logger.warning("Could not write optimization history", exc_info=True)

        # Success-only bookkeeping (mirrors session.optimize's success
        # path: skipped on failure — the except clause below re-raises
        # before reaching here — and, on this path, on abort too, since an
        # operator abort surfaces as an exception through the generator
        # chain rather than session.optimize's settled-and-quiet aborted
        # outcome).
        finish = getattr(opt_bridge, "finish", None)
        if finish is not None:
            try:
                finish()
            except Exception:
                logger.warning("optimization bridge finish() failed", exc_info=True)
    except BaseException:
        if scan_number is not None:
            log_claimed_scan_failure(
                scan_number, scan_folder, label="Optimization scan"
            )
        if spec.move_to_best_on_finish:
            # Abort restores *initial*, never best — same contract as
            # GeecsSession.optimize (docstring above). Best-effort: a
            # restore failure must not mask the original exception.
            try:
                yield from bps.mv(*_flatten_move_targets(variables, initial_values))
            except Exception:
                logger.warning(
                    "could not restore initial values after an optimize failure/abort",
                    exc_info=True,
                )
        raise

    return uid


# Same signature constraint as geecs_scan_request_plan (see the comment
# above its `def`): the RE Manager re-evaluates annotation strings in a bare
# namespace at `queue add`, so only builtin names may appear — `name: str`
# is safe, the keyword-only injection seams stay unannotated.
def geecs_run_action_plan(
    name: str,
    *,
    session=None,
    resolver=None,
):
    """Run one named ActionPlan as its own queue item (decision 2, #648).

    The queueserver replacement for on-demand action execution
    (:meth:`~geecs_bluesky.session.GeecsSession.run_action`, whose direct
    ``RE(...)`` call cannot run under a manager-owned RunEngine): the
    console's Actions menu submits this plan to the queue, so a manual
    action gets the same queue provenance, ordering, and idle-only
    semantics as any scan.  No run is opened — an action emits no
    documents, exactly like today's ``run_action``.

    Resolution, nested-``run`` flattening, signal prefetch, and the
    compiled step semantics are the same code paths the in-scan action
    slots use; connects happen via plan messages (the in-plan pattern of
    :func:`geecs_scan_request_plan`) and a finalize disconnects everything
    created — success, failure, and abort alike.

    Parameters
    ----------
    name :
        Action-plan name in the experiment's action library.
    session :
        The :class:`~geecs_bluesky.session.GeecsSession` whose signal
        factory builds the action targets.  Defaults to the worker-wide
        session installed by :func:`set_plan_session`.
    resolver :
        Name resolver; defaults to
        :class:`~geecs_bluesky.config_resolver.ConfigsRepoResolver` over
        the session's experiment.

    Raises
    ------
    GeecsConfigurationError
        No session configured, or an unknown plan name / bad nested
        reference (fail-fast, before any signal connects).
    """
    if session is None:
        session = _worker_session
        if session is None:
            raise GeecsConfigurationError(
                "geecs_run_action_plan has no session: install one with "
                "set_plan_session(...) at worker startup, or pass session=..."
            )
    # Same manual-move mutual exclusion as geecs_scan_request_plan (the
    # session contract must travel to the queue path — #652 review).
    session._refuse_if_manual_move("action not started")
    if resolver is None:
        resolver = ConfigsRepoResolver(session.experiment)

    registry = build_action_registry(resolver)
    factories = _DeferredConnectFactories(session)
    factory = factories.action_signal_factory()
    created: list = [factory]

    def _body():
        # Fail-fast: unknown names / nested cycles surface in resolution
        # and prefetch, before any connect message is emitted.
        stub, plans = compile_action_slot([name], resolver, registry, factory)
        if stub is None:  # pragma: no cover - [name] is never empty
            return
        prefetch_action_signals(plans, registry, factory)
        created.extend(factories.created)
        if factories.created:
            yield from _connect_in_batches(factories.created, mock=session._mock)
        logger.info("Running action %r (queue item)", name)
        yield from stub()
        logger.info("Action %r completed", name)

    return (yield from bpp.finalize_wrapper(_body(), lambda: _disconnect_plan(created)))
