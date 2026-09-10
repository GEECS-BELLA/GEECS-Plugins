"""The GEECS scan preamble: everything a ScanRequest needs before the run opens.

Extracted verbatim from ``plans/scan_request_plan.py`` (GEECS-Plugins#807
phase 2) so that **one implementation has two callers**: the funnel plan
(:func:`~geecs_bluesky.plans.scan_request_plan.geecs_scan_request_plan`,
which keeps working unchanged until phase 5 retires it) and the RunEngine
preprocessor that gives stock ``bluesky.plans`` verbs the same preamble.

Order of operations — everything here is **pre-claim** and fail-fast, so a
failure burns no scan number:

1. :func:`resolve_request` — authoritative validation (the same
   ``validate_scan_request`` clients run pre-submit), the trigger profile →
   :class:`~geecs_bluesky.shot_controller.ShotController` (unconnected), and
   the strict/free-run decision.  Blocking config-repo I/O, no plan messages;
   shared with optimize mode, which branches in the caller.
2. :func:`prepare_step_scan` — save sets, scalar policy and devices config;
   the unserved and CONNECTED preflights; action slots; axis resolution; the
   capture toggle; worker-side construction with deferred connects; the
   in-plan connect batches (devices, telemetry, shot-control setters); the
   fail-fast gates; and the run-metadata spec.  Returns a
   :class:`PreparedScan`.

The caller claims the scan number and builds the inner plan (or, for the
preprocessor, forwards a metadata-injected ``open_run``).  The deliberately
blocking I/O of step 1–2 runs on the RunEngine thread, as documented on the
funnel plan.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import bluesky.plan_stubs as bps
from ophyd_async.plan_stubs import ensure_connected

from geecs_bluesky.config_resolver import ConfigResolver
from geecs_bluesky.db_runtime import select_telemetry_variables
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.scan_request_runner import (
    _build_request_detectors,
    _defaults_flag,
    _preflight_connected,
    _preflight_unserved,
    assemble_action_slots,
    build_action_registry,
    build_movable,
    build_step_scan_spec,
    compile_action_slot,
    make_scalar_policy,
    prefetch_action_signals,
    resolve_and_apply_capture_toggle,
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

__all__ = [
    "PreparedScan",
    "ResolvedRequest",
    "prepare_step_scan",
    "resolve_request",
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

    Same selection (:func:`~geecs_bluesky.db_runtime.select_telemetry_variables`),
    same soft-tier contract (a device unreachable at scan start is dropped
    with a warning, never an abort; only connected devices are recorded),
    one :class:`~geecs_bluesky.devices.ca.telemetry.CaTelemetryGroup` per
    scan.  One concurrent gather, so wall time is the slowest device.

    Returns
    -------
    tuple
        ``(readables, recorded)`` — the group (or nothing) to append to the
        read set, and ``{device: [variables]}`` of what actually connected.
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


def _namespace_movable(namespace: Any, target: Any) -> Any:
    """The namespace child for one resolved axis, or a refusal.

    ``build_movable`` (``scan_request_runner.py``) dispatches a catalog
    target onto three topologies — pseudo, confirm-elsewhere, and
    tolerance-checked motor.  The namespace builds its children from the DB
    alone and knows none of that yet (the catalog's ``kind: motor`` opt-in
    arrives with the axes in phase 3), so a target needing a topology the
    namespace cannot express is **refused**.  Silently degrading a
    confirming or motor axis to a fire-and-forget setpoint would let a stock
    ``list_scan`` step to the next point before the readback converged —
    every row taken at the wrong position, and nothing in the data saying so.
    """
    from geecs_bluesky.devices.ca.motor import CaMotor

    label = getattr(target, "label", target)
    if not hasattr(target, "device"):
        raise GeecsConfigurationError(
            f"{label}: pseudo/composite scan variables are not supported on the "
            "stock-plan door yet (the namespace builds no pseudo axes until "
            "GEECS-Plugins#807 phase 3). Submit it through geecs_scan_request_plan."
        )
    if getattr(target, "confirm", None) is not None:
        raise GeecsConfigurationError(
            f"{label}: this scan variable confirms on {target.confirm!r}, a "
            "topology the device namespace does not build yet "
            "(GEECS-Plugins#807 phase 3). Submit it through "
            "geecs_scan_request_plan, which builds a CaConfirmSettable."
        )
    movable = namespace.variable(target.device, target.variable)
    if getattr(target, "kind", None) == "motor" and not isinstance(movable, CaMotor):
        raise GeecsConfigurationError(
            f"{label}: the catalog declares kind: motor, but the namespace built "
            "a plain setpoint because the DB carries no positive tolerance for "
            "it — a move would complete before the readback converged. Set the "
            "devicetype_variable tolerance in the GEECS DB."
        )
    return movable


@dataclass(frozen=True)
class ResolvedRequest:
    """What :func:`resolve_request` settles before the mode branch."""

    request: ScanRequest
    applied_defaults: Any
    defaults: Any
    controller: ShotController | None
    strict: bool


@dataclass
class PreparedScan:
    """Everything :func:`prepare_step_scan` produced, ready for the claim.

    ``as_claimed_kwargs()`` is the keyword set
    :meth:`~geecs_bluesky.session.GeecsSession.build_claimed_scan_plan`
    takes, so the funnel's call site stays a one-liner.
    """

    request: ScanRequest
    controller: ShotController | None
    strict: bool
    detectors: list
    motor_arg: Any
    spec: Any
    setup: Any = None
    per_step: Any = None
    closeout: Any = None
    telemetry_selected: dict = field(default_factory=dict)
    telemetry_readables: list = field(default_factory=list)

    def as_claimed_kwargs(self) -> dict:
        """The ``build_claimed_scan_plan`` keywords this preparation implies."""
        return {
            "detectors": self.detectors,
            "motor": self.motor_arg,
            "positions": self.spec.positions,
            "shots_per_step": self.request.capture.shots_per_step,
            "strict": self.strict,
            "controller": self.controller,
            "description": self.request.description,
            "md": self.spec.md,
            "scan_info_overrides": self.spec.scan_info,
            "setup": self.setup,
            "per_step": self.per_step,
            "closeout": self.closeout,
        }


def resolve_request(
    session: Any, resolver: ConfigResolver, request: dict | ScanRequest
) -> ResolvedRequest:
    """Validate the request and build its (unconnected) shot controller.

    Pre-claim and mode-independent: optimize and step/noscan share it, and
    the caller branches on ``resolved.request.mode`` afterwards.  Blocking
    config-repo I/O; emits no plan messages.
    """
    if not isinstance(request, ScanRequest):
        request = ScanRequest.model_validate(request)
    request, applied_defaults, defaults = validate_scan_request(request, resolver)

    controller = None
    if request.capture.trigger_profile:
        profile = resolver.resolve_trigger_profile(request.capture.trigger_profile)
        writes = trigger_writes_from_profile(profile)
        if writes.states:
            # Constructed worker-side, unconnected; the setter reachability
            # check joins the in-plan connect stage below.
            controller = ShotController.from_writes(
                writes,
                experiment=session.experiment,
                rep_rate_hz=session.rep_rate_hz,
            )
    strict = request.capture.acquisition is AcquisitionMode.STRICT
    return ResolvedRequest(
        request=request,
        applied_defaults=applied_defaults,
        defaults=defaults,
        controller=controller,
        strict=strict,
    )


def prepare_step_scan(
    session: Any,
    resolver: ConfigResolver,
    resolved: ResolvedRequest,
    created: list,
    *,
    submission: Any | None = None,
    namespace: Any | None = None,
):
    """The pre-claim preamble for a step/noscan request; returns a :class:`PreparedScan`.

    *namespace* is the device-source seam.  ``None`` (the funnel) builds
    fresh per-scan devices through the session factories, as it always has.
    A :class:`~geecs_bluesky.namespace.GeecsNamespace` (the preprocessor
    door) instead **selects** the save set's devices from the namespace, so
    the objects are the very ones a stock plan was handed.

    A plan (it yields the connect messages), so the caller drives it with
    ``yield from``.  Everything it constructs is appended to *created* — the
    caller-owned cleanup list its enclosing finalize disconnects — so a
    failure at any stage still tears down whatever already existed.
    """
    request = resolved.request
    applied_defaults = resolved.applied_defaults
    defaults = resolved.defaults
    controller = resolved.controller
    strict = resolved.strict

    save_set, rituals = resolve_save_sets_and_rituals(
        resolver, request.capture.save_sets
    )
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
        request.capture.background_telemetry
        if request.capture.background_telemetry is not None
        else _defaults_flag(defaults, "background_telemetry", True)
    )
    devices_config, capture_devices, native_image_save = (
        resolve_and_apply_capture_toggle(request, defaults, devices_config, session)
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

    if namespace is None:
        detectors = _build_request_detectors(
            factories, devices_config, free_run=not strict
        )
        movables = [build_movable(factories, target) for target in axis_resolved]
    else:
        # Stock-plan door: the objects must be the ones the plan itself is
        # handed, so the save set SELECTS namespace devices instead of
        # constructing new ones (GeecsNamespace.select).  Every device is
        # reset first: these nouns outlive the run, so a previous scan's
        # saving mode, save path, asset definitions or shot-ID origin would
        # otherwise still be set on the ones this save set does not name.
        namespace.reset_run_configuration()
        detectors = namespace.select(devices_config)
        movables = [_namespace_movable(namespace, target) for target in axis_resolved]
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
    # The stock-plan door reads only what the plan was handed, so the
    # telemetry tail has to be injected into the same event by the
    # preprocessor; it is kept addressable for that.

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
        submission=submission,
    )
    if request.mode is ScanRequestMode.NOSCAN:
        motor_arg: Any = None
    elif len(movables) == 1:
        motor_arg = movables[0]
    else:
        motor_arg = movables

    return PreparedScan(
        request=request,
        controller=controller,
        strict=strict,
        detectors=all_detectors,
        motor_arg=motor_arg,
        spec=spec,
        setup=setup,
        per_step=per_step,
        closeout=closeout,
        telemetry_selected=telemetry_selected,
        telemetry_readables=list(telemetry_readables),
    )
