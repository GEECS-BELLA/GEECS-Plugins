"""Run a ScanRequest: resolve config names, map onto the session machinery.

``session.run(request)`` (or ``BlueskyScanner.reinitialize``) hands a
:class:`~geecs_schemas.scan_request.ScanRequest` here.  :func:`run_scan_request`

- resolves every config *name* through a :class:`ConfigResolver`
  (:mod:`geecs_bluesky.config_resolver`, re-exported here);
- unions the named save sets into one effective SaveSet — the per-device
  union rule is documented on :func:`merge_save_sets`; everything downstream
  (devices config, telemetry exclusion, boundary warning) sees the merged set;
- adapts schemas to engine shapes (:func:`save_set_to_devices_config`,
  :func:`trigger_writes_from_profile`) — adapters live bluesky-side because
  ``geecs_schemas`` must never import ``geecs_bluesky``;
- assembles and compiles action slots in §4.4b nesting order
  (:func:`assemble_action_slots`), with fail-fast pre-claim name resolution
  and every plan signal pre-connected (:func:`prefetch_action_signals` — a
  lazy connect inside the RE loop would deadlock);
- executes noscan/step (multi-axis = outer-product grid, first axis
  outermost) and optimize modes on a
  :class:`~geecs_bluesky.session.GeecsSession`.

Deliberate v1 gaps (validated, then refused loudly — never silently wrong):
pseudo scan variables, ``all_scalars``, and optimize without either an
injected ``objective``/``suggester`` pair or an ``optimization_binder``
(the Xopt stack lives in ``geecs_scanner.optimization``, which this package
must not import — the binder is the GUI bridge's injected seam for it).

Configs speak GEECS device/variable names, never PVs (ratified convention);
PV derivation stays inside the device factories.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Callable

# ConfigsRepoResolver is re-exported: the existing import surface
# (bridge, tests, notebooks) gets both names from this module.
from geecs_bluesky.config_resolver import (  # noqa: F401
    ConfigResolver,
    ConfigsRepoResolver,
)
from geecs_bluesky.db_runtime import (
    GeecsDbScalarPolicy,
    ScalarPolicyProvider,
    resolve_entry_scalars,
    select_telemetry_variables,
)
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlWrites
from geecs_bluesky.plans.action_compiler import compile_action_plan
from geecs_bluesky.plans.run_wrapper import claim_scan
from geecs_bluesky.scan_log import log_claimed_scan_failure, scan_log
from geecs_schemas import (
    ActionBindings,
    ActionPlan,
    AcquisitionMode,
    PseudoScanVariable,
    SaveRole,
    SaveSet,
    SaveSetEntry,
    ScanRequest,
    ScanRequestMode,
    ScanVariable,
    ScanVariableSpec,
    TriggerProfile,
    TriggerState,
)
from geecs_schemas.action_plan import CheckStep, RunPlanStep, SetStep

logger = logging.getLogger(__name__)


# The resolver layer (ConfigResolver protocol + the production
# ConfigsRepoResolver) lives in geecs_bluesky.config_resolver; the names
# are re-exported here for the existing import surface.


# ---------------------------------------------------------------------------
# Schema → engine-shape adapters
# ---------------------------------------------------------------------------


def _state_write_triples(
    profile: TriggerProfile, state: "TriggerState", variant: str | None
) -> list[tuple[str | None, str, str]]:
    """Normalize one state's writes to ``(device, variable, value)`` triples.

    Handles both TriggerProfile generations (single-device dict shape and
    multi-device ordered write lists); order is preserved exactly
    (schema-documented: writes apply top to bottom).
    """
    writes = profile.writes_for(state, variant)
    if isinstance(writes, dict):
        device = getattr(profile, "device", None)
        return [(device, variable, value) for variable, value in writes.items()]
    triples: list[tuple[str | None, str, str]] = []
    for write in writes:
        if isinstance(write, dict):
            triples.append((write["device"], write["variable"], write["value"]))
        else:
            triples.append((write.device, write.variable, write.value))
    return triples


def trigger_writes_from_profile(
    profile: TriggerProfile, variant: str | None = None
) -> ShotControlWrites:
    """Adapt a TriggerProfile into the engine's ShotControlWrites.

    Each state becomes the profile's **ordered** write list (possibly
    spanning several devices); ``ShotController.from_writes`` replays them
    sequentially, each write completing before the next.

    Parameters
    ----------
    profile :
        The trigger profile to adapt.
    variant :
        Optional profile variant overlaid first (e.g. ``"laser_off"``).

    Raises
    ------
    GeecsConfigurationError
        Unknown *variant*, or the profile writes no device at all.
    """
    if variant is not None and variant not in profile.variants:
        raise GeecsConfigurationError(
            f"trigger profile {profile.name!r} has no variant {variant!r}. "
            f"Known variants: {sorted(profile.variants)}"
        )
    states: dict[str, list[tuple[str, str, str]]] = {}
    any_device = False
    for state in TriggerState:
        triples: list[tuple[str, str, str]] = []
        for device, variable, value in _state_write_triples(profile, state, variant):
            if device is None:
                raise GeecsConfigurationError(
                    f"trigger profile {profile.name!r} has a write to "
                    f"{variable!r} with no device — it cannot be sent"
                )
            triples.append((device, variable, value))
            any_device = True
        if triples:
            states[state.value] = triples
    if not any_device:
        raise GeecsConfigurationError(
            f"trigger profile {profile.name!r} names no trigger device — "
            "it cannot drive a scan's trigger"
        )
    name = getattr(profile, "name", "") or ""
    return ShotControlWrites(name=name, states=states)


def save_set_to_devices_config(
    save_set: SaveSet,
    scalar_policy: "ScalarPolicyProvider | None" = None,
) -> dict[str, dict[str, Any]]:
    """Derive the legacy ``devices_config`` shape from a SaveSet.

    Applies the intent→mechanics rules documented in ``geecs_schemas.save_set``:
    ``snapshot`` role → asynchronous, ``images`` → ``save_nonscalar_data``,
    and role overrides shape the **ordering** (the downstream classifier
    assigns free-run roles by position: reference first, contributors after
    the unmarked synchronous entries).  Each recorded ``variable_list`` is
    resolved per the ``db_scalars`` contract via
    :func:`~geecs_bluesky.db_runtime.resolve_entry_scalars`; with
    *scalar_policy* ``None`` (GUI bridge / off-network) only explicit scalars
    are recorded.

    Returns
    -------
    dict
        ``{device: {"synchronous": bool, "save_nonscalar_data": bool,
        "variable_list": [...]}}`` in role-derived order.

    Raises
    ------
    GeecsConfigurationError
        More than one ``reference`` entry, or contributors with no possible
        pacemaker.
    NotImplementedError
        ``all_scalars`` without an explicit list and no *scalar_policy*.
    """
    references = [e for e in save_set.entries if e.role is SaveRole.REFERENCE]
    if len(references) > 1:
        raise GeecsConfigurationError(
            f"save set {save_set.name!r} flags more than one entry as "
            f"role=reference ({[e.device for e in references]}); at most one "
            "device can be the free-run pacemaker"
        )
    contributors = [e for e in save_set.entries if e.role is SaveRole.CONTRIBUTOR]
    unmarked = [e for e in save_set.entries if e.role is None]
    snapshots = [e for e in save_set.entries if e.role is SaveRole.SNAPSHOT]
    if contributors and not references and not unmarked:
        raise GeecsConfigurationError(
            f"save set {save_set.name!r} marks every synchronous entry as "
            "role=contributor — flag one entry role=reference so free-run "
            "scans have a pacemaker"
        )

    config: dict[str, dict[str, Any]] = {}
    for entry in references + unmarked + contributors + snapshots:
        if entry.all_scalars and not entry.scalars and scalar_policy is None:
            raise NotImplementedError(
                f"save set {save_set.name!r}, device {entry.device!r}: "
                "all_scalars needs the DB-backed scalar enumeration — run the "
                "request through GeecsSession.run (which supplies the DB "
                "policy) or list the scalars explicitly"
            )
        variable_list = resolve_entry_scalars(
            entry.device,
            list(entry.scalars),
            db_scalars=entry.db_scalars,
            all_scalars=entry.all_scalars,
            provider=scalar_policy,
        )
        config[entry.device] = {
            "synchronous": entry.role is not SaveRole.SNAPSHOT,
            "save_nonscalar_data": entry.images,
            "variable_list": variable_list,
        }
    return config


def resolve_and_validate_actions(
    actions: ActionBindings, resolver: ConfigResolver
) -> dict[str, list[str]]:
    """Resolve every action name in the bindings against the library.

    Fail-fast, pre-claim: each name must exist (the resolver raises
    otherwise) before any hardware is touched.  The engine then compiles
    and executes the assembled slots (:func:`assemble_action_slots`).

    Parameters
    ----------
    actions :
        The request's ``setup`` / ``per_step`` / ``closeout`` bindings.
    resolver :
        Where names are looked up.

    Returns
    -------
    dict
        ``{slot: [names]}`` for the three slots, every name validated.
    """
    resolved: dict[str, list[str]] = {}
    for slot in ("setup", "per_step", "closeout"):
        names = list(getattr(actions, slot))
        for name in names:
            resolver.resolve_action_plan(name)
        resolved[slot] = names
    return resolved


# ---------------------------------------------------------------------------
# Action assembly + compilation (the §4.4b layers, executed)
# ---------------------------------------------------------------------------


def collect_save_set_rituals(save_set: SaveSet) -> dict[str, list[str]]:
    """Collect entry-level setup/closeout plan names, de-duplicated by name.

    A ritual shared by several entries runs **once** per scan.  Returns
    ``{"setup": [names], "closeout": [names]}`` in first-appearance order.
    """
    rituals: dict[str, list[str]] = {"setup": [], "closeout": []}
    for slot, names in rituals.items():
        seen: set[str] = set()
        for entry in save_set.entries:
            value = getattr(entry, slot, None)
            if not value:
                continue
            for name in value if isinstance(value, (list, tuple)) else [value]:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return rituals


def _merge_two_entries(existing: SaveSetEntry, addition: SaveSetEntry) -> SaveSetEntry:
    """Merge a second entry for the same device into the first.

    Applies the per-device union rule documented on :func:`merge_save_sets`.
    Reserved ``at_scan_start`` / ``at_scan_end`` maps merge key-wise
    (existing wins; inert fields, so this only affects the reserved warning).

    Raises
    ------
    GeecsConfigurationError
        The two entries give the same device different explicit roles.
    """

    def _union(first: list[str], second: list[str]) -> list[str]:
        merged = list(first)
        for item in second:
            if item not in merged:
                merged.append(item)
        return merged

    # Conflicting explicit roles must raise, never resolve by list order —
    # role sets the scan's synchronization semantics (pacemaker wiring).
    if (
        existing.role is not None
        and addition.role is not None
        and existing.role != addition.role
    ):
        raise GeecsConfigurationError(
            f"save-set union: device {existing.device!r} has conflicting "
            f"explicit roles across the named save sets "
            f"({existing.role.value!r} vs {addition.role.value!r}). Role sets "
            f"the acquisition semantics, so a device required by more than one "
            f"set must not disagree on it — give it the same role, or leave it "
            f"unset, in the overlapping sets."
        )

    return SaveSetEntry(
        device=existing.device,
        scalars=_union(list(existing.scalars), list(addition.scalars)),
        all_scalars=existing.all_scalars or addition.all_scalars,
        images=existing.images or addition.images,
        role=existing.role if existing.role is not None else addition.role,
        setup=_union(list(existing.setup), list(addition.setup)),
        closeout=_union(list(existing.closeout), list(addition.closeout)),
        db_scalars=existing.db_scalars or addition.db_scalars,
        at_scan_start={**addition.at_scan_start, **existing.at_scan_start},
        at_scan_end={**addition.at_scan_end, **existing.at_scan_end},
    )


def merge_save_sets(save_sets: list[SaveSet], name: str = "merged") -> SaveSet:
    """Union several resolved save sets into one effective save set.

    ``ScanRequest.save_sets`` names a list of save sets; the engine records
    the **union** of their devices so operators mix and match named
    diagnostic groups per scan. The union rule, applied device by device
    (first appearance across the list order preserved):

    - a device in only one set is carried over unchanged;
    - a device in more than one set is **merged** — ``scalars`` union
      (order-preserving, deduped), ``images`` / ``db_scalars`` /
      ``all_scalars`` OR together (True wins), the single non-``None`` ``role``
      is used (**conflicting explicit roles raise** — role sets the
      acquisition semantics, so overlapping sets must not disagree), and the
      entry-level ``setup`` / ``closeout`` ritual name lists union (deduped).

    A single-element list resolves to that set unchanged (cheap identity for
    the common single-set case).

    Parameters
    ----------
    save_sets :
        The resolved save sets to union, in ``ScanRequest.save_sets`` order.
    name :
        Name for the merged set (used in downstream error/warn messages).

    Returns
    -------
    SaveSet
        One save set whose entries are the deduped union of every input.
    """
    if len(save_sets) == 1:
        return save_sets[0]
    merged: dict[str, SaveSetEntry] = {}
    for save_set in save_sets:
        for entry in save_set.entries:
            existing = merged.get(entry.device)
            merged[entry.device] = (
                entry.model_copy(deep=True)
                if existing is None
                else _merge_two_entries(existing, entry)
            )
    return SaveSet(name=name, entries=list(merged.values()))


def resolve_save_sets_and_rituals(
    resolver: ConfigResolver, names: list[str], *, merged_name: str = "merged"
) -> tuple[SaveSet, dict[str, list[str]]]:
    """Resolve every named save set, union them, and collect all rituals.

    Rituals are collected across **all** sets, deduped by plan name (a shared
    ritual runs once), and every referenced plan name is validated fail-fast
    pre-claim.

    Returns
    -------
    tuple
        ``(merged_save_set, {"setup": [...], "closeout": [...]})``.
    """
    resolved = [resolver.resolve_save_set(name) for name in names]
    merged = merge_save_sets(resolved, name=merged_name)
    # Collect rituals across ALL sets (not just the merged entries): a ritual
    # is deduped by plan name across the whole selection so it runs once.
    rituals: dict[str, list[str]] = {"setup": [], "closeout": []}
    seen: dict[str, set[str]] = {"setup": set(), "closeout": set()}
    for save_set in resolved:
        per_set = collect_save_set_rituals(save_set)
        for slot in ("setup", "closeout"):
            for action_name in per_set[slot]:
                if action_name not in seen[slot]:
                    seen[slot].add(action_name)
                    rituals[slot].append(action_name)
    for slot_names in rituals.values():
        for action_name in slot_names:
            resolver.resolve_action_plan(action_name)
    return merged, rituals


def assemble_action_slots(
    actions: ActionBindings,
    applied_defaults: Mapping[str, Any],
    rituals: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    """Assemble the final ordered plan-name lists for the three action slots.

    The §4.4b layers nest like context managers (mirrored teardown, per the
    ``ExperimentDefaults`` schema):

    - **setup**: experiment defaults → save-set entry rituals → the scan's own
    - **per_step**: the scan's own only (deliberate — no other layer has one)
    - **closeout**: the exact reverse of setup

    *actions* is the post-defaults bindings (:func:`apply_experiment_defaults`
    prepends defaults to setup, appends to closeout); *applied_defaults* says
    how many entries came from defaults so *rituals* can be spliced between
    the layers.

    Returns
    -------
    dict
        ``{"setup": [...], "per_step": [...], "closeout": [...]}`` in final
        execution order.
    """
    n_setup_defaults = len(applied_defaults.get("actions.setup", []))
    merged_setup = list(actions.setup)  # defaults first, then the scan's own
    setup = (
        merged_setup[:n_setup_defaults]
        + list(rituals.get("setup", []))
        + merged_setup[n_setup_defaults:]
    )
    n_closeout_defaults = len(applied_defaults.get("actions.closeout", []))
    merged_closeout = list(actions.closeout)  # scan's own first, defaults last
    cut = len(merged_closeout) - n_closeout_defaults
    closeout = (
        merged_closeout[:cut]
        + list(rituals.get("closeout", []))
        + merged_closeout[cut:]
    )
    return {
        "setup": setup,
        "per_step": list(actions.per_step),
        "closeout": closeout,
    }


class _LazyResolverRegistry(Mapping):
    """Mapping façade over ``resolver.resolve_action_plan`` (fallback only).

    Used when a resolver does not expose ``action_plan_registry()``; nested
    ``run`` steps then resolve lazily by name.  Iteration/len are empty (the
    known-name list is the resolver's business), so a missing nested plan's
    error message lists no candidates — resolvers wanting better messages
    should implement ``action_plan_registry``.
    """

    def __init__(self, resolver: ConfigResolver) -> None:
        self._resolver = resolver

    def __getitem__(self, name: str) -> ActionPlan:
        # Only a genuine "not in the library" miss becomes a KeyError (which
        # the compiler turns into ActionPlanNotFoundError).  Any other fault
        # (transient IO, a bug in a resolver) must propagate — masking it as a
        # miss would misdirect debugging to "plan not found" with no candidates.
        try:
            return self._resolver.resolve_action_plan(name)
        except GeecsConfigurationError:
            raise KeyError(name) from None

    def get(self, name: str, default: Any = None) -> Any:
        """Resolve *name*, returning *default* only when the name is unknown."""
        try:
            return self._resolver.resolve_action_plan(name)
        except GeecsConfigurationError:
            return default

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


def build_action_registry(resolver: ConfigResolver) -> Mapping[str, ActionPlan]:
    """Return the named-plan registry nested ``run`` steps resolve against.

    Prefers the resolver's duck-typed ``action_plan_registry()``; falls back
    to a lazy per-name façade.
    """
    method = getattr(resolver, "action_plan_registry", None)
    if callable(method):
        return method()
    return _LazyResolverRegistry(resolver)


def prefetch_action_signals(
    plans: list[ActionPlan],
    registry: Mapping[str, ActionPlan],
    settables: Any,
) -> None:
    """Create/connect every signal the compiled *plans* will touch, up front.

    Plan generators execute **inside** the RunEngine loop, where a lazy
    connect would deadlock — so every target is connected here, pre-claim
    (doubling as fail-fast validation).  Nested ``run`` steps are walked
    recursively (visited-set bounded).
    """
    visited: set[str] = set()

    def _walk(plan: ActionPlan) -> None:
        for step in plan.steps:
            if isinstance(step, SetStep):
                settables.get_settable(step.device, step.variable)
            elif isinstance(step, CheckStep):
                settables.get_readable(step.device, step.variable)
            elif isinstance(step, RunPlanStep):
                if step.plan in visited:
                    continue
                visited.add(step.plan)
                nested = registry.get(step.plan)
                if nested is not None:
                    _walk(nested)

    for plan in plans:
        _walk(plan)


def compile_action_slot(
    names: list[str],
    resolver: ConfigResolver,
    registry: Mapping[str, ActionPlan],
    settables: Any,
) -> tuple[Callable | None, list[ActionPlan]]:
    """Compile one slot's plan names into a reusable plan-stub callable.

    The callable produces a **fresh** message generator per call — required
    for ``per_step`` and for ``finalize_wrapper`` re-instantiation.

    Returns
    -------
    tuple
        ``(stub, plans)`` — stub is ``None`` when the slot is empty; plans
        feed signal prefetching.
    """
    if not names:
        return None, []
    plans = [resolver.resolve_action_plan(name) for name in names]

    def _slot_plan():
        for plan in plans:
            yield from compile_action_plan(plan, registry=registry, settables=settables)

    return _slot_plan, plans


def apply_experiment_defaults(
    request: ScanRequest, defaults: Any | None
) -> tuple[ScanRequest, dict[str, Any]]:
    """Apply experiment defaults where the request is silent (with provenance).

    :class:`~geecs_schemas.ExperimentDefaults` merge rule — defaults are the
    outermost bracket: default setup *prepends*, default closeout *appends*,
    a default trigger profile fills only an unset one.  Never overrides an
    explicit request value.

    Returns
    -------
    tuple
        ``(request, applied)`` — the updated request and a ``{field: value}``
        provenance record of every default applied (goes into run metadata).
    """

    def _field(source: Any, name: str) -> Any:
        if isinstance(source, dict):
            return source.get(name)
        return getattr(source, name, None)

    applied: dict[str, Any] = {}
    if defaults is None:
        return request, applied

    updates: dict[str, Any] = {}
    default_profile = _field(defaults, "trigger_profile")
    if request.trigger_profile is None and default_profile:
        updates["trigger_profile"] = default_profile
        applied["trigger_profile"] = default_profile

    actions_default = _field(defaults, "actions")
    slot_updates: dict[str, list[str]] = {}
    for slot in ("setup", "closeout"):
        value = None
        if actions_default is not None:
            value = _field(actions_default, slot)
        if not value:
            continue
        names = list(value) if isinstance(value, (list, tuple)) else [value]
        own = list(getattr(request.actions, slot))
        # Mirrored bracket: default setup runs before the scan's own,
        # default closeout runs after it (the ExperimentDefaults merge rule).
        slot_updates[slot] = names + own if slot == "setup" else own + names
        applied[f"actions.{slot}"] = names
    if slot_updates:
        updates["actions"] = request.actions.model_copy(update=slot_updates)

    if updates:
        request = request.model_copy(update=updates)
    return request, applied


def resolve_experiment_defaults(resolver: ConfigResolver) -> Any | None:
    """Return the resolver's experiment defaults, or ``None`` (tolerantly).

    Resolvers without a ``resolve_experiment_defaults`` method are treated as
    having no defaults.
    """
    resolve = getattr(resolver, "resolve_experiment_defaults", None)
    return resolve() if callable(resolve) else None


def resolve_defaults_for(
    resolver: ConfigResolver, request: ScanRequest
) -> tuple[ScanRequest, dict[str, Any]]:
    """Apply the resolver's experiment defaults to *request* (tolerantly).

    Resolvers without a ``resolve_experiment_defaults`` method are treated
    as having no defaults.  Returns as :func:`apply_experiment_defaults`.
    """
    return apply_experiment_defaults(request, resolve_experiment_defaults(resolver))


def _defaults_flag(defaults: Any | None, name: str, fallback: bool) -> bool:
    """Read a boolean flag off the experiment defaults (model or mapping).

    Returns *fallback* when there are no defaults or the flag is absent.
    """
    if defaults is None:
        return fallback
    if isinstance(defaults, dict):
        value = defaults.get(name, fallback)
    else:
        value = getattr(defaults, name, fallback)
    return fallback if value is None else bool(value)


# ---------------------------------------------------------------------------
# Request execution on a GeecsSession
# ---------------------------------------------------------------------------


def resolve_movable_target(
    spec: ScanVariableSpec, name: str
) -> tuple[str, str, str, str | None]:
    """Return ``(device, variable, kind, confirm)`` for a plain scan variable.

    ``confirm`` is the entry's optional ``"Device:Variable"`` confirming
    target (``None`` when the set variable is also the readback).

    Raises
    ------
    NotImplementedError
        Pseudo (composite) variables — the pseudo-positioner is not built yet.
    """
    if isinstance(spec, PseudoScanVariable):
        raise NotImplementedError(
            f"scan variable {name!r} is a pseudo (composite) variable; "
            "composite pseudo-positioner execution is not built yet — "
            "sweep the underlying Device:Variable targets directly for now"
        )
    assert isinstance(spec, ScanVariable)
    device, _, variable = spec.target.partition(":")
    return device, variable, spec.kind, spec.confirm


def build_movable(
    session: Any, device: str, variable: str, kind: str, confirm: str | None
) -> Any:
    """Build the right movable for one resolved scan-variable target.

    ``confirm`` (a ``"Device:Variable"`` string) takes precedence over
    ``kind``: a variable with a confirming target is the topology-C case
    (:class:`~geecs_bluesky.devices.ca.confirm.CaConfirmSettable`) regardless
    of whether it is also declared ``kind: motor`` — the confirming poll is
    the more specific completion check.  Otherwise dispatches on ``kind`` as
    before: ``"motor"`` → :meth:`GeecsSession.motor`, else
    :meth:`GeecsSession.settable`.
    """
    if confirm is not None:
        confirm_device, _, confirm_variable = confirm.partition(":")
        return session.confirm_settable(
            device,
            variable,
            confirm_device=confirm_device,
            confirm_variable=confirm_variable,
        )
    if kind == "motor":
        return session.motor(device, variable)
    return session.settable(device, variable)


def _build_request_detectors(
    session: Any, devices_config: dict[str, dict[str, Any]], *, free_run: bool
) -> list:
    """Create session devices from a derived devices_config, roles by order.

    Mirrors ``BlueskyScanner._classify_device_roles``: free-run → first
    synchronous entry is the reference, later ones contributors; strict →
    all synchronous entries triggered; asynchronous → snapshots.  This is
    the *headless* build — failures propagate (fail loudly); operator
    drop/promote interaction is the scanner layer's job.  Returns connected
    devices, reference first.
    """
    detectors: list = []
    reference_assigned = False
    for device_name, cfg in devices_config.items():
        variables = list(cfg.get("variable_list") or [])
        save = bool(cfg.get("save_nonscalar_data", False))
        synchronous = bool(cfg.get("synchronous", False))
        if not synchronous:
            if not variables:
                logger.warning(
                    "Skipping asynchronous device %s: no scalars to record",
                    device_name,
                )
                continue
            detectors.append(session.snapshot(device_name, variables))
        elif free_run and reference_assigned:
            detectors.append(
                session.contributor(device_name, variables, save_images=save)
            )
        else:
            detectors.append(session.detector(device_name, variables, save_images=save))
            reference_assigned = True
    return detectors


# ---------------------------------------------------------------------------
# DB-integration runtime (M3c): db_scalars + background telemetry (get-side)
# ---------------------------------------------------------------------------


def make_scalar_policy(session: Any) -> ScalarPolicyProvider | None:
    """Build the get-side DB scalar policy provider for *session*'s experiment.

    Returns a :class:`~geecs_bluesky.db_runtime.GeecsDbScalarPolicy` bound to
    the session's experiment.  The provider itself is failure-tolerant (a DB
    lookup that fails degrades to empty policy with a warning), so this never
    raises for a missing DB; ``None`` is returned only when the session does
    not expose an ``experiment`` attribute (defensive — every real session
    does).

    Parameters
    ----------
    session :
        The :class:`~geecs_bluesky.session.GeecsSession` (duck-typed).

    Returns
    -------
    ScalarPolicyProvider or None
        A DB-backed policy provider, or ``None`` when no experiment is known.
    """
    experiment = getattr(session, "experiment", None)
    if not experiment:
        return None
    return GeecsDbScalarPolicy(experiment)


def warn_if_reserved_boundary_overrides(save_set: SaveSet | None) -> None:
    """Warn once if any entry sets the reserved (not-honored) set-side fields.

    ``at_scan_start`` / ``at_scan_end`` are reserved and inert — the DB
    set-side is intentionally disabled (rationale:
    ``GeecsBluesky/CLAUDE.md``, M3c set-side section).  One WARNING names
    the offending device(s); not an error.
    """
    if save_set is None:
        return
    devices = [
        entry.device
        for entry in save_set.entries
        if getattr(entry, "at_scan_start", None) or getattr(entry, "at_scan_end", None)
    ]
    if devices:
        logger.warning(
            "save set %r sets the reserved DB scan start/end fields "
            "(at_scan_start / at_scan_end) on %s, but the set-side is disabled "
            "in this version and these values are NOT applied — triggering is "
            "owned by the TriggerProfile/shot controller and camera saving by "
            "the scanner's save-windowing (kept reserved for a possible future "
            "re-enable)",
            save_set.name,
            ", ".join(devices),
        )


def build_telemetry_readables(
    session: Any,
    save_set: SaveSet | None,
    scalar_policy: ScalarPolicyProvider | None,
) -> tuple[list, dict[str, list[str]]]:
    """Build the Tier-2 background-telemetry readables (soft, dropped-if-dead).

    One soft readable per experiment device with a ``get='yes'`` variable not
    in *save_set*; ``session.telemetry`` returns ``None`` for a device
    unreachable at scan start (dropped with a log line, never an abort).
    ``scalar_policy`` ``None`` means no telemetry.

    Returns
    -------
    tuple
        ``(readables, recorded)`` — connected devices and the
        ``{device: [variables]}`` map of **only those that connected**
        (run-metadata key must match the columns that actually exist).
    """
    if scalar_policy is None:
        return [], {}
    selected = select_telemetry_variables(
        save_set, scalar_policy.subscribed_by_device()
    )
    readables: list = []
    recorded: dict[str, list[str]] = {}
    for device, variables in selected.items():
        readable = session.telemetry(device, variables)
        if readable is not None:
            readables.append(readable)
            recorded[device] = list(variables)
    # Record only the devices that actually connected: a device dropped as
    # unreachable at scan start contributes no columns, so the start-doc
    # metadata must not advertise them (EVENT_SCHEMA.md contract — the key
    # reflects what was recorded, not what was selected).
    return readables, recorded


def run_scan_request(
    session: Any,
    request: ScanRequest,
    resolver: ConfigResolver,
    *,
    objective: Any | None = None,
    suggester: Any | None = None,
    optimization_binder: Callable[..., tuple[Any, Any]] | None = None,
    preflight: Callable[[list, bool], list | None] | None = None,
    on_scan_start: Callable[[int, int], None] | None = None,
) -> str | None:
    """Execute *request* on *session*; return the run uid.

    Resolution order is fail-fast: every action name (request-level, entry
    rituals, defaults) is resolved before any hardware is touched, the
    trigger profile is attached (generalized multi-device ordered writes),
    action plans are compiled and their signals pre-connected, devices are
    built, then the scan runs — all of it before a scan number is claimed
    (the claim happens inside ``session.scan``).  Devices *and* the action
    signal factory created here are disconnected afterwards (the run owns
    what it creates).

    Multi-axis requests run as an outer-product grid (first axis
    outermost/slowest): one movable per axis, one bin per grid point,
    ``per_step`` actions at every grid point, all axis readbacks in the
    event rows; the run metadata carries ``scan_axes`` / ``grid_shape`` /
    ``num_grid_points`` and ScanInfo's 1-D fields describe the outermost
    axis.

    Parameters
    ----------
    session :
        A :class:`~geecs_bluesky.session.GeecsSession`.
    request :
        The scan request to run.
    resolver :
        Resolves the request's names to schema models.
    objective, suggester :
        Ready-made optimization callables for ``optimize`` mode (see the
        module docstring's gap list): the evaluator/generator specs cannot
        be instantiated in this package.
    optimization_binder :
        Alternative to *objective*/*suggester* for ``optimize`` mode: a
        scanner-layer hook (the GUI bridge's injected
        ``optimization_loader`` seam) called as
        ``binder(devices=..., scan_tag=..., scan_folder=...) ->
        (objective, suggester)`` with the connected movables + detectors
        and the freshly claimed scan (the runner claims pre-bind; see
        :func:`_run_optimize_request`).  Ignored when *objective* and
        *suggester* are given.
    preflight :
        Optional scanner-layer hook (the GUI bridge's operator-dialog
        seam), called pre-claim with the fully assembled detector list and
        a strict flag: ``preflight(detectors, strict) -> list | None``.
        The returned (possibly reduced) list becomes the scan's detectors;
        ``None`` aborts the run (created devices are still disconnected).
        Not called on the optimize path — an optimize preflight is a later
        seam.  Headless callers omit it (behavior unchanged when ``None``).
    on_scan_start :
        Optional progress-totals hook (the GUI bridge's progress seam),
        called with ``(total_steps, total_shots)`` immediately before the
        session scan starts.  On the optimize path the totals are the
        upper bound ``(max_iterations, max_iterations × shots_per_step)``
        — the suggester may stop early.

    Returns
    -------
    str or None
        The Bluesky run uid (``None`` when nothing was persisted).

    Raises
    ------
    NotImplementedError
        Pseudo scan variables, and optimize mode without an injected
        objective/suggester or optimization_binder (the documented v1
        gaps — validated first, refused loudly).
    GeecsConfigurationError
        Unresolvable names, or a step/noscan request without a save set.
    """
    defaults = resolve_experiment_defaults(resolver)
    request, applied_defaults = apply_experiment_defaults(request, defaults)
    resolved_actions = resolve_and_validate_actions(request.actions, resolver)

    if request.trigger_profile:
        profile = resolver.resolve_trigger_profile(request.trigger_profile)
        session.shot_control(
            trigger_writes_from_profile(profile, request.trigger_variant)
        )
    else:
        session.shot_control(None)

    mode = "strict" if request.acquisition is AcquisitionMode.STRICT else "free_run"

    if request.mode is ScanRequestMode.OPTIMIZE:
        # Optimize has no action hooks yet: skip actions (never refuse — that
        # would block optimization wherever defaults define bracket actions),
        # log loudly, and record the skip in run metadata (never silent).
        # Rationale: GeecsBluesky/CLAUDE.md (engine consolidation).
        skipped_actions = {k: v for k, v in resolved_actions.items() if v}
        return _run_optimize_request(
            session,
            request,
            resolver,
            mode,
            objective=objective,
            suggester=suggester,
            optimization_binder=optimization_binder,
            on_scan_start=on_scan_start,
            applied_defaults=applied_defaults,
            skipped_actions=skipped_actions,
        )

    if not request.save_sets:
        raise GeecsConfigurationError(
            f"a {request.mode.value!r} ScanRequest needs at least one save "
            "set in save_sets — without one the scan would record nothing"
        )
    # Multiple named save sets union into one effective save set (devices
    # deduped/merged; rituals collected across all sets, deduped by name).
    save_set, rituals = resolve_save_sets_and_rituals(resolver, request.save_sets)

    # M3c get-side runtime: failure-tolerant policy provider (a DB blip never
    # aborts a scan); the DB set-side stays disabled (reserved fields warn).
    scalar_policy = make_scalar_policy(session)
    devices_config = save_set_to_devices_config(save_set, scalar_policy)
    slots = assemble_action_slots(request.actions, applied_defaults, rituals)
    warn_if_reserved_boundary_overrides(save_set)

    # Resolve the scan-variable movable targets up front (full movable
    # construction happens later; only the (device, variable, kind, confirm)
    # quadruples are needed here for the standard-scan build below).
    axis_resolved: list[tuple[str, str, str, str | None]] = []
    for axis in request.axes:
        spec = resolver.resolve_scan_variable(axis.variable)
        device, variable, kind, confirm = resolve_movable_target(spec, axis.variable)
        axis_resolved.append((device, variable, kind, confirm))

    telemetry_enabled = (
        request.background_telemetry
        if request.background_telemetry is not None
        else _defaults_flag(defaults, "background_telemetry", True)
    )

    created: list = []
    try:
        # Compile the action slots first: signal prefetch fail-fasts on an
        # unreachable action target before detectors are even built, and
        # everything stays pre-claim.
        setup = per_step = closeout = None
        if any(slots.values()):
            factory = session.action_signal_factory()
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
            prefetch_action_signals(
                setup_plans + per_step_plans + closeout_plans, registry, factory
            )

        detectors = _build_request_detectors(
            session, devices_config, free_run=mode == "free_run"
        )
        created.extend(detectors)

        telemetry_selected: dict[str, list[str]] = {}
        if telemetry_enabled:
            telemetry_readables, telemetry_selected = build_telemetry_readables(
                session, save_set, scalar_policy
            )
            # Telemetry is soft: appended to the read set as extra snapshot
            # columns, never as the reference (index 0 stays the save set's).
            detectors = list(detectors) + telemetry_readables
            created.extend(telemetry_readables)

        if preflight is not None:
            # Scanner-layer seam (operator dialogs): runs pre-claim by
            # construction — the claim happens inside session.scan below.
            checked = preflight(detectors, mode == "strict")
            if checked is None:
                logger.warning(
                    "ScanRequest preflight aborted the scan (pre-claim; no "
                    "scan number was burned)"
                )
                return None
            detectors = list(checked)

        md: dict[str, Any] = {"scan_request_mode": request.mode.value}
        # Provenance: which named save sets were unioned for this scan.
        md["save_sets"] = list(request.save_sets)
        if applied_defaults:
            # Provenance: the run records exactly which experiment defaults
            # filled in fields the submitter left unset.
            md["applied_defaults"] = applied_defaults
        if any(slots.values()):
            # Provenance: the assembled per-slot execution order (defaults +
            # entry rituals + the request's own, mirrored on closeout).
            md["action_plans"] = {k: v for k, v in slots.items() if v}
        if telemetry_enabled and telemetry_selected:
            md["background_telemetry"] = {
                dev: list(vars_) for dev, vars_ in telemetry_selected.items()
            }
        scan_info: dict[str, Any] = {
            "shots": request.shots_per_step,
            "background": request.background,
        }

        if request.mode is ScanRequestMode.NOSCAN:
            scan_info["scan_mode"] = "noscan"
            if on_scan_start is not None:
                on_scan_start(1, request.shots_per_step)
            return session.scan(
                detectors=detectors,
                motor=None,
                positions=[None],
                shots_per_step=request.shots_per_step,
                mode=mode,
                description=request.description,
                md=md,
                scan_info=scan_info,
                setup=setup,
                per_step=per_step,
                closeout=closeout,
            )

        movables: list = []
        targets: list[str] = []
        value_lists: list[list[float]] = []
        for (device, variable, kind, confirm), axis in zip(axis_resolved, request.axes):
            movable = build_movable(session, device, variable, kind, confirm)
            created.append(movable)
            movables.append(movable)
            targets.append(f"{device}:{variable}")
            value_lists.append(axis.positions.to_values())

        scan_info["scan_mode"] = "standard"
        scan_info["scan_parameter"] = ",".join(targets)
        if len(request.axes) == 1:
            motor_arg: Any = movables[0]
            positions: list[Any] = value_lists[0]
            md["scan_variable"] = request.axes[0].variable
        else:
            # Outer product, first axis outermost/slowest (the schema's
            # documented grid semantics); one bin per grid point.
            motor_arg = movables
            positions = [tuple(point) for point in itertools.product(*value_lists)]
            md["scan_variable"] = ",".join(a.variable for a in request.axes)
            md["scan_axes"] = [a.variable for a in request.axes]
            md["grid_shape"] = list(request.grid_shape())
            md["num_grid_points"] = request.n_steps()
            # ScanInfo is a legacy 1-D format: Start/End/Step describe the
            # outermost axis; the grid truth lives in the run metadata.
            outer = value_lists[0]
            scan_info["start"] = outer[0]
            scan_info["end"] = outer[-1]
            scan_info["step"] = (outer[1] - outer[0]) if len(outer) > 1 else 0
        if on_scan_start is not None:
            on_scan_start(len(positions), len(positions) * request.shots_per_step)
        return session.scan(
            detectors=detectors,
            motor=motor_arg,
            positions=positions,
            shots_per_step=request.shots_per_step,
            mode=mode,
            description=request.description,
            md=md,
            scan_info=scan_info,
            setup=setup,
            per_step=per_step,
            closeout=closeout,
        )
    finally:
        if created and hasattr(session, "disconnect"):
            session.disconnect(*created)


def _run_optimize_request(
    session: Any,
    request: ScanRequest,
    resolver: ConfigResolver,
    mode: str,
    *,
    objective: Any | None,
    suggester: Any | None,
    optimization_binder: Callable[..., tuple[Any, Any]] | None = None,
    on_scan_start: Callable[[int, int], None] | None = None,
    applied_defaults: dict[str, Any] | None = None,
    skipped_actions: dict[str, list[str]] | None = None,
) -> str | None:
    """Map an optimize-mode request onto :meth:`GeecsSession.optimize`.

    Consumes ``optimization.variables`` (names resolved through the
    scan-variable catalog; ``Device:Variable`` strings pass through),
    ``max_iterations``, and ``move_to_best_on_finish`` (→ ``on_finish``).
    The variable *bounds*, ``objectives``/``observables``/``constraints``,
    and the evaluator/generator specs are the suggester's business — they
    are **not** consumed here (the injected objective/suggester — or the
    binder's stack — is expected to have been built from them).

    Parameters
    ----------
    session, request, resolver :
        As in :func:`run_scan_request`.
    mode :
        ``"strict"`` or ``"free_run"``.
    objective, suggester :
        The ready-made optimization callables.
    optimization_binder, on_scan_start :
        As in :func:`run_scan_request`.  With a binder (and no ready-made
        callables) the runner claims the scan itself, pre-bind, so the
        binder's analyzers get the real ``ScanTag`` — mirroring the legacy
        exec_config optimization path; the claim still happens *after*
        every fail-fast resolution and device connect, and the runner then
        owns the ``scan.log`` attach for the run.

    Returns
    -------
    str or None
        The run uid.

    Raises
    ------
    NotImplementedError
        When *objective* or *suggester* is missing and no
        *optimization_binder* was given.
    """
    spec = request.optimization
    assert spec is not None  # guaranteed by ScanRequest validation
    if (objective is None or suggester is None) and optimization_binder is None:
        raise NotImplementedError(
            "optimize-mode ScanRequest execution needs a ready-made "
            "objective and suggester (run(request, resolver, objective=..., "
            "suggester=...)) or an optimization_binder: instantiating them "
            "from the request's evaluator/generator specs lives in the GUI "
            "optimization stack (geecs_scanner.optimization), which "
            "geecs_bluesky cannot import"
        )

    detectors: list = []
    created: list = []
    try:
        skipped = {k: list(v) for k, v in (skipped_actions or {}).items() if v}
        # db_scalars applies to optimize too; telemetry does not run here yet
        # (no scan-boundary hook); the DB set-side stays disabled everywhere.
        scalar_policy = make_scalar_policy(session)
        if request.save_sets:
            save_set, rituals = resolve_save_sets_and_rituals(
                resolver, request.save_sets
            )
            # Reserved DB set-side overrides are inert here too — warn once, as
            # on the scan/noscan path, so the promise holds in every mode.
            warn_if_reserved_boundary_overrides(save_set)
            ritual_names = [n for names in rituals.values() for n in names]
            if ritual_names:
                # Save-set entry rituals can't run in optimize mode yet either;
                # skip and record rather than refuse (see run_scan_request).
                skipped["save_set_rituals"] = ritual_names
            detectors = _build_request_detectors(
                session,
                save_set_to_devices_config(save_set, scalar_policy),
                free_run=mode == "free_run",
            )
            created.extend(detectors)

        variables: dict[str, Any] = {}
        for name in spec.variables:
            if ":" in name:
                device, _, variable = name.partition(":")
                movable = session.settable(device, variable)
            else:
                var_spec = resolver.resolve_scan_variable(name)
                device, variable, kind, confirm = resolve_movable_target(var_spec, name)
                movable = build_movable(session, device, variable, kind, confirm)
            variables[name] = movable
            created.append(movable)

        md: dict[str, Any] = {"scan_request_mode": request.mode.value}
        if request.save_sets:
            # Provenance: which named save sets were unioned for this scan.
            md["save_sets"] = list(request.save_sets)
        if applied_defaults:
            md["applied_defaults"] = applied_defaults
        if skipped:
            md["skipped_action_plans"] = skipped
            logger.warning(
                "Optimize mode does not run action plans yet — skipping the "
                "following for this optimization (setup/per_step/closeout "
                "and save-set rituals do not run during optimization): %s",
                skipped,
            )
        # Telemetry is not wired into optimize yet; db_scalars above applies.
        # The DB set-side is disabled
        # everywhere in this version.  Recorded for provenance.
        md["db_scan_runtime"] = {
            "db_scalars": "applied",
            "background_telemetry": "not_run_in_optimize",
        }

        scan_number: int | None = None
        scan_folder: str | None = None
        claimed_here = False
        try:
            if objective is None or suggester is None:
                # Binder path (checked non-None at entry): claim first so the
                # binder's analyzers get the real ScanTag (docstring above).
                scan_tag, scan_folder = claim_scan(
                    getattr(session, "experiment", "") or ""
                )
                scan_number = scan_tag.number if scan_tag is not None else None
                claimed_here = scan_number is not None
                objective, suggester = optimization_binder(
                    devices=list(variables.values()) + detectors,
                    scan_tag=scan_tag,
                    scan_folder=scan_folder,
                )

            max_iterations = spec.max_iterations or 20
            if on_scan_start is not None:
                # Upper-bound totals: the suggester may stop early.
                on_scan_start(max_iterations, max_iterations * request.shots_per_step)

            # The runner claimed → the runner attaches scan.log (the session
            # only self-attaches when *it* claimed the number).
            with scan_log(scan_number, scan_folder) if claimed_here else nullcontext():
                uid, _history = session.optimize(
                    variables=variables,
                    detectors=detectors,
                    objective=objective,
                    suggester=suggester,
                    shots_per_iteration=request.shots_per_step,
                    max_iterations=max_iterations,
                    mode=mode,
                    description=request.description,
                    md=md,
                    on_finish="best" if spec.move_to_best_on_finish else "hold",
                    scan_number=scan_number,
                    scan_folder=scan_folder,
                )
            return uid
        except BaseException:
            if claimed_here:
                log_claimed_scan_failure(
                    scan_number, scan_folder, label="Optimization scan"
                )
            raise
    finally:
        if created and hasattr(session, "disconnect"):
            session.disconnect(*created)
