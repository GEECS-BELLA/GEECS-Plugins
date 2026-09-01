"""Run a ScanRequest: resolve config names, map onto the session machinery.

``session.run(request)`` hands a
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

Pseudo (composite) scan variables execute: :func:`resolve_movable_target`
compiles every component's ``forward`` formula fail-fast pre-claim and
:func:`build_movable` builds a
:class:`~geecs_bluesky.devices.ca.pseudo.CaPseudoMovable` for them, on both
the step-axis and optimize movable paths (spec + formulas recorded in run
metadata under ``pseudo_variables``).

Deliberate v1 gaps (validated, then refused loudly — never silently wrong):
``all_scalars``, and optimize without either an
injected ``objective``/``suggester`` pair or an ``optimization_binder``
(the Xopt stack lives in ``geecs_bluesky.optimization`` behind the
``optimize`` extra — the binder is the caller's injected seam for it).

Configs speak GEECS device/variable names, never PVs (ratified convention);
PV derivation stays inside the device factories.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Callable

# ConfigsRepoResolver is re-exported: the existing import surface
# (bridge, tests, notebooks) gets both names from this module.
from geecs_bluesky.config_resolver import (  # noqa: F401
    ConfigResolver,
    ConfigsRepoResolver,
)
from geecs_bluesky.db_runtime import (
    GeecsDbDeviceTypes,
    GeecsDbScalarPolicy,
    GeecsDbServedSetProvider,
    ScalarPolicyProvider,
    resolve_entry_scalars,
    select_telemetry_variables,
)
from geecs_bluesky.exceptions import GeecsConfigurationError, GeecsDeviceDownError
from geecs_bluesky.forward_expr import CompiledForward, compile_forward
from geecs_bluesky.models.shot_control import ShotControlWrites
from geecs_bluesky.plans.action_compiler import compile_action_plan
from geecs_bluesky.plans.run_wrapper import claim_scan
from geecs_bluesky.preflight import run_unserved_variables_check
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
    *scalar_policy* ``None`` (no DB / off-network) only explicit scalars
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


def _requirement_field(requirement: Any, name: str, default: Any) -> Any:
    """Read *name* from a duck-typed requirement entry (mapping or attributes)."""
    if isinstance(requirement, Mapping):
        return requirement.get(name, default)
    return getattr(requirement, name, default)


def merge_optimizer_device_requirements(
    devices_config: dict[str, dict[str, Any]],
    requirements: Any,
) -> dict[str, dict[str, Any]]:
    """Merge optimizer ``device_requirements`` into *devices_config*, in place.

    The one surviving definition of the optimizer-requirements merge
    (the bridge-side twin died with the exec_config path, G3): every
    device the objective's evaluator needs is acquired and natively saved
    even when the request's save sets do not name it — or when the request
    names no save sets at all.  This reverses the #520 deferral ("the
    request's save sets must name the objective's diagnostics"): in the
    field the evaluator's auto-generated requirements were silently
    ignored, the diagnostic never saved, and every objective evaluated to
    NaN (live incident 2026-07-15, ``TopViewMax`` / ``UC_TopView``).

    Union semantics per device, mirroring :func:`merge_save_sets`:
    ``variable_list`` unions (order-preserving, deduped; the save-set
    variables stay first), ``save_nonscalar_data`` ORs (True wins), and a
    device already configured keeps its ``synchronous`` flag — the save
    sets own the acquisition-role/ordering semantics, so the pacemaker
    choice is unchanged and a *new* required device is appended after the
    save-set devices (exactly the legacy merge's behavior).

    Device names match case-insensitively (``str.casefold``) — GEECS is
    case-inconsistent about device-name spelling and CA PV names are not —
    with a hit merged under the configured spelling, logged at INFO.

    *requirements* is duck-typed and opaque: a ``{"Devices": {name: cfg}}``
    mapping (the shape the optimization stack auto-generates from its
    evaluator's analyzers and exposes on the loader-returned bridge), or
    ``None``/empty for a no-op.  ``geecs_bluesky`` never imports the stack
    that builds it (dependency direction, AST-pinned).

    Parameters
    ----------
    devices_config :
        The effective devices config derived from the request's save sets
        (possibly empty).  Mutated in place.
    requirements :
        The optimizer's device requirements, or ``None``.

    Returns
    -------
    dict
        What was actually provisioned, for run-metadata provenance
        (recorded as ``provisioned_device_requirements``): the full entry
        for a newly added device; only the added variables / flipped save
        flag for an already-configured one.  Empty when nothing changed.
    """
    devices = requirements.get("Devices") if isinstance(requirements, Mapping) else None
    provisioned: dict[str, dict[str, Any]] = {}
    if not devices:
        return provisioned
    for device_name, requirement in devices.items():
        req_vars = [
            str(v) for v in (_requirement_field(requirement, "variable_list", []) or [])
        ]
        req_save = bool(_requirement_field(requirement, "save_nonscalar_data", False))
        configured_name = device_name
        existing = devices_config.get(device_name)
        if existing is None:
            folded = str(device_name).casefold()
            match = next(
                (name for name in devices_config if name.casefold() == folded),
                None,
            )
            if match is not None:
                configured_name = match
                existing = devices_config[match]
                logger.info(
                    "Optimization: required device %s differs only in case "
                    "from configured device %s; merging under the configured "
                    "spelling %s (CA PV names are case-sensitive)",
                    device_name,
                    match,
                    match,
                )
        if existing is None:
            entry = {
                "synchronous": bool(
                    _requirement_field(requirement, "synchronous", False)
                ),
                "save_nonscalar_data": req_save,
                "variable_list": req_vars,
            }
            devices_config[device_name] = entry
            provisioned[device_name] = {
                "synchronous": entry["synchronous"],
                "save_nonscalar_data": entry["save_nonscalar_data"],
                "variable_list": list(req_vars),
            }
            logger.info(
                "Optimization: auto-provisioned required device %s "
                "(synchronous=%s, save_nonscalar=%s, variables=%s) — verify "
                "the spelling matches the GEECS database; CA PV names are "
                "case-sensitive",
                device_name,
                entry["synchronous"],
                entry["save_nonscalar_data"],
                req_vars,
            )
        else:
            added: dict[str, Any] = {}
            current = list(existing.get("variable_list") or [])
            missing = [v for v in req_vars if v not in current]
            if missing:
                existing["variable_list"] = current + missing
                added["variable_list"] = missing
            if req_save and not existing.get("save_nonscalar_data", False):
                existing["save_nonscalar_data"] = True
                added["save_nonscalar_data"] = True
            if added:
                provisioned[configured_name] = added
                logger.info(
                    "Optimization: merged optimizer requirements into "
                    "configured device %s (save-set settings preserved): %s",
                    configured_name,
                    added,
                )
    return provisioned


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
    if request.capture.trigger_profile is None and default_profile:
        updates["capture"] = request.capture.model_copy(
            update={"trigger_profile": default_profile}
        )
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


def metadata_applied_defaults(
    applied_defaults: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return event-model-safe provenance records for applied defaults.

    The execution-side merge logic keeps dotted field names such as
    ``actions.setup`` because they are concise and useful internally.
    Event-model validates metadata keys recursively, though, so those
    dotted names cannot be emitted as dictionary keys in the start
    document.  Emit them as values instead.
    """
    return [
        {"field": field, "value": value} for field, value in applied_defaults.items()
    ]


def metadata_submission(submission: Any | None) -> dict[str, Any] | None:
    """Return the submission-provenance record for run metadata.

    A ``SubmissionRecord`` is built by the submitting client at queue time —
    who queued the request, when, and the pre-submit preflight outcomes.
    Since geecs-schemas 0.14.0 it is not part of the request document; it
    travels beside the request (the funnel plan's ``submission`` parameter).
    The engine records it verbatim and never acts on it.  ``None`` in,
    ``None`` out (headless callers, saved presets) — callers gate the
    metadata key on the return value.  Accepts the record as a model or as
    its JSON-dict form (the queue's wire shape); validation here keeps
    garbage out of the start document.
    """
    if submission is None:
        return None
    from geecs_schemas import SubmissionRecord

    return SubmissionRecord.model_validate(submission).model_dump(mode="json")


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


def validate_scan_request(
    request: ScanRequest, resolver: ConfigResolver
) -> tuple[ScanRequest, dict[str, Any]]:
    """Fail-fast dry-run of everything :func:`run_scan_request` must resolve.

    THE one definition of "what must resolve" (issue #529): the GUI
    bridge's ``reinitialize`` calls this for submission-time errors, and
    :func:`run_scan_request` runs it as its own first phase — so the
    bridge's fail-fast can never drift from execution.  **When a new
    resolvable field is added to the runner, its resolution is added
    here**, never re-implemented in a caller.

    Pure resolution: no session, no hardware, no side effects.  Resolved
    products beyond the returned pair are discarded — execution re-resolves
    what it needs.  Checks, in order: experiment defaults apply, every
    action name (request-level; unknown nested ``run`` references included),
    the trigger profile + variant, the save-set rule (a non-optimize
    request needs at least one — optimize may run save-set-less because the
    optimizer's ``device_requirements`` are auto-provisioned at execution
    time, where an empty *effective* set still refuses pre-claim), every
    named save set + entry rituals, every step-axis movable target
    (pseudo-variable forward formulas compiled here — a bad expression
    fails now, not mid-scan), and every optimize VOCS catalog name (bare names
    only — ``Device:Variable`` strings pass through, matching the runner's
    dispatch).

    Parameters
    ----------
    request :
        The scan request to validate.
    resolver :
        Resolves the request's names to schema models.

    Returns
    -------
    tuple[ScanRequest, dict, Any]
        The **post-defaults** validated copy of *request*, the applied-
        defaults provenance record (as :func:`apply_experiment_defaults`),
        and the raw defaults object itself — returned so execution reads
        the defaults file exactly once per run (flags that are not request
        fields, e.g. ``background_telemetry``, must come from the same
        snapshot the validation applied, not a second read that could see
        a concurrent edit).

    Raises
    ------
    GeecsConfigurationError
        Unresolvable names, an unknown trigger variant, a step/noscan
        request without a save set, or a pseudo scan variable whose
        ``forward`` expression fails to compile.
    """
    defaults = resolve_experiment_defaults(resolver)
    validated, applied = apply_experiment_defaults(request, defaults)
    resolve_and_validate_actions(validated.actions, resolver)

    if validated.capture.trigger_profile:
        # Adapt (and discard) the writes so an unknown trigger_variant
        # fails here, not at execution time.
        profile = resolver.resolve_trigger_profile(validated.capture.trigger_profile)
        trigger_writes_from_profile(profile, validated.capture.trigger_variant)

    if not validated.capture.save_sets:
        if validated.mode is not ScanRequestMode.OPTIMIZE:
            raise GeecsConfigurationError(
                f"a {validated.mode.value!r} ScanRequest needs at least "
                "one save set in capture.save_sets — without one the scan "
                "would record nothing"
            )
    else:
        resolve_save_sets_and_rituals(resolver, validated.capture.save_sets)

    if validated.mode is ScanRequestMode.STEP:
        for axis in validated.axes:
            spec = resolver.resolve_scan_variable(axis.variable)
            resolve_movable_target(spec, axis.variable)
    if validated.mode is ScanRequestMode.OPTIMIZE and validated.optimization:
        for name in validated.optimization.variables:
            if ":" not in name:
                spec = resolver.resolve_scan_variable(name)
                resolve_movable_target(spec, name)

    return validated, applied, defaults


@dataclass(frozen=True)
class PlainMovableTarget:
    """One plain scan-variable target: a single ``Device:Variable`` write."""

    device: str
    variable: str
    kind: str
    confirm: str | None

    @property
    def label(self) -> str:
        """The ``Device:Variable`` string recorded as the scan parameter."""
        return f"{self.device}:{self.variable}"


@dataclass(frozen=True)
class PseudoMovableTarget:
    """One pseudo scan variable: components with compiled forward formulas."""

    variable_name: str
    mode: str
    components: tuple[tuple[str, str, CompiledForward], ...]

    @property
    def label(self) -> str:
        """The catalog friendly name recorded as the scan parameter."""
        return self.variable_name

    def metadata(self) -> dict[str, Any]:
        """The provenance record for ``md["pseudo_variables"]``."""
        return {
            "mode": self.mode,
            "targets": [
                {"target": f"{device}:{variable}", "forward": forward.source}
                for device, variable, forward in self.components
            ],
        }


MovableTarget = PlainMovableTarget | PseudoMovableTarget


@dataclass(frozen=True)
class StepScanSpec:
    """The pure execution picture of a step/noscan request (no devices).

    Everything :func:`run_scan_request` derives from the request *before*
    touching hardware, packaged so the legacy entry point and the
    queueserver plan preamble
    (:func:`~geecs_bluesky.plans.scan_request_plan.geecs_scan_request_plan`)
    share one definition of the run metadata / ScanInfo / grid shapes and
    cannot drift (the acceptance contract is document equality).

    Attributes
    ----------
    md :
        The run-metadata dict (``scan_request_mode``, provenance keys,
        grid/pseudo metadata) — the caller merges ``description`` on top.
    scan_info :
        The ScanInfo ini overrides (``scan_mode``, ``scan_parameter``,
        legacy 1-D grid fields).
    positions :
        The positions list handed to the scan plan: ``[None]`` for noscan,
        the axis values for a single axis, grid-point tuples for a grid.
    n_steps, n_shots :
        Progress totals (recorded for consumers of the built spec; the
        GUI's live totals now come from the run start document).
    """

    md: dict[str, Any]
    scan_info: dict[str, Any]
    positions: list[Any]
    n_steps: int
    n_shots: int


def build_step_scan_spec(
    request: ScanRequest,
    axis_resolved: list["MovableTarget"],
    *,
    applied_defaults: Mapping[str, Any],
    slots: Mapping[str, list[str]],
    dropped_unserved: Mapping[str, list[str]],
    dropped_unserved_devices: list[str],
    disconnected_devices: list[str],
    telemetry_selected: Mapping[str, list[str]],
    capture_devices: list[str] | None = None,
    native_image_save: bool = True,
    submission: Any | None = None,
) -> StepScanSpec:
    """Assemble the pure run picture of a step/noscan request.

    Relocated verbatim from :func:`run_scan_request`'s inline metadata
    assembly (queueserver round 1) so the plan preamble reuses it; behavior
    is pinned by the existing runner suite.  Multi-axis requests become an
    outer-product grid (first axis outermost/slowest, one bin per grid
    point) with the legacy 1-D ScanInfo fields describing the outermost
    axis.

    Parameters
    ----------
    request :
        The **post-defaults** validated request (step or noscan mode).
    axis_resolved :
        The resolved movable target per ``request.axes`` entry (empty for
        noscan) — :func:`resolve_movable_target` output, in axis order.
    applied_defaults, slots, dropped_unserved, dropped_unserved_devices,
    telemetry_selected :
        The provenance records accumulated by the prologue; each lands in
        run metadata only when non-empty (pass ``{}``/``[]`` to omit —
        e.g. ``telemetry_selected`` must already be gated on the
        telemetry-enabled flag).
    submission :
        The client's :class:`~geecs_schemas.SubmissionRecord` (model or
        JSON dict), traveling beside the request since geecs-schemas
        0.14.0; ``None`` when the client stamped nothing.
    """
    md: dict[str, Any] = {"scan_request_mode": request.mode.value}
    # Provenance: which named save sets were unioned for this scan.
    md["save_sets"] = list(request.capture.save_sets)
    if capture_devices:
        # The capture daemon's device list (it prefers this over inferring
        # from nonscalar_save_paths) + the effective toggle, for provenance
        # and the dual-write diff. run_wrapper mkdirs these device dirs
        # pre-start-doc, engine-side.
        md["capture_devices"] = list(capture_devices)
        md["native_image_save"] = bool(native_image_save)
    elif not native_image_save:
        # Off was requested but nothing was eligible (DB blip / no registry
        # cameras): record the unhonored intent so the run's provenance —
        # and the dual-write diff — can see the request was inert.
        md["native_image_save"] = False
    if dropped_unserved:
        # Provenance: variables (and whole devices) dropped by the
        # unserved-variables pre-flight — the run proceeded without them.
        md["dropped_unserved_variables"] = {
            dev: list(vars_) for dev, vars_ in dropped_unserved.items()
        }
    if dropped_unserved_devices:
        md["dropped_unserved_devices"] = list(dropped_unserved_devices)
    if disconnected_devices:
        # Provenance: snapshot devices the CONNECTED re-check found down
        # at execution — the run proceeded without their columns (#664;
        # dead synchronous devices refuse pre-claim and never get here).
        md["disconnected_devices"] = list(disconnected_devices)
    if applied_defaults:
        # Provenance: the run records exactly which experiment defaults
        # filled in fields the submitter left unset.
        md["applied_defaults"] = metadata_applied_defaults(applied_defaults)
    submission_md = metadata_submission(submission)
    if submission_md is not None:
        # Provenance: the submitting client's record — who queued the
        # request, when, and the pre-submit preflight outcomes (#648).
        md["submission"] = submission_md
    if any(slots.values()):
        # Provenance: the assembled per-slot execution order (defaults +
        # entry rituals + the request's own, mirrored on closeout).
        md["action_plans"] = {k: list(v) for k, v in slots.items() if v}
    if telemetry_selected:
        md["background_telemetry"] = {
            dev: list(vars_) for dev, vars_ in telemetry_selected.items()
        }
    scan_info: dict[str, Any] = {
        "shots": request.capture.shots_per_step,
        "background": request.background,
    }

    if request.mode is ScanRequestMode.NOSCAN:
        scan_info["scan_mode"] = "noscan"
        return StepScanSpec(md, scan_info, [None], 1, request.capture.shots_per_step)

    targets = [target.label for target in axis_resolved]
    value_lists = [axis.positions.to_values() for axis in request.axes]
    pseudo_meta = {
        target.variable_name: target.metadata()
        for target in axis_resolved
        if isinstance(target, PseudoMovableTarget)
    }
    if pseudo_meta:
        md["pseudo_variables"] = pseudo_meta

    scan_info["scan_mode"] = "standard"
    scan_info["scan_parameter"] = ",".join(targets)
    if len(request.axes) == 1:
        positions: list[Any] = list(value_lists[0])
        md["scan_variable"] = request.axes[0].variable
    else:
        # Outer product, first axis outermost/slowest (the schema's
        # documented grid semantics); one bin per grid point.
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
    return StepScanSpec(
        md,
        scan_info,
        positions,
        len(positions),
        len(positions) * request.capture.shots_per_step,
    )


def resolve_movable_target(spec: ScanVariableSpec, name: str) -> MovableTarget:
    """Resolve a catalog entry into its movable target, fail-fast.

    A plain :class:`~geecs_schemas.scan_variables.ScanVariable` becomes a
    :class:`PlainMovableTarget` (``confirm`` is the entry's optional
    confirming ``"Device:Variable"``, ``None`` when the set variable is also
    the readback).  A :class:`~geecs_schemas.scan_variables.PseudoScanVariable`
    becomes a :class:`PseudoMovableTarget` with every component's ``forward``
    formula compiled here — so a bad expression fails at validation time,
    pre-claim, never mid-scan.

    Raises
    ------
    GeecsConfigurationError
        A pseudo component's ``forward`` expression fails to compile.
    """
    if isinstance(spec, PseudoScanVariable):
        components = []
        for component in spec.targets:
            device, _, variable = component.target.partition(":")
            try:
                forward = compile_forward(component.forward)
            except GeecsConfigurationError as exc:
                raise GeecsConfigurationError(
                    f"pseudo scan variable {name!r}, target {component.target!r}: {exc}"
                ) from exc
            components.append((device, variable, forward))
        return PseudoMovableTarget(
            variable_name=name,
            mode=spec.mode.value,
            components=tuple(components),
        )
    assert isinstance(spec, ScanVariable)
    device, _, variable = spec.target.partition(":")
    return PlainMovableTarget(device, variable, spec.kind, spec.confirm)


def build_movable(session: Any, target: MovableTarget) -> Any:
    """Build the right movable for one resolved scan-variable target.

    A :class:`PseudoMovableTarget` builds via :meth:`GeecsSession.pseudo_movable`
    (one number fanned out to every component).  For a plain target,
    ``confirm`` (a ``"Device:Variable"`` string) takes precedence over
    ``kind``: a variable with a confirming target is the topology-C case
    (:class:`~geecs_bluesky.devices.ca.confirm.CaConfirmSettable`) regardless
    of whether it is also declared ``kind: motor`` — the confirming poll is
    the more specific completion check.  Otherwise dispatches on ``kind`` as
    before: ``"motor"`` → :meth:`GeecsSession.motor`, else
    :meth:`GeecsSession.settable`.
    """
    if isinstance(target, PseudoMovableTarget):
        return session.pseudo_movable(
            target.variable_name, list(target.components), target.mode
        )
    if target.confirm is not None:
        confirm_device, _, confirm_variable = target.confirm.partition(":")
        return session.confirm_settable(
            target.device,
            target.variable,
            confirm_device=confirm_device,
            confirm_variable=confirm_variable,
        )
    if target.kind == "motor":
        return session.motor(target.device, target.variable)
    return session.settable(target.device, target.variable)


def _build_request_detectors(
    session: Any, devices_config: dict[str, dict[str, Any]], *, free_run: bool
) -> list:
    """Create session devices from a derived devices_config, roles by order.

    Role assignment by config order (the one definition since G3
    deleted the bridge-side twin): free-run → first
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
        save_control_only = bool(cfg.get("save_control_only", False))
        synchronous = bool(cfg.get("synchronous", False))
        if not synchronous:
            if not variables:
                logger.warning(
                    "Skipping asynchronous device %s: no scalars to record",
                    device_name,
                )
                continue
            detectors.append(
                session.snapshot(
                    device_name, variables, save_control_only=save_control_only
                )
            )
        elif free_run and reference_assigned:
            detectors.append(
                session.contributor(
                    device_name,
                    variables,
                    save_images=save,
                    save_control_only=save_control_only,
                )
            )
        else:
            detectors.append(
                session.detector(
                    device_name,
                    variables,
                    save_images=save,
                    save_control_only=save_control_only,
                )
            )
            reference_assigned = True
    return detectors


# ---------------------------------------------------------------------------
# Native-image-save toggle (capture arc, Planning/data_capture/): whether
# capture-eligible cameras write native per-shot files or the capture daemon
# owns their images. One devicetype-scoped decision, never per-entry config.
# ---------------------------------------------------------------------------


def resolve_native_image_save(request: ScanRequest, defaults: Any) -> bool:
    """Effective native-image-save flag: request override else experiment default."""
    return (
        request.capture.native_image_save
        if request.capture.native_image_save is not None
        else _defaults_flag(defaults, "native_image_save", True)
    )


def select_capture_devices(
    experiment: str,
    devices_config: dict[str, dict[str, Any]],
    *,
    provider: Any | None = None,
) -> list[str]:
    """Image-saving devices whose devicetype the capture daemon owns.

    Devicetypes come from the failure-tolerant
    :class:`~geecs_bluesky.db_runtime.GeecsDbDeviceTypes` provider — a DB
    failure yields an empty mapping, so nothing is capture-eligible and
    native saving is never switched off on a DB blip (fail-open keeps data).
    """
    from geecs_bluesky.capture.discovery import CAPTURE_DEVICE_TYPES

    if provider is None:
        if not experiment:
            # No experiment context (hermetic sessions, defensive) — nothing
            # is capture-eligible; native saving stays untouched.
            return []
        provider = GeecsDbDeviceTypes(experiment)
    types = provider.by_device()
    return [
        name
        for name, cfg in devices_config.items()
        if cfg.get("save_nonscalar_data") and types.get(name) in CAPTURE_DEVICE_TYPES
    ]


def preflight_capture_liveness(
    capture_devices: list[str], native_image_save: bool
) -> None:
    """Refuse a toggle-off scan when the capture daemon looks absent.

    Pre-claim and fail-CLOSED (the opposite convention from the DB
    providers, deliberately): with native saving off, a dead daemon means
    the captured cameras' images exist NOWHERE — refusing before a scan
    number is burned is the only safe answer. Beyond freshness, the
    heartbeat's ``targets`` roster must cover every requested capture
    device — a daemon started before a camera joined the DB roster is
    alive but not monitoring it, which is the same nowhere. Only consulted
    when the operator explicitly requested off and capture devices exist;
    the default dual-write path never touches it.
    """
    if native_image_save or not capture_devices:
        return
    from geecs_bluesky.capture.heartbeat import (
        STALE_AFTER_S,
        heartbeat_path,
        read_heartbeat,
    )

    payload = read_heartbeat()
    age = (
        max(0.0, time.time() - float(payload["time"])) if payload is not None else None
    )
    if age is None or age > STALE_AFTER_S:
        raise GeecsConfigurationError(
            "native_image_save=off refused: the capture daemon looks absent "
            f"(heartbeat {heartbeat_path()} "
            f"{'missing/unreadable' if age is None else f'{age:.0f}s old'}, "
            f"stale after {STALE_AFTER_S:.0f}s; the check reads the daemon's "
            "heartbeat on THIS host — a daemon on another machine cannot "
            "satisfy it) — with native saving off, "
            f"{', '.join(capture_devices)} would be recorded NOWHERE. Start "
            "the capture daemon, or run with native_image_save unset/true."
        )
    roster = payload.get("targets")
    if not isinstance(roster, list):
        # The daemon always writes a device-name roster; a fresh heartbeat
        # without one is corrupt or from something that is not the daemon —
        # fail closed, coverage cannot be verified (codex gate P2).
        raise GeecsConfigurationError(
            "native_image_save=off refused: the heartbeat at "
            f"{heartbeat_path()} carries no device roster, so coverage of "
            f"{', '.join(capture_devices)} cannot be verified. Restart the "
            "capture daemon, or run with native_image_save unset/true."
        )
    missing = sorted(set(capture_devices) - {str(t) for t in roster})
    if missing:
        raise GeecsConfigurationError(
            "native_image_save=off refused: the capture daemon is alive "
            f"but not monitoring {', '.join(missing)} (its heartbeat "
            "roster predates them) — their images would be recorded "
            "NOWHERE. Restart the capture daemon to re-discover the "
            "roster, or run with native_image_save unset/true."
        )


def apply_native_image_save_off(
    devices_config: dict[str, dict[str, Any]], capture_devices: list[str]
) -> dict[str, dict[str, Any]]:
    """Return a copy of *devices_config* with native saving off for *capture_devices*.

    The devices stay full scan participants (scalars, shot-id columns,
    strict-row membership); only the native file save — ``localsavingpath``
    / ``save`` writes and the PNG-pointing asset documents — is suppressed
    (``session._configure_saving`` gates on the resulting
    ``save_nonscalar_data``).
    """
    updated = {name: dict(cfg) for name, cfg in devices_config.items()}
    for name in capture_devices:
        if name in updated:
            updated[name]["save_nonscalar_data"] = False
            # Active off-write surface: the detector gets ONLY the `save`
            # control child, and the run wrapper commands "off" at scan
            # start — a flag left on out-of-band must never keep writing
            # native files to a stale path (codex finding on PR #697).
            updated[name]["save_control_only"] = True
    return updated


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


def make_served_set_provider(session: Any) -> GeecsDbServedSetProvider | None:
    """Build the gateway served-set provider for *session*'s experiment.

    The unserved-variables pre-flight check resolves every devices-config
    variable against this provider's served set (``get='yes'`` union
    settable variables of enabled devices — the gateway's serving rule).
    The provider is failure-tolerant: a DB failure makes
    ``served_by_device()`` return ``None`` and the check is skipped with a
    warning — a scan never aborts because the DB blipped.

    Parameters
    ----------
    session :
        The :class:`~geecs_bluesky.session.GeecsSession` (duck-typed).

    Returns
    -------
    GeecsDbServedSetProvider or None
        A DB-backed provider, or ``None`` when no experiment is known
        (the check is then skipped entirely).
    """
    experiment = getattr(session, "experiment", None)
    if not experiment:
        return None
    return GeecsDbServedSetProvider(experiment)


def _preflight_unserved(
    session: Any,
    devices_config: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, list[str]], list[str]]:
    """Run the unserved-variables check over *devices_config* (pre-claim).

    Thin glue between :func:`make_served_set_provider` and
    :func:`~geecs_bluesky.preflight.run_unserved_variables_check`; returns
    as the latter.  Headless by construction (queueserver decision 3): the
    operator was asked client-side at submission, so a raised question
    takes its continue-and-drop default with a WARNING.
    """
    provider = make_served_set_provider(session)
    return run_unserved_variables_check(
        devices_config,
        provider.served_by_device if provider is not None else None,
    )


def _preflight_connected(
    session: Any,
    devices_config: dict[str, dict[str, Any]],
) -> list[str]:
    """Pre-claim CONNECTED liveness re-check over *devices_config* (#664).

    The client-side pre-submit preflight reads the same PVs, but the
    submission-to-execution gap under the queue is long (a device can die
    while the item waits, or the client skipped its checks) — so the
    worker re-checks at execution, exactly like it re-runs validation and
    the unserved-variables check.  Headless dispositions:

    - A **synchronous** device reading the exact ``Disconnected`` choice
      string refuses the scan pre-claim (:class:`GeecsDeviceDownError`
      naming it) — in both modes.  Strict rows await every synchronous
      device; free-run rows are paced by the reference AND every dead
      sync contributor would fail t0 sync post-claim anyway (its stale
      cached timestamp blows the spread window, and the seed gate refuses
      it by name) — so a warn-and-continue disposition for free-run
      contributors would be a fiction, deferring the same death past the
      claim.  Refusing pre-claim burns no scan number.
    - A Disconnected **asynchronous** (snapshot) device warns and
      continues — its columns are sampled best-effort and a queued scan
      survives without them.  The list is recorded in run metadata as
      ``disconnected_devices``.
    - Fail-open everywhere else (the liveness doctrine): an unreadable
      PV, no experiment, or missing CA support is never a verdict.

    The probe itself (one concurrent batch read, one timeout budget — the
    queue-plan paths run this inside the RE loop, so per-device
    sequential timeouts are not acceptable) is the shared
    :func:`~geecs_bluesky.devices.ca.liveness.probe_disconnected`.

    Returns
    -------
    list of str
        The Disconnected devices the scan continues without.
    """
    experiment = getattr(session, "experiment", None)
    if not experiment or not devices_config:
        return []
    try:
        from geecs_bluesky.devices.ca.liveness import probe_disconnected

        down = probe_disconnected(experiment, devices_config)
    except Exception as exc:  # probe machinery broke — fail open, never block
        logger.warning("CONNECTED pre-flight skipped: %s", exc)
        return []
    if not down:
        return []
    fatal = [d for d in down if devices_config[d].get("synchronous")]
    if fatal:
        raise GeecsDeviceDownError(
            f"gateway reports {', '.join(sorted(fatal))} as Disconnected — "
            "a synchronous device the scan's rows cannot complete without; "
            "restart the device and resubmit (pre-claim — no scan number "
            "was burned)",
            device_name=fatal[0],
        )
    logger.warning(
        "gateway reports %s as Disconnected — continuing; their columns "
        "will be missing or invalid (recorded as disconnected_devices)",
        ", ".join(sorted(down)),
    )
    return down


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
        ``(readables, recorded)`` — a single-element list holding the
        :class:`~geecs_bluesky.devices.ca.telemetry.CaTelemetryGroup` of
        connected devices (empty list when none connected), and the
        ``{device: [variables]}`` map of **only those that connected**
        (run-metadata key must match the columns that actually exist).
        The group costs one RunEngine ``read`` message per row instead of
        one per device; event columns are identical to the ungrouped
        layout (members keep their own names).
    """
    if scalar_policy is None:
        return [], {}
    selected = select_telemetry_variables(
        save_set, scalar_policy.subscribed_by_device()
    )
    members: list = []
    recorded: dict[str, list[str]] = {}
    if hasattr(session, "telemetry_batch"):
        # Concurrent connects: wall time = slowest device, not the sum
        # (~87 sequential connects cost ~9 s of start latency; measured
        # live 2026-07-13).  Fake sessions without the batch method fall
        # through to the sequential per-device factory below.
        members = list(session.telemetry_batch(selected))
        for readable in members:
            device = getattr(readable, "_geecs_device_name", None)
            if device in selected:
                recorded[device] = list(selected[device])
    else:
        for device, variables in selected.items():
            readable = session.telemetry(device, variables)
            if readable is not None:
                members.append(readable)
                recorded[device] = list(variables)
    # Record only the devices that actually connected: a device dropped as
    # unreachable at scan start contributes no columns, so the start-doc
    # metadata must not advertise them (EVENT_SCHEMA.md contract — the key
    # reflects what was recorded, not what was selected).
    if not members:
        return [], recorded
    # Lazy import: the runner stays free of device-layer (aioca) imports so
    # it remains importable without the `ca` extra.
    from geecs_bluesky.devices.ca.telemetry import CaTelemetryGroup

    return [CaTelemetryGroup(members)], recorded


def _stopped_during_init(
    session: Any, should_abort: Callable[[], bool] | None, stage: str
) -> bool:
    """Return ``True`` when an operator stop arrived during initialization.

    The init-stage checkpoints of :func:`run_scan_request`: all of them run
    **pre-claim**, so a stop caught here burns no scan number.  On trip the
    session's ``last_run_aborted`` is set (the quiet aborted-outcome
    contract from #563 — callers distinguish "stopped" from "failed"
    without an exception) and one INFO line is logged; the caller's
    ``finally`` still owns disconnection of everything it created.
    """
    if should_abort is None or not should_abort():
        return False
    logger.info(
        "scan stopped during initialization (%s); nothing claimed — "
        "no scan number was burned",
        stage,
    )
    session.last_run_aborted = True
    return True


def resolve_and_apply_capture_toggle(
    request: "ScanRequest",
    defaults: Any,
    devices_config: dict[str, dict[str, Any]],
    session: Any,
) -> tuple[dict[str, dict[str, Any]], list[str], bool]:
    """Resolve the ``native_image_save`` toggle and apply it — the ONE block.

    Shared by both execution paths (the headless runner and the queue
    plan) like every other prologue step, so the toggle contract cannot
    drift between them (ultra review of PR #693). Returns the possibly
    rewritten devices config, the capture-eligible device list, and the
    effective flag. Refuses toggle-off pre-claim via
    :func:`preflight_capture_liveness`.
    """
    native_image_save = resolve_native_image_save(request, defaults)
    capture_devices = select_capture_devices(
        getattr(session, "experiment", ""), devices_config
    )
    if not native_image_save:
        if capture_devices:
            preflight_capture_liveness(capture_devices, native_image_save)
            devices_config = apply_native_image_save_off(
                devices_config, capture_devices
            )
            logger.info(
                "native_image_save=off: capture daemon owns images for %s",
                ", ".join(capture_devices),
            )
        else:
            logger.warning(
                "native_image_save=off requested but no capture-eligible "
                "devices resolved (DB unreachable, or no registry-devicetype "
                "cameras in the save set) — native saving unchanged"
            )
    return devices_config, capture_devices, native_image_save


def run_scan_request(
    session: Any,
    request: ScanRequest,
    resolver: ConfigResolver,
    *,
    objective: Any | None = None,
    suggester: Any | None = None,
    optimization_binder: Callable[..., tuple[Any, Any]] | None = None,
    device_requirements: Any | None = None,
    should_abort: Callable[[], bool] | None = None,
    submission: Any | None = None,
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
        scanner-layer hook (the queueserver worker's registered
        ``optimization_loader`` seam) called as
        ``binder(devices=..., scan_tag=..., scan_folder=...) ->
        (objective, suggester)`` with the connected movables + detectors
        and the freshly claimed scan (the runner claims pre-bind; see
        :func:`_run_optimize_request`).  Ignored when *objective* and
        *suggester* are given.
    device_requirements :
        Optional optimizer device requirements for ``optimize`` mode
        (duck-typed ``{"Devices": {name: cfg}}`` mapping — e.g. the
        loader-returned bridge's ``device_requirements`` attribute,
        auto-generated by the optimization stack from its evaluator's
        analyzers).  Merged into the effective devices config so the
        objective's diagnostics acquire and save even when the save sets
        do not name them (:func:`merge_optimizer_device_requirements`);
        the merged additions run through the same unserved-variables
        pre-flight as everything else and are recorded in run metadata as
        ``provisioned_device_requirements``.  ``None``/empty is a no-op;
        ignored on non-optimize modes.
    should_abort :
        Optional external-stop probe, consulted between the
        initialization stages — after configuration resolution, after
        device connect, and immediately before the scan-number claim —
        because ``RE.abort()`` cannot stop a scan that has not reached the
        RunEngine yet.  Every checkpoint is pre-claim, so an init-stage
        stop burns no scan number; on trip the runner disconnects what it
        created, logs one INFO line, sets ``session.last_run_aborted``
        (the #563 aborted-outcome contract) and returns ``None``.  The
        callable is also handed to ``session.scan``/``session.optimize``,
        whose in-plan gate closes the residual window between the last
        checkpoint here and the engine reporting ``running``.  ``None``
        (headless default) checks nothing.
    submission :
        Optional client :class:`~geecs_schemas.SubmissionRecord` (model or
        JSON dict) traveling beside the request (geecs-schemas 0.14.0
        split it out of the document); recorded verbatim in run metadata,
        never acted on.

    Returns
    -------
    str or None
        The Bluesky run uid (``None`` when nothing was persisted).

    Raises
    ------
    NotImplementedError
        Optimize mode without an injected objective/suggester or
        optimization_binder (a documented v1 gap — validated first,
        refused loudly).
    GeecsConfigurationError
        Unresolvable names, a step/noscan request without a save set, or
        an optimize request with an empty effective device set (no save
        sets and no optimizer device requirements to provision).
    """
    # Phase 1 — the one fail-fast definition (issue #529): everything that
    # must resolve does so here, before any session state is touched.
    # Clients run the same function pre-submit (the console's preflight),
    # so submission-time and execution-time validation cannot drift.
    # The returned raw defaults object serves execution-time flags that are
    # not request fields (background_telemetry) from the SAME file snapshot
    # the validation applied — one read per run, no torn-edit window.
    request, applied_defaults, defaults = validate_scan_request(request, resolver)
    resolved_actions = resolve_and_validate_actions(request.actions, resolver)

    if request.capture.trigger_profile:
        profile = resolver.resolve_trigger_profile(request.capture.trigger_profile)
        session.shot_control(
            trigger_writes_from_profile(profile, request.capture.trigger_variant)
        )
    else:
        session.shot_control(None)

    mode = (
        "strict"
        if request.capture.acquisition is AcquisitionMode.STRICT
        else "free_run"
    )

    if request.mode is ScanRequestMode.OPTIMIZE:
        # Optimize has no action hooks yet: skip actions (never refuse — that
        # would block optimization wherever defaults define bracket actions),
        # log loudly, and record the skip in run metadata (never silent).
        # Rationale: GeecsBluesky/CLAUDE.md (engine consolidation).
        skipped_actions = {k: v for k, v in resolved_actions.items() if v}
        if request.capture.native_image_save is False:
            # v0: optimize keeps native saving unconditionally (its
            # evaluators read per-shot files from disk). Never silent —
            # once an experiment default flips off, every optimize scan
            # overrides it and the operator must be able to see that.
            logger.warning(
                "optimize mode keeps native image saving — the request's "
                "native_image_save=false is ignored (v0: evaluators read "
                "per-shot files)"
            )
        return _run_optimize_request(
            session,
            request,
            resolver,
            mode,
            objective=objective,
            suggester=suggester,
            optimization_binder=optimization_binder,
            device_requirements=device_requirements,
            applied_defaults=applied_defaults,
            skipped_actions=skipped_actions,
            should_abort=should_abort,
            submission=submission,
        )

    # A step/noscan request without a save set was already refused by
    # validate_scan_request (phase 1) — no second copy of that rule here.
    # Multiple named save sets union into one effective save set (devices
    # deduped/merged; rituals collected across all sets, deduped by name).
    save_set, rituals = resolve_save_sets_and_rituals(
        resolver, request.capture.save_sets
    )

    # M3c get-side runtime: failure-tolerant policy provider (a DB blip never
    # aborts a scan); the DB set-side stays disabled (reserved fields warn).
    scalar_policy = make_scalar_policy(session)
    devices_config = save_set_to_devices_config(save_set, scalar_policy)
    # Unserved-variables pre-flight (pre-claim, pre-device-build): a variable
    # the gateway does not serve has no PV, so its detector could never
    # connect — drop it now with a WARNING (headless; the operator was
    # asked client-side pre-submit, decision 3)
    # instead of dying in a 20 s NotConnectedError during connect.
    checked_config, dropped_unserved, dropped_unserved_devices = _preflight_unserved(
        session, devices_config
    )
    if checked_config is None:
        logger.warning(
            "ScanRequest preflight aborted the scan (unserved save-set "
            "variables; pre-claim — no scan number was burned)"
        )
        return None
    devices_config = checked_config
    # CONNECTED liveness re-check (#664): the client asked pre-submit, but
    # the queue's submission-to-execution gap is long — re-check here,
    # refusing only when a row could never complete.
    disconnected_devices = _preflight_connected(session, devices_config)
    if _stopped_during_init(session, should_abort, "after configuration resolution"):
        return None
    slots = assemble_action_slots(request.actions, applied_defaults, rituals)
    warn_if_reserved_boundary_overrides(save_set)

    # Resolve the scan-variable movable targets up front (full movable
    # construction happens later; only the resolved targets are needed here
    # for the standard-scan build below).
    axis_resolved: list[MovableTarget] = []
    for axis in request.axes:
        spec = resolver.resolve_scan_variable(axis.variable)
        axis_resolved.append(resolve_movable_target(spec, axis.variable))

    telemetry_enabled = (
        request.capture.background_telemetry
        if request.capture.background_telemetry is not None
        else _defaults_flag(defaults, "background_telemetry", True)
    )

    devices_config, capture_devices, native_image_save = (
        resolve_and_apply_capture_toggle(request, defaults, devices_config, session)
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

        if _stopped_during_init(session, should_abort, "after device connect"):
            return None

        movables: list = []
        for target in axis_resolved:
            movable = build_movable(session, target)
            created.append(movable)
            movables.append(movable)
        if request.mode is ScanRequestMode.NOSCAN:
            motor_arg: Any = None
        elif len(request.axes) == 1:
            motor_arg = movables[0]
        else:
            motor_arg = movables

        # The pure run picture (md / ScanInfo / positions / totals) — one
        # definition shared with the queueserver plan preamble.
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

        # The last pre-claim checkpoint: the claim happens inside
        # session.scan, immediately below.
        if _stopped_during_init(session, should_abort, "before scan-number claim"):
            return None
        return session.scan(
            detectors=detectors,
            motor=motor_arg,
            positions=spec.positions,
            shots_per_step=request.capture.shots_per_step,
            mode=mode,
            description=request.description,
            md=spec.md,
            scan_info=spec.scan_info,
            setup=setup,
            per_step=per_step,
            closeout=closeout,
            should_abort=should_abort,
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
    device_requirements: Any | None = None,
    applied_defaults: dict[str, Any] | None = None,
    skipped_actions: dict[str, list[str]] | None = None,
    should_abort: Callable[[], bool] | None = None,
    submission: Any | None = None,
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
    optimization_binder, device_requirements, should_abort :
        As in :func:`run_scan_request` (the unserved-variables check runs
        here too, pre-claim, over the effective devices config — save-set
        devices *and* optimizer-provisioned ones; the *should_abort*
        init-stage checkpoints run after device connect and immediately
        before the claim — which on this path is the runner's own,
        pre-bind).  With a
        binder (and no ready-made callables) the runner claims the scan
        itself, pre-bind, so the
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
    GeecsConfigurationError
        The effective device set is empty — no save sets named and no
        optimizer device requirements to provision (pre-claim; the
        objective would have nothing to read).
    """
    spec = request.optimization
    assert spec is not None  # guaranteed by ScanRequest validation
    if (objective is None or suggester is None) and optimization_binder is None:
        raise NotImplementedError(
            "optimize-mode ScanRequest execution needs a ready-made "
            "objective and suggester (run(request, resolver, objective=..., "
            "suggester=...)) or an optimization_binder: instantiating them "
            "from the request's evaluator/generator specs lives in the "
            "optimization stack (geecs_bluesky.optimization, the `optimize` "
            "extra), wired in by a caller-provided loader/binder"
        )

    detectors: list = []
    created: list = []
    dropped_unserved: dict[str, list[str]] = {}
    dropped_unserved_devices: list[str] = []
    disconnected_devices: list[str] = []
    try:
        skipped = {k: list(v) for k, v in (skipped_actions or {}).items() if v}
        # db_scalars applies to optimize too; telemetry does not run here yet
        # (no scan-boundary hook); the DB set-side stays disabled everywhere.
        scalar_policy = make_scalar_policy(session)
        devices_config: dict[str, dict[str, Any]] | None = {}
        if request.capture.save_sets:
            save_set, rituals = resolve_save_sets_and_rituals(
                resolver, request.capture.save_sets
            )
            # Reserved DB set-side overrides are inert here too — warn once, as
            # on the scan/noscan path, so the promise holds in every mode.
            warn_if_reserved_boundary_overrides(save_set)
            ritual_names = [n for names in rituals.values() for n in names]
            if ritual_names:
                # Save-set entry rituals can't run in optimize mode yet either;
                # skip and record rather than refuse (see run_scan_request).
                skipped["save_set_rituals"] = ritual_names
            devices_config = save_set_to_devices_config(save_set, scalar_policy)
        # Auto-provision the optimizer's device requirements: the objective's
        # diagnostics acquire and save even when the save sets don't name
        # them — or when the request names no save sets at all (field
        # incident 2026-07-15: the evaluator's auto-generated requirements
        # were ignored, the diagnostic never saved, every objective was NaN).
        provisioned = merge_optimizer_device_requirements(
            devices_config, device_requirements
        )
        if not devices_config:
            raise GeecsConfigurationError(
                "an 'optimize' ScanRequest needs at least one recording "
                "device — name save sets in save_sets, or use an optimizer "
                "whose evaluator declares device_requirements (auto-generated "
                "from its analyzers); without either the objective has "
                "nothing to read"
            )
        # Unserved-variables pre-flight, exactly as on the scan/noscan
        # path (pre-claim: the optimize claim happens further down, so
        # an abort here burns no scan number).  Provisioned devices go
        # through the same check as save-set ones.  The detector-level
        # operator preflight hook still does not run on optimize (its
        # seam is unchanged); this config-level check does.
        (
            devices_config,
            dropped_unserved,
            dropped_unserved_devices,
        ) = _preflight_unserved(session, devices_config)
        if devices_config is not None:
            # CONNECTED liveness re-check (#664), same terms as the
            # scan/noscan path.
            disconnected_devices = _preflight_connected(session, devices_config)
        if devices_config is None:
            logger.warning(
                "ScanRequest preflight aborted the optimization (unserved "
                "save-set variables; pre-claim — no scan number was burned)"
            )
            return None
        detectors = _build_request_detectors(
            session,
            devices_config,
            free_run=mode == "free_run",
        )
        created.extend(detectors)

        variables: dict[str, Any] = {}
        pseudo_meta: dict[str, Any] = {}
        for name in spec.variables:
            if ":" in name:
                device, _, variable = name.partition(":")
                movable = session.settable(device, variable)
            else:
                var_spec = resolver.resolve_scan_variable(name)
                target = resolve_movable_target(var_spec, name)
                if isinstance(target, PseudoMovableTarget):
                    pseudo_meta[target.variable_name] = target.metadata()
                movable = build_movable(session, target)
            variables[name] = movable
            created.append(movable)

        if _stopped_during_init(session, should_abort, "after device connect"):
            return None

        md: dict[str, Any] = {"scan_request_mode": request.mode.value}
        if pseudo_meta:
            md["pseudo_variables"] = pseudo_meta
        if request.capture.save_sets:
            # Provenance: which named save sets were unioned for this scan.
            md["save_sets"] = list(request.capture.save_sets)
        if provisioned:
            # Provenance: what the optimizer's device_requirements added to
            # the effective device set (whole entries for new devices, the
            # added variables/flags for save-set ones).  Recorded pre-drop —
            # dropped_unserved_* below says what pre-flight then removed.
            md["provisioned_device_requirements"] = provisioned
        if dropped_unserved:
            # Provenance: variables (and whole devices) dropped by the
            # unserved-variables pre-flight — the run proceeded without them.
            md["dropped_unserved_variables"] = {
                dev: list(vars_) for dev, vars_ in dropped_unserved.items()
            }
        if dropped_unserved_devices:
            md["dropped_unserved_devices"] = list(dropped_unserved_devices)
        if disconnected_devices:
            md["disconnected_devices"] = list(disconnected_devices)
        if applied_defaults:
            md["applied_defaults"] = metadata_applied_defaults(applied_defaults)
        submission_md = metadata_submission(submission)
        if submission_md is not None:
            md["submission"] = submission_md
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
            # The last pre-claim checkpoint: on this path the runner claims
            # the scan itself (pre-bind), immediately below.
            if _stopped_during_init(session, should_abort, "before scan-number claim"):
                return None
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

            # The runner claimed → the runner attaches scan.log (the session
            # only self-attaches when *it* claimed the number).
            with scan_log(scan_number, scan_folder) if claimed_here else nullcontext():
                uid, _history = session.optimize(
                    variables=variables,
                    detectors=detectors,
                    objective=objective,
                    suggester=suggester,
                    shots_per_iteration=request.capture.shots_per_step,
                    max_iterations=max_iterations,
                    mode=mode,
                    description=request.description,
                    md=md,
                    on_finish="best" if spec.move_to_best_on_finish else "hold",
                    scan_number=scan_number,
                    scan_folder=scan_folder,
                    should_abort=should_abort,
                )
            if claimed_here and getattr(session, "last_run_aborted", False):
                # session.optimize returned the aborted outcome quietly
                # (operator-requested); note the claimed-but-partial folder
                # calmly (WARNING) instead of the failure ERROR.
                log_claimed_scan_failure(
                    scan_number,
                    scan_folder,
                    label="Optimization scan",
                    aborted=True,
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
