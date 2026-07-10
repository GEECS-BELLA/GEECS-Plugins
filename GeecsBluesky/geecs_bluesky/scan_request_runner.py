"""Run a ScanRequest: resolve config names, map onto the existing machinery.

The one submission object of the target architecture (vision doc §4.1) meets
the engine here: a client builds a
:class:`~geecs_schemas.scan_request.ScanRequest` and calls
``session.run(request)`` (or hands it to ``BlueskyScanner.reinitialize``).
This module owns the mapping:

- A :class:`ConfigResolver` turns the request's *names* into schema models —
  each name in ``save_sets`` → a :class:`~geecs_schemas.save_set.SaveSet`,
  ``trigger_profile`` → :class:`~geecs_schemas.trigger_profile.TriggerProfile`,
  an axis variable → a :class:`~geecs_schemas.scan_variables.ScanVariable`,
  action names → :class:`~geecs_schemas.action_plan.ActionPlan`.
- **Multiple save sets union.**  ``ScanRequest.save_sets`` is a list; the
  engine resolves each named set and unions them into one effective SaveSet
  (:func:`merge_save_sets`) so operators mix and match named diagnostic
  groups per scan.  The per-device union rule: ``scalars`` union
  (order-preserving, deduped), ``images`` / ``db_scalars`` / ``all_scalars``
  OR together (True wins), the single non-``None`` ``role`` is used (a device
  the sets require with *conflicting* explicit roles is an error), and the
  entry-level ``setup`` / ``closeout`` ritual name lists union (deduped).
  Entry-level rituals are collected across *all* named sets, deduped by plan
  name so a shared ritual runs once (:func:`resolve_save_sets_and_rituals`).
  Everything downstream — ``save_set_to_devices_config``, telemetry
  exclusion, the reserved-boundary warning — operates on the merged set.
- :class:`ConfigsRepoResolver` (in :mod:`geecs_bluesky.config_resolver`,
  re-exported here) reads the real configs repository — new-schema YAML
  directly, anything else through the legacy converters.
- Pure mapping helpers derive the engine shapes from the schemas:
  :func:`save_set_to_devices_config` (SaveSet → the ``devices_config`` dict
  ``BlueskyScanner._build_session_devices`` / the session factories expect,
  applying the documented intent→mechanics derivation rules) and
  :func:`trigger_writes_from_profile` (TriggerProfile →
  :class:`~geecs_bluesky.models.shot_control.ShotControlWrites`: per-state
  **ordered** multi-device write lists — order preserved exactly as the
  schema documents).  The adapters live *here* — bluesky-side — because
  ``geecs_schemas`` must never import ``geecs_bluesky`` (dependency
  direction).
- **Actions execute.**  Request-level ``setup``/``per_step``/``closeout``
  bindings, SaveSet entry rituals, and ExperimentDefaults plans are
  assembled (:func:`assemble_action_slots`), compiled to plan stubs
  (:func:`~geecs_bluesky.plans.action_compiler.compile_action_plan`) against
  the session's CA signal factory, and handed to the orchestration hooks.
  Names still resolve fail-fast **pre-claim** (an unknown plan name fails
  before any hardware is touched or a scan number is used), and every
  signal a plan will touch is pre-connected
  (:func:`prefetch_action_signals`) before the RunEngine runs — a lazy
  connect inside the RE loop would deadlock.
- **Multi-axis step scans execute** as an outer-product grid (first axis
  outermost/slowest — the schema's documented semantics): N movables, one
  bin per grid point, per-step actions at every grid point, all axis
  readbacks in the event rows.
- :func:`run_scan_request` executes the request on a
  :class:`~geecs_bluesky.session.GeecsSession`.

Deliberate v1 gaps (validated, then refused loudly — never silently wrong):

- **Pseudo (composite) scan variables** raise ``NotImplementedError`` — the
  composite pseudo-positioner device is not built yet.
- **``all_scalars``** on a save-set entry raises ``NotImplementedError`` —
  enumerating a device's scalar variables needs the DB-backed validation
  pass.
- **Optimize mode** maps onto :meth:`GeecsSession.optimize` as far as its
  signature allows: variables, iteration/shot counts, ``on_finish``.  The
  ``evaluator``/``generator`` specs cannot be instantiated in this package
  (the config-driven Xopt/evaluator stack lives in
  ``geecs_scanner.optimization``, which geecs_bluesky must not import), so a
  ready-made ``objective`` and ``suggester`` must be injected; without them
  optimize mode raises ``NotImplementedError``.  Action bindings on an
  optimize-mode request are likewise refused (``GeecsSession.optimize`` has
  no action hooks yet).

Scan variables in configs speak **GEECS device/variable names, never PVs**
(maintainer-ratified convention): all PV derivation stays inside the device
factories via ``ca_pv``.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
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

#: GUI-bridge refusal (the *bridge* still stages requests through the legacy
#: exec-config machinery; the engine itself executes grids — see
#: ``run_scan_request``).
MULTI_AXIS_MESSAGE = (
    "the GUI bridge does not execute multi-axis grids yet — run the request "
    "headless via GeecsSession.run, where grid execution landed with the "
    "M3b milestone; GUI submission is a later milestone"
)


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

    Handles both TriggerProfile generations: the single-device shape
    (top-level ``device`` + per-state ``{variable: value}``) and the
    multi-device shape (per-state ordered write lists, each write carrying
    its own ``device``).  Order is preserved exactly (schema-documented:
    writes are applied top to bottom).

    Parameters
    ----------
    profile :
        The trigger profile.
    state :
        The state whose writes to normalize.
    variant :
        Optional variant overlay.

    Returns
    -------
    list of tuple
        The state's writes as ``(device, variable, value)``.
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
    """Adapt a TriggerProfile into the engine's generalized ShotControlWrites.

    A state transition becomes the profile's **ordered** write list — order
    preserved verbatim (schema-documented: writes are applied top to
    bottom), spanning as many devices as the profile names.  A single-device
    profile is simply the one-device case of the same structure; there is no
    separate pivot any more (the engine's ``ShotController.from_writes``
    replays these lists sequentially, each write completing before the
    next).  This adapter lives bluesky-side because ``geecs_schemas`` must
    not import ``geecs_bluesky``.

    Parameters
    ----------
    profile :
        The trigger profile to adapt.
    variant :
        Optional profile variant overlaid first (e.g. ``"laser_off"``).

    Returns
    -------
    ShotControlWrites
        Per-state ordered ``(device, variable, value)`` write lists.

    Raises
    ------
    GeecsConfigurationError
        If *variant* is not defined on the profile, or no state writes any
        device at all (the profile cannot drive a scan's trigger).
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

    Applies the documented intent→mechanics derivation rules
    (``geecs_schemas.save_set`` module docstring):

    - ``synchronous`` is derived from the entry's role: ``snapshot`` →
      asynchronous, everything else → synchronous.
    - ``images`` → ``save_nonscalar_data``; the recorded ``variable_list`` is
      the entry's resolved scalar set (``acq_timestamp`` stays implicit — the
      device layer always records it).
    - Role overrides shape the **ordering** (the downstream classifier
      assigns free-run roles by position): a ``reference``-flagged entry is
      moved first; ``contributor``-flagged entries are placed after the
      unmarked synchronous ones so they never inherit pacemaker duty.

    The recorded ``variable_list`` follows the ``SaveSetEntry`` ``db_scalars``
    contract (M3c), delegated to
    :func:`~geecs_bluesky.db_runtime.resolve_entry_scalars`: with a
    *scalar_policy* provider, ``db_scalars=True`` unions the device's DB
    ``get='yes'`` variables (or every DB variable when ``all_scalars=True``)
    with the explicit list; ``db_scalars=False`` records only the explicit
    list (the legacy-converter pin).  Without a provider (the GUI-bridge path
    or no DB access) only the explicit list is recorded — the M3b behavior,
    with ``all_scalars`` still a documented gap.

    Parameters
    ----------
    save_set :
        The save set to translate.
    scalar_policy :
        Optional DB policy provider (M3c); ``None`` keeps the M3b
        explicit-only behavior.

    Returns
    -------
    dict
        ``{device_name: {"synchronous": bool, "save_nonscalar_data": bool,
        "variable_list": [...]}}`` in role-derived order.

    Raises
    ------
    GeecsConfigurationError
        If more than one entry claims the ``reference`` role, or if every
        synchronous entry is ``contributor``-flagged (no pacemaker left).
    NotImplementedError
        For ``all_scalars`` entries without an explicit ``scalars`` list when
        no *scalar_policy* is available — enumerating a device's scalars needs
        the DB (which the provider supplies).
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
    and executes the assembled slots (:func:`assemble_action_slots`); the
    GUI bridge still refuses them (:func:`raise_if_actions_present`).

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


def raise_if_actions_present(resolved: dict[str, list[str]]) -> None:
    """Refuse validated action bindings — **GUI-bridge path only**.

    The engine executes actions (see :func:`run_scan_request`); the GUI
    bridge still stages requests through the legacy exec-config machinery
    and does not run them yet.  Names were already resolved and are valid.
    """
    present = {slot: names for slot, names in resolved.items() if names}
    if present:
        raise NotImplementedError(
            f"the GUI bridge does not execute action bindings yet — run the "
            f"request headless via GeecsSession.run (action execution landed "
            f"with the M3b milestone); the request's action names were "
            f"resolved and are valid: {present}"
        )


# ---------------------------------------------------------------------------
# Action assembly + compilation (the §4.4b layers, executed)
# ---------------------------------------------------------------------------


def collect_save_set_rituals(save_set: SaveSet) -> dict[str, list[str]]:
    """Collect entry-level setup/closeout plan names, de-duplicated by name.

    The SaveSet schema's contract: the plans named by all entries are
    collected together (entry order preserved), de-duplicated by name, and
    each runs **once** per scan — two cameras sharing a ``prep_visa_line``
    ritual do not run it twice.  Older SaveSets without the fields
    contribute nothing (duck-typed).

    Parameters
    ----------
    save_set :
        The save set to inspect.

    Returns
    -------
    dict
        ``{"setup": [names], "closeout": [names]}``, each list de-duplicated
        in first-appearance order.
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
    """Merge a second entry for the same device into the first (union rule).

    The documented per-device union rule (see :func:`merge_save_sets`):
    ``scalars`` union order-preserving and deduped, the boolean flags
    (``images`` / ``db_scalars`` / ``all_scalars``) OR together (True wins),
    the single non-``None`` ``role`` is used — **conflicting explicit roles
    across the sets raise** (:class:`GeecsConfigurationError`) rather than
    resolving by list order, since role sets the acquisition semantics — and
    the entry-level ``setup`` / ``closeout`` ritual name lists union (deduped,
    first appearance). Reserved ``at_scan_start`` / ``at_scan_end`` maps merge
    key-wise (existing wins on a key clash) — they are inert in this version,
    so the choice only affects the one warning they draw.

    Raises
    ------
    GeecsConfigurationError
        If the two entries give the same device different explicit roles.
    """

    def _union(first: list[str], second: list[str]) -> list[str]:
        merged = list(first)
        for item in second:
            if item not in merged:
                merged.append(item)
        return merged

    # Role sets the acquisition guarantees (reference / contributor / snapshot
    # pacemaker wiring in save_set_to_devices_config). Two named sets that
    # both require the same device but disagree on its explicit role would
    # otherwise be resolved by save_sets list order — silently giving the scan
    # the wrong synchronization semantics. Refuse it (matching the legacy
    # preset composer), rather than pick by order.
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

    Resolves each name in *names* to a :class:`SaveSet`, merges them into one
    effective save set (:func:`merge_save_sets`), and collects the entry-level
    setup/closeout rituals across **all** sets, de-duplicated by plan name
    (so a ritual shared by two sets still runs once). Every referenced ritual
    plan name is validated against the action library now (fail-fast,
    pre-claim).

    Parameters
    ----------
    resolver :
        Where names are looked up.
    names :
        The save-set names (``ScanRequest.save_sets``); must be non-empty.
    merged_name :
        Name for the merged set (used in downstream messages).

    Returns
    -------
    tuple
        ``(merged_save_set, rituals)`` — the unioned save set and its
        de-duplicated ``{"setup": [...], "closeout": [...]}`` ritual names
        collected across every named set.
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

    The §4.4b setup layers nest like context managers (mirrored teardown —
    the ordering decision ratified with this milestone and documented in
    the ``ExperimentDefaults`` schema):

    - **setup**: experiment defaults → save-set entry rituals → the scan's
      own ``actions.setup`` (defaults outermost, the specific scan
      innermost).
    - **per_step**: the scan's own ``actions.per_step`` only (neither
      defaults nor entries have a per-step slot — deliberate).
    - **closeout**: the exact reverse — the scan's own ``actions.closeout``
      → entry rituals → experiment defaults.

    *actions* is the post-defaults request bindings (defaults already merged
    by :func:`apply_experiment_defaults`: prepended to setup, appended to
    closeout); *applied_defaults* tells this function how many entries of
    each list came from defaults, so entry rituals can be spliced between
    the layers.  Within the entry layer, first-appearance entry order is
    kept for both slots (the schema orders the layers, not the entries).

    Parameters
    ----------
    actions :
        The request's action bindings, after defaults were applied.
    applied_defaults :
        The provenance record from :func:`apply_experiment_defaults`.
    rituals :
        The save set's ``{"setup": [...], "closeout": [...]}`` names.

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

    Prefers the resolver's ``action_plan_registry()`` (duck-typed —
    :class:`ConfigsRepoResolver` implements it: the experiment's action
    library plus any plans extracted from converted save elements); falls
    back to a lazy per-name façade.

    Parameters
    ----------
    resolver :
        The name resolver.

    Returns
    -------
    Mapping
        ``{plan_name: ActionPlan}`` (possibly lazy).
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

    Compiled plan generators execute **inside** the RunEngine's event loop,
    where a blocking signal connect would deadlock — so every
    ``(device, variable)`` a ``set``/``check`` step names is touched here
    first (the factory caches per target, so execution finds everything
    connected).  Doubles as fail-fast validation: an unreachable target
    fails now, before the scan claims a number.  Nested ``run`` steps are
    walked recursively (visited-set bounded, so cycles terminate — the
    compiler itself raises on them at execution).

    Parameters
    ----------
    plans :
        The resolved plans of every slot about to run.
    registry :
        Named plans for ``run``-step recursion.
    settables :
        The :class:`~geecs_bluesky.plans.action_compiler.SettableFactory`.
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

    The returned callable yields the slot's plans in order, producing a
    **fresh** message generator per call — required for ``per_step`` (run at
    every step boundary) and for ``finalize_wrapper`` (which may instantiate
    its closeout more than once).

    Parameters
    ----------
    names :
        The slot's plan names, in final execution order.
    resolver :
        Resolves each name to its :class:`ActionPlan`.
    registry :
        Named plans for nested ``run`` steps.
    settables :
        The signal factory the compiled steps draw from.

    Returns
    -------
    tuple
        ``(stub, plans)`` — the plan-stub callable (``None`` when the slot
        is empty) and the resolved plans (for signal prefetching).
    """
    if not names:
        return None, []
    plans = [resolver.resolve_action_plan(name) for name in names]

    def _slot_plan():
        for plan in plans:
            yield from compile_action_plan(plan, registry=registry, settables=settables)

    return _slot_plan, plans


def resolve_save_sets_checked(resolver: ConfigResolver, names: list[str]) -> SaveSet:
    """Resolve and union several save sets, refusing rituals — **GUI-bridge only**.

    Every name in *names* is resolved, the sets are unioned into one effective
    SaveSet (:func:`merge_save_sets`), and any entry-level ritual (collected
    across all named sets) is *validated* then refused — the GUI bridge stages
    requests through the legacy exec-config machinery and does not run rituals
    yet. The bridge still refuses actions/multi-axis elsewhere; this only
    makes the save-set resolution list-aware.

    Parameters
    ----------
    resolver :
        Where names are looked up.
    names :
        The save-set names (``ScanRequest.save_sets``).

    Returns
    -------
    SaveSet
        The unioned save set (guaranteed free of entry-level actions).
    """
    save_set, rituals = resolve_save_sets_and_rituals(resolver, names)
    entry_actions = [n for slot in rituals.values() for n in slot]
    if entry_actions:
        raise NotImplementedError(
            f"the GUI bridge does not execute save-set entry rituals yet — "
            f"run the request headless via GeecsSession.run (ritual "
            f"execution landed with the M3b milestone); save sets {names!r} "
            f"reference entry-level setup/closeout plans, which were "
            f"resolved and are valid: {entry_actions}"
        )
    return save_set


def apply_experiment_defaults(
    request: ScanRequest, defaults: Any | None
) -> tuple[ScanRequest, dict[str, Any]]:
    """Apply experiment defaults where the request is silent (with provenance).

    The merge rule is the :class:`~geecs_schemas.ExperimentDefaults` one —
    **defaults run first on the way in and last on the way out**: a default
    trigger profile is used only when the request names none; default setup
    plans are *prepended* to the request's own setup list, and default
    closeout plans are *appended* after the request's own closeout list
    (teardown mirrors setup — the defaults are the outermost bracket).
    What was applied is returned for provenance (recorded into the run
    metadata by :func:`run_scan_request`) — a run's metadata must show the
    configuration it actually used, not require reconstructing which
    defaults file was in force.

    Parameters
    ----------
    request :
        The submitted request.
    defaults :
        The experiment defaults — an
        :class:`~geecs_schemas.ExperimentDefaults` model or a plain mapping
        of the same shape; ``None`` means no defaults.

    Returns
    -------
    tuple
        ``(request, applied)`` — the (possibly copied and updated) request
        and a ``{field: value}`` record of every default applied (empty
        when nothing was).
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
    as having no defaults.

    Parameters
    ----------
    resolver :
        The name resolver.
    request :
        The submitted request.

    Returns
    -------
    tuple
        As :func:`apply_experiment_defaults`.
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

    Parameters
    ----------
    spec :
        The catalog entry.
    name :
        The friendly name (for error messages).

    Returns
    -------
    tuple
        Device name, variable name, the entry's kind, and its optional
        ``confirm`` target (``"Device:Variable"``, or ``None`` when the set
        variable is also the readback — the common case).

    Raises
    ------
    NotImplementedError
        For pseudo (composite) variables — the composite pseudo-positioner
        device is not built yet.
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

    Mirrors the role rules of ``BlueskyScanner._classify_device_roles``:
    free-run → first synchronous entry is the reference (built with
    :meth:`GeecsSession.detector`), later synchronous entries are
    contributors; strict → every synchronous entry is a triggered detector;
    asynchronous entries are snapshots.  This is the *headless* build:
    failures propagate (fail loudly) — operator drop/promote interaction is
    the scanner layer's job.

    Parameters
    ----------
    session :
        The :class:`~geecs_bluesky.session.GeecsSession` (duck-typed).
    devices_config :
        Output of :func:`save_set_to_devices_config`.
    free_run :
        Whether the scan runs in free-run acquisition.

    Returns
    -------
    list
        Connected devices, reference first.
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

    ``SaveSetEntry.at_scan_start`` / ``at_scan_end`` are **reserved and not
    applied in this version**: the engine sets up triggering via the
    TriggerProfile/shot controller and camera saving via its own
    save-windowing, so DB set-side scan start/end writes are intentionally
    disabled (they would race the shot controller on the DG645).  A config
    that still sets them is not an error — it is honored by a future
    re-enable — but the operator should know the values are inert now, so we
    log one WARNING naming the device(s).

    Parameters
    ----------
    save_set :
        The scan's save set (``None`` = nothing to check).
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

    Selects every experiment device with a ``get='yes'`` variable not in the
    save set (:func:`~geecs_bluesky.db_runtime.select_telemetry_variables`),
    then builds one soft telemetry readable per device via
    ``session.telemetry`` — which returns ``None`` for a device unreachable at
    scan start (dropped with a log line, never an abort).  Devices that connect
    are appended to the scan's read set as extra columns; the softness (never
    waited on) lives in the device's own tolerant ``read``.

    Parameters
    ----------
    session :
        The session (its ``telemetry`` factory connects each device softly).
    save_set :
        The scan's save set (its devices are excluded from telemetry).
    scalar_policy :
        Supplies ``get='yes'`` variables; ``None`` means no telemetry.

    Returns
    -------
    tuple
        ``(readables, recorded)`` — the connected telemetry devices and the
        ``{device: [variables]}`` map of **only those that connected**
        (recorded in run metadata; devices dropped as unreachable are excluded
        so the metadata matches the columns that actually exist).
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
        Required for ``optimize`` mode (see the module docstring's gap
        list): the evaluator/generator specs cannot be instantiated in this
        package.

    Returns
    -------
    str or None
        The Bluesky run uid (``None`` when nothing was persisted).

    Raises
    ------
    NotImplementedError
        Pseudo scan variables, and optimize mode without an injected
        objective/suggester or with action bindings (the documented v1
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
        # Optimize mode has no action hooks yet (GeecsSession.optimize runs an
        # adaptive scan with no setup/per_step/closeout seam).  Rather than
        # refuse — which would block every optimization the moment an
        # experiment defines default bracket actions — we run the optimization
        # and skip the actions, logging loudly and recording the skip in run
        # metadata so the omission is never silent.  Safety/setup actions are
        # intentionally not run during optimization for now (a new capability,
        # not present in the legacy scanner either).
        skipped_actions = {k: v for k, v in resolved_actions.items() if v}
        return _run_optimize_request(
            session,
            request,
            resolver,
            mode,
            objective=objective,
            suggester=suggester,
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

    # M3c DB-integration runtime tier — get-side only.  The policy provider is
    # failure-tolerant (empty policy on a missing/unreachable DB), so it never
    # aborts a scan.  The DB set-side (scan start/end writes) is intentionally
    # disabled in this version (see the module docstring); a config that still
    # sets the reserved at_scan_start / at_scan_end fields gets one warning.
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
    applied_defaults: dict[str, Any] | None = None,
    skipped_actions: dict[str, list[str]] | None = None,
) -> str | None:
    """Map an optimize-mode request onto :meth:`GeecsSession.optimize`.

    Consumes ``optimization.variables`` (names resolved through the
    scan-variable catalog; ``Device:Variable`` strings pass through),
    ``max_iterations``, and ``move_to_best_on_finish`` (→ ``on_finish``).
    The variable *bounds*, ``objectives``/``observables``/``constraints``,
    and the evaluator/generator specs are the suggester's business — they
    are **not** consumed here (documented gap; the injected suggester is
    expected to have been built from them by the caller's stack).

    Parameters
    ----------
    session, request, resolver :
        As in :func:`run_scan_request`.
    mode :
        ``"strict"`` or ``"free_run"``.
    objective, suggester :
        The ready-made optimization callables (required).

    Returns
    -------
    str or None
        The run uid.

    Raises
    ------
    NotImplementedError
        When *objective* or *suggester* is missing.
    """
    spec = request.optimization
    assert spec is not None  # guaranteed by ScanRequest validation
    if objective is None or suggester is None:
        raise NotImplementedError(
            "optimize-mode ScanRequest execution needs a ready-made "
            "objective and suggester (run(request, resolver, objective=..., "
            "suggester=...)): instantiating them from the request's "
            "evaluator/generator specs lives in the GUI optimization stack "
            "(geecs_scanner.optimization), which geecs_bluesky cannot import"
        )

    detectors: list = []
    created: list = []
    try:
        skipped = {k: list(v) for k, v in (skipped_actions or {}).items() if v}
        # db_scalars resolution applies to optimize too (recorded-scalar
        # consistency); background telemetry does not run in optimize mode yet
        # (no scan-boundary hook on GeecsSession.optimize).  The DB set-side
        # (scan start/end writes) is disabled everywhere in this version.
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
        # Background telemetry is not wired into optimize mode yet
        # (GeecsSession.optimize has no scan-boundary hook); db_scalars
        # resolution above still applies.  The DB set-side is disabled
        # everywhere in this version.  Recorded for provenance.
        md["db_scan_runtime"] = {
            "db_scalars": "applied",
            "background_telemetry": "not_run_in_optimize",
        }
        uid, _history = session.optimize(
            variables=variables,
            detectors=detectors,
            objective=objective,
            suggester=suggester,
            shots_per_iteration=request.shots_per_step,
            max_iterations=spec.max_iterations or 20,
            mode=mode,
            description=request.description,
            md=md,
            on_finish="best" if spec.move_to_best_on_finish else "hold",
        )
        return uid
    finally:
        if created and hasattr(session, "disconnect"):
            session.disconnect(*created)
