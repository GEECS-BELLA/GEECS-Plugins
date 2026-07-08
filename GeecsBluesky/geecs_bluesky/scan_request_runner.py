"""Run a ScanRequest: resolve config names, map onto the existing machinery.

The one submission object of the target architecture (vision doc §4.1) meets
the engine here: a client builds a
:class:`~geecs_schemas.scan_request.ScanRequest` and calls
``session.run(request)`` (or hands it to ``BlueskyScanner.reinitialize``).
This module owns the mapping:

- A :class:`ConfigResolver` turns the request's *names* into schema models —
  ``save_set`` → :class:`~geecs_schemas.save_set.SaveSet`,
  ``trigger_profile`` → :class:`~geecs_schemas.trigger_profile.TriggerProfile`,
  an axis variable → a :class:`~geecs_schemas.scan_variables.ScanVariable`,
  action names → :class:`~geecs_schemas.action_plan.ActionPlan`.
- :class:`ConfigsRepoResolver` reads the real configs repository
  (``scanner_configs/experiments/<Experiment>/``): a YAML file carrying a
  ``schema_version`` key loads as the new schema directly; anything else is
  converted from its legacy dialect via :mod:`geecs_schemas.convert` — so the
  whole existing config corpus is usable immediately, no flag day.
- Pure mapping helpers derive the legacy engine shapes from the schemas:
  :func:`save_set_to_devices_config` (SaveSet → the ``devices_config`` dict
  ``BlueskyScanner._build_session_devices`` / the session factories expect,
  applying the documented intent→mechanics derivation rules) and
  :func:`shot_control_config_from_trigger_profile` (TriggerProfile →
  :class:`~geecs_bluesky.models.shot_control.ShotControlConfig`).  The
  adapter lives *here* — bluesky-side — because ``geecs_schemas`` must never
  import ``geecs_bluesky`` (dependency direction).
- :func:`run_scan_request` executes the request on a
  :class:`~geecs_bluesky.session.GeecsSession`.

Deliberate v1 gaps (validated, then refused loudly — never silently wrong):

- **Multi-axis step scans** are accepted by the schema but raise
  ``NotImplementedError`` here — grid execution lands with the actions
  milestone.
- **Actions** (``setup`` / ``per_step`` / ``closeout``, request-level and
  SaveSet entry-level alike): the names are resolved and validated against
  the action library *now*, then ``NotImplementedError`` is raised — the
  ActionPlan→plan-stub compiler is being built in a parallel milestone.
- **Multi-device trigger profiles**: the adapter takes the single-device
  fast path (all writes in all states name one device — whether declared
  top-level or per write); writes spanning devices raise
  ``NotImplementedError`` — the engine's ShotController drives one device.
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
  optimize mode raises ``NotImplementedError``.

Scan variables in configs speak **GEECS device/variable names, never PVs**
(maintainer-ratified convention): all PV derivation stays inside the device
factories via ``ca_pv``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.scanner_configs import SHOT_CONTROL_FOLDER, scanner_configs_base
from geecs_schemas import (
    ActionBindings,
    ActionPlan,
    ActionPlanLibrary,
    AcquisitionMode,
    ExperimentDefaults,
    PseudoScanVariable,
    SaveRole,
    SaveSet,
    ScanRequest,
    ScanRequestMode,
    ScanVariable,
    ScanVariables,
    ScanVariableSpec,
    TriggerProfile,
    TriggerState,
)
from geecs_schemas.convert import (
    convert_action_library,
    convert_save_element,
    convert_scan_variables,
    convert_shot_control,
)

logger = logging.getLogger(__name__)

MULTI_AXIS_MESSAGE = "multi-axis execution lands with the actions milestone"


# ---------------------------------------------------------------------------
# ConfigResolver protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ConfigResolver(Protocol):
    """Resolves the names a ScanRequest carries into schema models."""

    def resolve_save_set(self, name: str) -> SaveSet:
        """Return the save set called *name*."""
        ...

    def resolve_trigger_profile(self, name: str) -> TriggerProfile:
        """Return the trigger profile called *name*."""
        ...

    def resolve_scan_variable(self, name: str) -> ScanVariableSpec:
        """Return the scan-variable entry called *name*."""
        ...

    def resolve_action_plan(self, name: str) -> ActionPlan:
        """Return the action plan called *name*."""
        ...

    def resolve_experiment_defaults(self) -> ExperimentDefaults | None:
        """Return the experiment's defaults, or ``None`` if none are declared.

        Defaults apply where the request is silent (default trigger
        profile; default setup/closeout plans prepended); what was applied
        is recorded for provenance (see
        :func:`apply_experiment_defaults`).  Resolvers without this method
        are tolerated (no defaults).
        """
        ...


class ConfigsRepoResolver:
    """Resolver over the real configs-repo layout, converter-backed.

    Reads ``scanner_configs/experiments/<experiment>/`` (the same resolution
    roots as :func:`geecs_bluesky.scanner_configs.scanner_configs_base`):

    - ``save_devices/<name>.yaml`` — save sets (legacy save elements)
    - ``shot_control_configurations/<name>.yaml`` — trigger profiles
    - ``scan_devices/scan_variables.yaml`` (new schema) or the legacy
      ``scan_devices/scan_devices.yaml`` + ``scan_devices/composite_variables.yaml``
      pair — the scan-variable catalog
    - ``action_library/actions.yaml`` — the action-plan library

    A file whose top level carries ``schema_version`` is loaded as the new
    schema; anything else goes through the matching legacy converter — so
    the existing corpus works unchanged and files can migrate one at a time.

    Parameters
    ----------
    experiment :
        Experiment folder name under ``scanner_configs/experiments``.
    experiments_root :
        Override for the experiments root (tests); defaults to the
        production resolution (``GEECS_SCANNER_CONFIG_DIR`` env var or
        config.ini), resolved lazily on first use.
    """

    SAVE_SET_FOLDER = "save_devices"
    TRIGGER_FOLDER = SHOT_CONTROL_FOLDER
    SCAN_VARIABLES_FOLDER = "scan_devices"
    ACTION_FOLDER = "action_library"

    def __init__(
        self, experiment: str, experiments_root: str | Path | None = None
    ) -> None:
        self._experiment = experiment
        self._experiments_root = (
            Path(experiments_root) if experiments_root is not None else None
        )
        self._scan_variables_cache: ScanVariables | None = None
        self._action_library_cache: ActionPlanLibrary | None = None
        # Plans extracted by the save-element converter (legacy
        # setup_action / closeout_action / scan_setup become named plans
        # referenced from the entries) — they live beside the element, not
        # in the experiment's action library, so name resolution falls back
        # to them after the library.
        self._extracted_element_actions: dict[str, ActionPlan] = {}

    @property
    def _root(self) -> Path:
        root = self._experiments_root or scanner_configs_base()
        return root / self._experiment

    @staticmethod
    def _strip_yaml_suffix(name: str) -> str:
        for suffix in (".yaml", ".yml"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def _load_yaml(self, path: Path, kind: str, name: str) -> dict:
        """Load one YAML mapping, failing loudly with the config kind/name."""
        if not path.exists():
            raise GeecsConfigurationError(
                f"{kind} {name!r} not found for experiment "
                f"{self._experiment!r}: no file at {path}"
            )
        document = yaml.safe_load(path.read_text())
        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise GeecsConfigurationError(
                f"{kind} {name!r}: expected a YAML mapping at the top of "
                f"{path}, got {type(document).__name__}"
            )
        return document

    def resolve_save_set(self, name: str) -> SaveSet:
        """Load the save set *name* (new schema, else converted save element).

        Parameters
        ----------
        name :
            File stem under ``save_devices/`` (``.yaml`` optional).

        Returns
        -------
        SaveSet
            The validated save set.

        Raises
        ------
        GeecsConfigurationError
            If the file is missing or is an action-only legacy element
            (nothing to record).
        """
        stem = self._strip_yaml_suffix(name)
        path = self._root / self.SAVE_SET_FOLDER / f"{stem}.yaml"
        document = self._load_yaml(path, "save set", name)
        if "schema_version" in document:
            return SaveSet.model_validate(document)
        result = convert_save_element(document, name=stem)
        for note in result.notes:
            logger.info("save set %s (converted from legacy): %s", stem, note)
        # The converter extracts element-level setup/closeout (and per-device
        # scan_setup) into named plans referenced from the entries; remember
        # them so resolve_action_plan can validate those references.
        self._extracted_element_actions.update(result.actions)
        if result.save_set is None:
            raise GeecsConfigurationError(
                f"save set {name!r} ({path}) is an action-only legacy element "
                "— it lists no devices to record"
            )
        return result.save_set

    def resolve_trigger_profile(self, name: str) -> TriggerProfile:
        """Load the trigger profile *name* (new schema, else converted).

        Parameters
        ----------
        name :
            File stem under ``shot_control_configurations/``.

        Returns
        -------
        TriggerProfile
            The validated profile.

        Raises
        ------
        GeecsConfigurationError
            If the file is missing or names no trigger device.
        """
        stem = self._strip_yaml_suffix(name)
        path = self._root / self.TRIGGER_FOLDER / f"{stem}.yaml"
        document = self._load_yaml(path, "trigger profile", name)
        if "schema_version" in document:
            return TriggerProfile.model_validate(document)
        profile = convert_shot_control(document, name=stem)
        if profile is None:
            raise GeecsConfigurationError(
                f"trigger profile {name!r} ({path}) is empty / names no "
                "device — it cannot drive a scan's trigger"
            )
        return profile

    def _scan_variables_catalog(self) -> ScanVariables:
        """Load (and cache) the experiment's scan-variable catalog."""
        if self._scan_variables_cache is not None:
            return self._scan_variables_cache
        folder = self._root / self.SCAN_VARIABLES_FOLDER
        new_schema = folder / "scan_variables.yaml"
        if new_schema.exists():
            document = self._load_yaml(new_schema, "scan variables", "catalog")
            catalog = ScanVariables.model_validate(document)
        else:
            scan_devices = folder / "scan_devices.yaml"
            composites = folder / "composite_variables.yaml"
            if not scan_devices.exists() and not composites.exists():
                raise GeecsConfigurationError(
                    f"no scan-variable catalog for experiment "
                    f"{self._experiment!r}: expected {new_schema} or the "
                    f"legacy pair in {folder}"
                )
            catalog = convert_scan_variables(
                scan_devices if scan_devices.exists() else None,
                composites if composites.exists() else None,
            )
        self._scan_variables_cache = catalog
        return catalog

    def resolve_scan_variable(self, name: str) -> ScanVariableSpec:
        """Look up the scan variable *name* in the experiment catalog.

        Parameters
        ----------
        name :
            Friendly variable name (a key of the catalog).

        Returns
        -------
        ScanVariable or PseudoScanVariable
            The catalog entry.

        Raises
        ------
        GeecsConfigurationError
            If the name is not in the catalog (known names are listed).
        """
        catalog = self._scan_variables_catalog()
        try:
            return catalog.variables[name]
        except KeyError:
            raise GeecsConfigurationError(
                f"scan variable {name!r} is not in the "
                f"{self._experiment!r} catalog. Known variables: "
                f"{sorted(catalog.variables)}"
            ) from None

    def _action_library(self) -> ActionPlanLibrary:
        """Load (and cache) the experiment's action-plan library."""
        if self._action_library_cache is not None:
            return self._action_library_cache
        path = self._root / self.ACTION_FOLDER / "actions.yaml"
        document = self._load_yaml(path, "action library", "actions")
        if "schema_version" in document:
            library = ActionPlanLibrary.model_validate(document)
        else:
            library = convert_action_library(document)
        self._action_library_cache = library
        return library

    def resolve_action_plan(self, name: str) -> ActionPlan:
        """Look up the action plan *name* in the experiment library.

        Parameters
        ----------
        name :
            Plan name (a key of the action library).

        Returns
        -------
        ActionPlan
            The named plan.

        Raises
        ------
        GeecsConfigurationError
            If the name is not in the library (known names are listed).
        """
        # Plans the save-element converter extracted resolve first: they
        # live beside their element (their `<element>_setup` names cannot
        # collide with library names in practice), and an experiment may
        # have converted elements without having an action library at all.
        plan = self._extracted_element_actions.get(name)
        if plan is not None:
            return plan
        library = self._action_library()
        try:
            return library.plans[name]
        except KeyError:
            raise GeecsConfigurationError(
                f"action plan {name!r} is not in the {self._experiment!r} "
                f"action library. Known plans: {sorted(library.plans)}"
            ) from None

    DEFAULTS_FILE = "experiment_defaults.yaml"

    def resolve_experiment_defaults(self) -> ExperimentDefaults | None:
        """Load and validate the experiment's defaults file, if one exists.

        Reads ``<experiment>/experiment_defaults.yaml`` into the
        :class:`~geecs_schemas.ExperimentDefaults` model (there is no
        legacy dialect behind it — the legacy scanner kept these choices in
        GUI state).

        Returns
        -------
        ExperimentDefaults or None
            The validated defaults, or ``None`` when no file exists.
        """
        path = self._root / self.DEFAULTS_FILE
        if not path.exists():
            return None
        document = self._load_yaml(path, "experiment defaults", self.DEFAULTS_FILE)
        return ExperimentDefaults.model_validate(document)


# ---------------------------------------------------------------------------
# Schema → engine-shape adapters
# ---------------------------------------------------------------------------


MULTI_DEVICE_TRIGGER_MESSAGE = (
    "multi-device trigger profiles land with a later milestone"
)


def _state_write_triples(
    profile: TriggerProfile, state: "TriggerState", variant: str | None
) -> list[tuple[str | None, str, str]]:
    """Normalize one state's writes to ``(device, variable, value)`` triples.

    Handles both TriggerProfile generations: the single-device shape
    (top-level ``device`` + per-state ``{variable: value}``) and the
    multi-device shape (per-state ordered write lists, each write carrying
    its own ``device``).

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


def shot_control_config_from_trigger_profile(
    profile: TriggerProfile, variant: str | None = None
) -> ShotControlConfig:
    """Pivot a TriggerProfile into the engine's ShotControlConfig shape.

    The two models share semantics (named states, verbatim wire values,
    omission = no-op); only the layout differs — the profile is per-state,
    the engine config is per-variable ``{variable: {state: value}}``.
    ``defines_state`` / ``values_for_state`` answers are preserved exactly.
    This adapter lives bluesky-side because ``geecs_schemas`` must not
    import ``geecs_bluesky``.

    **Single-device fast path only** (schema-accepts / engine-pending, the
    same pattern as multi-axis): a profile whose writes all target one
    device adapts exactly as before — whether it declares the device at the
    top level (single-device shape) or per write (multi-device shape).  A
    profile whose writes span several devices is accepted by the schema but
    raises ``NotImplementedError`` here: the engine's ``ShotController``
    drives one device today.

    Parameters
    ----------
    profile :
        The trigger profile to adapt.
    variant :
        Optional profile variant overlaid before pivoting (e.g.
        ``"laser_off"``).

    Returns
    -------
    ShotControlConfig
        The engine-facing shot-control configuration.

    Raises
    ------
    GeecsConfigurationError
        If *variant* is not defined on the profile, or no trigger device
        can be determined at all.
    NotImplementedError
        If the profile's writes span more than one device.
    """
    if variant is not None and variant not in profile.variants:
        raise GeecsConfigurationError(
            f"trigger profile {profile.name!r} has no variant {variant!r}. "
            f"Known variants: {sorted(profile.variants)}"
        )
    variables: dict[str, dict[str, str]] = {}
    devices: list[str] = []
    for state in TriggerState:
        for device, variable, value in _state_write_triples(profile, state, variant):
            if device is not None and device not in devices:
                devices.append(device)
            variables.setdefault(variable, {})[state.value] = value
    if len(devices) > 1:
        raise NotImplementedError(
            f"{MULTI_DEVICE_TRIGGER_MESSAGE} — trigger profile "
            f"{profile.name!r} writes to {devices}; the engine's shot "
            "controller drives exactly one device today"
        )
    device = devices[0] if devices else getattr(profile, "device", None)
    if not device:
        raise GeecsConfigurationError(
            f"trigger profile {profile.name!r} names no trigger device — "
            "it cannot drive a scan's trigger"
        )
    return ShotControlConfig(device=device, variables=variables)


def save_set_to_devices_config(save_set: SaveSet) -> dict[str, dict[str, Any]]:
    """Derive the legacy ``devices_config`` shape from a SaveSet.

    Applies the documented intent→mechanics derivation rules
    (``geecs_schemas.save_set`` module docstring):

    - ``synchronous`` is derived from the entry's role: ``snapshot`` →
      asynchronous, everything else → synchronous.
    - ``images`` → ``save_nonscalar_data``; ``scalars`` → ``variable_list``
      (``acq_timestamp`` stays implicit — the device layer always records it).
    - Role overrides shape the **ordering** (the downstream classifier
      assigns free-run roles by position): a ``reference``-flagged entry is
      moved first; ``contributor``-flagged entries are placed after the
      unmarked synchronous ones so they never inherit pacemaker duty.

    Parameters
    ----------
    save_set :
        The save set to translate.

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
        For ``all_scalars`` entries without an explicit ``scalars`` list —
        enumerating a device's scalars needs the DB-backed validation pass.
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
        if entry.all_scalars and not entry.scalars:
            raise NotImplementedError(
                f"save set {save_set.name!r}, device {entry.device!r}: "
                "all_scalars needs the DB-backed scalar enumeration, which "
                "lands with save-set validation — list the scalars "
                "explicitly for now"
            )
        config[entry.device] = {
            "synchronous": entry.role is not SaveRole.SNAPSHOT,
            "save_nonscalar_data": entry.images,
            "variable_list": list(entry.scalars),
        }
    return config


def resolve_and_validate_actions(
    actions: ActionBindings, resolver: ConfigResolver
) -> dict[str, list[str]]:
    """Resolve every action name in the bindings against the library.

    Validation now, execution later: each name must exist (the resolver
    raises otherwise); the caller refuses to *run* them until the
    ActionPlan→plan-stub compiler lands.

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
    """Refuse (loudly) to execute validated action bindings — v1 gap."""
    present = {slot: names for slot, names in resolved.items() if names}
    if present:
        raise NotImplementedError(
            f"action execution lands with the actions milestone (the "
            f"ActionPlan compiler is being built in a parallel milestone); "
            f"the request's action names were resolved and are valid: "
            f"{present}"
        )


def collect_save_set_action_names(save_set: SaveSet) -> list[str]:
    """Return entry-level setup/closeout ActionPlan references, if any.

    Newer SaveSet schemas let an entry reference setup/closeout action
    plans by name; older SaveSets have no such fields (this returns
    ``[]``).  Duck-typed so both schema generations pass through.

    Parameters
    ----------
    save_set :
        The save set to inspect.

    Returns
    -------
    list of str
        Every referenced plan name, in entry order.
    """
    names: list[str] = []
    for entry in save_set.entries:
        for slot in ("setup", "closeout"):
            value = getattr(entry, slot, None)
            if not value:
                continue
            names.extend(value if isinstance(value, (list, tuple)) else [value])
    return names


def resolve_save_set_checked(resolver: ConfigResolver, name: str) -> SaveSet:
    """Resolve a save set and give its entry-level action refs the M3b treatment.

    Entry-level setup/closeout references are *validated* against the
    action library now (unknown names fail loudly), then refused with
    ``NotImplementedError`` — exactly like the request-level action
    bindings, until the ActionPlan compiler lands.

    Parameters
    ----------
    resolver :
        Where names are looked up.
    name :
        The save set name.

    Returns
    -------
    SaveSet
        The resolved save set (guaranteed free of entry-level actions).
    """
    save_set = resolver.resolve_save_set(name)
    entry_actions = collect_save_set_action_names(save_set)
    for action_name in entry_actions:
        resolver.resolve_action_plan(action_name)
    if entry_actions:
        raise NotImplementedError(
            f"action execution lands with the actions milestone; save set "
            f"{name!r} references entry-level setup/closeout plans, which "
            f"were resolved and are valid: {entry_actions}"
        )
    return save_set


def apply_experiment_defaults(
    request: ScanRequest, defaults: Any | None
) -> tuple[ScanRequest, dict[str, Any]]:
    """Apply experiment defaults where the request is silent (with provenance).

    The merge rule is the :class:`~geecs_schemas.ExperimentDefaults` one —
    **defaults run first, then the scan's own**: a default trigger profile
    is used only when the request names none; default setup/closeout plans
    are *prepended* to the request's own lists.  What was applied is
    returned for provenance (recorded into the run metadata by
    :func:`run_scan_request`) — a run's metadata must show the
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
        # Defaults run first, then the scan's own plans.
        slot_updates[slot] = names + list(getattr(request.actions, slot))
        applied[f"actions.{slot}"] = names
    if slot_updates:
        updates["actions"] = request.actions.model_copy(update=slot_updates)

    if updates:
        request = request.model_copy(update=updates)
    return request, applied


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
    resolve = getattr(resolver, "resolve_experiment_defaults", None)
    defaults = resolve() if callable(resolve) else None
    return apply_experiment_defaults(request, defaults)


# ---------------------------------------------------------------------------
# Request execution on a GeecsSession
# ---------------------------------------------------------------------------


def resolve_movable_target(spec: ScanVariableSpec, name: str) -> tuple[str, str, str]:
    """Return ``(device, variable, kind)`` for a plain scan variable.

    Parameters
    ----------
    spec :
        The catalog entry.
    name :
        The friendly name (for error messages).

    Returns
    -------
    tuple
        Device name, variable name, and the entry's kind.

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
    return device, variable, spec.kind


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


def run_scan_request(
    session: Any,
    request: ScanRequest,
    resolver: ConfigResolver,
    *,
    objective: Any | None = None,
    suggester: Any | None = None,
) -> str | None:
    """Execute *request* on *session*; return the run uid.

    Resolution order is fail-fast: action names and the multi-axis limit are
    checked before any hardware is touched, then the trigger profile is
    attached, then devices are built, then the scan runs.  Devices created
    here are disconnected afterwards (the run owns what it creates).

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
        Multi-axis requests, requests with action bindings, pseudo scan
        variables, and optimize mode without an injected objective/suggester
        (all documented v1 gaps — validated first, refused loudly).
    GeecsConfigurationError
        Unresolvable names, or a step/noscan request without a save set.
    """
    request, applied_defaults = resolve_defaults_for(resolver, request)
    resolved_actions = resolve_and_validate_actions(request.actions, resolver)
    raise_if_actions_present(resolved_actions)
    if request.mode is ScanRequestMode.STEP and len(request.axes) > 1:
        raise NotImplementedError(MULTI_AXIS_MESSAGE)

    if request.trigger_profile:
        profile = resolver.resolve_trigger_profile(request.trigger_profile)
        session.shot_control(
            shot_control_config_from_trigger_profile(profile, request.trigger_variant)
        )
    else:
        session.shot_control(None)

    mode = "strict" if request.acquisition is AcquisitionMode.STRICT else "free_run"

    if request.mode is ScanRequestMode.OPTIMIZE:
        return _run_optimize_request(
            session,
            request,
            resolver,
            mode,
            objective=objective,
            suggester=suggester,
            applied_defaults=applied_defaults,
        )

    if not request.save_set:
        raise GeecsConfigurationError(
            f"a {request.mode.value!r} ScanRequest needs a save_set — "
            "without one the scan would record nothing"
        )
    save_set = resolve_save_set_checked(resolver, request.save_set)
    devices_config = save_set_to_devices_config(save_set)

    created: list = []
    try:
        detectors = _build_request_detectors(
            session, devices_config, free_run=mode == "free_run"
        )
        created.extend(detectors)

        md: dict[str, Any] = {"scan_request_mode": request.mode.value}
        if applied_defaults:
            # Provenance: the run records exactly which experiment defaults
            # filled in fields the submitter left unset.
            md["applied_defaults"] = applied_defaults
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
            )

        axis = request.axes[0]
        spec = resolver.resolve_scan_variable(axis.variable)
        device, variable, kind = resolve_movable_target(spec, axis.variable)
        movable = (
            session.motor(device, variable)
            if kind == "motor"
            else session.settable(device, variable)
        )
        created.append(movable)
        positions = axis.positions.to_values()
        scan_info["scan_mode"] = "standard"
        scan_info["scan_parameter"] = f"{device}:{variable}"
        md["scan_variable"] = axis.variable
        return session.scan(
            detectors=detectors,
            motor=movable,
            positions=positions,
            shots_per_step=request.shots_per_step,
            mode=mode,
            description=request.description,
            md=md,
            scan_info=scan_info,
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
        if request.save_set:
            save_set = resolve_save_set_checked(resolver, request.save_set)
            detectors = _build_request_detectors(
                session,
                save_set_to_devices_config(save_set),
                free_run=mode == "free_run",
            )
            created.extend(detectors)

        variables: dict[str, Any] = {}
        for name in spec.variables:
            if ":" in name:
                device, _, variable = name.partition(":")
            else:
                var_spec = resolver.resolve_scan_variable(name)
                device, variable, _kind = resolve_movable_target(var_spec, name)
            movable = session.settable(device, variable)
            variables[name] = movable
            created.append(movable)

        md: dict[str, Any] = {"scan_request_mode": request.mode.value}
        if applied_defaults:
            md["applied_defaults"] = applied_defaults
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
