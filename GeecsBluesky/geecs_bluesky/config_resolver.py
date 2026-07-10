"""Resolve the names a ScanRequest carries into validated schema models.

:class:`ConfigResolver` is the protocol; :class:`ConfigsRepoResolver` is the
production implementation over the real configs-repo layout
(``scanner_configs/experiments/<Experiment>/``).  A YAML file carrying a
``schema_version`` key loads as the new schema directly; anything else is
converted from its legacy dialect via :mod:`geecs_schemas.convert` — so the
whole existing config corpus is usable immediately, no flag day.

Execution of a resolved request lives in
:mod:`geecs_bluesky.scan_request_runner`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.scanner_configs import SHOT_CONTROL_FOLDER, scanner_configs_base
from geecs_schemas import (
    ActionPlan,
    ActionPlanLibrary,
    ExperimentDefaults,
    SaveSet,
    ScanVariables,
    ScanVariableSpec,
    TriggerProfile,
)
from geecs_schemas.convert import (
    convert_action_library,
    convert_save_element,
    convert_scan_variables,
    convert_shot_control,
)

logger = logging.getLogger(__name__)


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
        :func:`geecs_bluesky.scan_request_runner.apply_experiment_defaults`).
        Resolvers without this method are tolerated (no defaults).
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

        Raises
        ------
        GeecsConfigurationError
            Missing file, or an action-only legacy element (nothing to record).
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

        Raises
        ------
        GeecsConfigurationError
            Missing file, or a profile that names no trigger device.
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

        Raises
        ------
        GeecsConfigurationError
            Unknown name (the error lists the known variables).
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

        Raises
        ------
        GeecsConfigurationError
            Unknown name (the error lists the known plans).
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

    def action_plan_registry(self) -> dict[str, ActionPlan]:
        """Return every named plan visible to nested ``run`` steps.

        The action library plus converter-extracted element plans (extracted
        plans win on collision, matching :meth:`resolve_action_plan`).
        """
        plans: dict[str, ActionPlan] = {}
        try:
            plans.update(self._action_library().plans)
        except GeecsConfigurationError:
            pass  # no actions.yaml — extracted element plans may still exist
        plans.update(self._extracted_element_actions)
        return plans

    DEFAULTS_FILE = "experiment_defaults.yaml"

    def resolve_experiment_defaults(self) -> ExperimentDefaults | None:
        """Load ``<experiment>/experiment_defaults.yaml``; ``None`` if absent.

        No legacy dialect exists behind it (the legacy scanner kept these
        choices in GUI state).
        """
        path = self._root / self.DEFAULTS_FILE
        if not path.exists():
            return None
        document = self._load_yaml(path, "experiment defaults", self.DEFAULTS_FILE)
        return ExperimentDefaults.model_validate(document)
