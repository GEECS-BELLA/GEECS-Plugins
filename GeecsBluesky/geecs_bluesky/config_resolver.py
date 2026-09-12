"""Resolve the configs-repo names a client uses into validated schema models.

:class:`ConfigResolver` is the protocol; :class:`ConfigsRepoResolver` is the
production implementation over the real configs-repo layout
(``scanner_configs/experiments/<Experiment>/``).  A YAML file carrying a
``schema_version`` key loads as the new schema directly; trigger profiles
without one are converted from their legacy dialect via
:mod:`geecs_schemas.convert`.  Presets, scan-variable catalogs and action
libraries are new-schema only (the legacy save elements / scan presets were
regenerated once as ``Preset`` documents, GEECS-Plugins#807; the
scan-device pair and its converter were retired 2026-09, #779; the action
libraries were regenerated once as ``ActionPlanLibrary`` documents and
their converter deleted, GEECS-Schemas 0.22.0).

The client seam expands a preset into a stock plan queue item
(:mod:`geecs_bluesky.qs_client.presets`).
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
    Preset,
    ScanVariables,
    ScanVariableSpec,
    TriggerProfile,
)
from geecs_schemas.convert import convert_shot_control

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConfigResolver protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ConfigResolver(Protocol):
    """Resolves the configs-repo names a client uses into schema models."""

    def resolve_preset(self, name: str) -> Preset:
        """Return the preset called *name*."""
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
        the client-side request expansion, phase 1 PR 2).
        Resolvers without this method are tolerated (no defaults).
        """
        ...


class ConfigsRepoResolver:
    """Resolver over the real configs-repo layout, converter-backed.

    Reads ``scanner_configs/experiments/<experiment>/`` (the same resolution
    roots as :func:`geecs_bluesky.scanner_configs.scanner_configs_base`):

    - ``presets/<name>.yaml`` — presets (``geecs_schemas.Preset``: the
      device group + the plan call; new schema only)
    - ``shot_control_configurations/<name>.yaml`` — trigger profiles
    - ``scan_devices/scan_variables.yaml`` — the scan-variable catalog
      (new schema only; the legacy ``scan_devices.yaml`` +
      ``composite_variables.yaml`` pair and its converter were retired
      2026-09, GEECS-Plugins#779)
    - ``action_library/actions.yaml`` — the action-plan library (new
      schema only; the legacy ``actions:`` dialect is refused)
    - ``optimizer_configs/<name>.yaml`` — listed (for clients) but not
      resolved here: ``OptimizationSpec`` documents validated by their
      consumers.

    A trigger profile whose top level carries ``schema_version`` is
    loaded as the new schema; anything else goes through the legacy
    converter.  Named configs resolve from
    either the ``.yaml`` or ``.yml`` spelling (console parity), so every
    listed name round-trips through resolution.

    Parameters
    ----------
    experiment :
        Experiment folder name under ``scanner_configs/experiments``.
    experiments_root :
        Override for the experiments root (tests); defaults to the
        production resolution (``GEECS_SCANNER_CONFIG_DIR`` env var or
        config.ini), resolved lazily on first use.
    """

    TRIGGER_FOLDER = SHOT_CONTROL_FOLDER
    SCAN_VARIABLES_FOLDER = "scan_devices"
    ACTION_FOLDER = "action_library"
    PRESET_FOLDER = "presets"
    OPTIMIZER_FOLDER = "optimizer_configs"

    def __init__(
        self, experiment: str, experiments_root: str | Path | None = None
    ) -> None:
        self._experiment = experiment
        self._experiments_root = (
            Path(experiments_root) if experiments_root is not None else None
        )
        self._scan_variables_cache: ScanVariables | None = None

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

    def _named_yaml_path(self, folder: str, stem: str) -> Path:
        """The config file for *stem* in *folder*: ``.yaml``, else its ``.yml`` twin.

        The console resolves both spellings (``NamedConfigStore._named_path``
        parity) and the listings count both, so resolution must round-trip
        every listed name.  When neither exists, the ``.yaml`` path is
        returned so the not-found error names the canonical spelling.
        """
        base = self._root / folder
        path = base / f"{stem}.yaml"
        if not path.exists():
            twin = base / f"{stem}.yml"
            if twin.exists():
                return twin
        return path

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

    # ------------------------------------------------------------------
    # Listings (folder scans — no YAML parsing, never raise)
    # ------------------------------------------------------------------

    def _list_folder(self, folder: str) -> list[str]:
        """Sorted YAML stems of one config folder; ``[]`` when anything is missing.

        Never raises — an unresolvable configs root, a missing experiment
        folder, a missing kind folder, or an I/O failure mid-scan (an SMB
        visibility blip on a mounted configs share, a permissions problem)
        all read as an empty listing: clients render "nothing available",
        they do not crash.  A listed name is a *file*, not a promise:
        resolution/validation can still refuse it.
        """
        try:
            path = self._root / folder
            if not path.is_dir():
                return []
            return sorted(
                entry.stem
                for entry in path.iterdir()
                if entry.suffix in (".yaml", ".yml")
            )
        except Exception:  # root unresolvable / I/O failure — empty, never raise
            logger.debug(
                "config listing failed for %s (read as empty)", folder, exc_info=True
            )
            return []

    def list_trigger_profiles(self) -> list[str]:
        """Names accepted by :meth:`resolve_trigger_profile` (sorted; ``[]`` if none)."""
        return self._list_folder(self.TRIGGER_FOLDER)

    def list_presets(self) -> list[str]:
        """Names accepted by :meth:`resolve_preset` (sorted; ``[]`` if none)."""
        return self._list_folder(self.PRESET_FOLDER)

    def resolve_preset(self, name: str) -> Preset:
        """Load preset *name* as a validated :class:`~geecs_schemas.Preset`.

        One YAML per name under ``presets/`` — the console's PresetStore
        writes them; the queue client's ``submit_preset`` reads them here
        so the folder layout keeps one owner.

        Raises
        ------
        GeecsConfigurationError
            Missing file or a document that is not a mapping.
        pydantic.ValidationError
            A document that is not a valid ``Preset``.
        """
        stem = self._strip_yaml_suffix(name)
        path = self._named_yaml_path(self.PRESET_FOLDER, stem)
        document = self._load_yaml(path, "preset", name)
        return Preset.model_validate(document)

    def list_optimizer_configs(self) -> list[str]:
        """Optimizer-config names (``OptimizationSpec`` documents; sorted; ``[]`` if none)."""
        return self._list_folder(self.OPTIMIZER_FOLDER)

    def resolve_trigger_profile(self, name: str) -> TriggerProfile:
        """Load the trigger profile *name* (new schema, else converted).

        Raises
        ------
        GeecsConfigurationError
            Missing file, or a profile that names no trigger device.
        """
        stem = self._strip_yaml_suffix(name)
        path = self._named_yaml_path(self.TRIGGER_FOLDER, stem)
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
        path = self._root / self.SCAN_VARIABLES_FOLDER / "scan_variables.yaml"
        if not path.exists():
            raise GeecsConfigurationError(
                f"no scan-variable catalog for experiment "
                f"{self._experiment!r}: expected {path}"
            )
        document = self._load_yaml(path, "scan variables", "catalog")
        catalog = ScanVariables.model_validate(document)
        self._scan_variables_cache = catalog
        return catalog

    def scan_variable_catalog(self) -> ScanVariables:
        """The experiment's validated scan-variable catalog (public accessor).

        The whole :class:`~geecs_schemas.scan_variables.ScanVariables`
        document — consumers list names or branch on each spec's shape
        (plain vs pseudo).  Promoted from the private cache method for
        GEECS-Console's movable panel (its CLAUDE.md carried the debt note);
        cached per resolver, like every other catalog here.

        Raises
        ------
        GeecsConfigurationError
            No catalog files for the experiment, or an invalid document.
        """
        return self._scan_variables_catalog()

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
        """Load the experiment's action-plan library — from disk on every call.

        Not cached: the worker holds one resolver for its lifetime and
        ``run_action`` resolves through it, so a plan edited in
        ``actions.yaml`` (the Console's action-library editor writes it)
        must be what the next queue item runs.  One small YAML per item.
        """
        path = self._root / self.ACTION_FOLDER / "actions.yaml"
        document = self._load_yaml(path, "action library", "actions")
        if not document:
            # An empty file (a fresh experiment's placeholder) is an empty
            # library — the Console's store reads it the same way.
            return ActionPlanLibrary(plans={})
        # A legacy 'actions:' document is refused by the schema itself
        # (ActionPlanLibrary's before-validator names the regeneration).
        return ActionPlanLibrary.model_validate(document)

    def resolve_action_plan(self, name: str) -> ActionPlan:
        """Look up the action plan *name* in the experiment library.

        Raises
        ------
        GeecsConfigurationError
            Unknown name (the error lists the known plans).
        """
        library = self._action_library()
        try:
            return library.plans[name]
        except KeyError:
            raise GeecsConfigurationError(
                f"action plan {name!r} is not in the {self._experiment!r} "
                f"action library. Known plans: {sorted(library.plans)}"
            ) from None

    def action_plan_registry(self) -> dict[str, ActionPlan]:
        """Return every named plan visible to nested ``run`` steps (the library).

        Empty when the experiment has no ``actions.yaml`` at all; an
        unreadable or legacy-dialect file **raises** (a listing that read
        empty would hide the regeneration a legacy file needs).
        """
        if not (self._root / self.ACTION_FOLDER / "actions.yaml").exists():
            return {}
        return dict(self._action_library().plans)

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
