"""Resolve the configs-repo names a client uses into validated schema models.

:class:`ConfigResolver` is the protocol; :class:`ConfigsRepoResolver` is the
production implementation over the real configs-repo layout
(``scanner_configs/experiments/<Experiment>/``).  A YAML file carrying a
``schema_version`` key loads as the new schema directly; trigger profiles
without one are converted from their legacy dialect via
:mod:`geecs_schemas.convert`.  Presets, scan-variable catalogs and action
libraries are new-schema only (the legacy save elements, scan presets and
action libraries were regenerated once as ``Preset`` and
``ActionPlanLibrary`` documents; no converter remains for them).

The client seam expands a preset into a plan queue item
(:mod:`geecs_bluesky.qs_client.presets`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import yaml
from pydantic import JsonValue

if TYPE_CHECKING:
    from geecs_schemas.analysis import AnalysisDocument

from geecs_bluesky._fs import replace_text
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_core.configs_repo import (
    scanner_configs_base,
    scanner_configs_write_base,
    settled_exists,
)
from geecs_schemas import (
    ActionPlan,
    ActionPlanLibrary,
    EXPERIMENT_DEFAULTS_FILENAME,
    NATIVE_DEVICES_FILENAME,
    OPTIMIZER_CONFIGS_FOLDER,
    PRESETS_FOLDER,
    SCAN_VARIABLES_FILENAME,
    SCAN_VARIABLES_FOLDER,
    SHOT_OFFSETS_FILENAME,
    TRIGGER_PROFILES_FOLDER,
    ExperimentDefaults,
    NativeDevices,
    Preset,
    OptimizerConfig,
    ScanVariables,
    ScanVariableSpec,
    ShotOffsets,
    TriggerProfile,
)
from geecs_schemas.action_plan import ACTION_LIBRARY_FILENAME, ACTION_LIBRARY_FOLDER
from geecs_schemas.convert import convert_shot_control

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConfigResolver protocol
# ---------------------------------------------------------------------------


def _write_yaml_atomically(path: Path, document: dict) -> None:
    """Write *document* as YAML to *path*, atomically and fsynced (:func:`replace_text`).

    The one write primitive of this module (``write_shot_offsets``): a
    reader — the configurator claiming the file from its inbox, another
    host's resolver — never sees a half-written file.
    """
    replace_text(path, yaml.safe_dump(document, sort_keys=False))


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

    def resolve_shot_offsets(self) -> ShotOffsets | None:
        """Return the measured drain offsets, or ``None`` if never measured.

        Resolvers without this method are tolerated (no calibration), so
        callers check for it rather than assuming it — see
        ``geecs_bluesky.plans.calibration._can_write`` for the write side,
        which must be checked *before* a measurement is spent.
        """
        ...

    def resolve_experiment_defaults(self) -> ExperimentDefaults | None:
        """Return the experiment's defaults, or ``None`` if none are declared.

        Defaults apply where the request is silent (default trigger
        profile; default setup/closeout plans prepended); what was applied
        is recorded for provenance (see the client-side preset expansion,
        :mod:`geecs_bluesky.qs_client.presets`).
        Resolvers without this method are tolerated (no defaults).
        """
        ...


class ConfigsRepoResolver:
    """Resolver over the real configs-repo layout, converter-backed.

    Reads ``scanner_configs/experiments/<experiment>/`` (the same resolution
    roots as :func:`geecs_core.configs_repo.scanner_configs_base`):

    - ``presets/<name>.yaml`` — presets (``geecs_schemas.Preset``: the
      device group + the plan call; new schema only)
    - ``shot_control_configurations/<name>.yaml`` — trigger profiles
    - ``scan_devices/scan_variables.yaml`` — the scan-variable catalog
      (new schema only)
    - ``action_library/actions.yaml`` — the action-plan library (new
      schema only; the legacy ``actions:`` dialect is refused)
    - ``optimizer_configs/<name>.yaml`` — validated native ``OptimizerConfig``
      documents, read fresh for each request.

    A trigger profile whose top level carries ``schema_version`` is
    loaded as the new schema; anything else goes through the legacy
    converter.  Named configs resolve from
    either the ``.yaml`` or ``.yml`` spelling, so every listed name
    round-trips through resolution.

    Parameters
    ----------
    experiment :
        Experiment folder name under ``scanner_configs/experiments``.
    experiments_root :
        Override for the experiments root (tests); defaults to the
        production resolution (:func:`geecs_core.configs_repo.scanner_configs_base`),
        resolved lazily on first use.
    experiments_write_root :
        Where the one write (:meth:`write_shot_offsets`) lands: defaults to
        *experiments_root* when that is given, else the production write
        root (:func:`geecs_core.configs_repo.scanner_configs_write_base`:
        ``GEECS_CONFIGS_WRITE_ROOT``, the configurator's inbox on the box,
        else the read root). Reads never look there.
    """

    # The folder names are GEECS-Schemas' constants (the one home); these
    # aliases keep the resolver's own spelling for its callers and tests.
    TRIGGER_FOLDER = TRIGGER_PROFILES_FOLDER
    SCAN_VARIABLES_FOLDER = SCAN_VARIABLES_FOLDER
    ACTION_FOLDER = ACTION_LIBRARY_FOLDER
    PRESET_FOLDER = PRESETS_FOLDER
    OPTIMIZER_FOLDER = OPTIMIZER_CONFIGS_FOLDER

    def __init__(
        self,
        experiment: str,
        experiments_root: str | Path | None = None,
        *,
        experiments_write_root: str | Path | None = None,
    ) -> None:
        self._experiment = experiment
        self._experiments_root = (
            Path(experiments_root) if experiments_root is not None else None
        )
        self._experiments_write_root = (
            Path(experiments_write_root)
            if experiments_write_root is not None
            else self._experiments_root
        )
        # (mtime_ns, size) of the catalog file the cached document came from.
        self._scan_variables_cache: tuple[tuple[int, int], ScanVariables] | None = None

    @property
    def _root(self) -> Path:
        root = self._experiments_root or scanner_configs_base()
        if root is None:
            raise RuntimeError(
                "Cannot resolve the scanner configs base. Set GEECS_SCANNER_CONFIG_DIR, or "
                "config.ini [Paths] scanner_config_root_path pointing at GEECS-Plugins-Configs."
            )
        return root / self._experiment

    @staticmethod
    def _strip_yaml_suffix(name: str) -> str:
        for suffix in (".yaml", ".yml"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def _named_yaml_path(self, folder: str, stem: str) -> Path:
        """The config file for *stem* in *folder*: ``.yaml``, else its ``.yml`` twin.

        The listings count both spellings, so resolution must round-trip
        every listed name.  When neither exists, the ``.yaml`` path is
        returned so the not-found error names the canonical spelling.
        """
        base = self._root / folder
        path = base / f"{stem}.yaml"
        if not settled_exists(path):
            twin = base / f"{stem}.yml"
            if settled_exists(twin):
                return twin
        return path

    def _load_yaml(self, path: Path, kind: str, name: str) -> dict:
        """Load one YAML mapping, failing loudly with the config kind/name."""
        if not settled_exists(path):
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

    def _list_folder(self, folder: str, *, strict: bool = False) -> list[str]:
        """Sorted YAML stems of one config folder; ``[]`` when anything is missing.

        With ``strict=True``, unresolved/missing experiment roots and I/O
        errors propagate so clients can explain an unavailable library. A
        missing kind folder within an existing experiment still reads empty.
        By default, never raises — an unresolvable configs root, a missing experiment
        folder, a missing kind folder, or an I/O failure mid-scan (an SMB
        visibility blip on a mounted configs share, a permissions problem)
        all read as an empty listing: clients render "nothing available",
        they do not crash.  A listed name is a *file*, not a promise:
        resolution/validation can still refuse it.
        """
        try:
            root = self._root
            if strict:
                root.stat()
            path = root / folder
            if not path.is_dir():
                return []
            return sorted(
                entry.stem
                for entry in path.iterdir()
                if entry.suffix in (".yaml", ".yml")
            )
        except Exception:  # Root resolution and directory traversal can both fail.
            if strict:
                raise
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

        One YAML per name under ``presets/`` — saved through the
        configurator (the scanner posts to it); the queue client's ``submit_preset`` reads them here so the
        folder layout keeps one owner.

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

    @property
    def analysis_config_dir(self) -> Path:
        """The shared Scan/ImageAnalysis config root, independent of scanner configs."""
        from geecs_data_utils.config_roots import scan_analysis_config

        root = scan_analysis_config.base_dir
        if root is None:
            raise GeecsConfigurationError(
                "set SCAN_ANALYSIS_CONFIG_DIR or [Paths] scan_analysis_configs_path"
            )
        return root

    def optimizer_config_path(self, name: str) -> Path:
        """Return the optimizer config path; its seed dumps resolve relative to it."""
        stem = self._strip_yaml_suffix(name)
        if not stem or stem in (".", "..") or any(c in stem for c in ("/", "\\")):
            raise GeecsConfigurationError("optimizer config must be a file stem")
        return self._named_yaml_path(self.OPTIMIZER_FOLDER, stem)

    def resolve_optimizer_config(self, name: str) -> OptimizerConfig:
        """Load and validate the native optimizer document fresh for each request."""
        path = self.optimizer_config_path(name)
        try:
            return OptimizerConfig.model_validate(
                self._load_yaml(path, "optimizer config", name)
            )
        except (ValueError, OSError, yaml.YAMLError) as exc:
            raise GeecsConfigurationError(f"optimizer config {name!r}: {exc}") from exc

    def diagnostic_device(self, stem: str) -> str:
        """Resolve a diagnostic's device without importing the analysis runtime."""
        return self.resolve_diagnostic(stem).device

    def resolve_diagnostic(
        self, stem: str, *, overrides: dict[str, JsonValue] | None = None
    ) -> AnalysisDocument:
        """Read and validate a diagnostic (a recipe, or a .himg job) fresh from the shared read-only source."""
        from geecs_data_utils.analysis_configs import read_diagnostic
        from geecs_schemas.analysis import load_analysis_document

        if not stem or stem in (".", "..") or any(c in stem for c in ("/", "\\")):
            raise GeecsConfigurationError("diagnostic must be a file stem")
        try:
            path, document = read_diagnostic(
                stem, config_dir=self.analysis_config_dir, overrides=overrides
            )
            diagnostic = load_analysis_document(document)
            diagnostic._source_id = path.stem
            return diagnostic
        except (KeyError, ValueError, OSError, yaml.YAMLError) as exc:
            raise GeecsConfigurationError(f"diagnostic {stem!r}: {exc}") from exc

    def list_optimizer_configs(self) -> list[str]:
        """Optimizer-config names (``OptimizerConfig`` documents; sorted; ``[]`` if none)."""
        return self.optimizer_config_listing()[0]

    def optimizer_config_listing(self) -> tuple[list[str], dict[str, str]]:
        """Return usable names and unavailable names mapped to validation reasons."""
        available = []
        unavailable = {}
        for name in self._list_folder(self.OPTIMIZER_FOLDER, strict=True):
            try:
                self.resolve_optimizer_config(name)
            except GeecsConfigurationError as exc:
                unavailable[name] = str(exc)
            else:
                available.append(name)
        return available, unavailable

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
        """Load the experiment's scan-variable catalog, cached until the file changes.

        The resolver lives as long as its process (the web scanner's — the
        catalog's only runtime reader), so a lifetime cache made every
        catalog edit wait for a restart.  A ``stat`` per call (the parse only on a
        miss) keeps the cost away from the hot paths — preflight, submit,
        move — while an edited file, new mtime or size, is re-read on the
        next call.  Accepted blind spot, shared with the portal's config
        fingerprint: a same-length edit saved within one mtime tick of
        the previous read (SMB shares tick at 1–2 s) is served stale until
        the next tick or byte-count change.
        """
        path = self._root / self.SCAN_VARIABLES_FOLDER / SCAN_VARIABLES_FILENAME
        try:
            st = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            # ``Path.exists`` read both as "missing"; keep that message.
            raise GeecsConfigurationError(
                f"no scan-variable catalog for experiment "
                f"{self._experiment!r}: expected {path}"
            ) from None
        stamp = (st.st_mtime_ns, st.st_size)
        cached = self._scan_variables_cache
        if cached is not None and cached[0] == stamp:
            return cached[1]
        document = self._load_yaml(path, "scan variables", "catalog")
        catalog = ScanVariables.model_validate(document)
        self._scan_variables_cache = (stamp, catalog)
        return catalog

    def scan_variable_catalog(self) -> ScanVariables:
        """The experiment's validated scan-variable catalog (public accessor).

        The whole :class:`~geecs_schemas.scan_variables.ScanVariables`
        document — consumers list names or branch on each spec's shape
        (plain vs pseudo).  Cached per resolver until the file's mtime or
        size changes (the other kinds are re-read on every call).

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
        ``run_action`` resolves through it, so a plan edited by hand in
        ``actions.yaml`` must be what the next queue item runs.  One small
        YAML per item.
        """
        path = self._root / self.ACTION_FOLDER / ACTION_LIBRARY_FILENAME
        document = self._load_yaml(path, "action library", "actions")
        if not document:
            # An empty file (a fresh experiment's placeholder) is an empty
            # library.  ``_load_yaml`` maps YAML ``None`` to ``{}``, so a
            # literal ``{}`` reads the same way here.
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
        if not settled_exists(
            self._root / self.ACTION_FOLDER / ACTION_LIBRARY_FILENAME
        ):
            return {}
        return dict(self._action_library().plans)

    def resolve_experiment_defaults(self) -> ExperimentDefaults | None:
        """Load ``<experiment>/experiment_defaults.yaml``; ``None`` if absent.

        No legacy dialect exists behind it (the legacy scanner kept these
        choices in GUI state).
        """
        path = self._root / EXPERIMENT_DEFAULTS_FILENAME
        if not settled_exists(path):
            return None
        document = self._load_yaml(
            path, "experiment defaults", EXPERIMENT_DEFAULTS_FILENAME
        )
        return ExperimentDefaults.model_validate(document)

    def resolve_native_devices(self) -> NativeDevices:
        """Load ``<experiment>/native_devices.yaml``; no file = no native devices.

        The registry of devices served by native EPICS IOCs
        (``NATIVE_DEVICES.md``), read at environment open and by each
        preset expansion (the ``save_images`` refusal).  No
        resolvable configs root reads as no file (logged): such a worker
        loads no trigger profile either.

        Raises
        ------
        GeecsConfigurationError
            The experiments root is unreachable, or the file exists but is
            unreadable, not a mapping, or not a valid ``NativeDevices``
            document — configured but broken is not "none".
        """
        try:
            path = self._root / NATIVE_DEVICES_FILENAME
        except RuntimeError as exc:
            logger.warning("no native devices: %s", exc)
            return NativeDevices()
        if not path.parent.parent.is_dir():
            # An unmounted share must not read as "no native devices".
            raise GeecsConfigurationError(
                f"configs directory {path.parent.parent} is not reachable"
            )
        if not settled_exists(path):
            return NativeDevices()
        try:
            document = self._load_yaml(
                path, "native device registry", NATIVE_DEVICES_FILENAME
            )
            return NativeDevices.model_validate(document)
        except (ValueError, OSError, yaml.YAMLError) as exc:
            raise GeecsConfigurationError(
                f"native device registry {path} is invalid: {exc}"
            ) from exc

    @property
    def shot_offsets_path(self) -> Path:
        """Where this experiment's measured drain offsets live (written or not)."""
        return self._root / SHOT_OFFSETS_FILENAME

    @property
    def shot_offsets_write_path(self) -> Path:
        """Where :meth:`write_shot_offsets` writes: the same file under the write root.

        The read path itself unless ``GEECS_CONFIGS_WRITE_ROOT`` names
        another root (the configurator's inbox on the services box, which
        commits the file to main; the worker reads it back from the live
        checkout at its next environment open).
        """
        root = self._experiments_write_root or scanner_configs_write_base()
        if root is None:
            raise GeecsConfigurationError(
                "cannot resolve where to write: set GEECS_CONFIGS_WRITE_ROOT, or the "
                "configs root (config.ini [Paths] scanner_config_root_path)"
            )
        return root / self._experiment / SHOT_OFFSETS_FILENAME

    def resolve_shot_offsets(self) -> ShotOffsets | None:
        """Load ``<experiment>/shot_offsets.yaml``; ``None`` if never measured.

        The per-device edge-to-stamp latencies the ``measure_shot_offsets``
        calibration plan writes.  ``None`` — the state of
        every experiment until the plan is first run — leaves every
        detector's ``drain_offset`` at ``0.0``, which is what the join
        assumed before this document existed.

        Raises
        ------
        GeecsConfigurationError
            The file exists but is not a YAML mapping.
        pydantic.ValidationError
            The file exists but is not a valid ``ShotOffsets`` document.
            Deliberately loud rather than falling back to zeros: a
            calibration that silently reverted to 0.0 would misjoin rows at
            a tight rep rate with nothing in the log to say why.
        """
        path = self.shot_offsets_path
        if not settled_exists(path):
            return None
        document = self._load_yaml(path, "shot offsets", SHOT_OFFSETS_FILENAME)
        return ShotOffsets.model_validate(document)

    def write_shot_offsets(self, offsets: ShotOffsets) -> Path:
        """Write the measured drain offsets, replacing any previous measurement.

        Written atomically (a temporary file in the same directory, fsynced,
        then ``os.replace``) so a reader — the worker reopening its
        environment, another host's resolver — never sees a half-written
        document, and a failure part-way leaves the previous calibration
        intact and no temporary behind.  The destination's permissions are
        preserved (0644 for a new file): the configs repo is a shared
        checkout, and a file only the service account could read would
        block the very review this write exists to invite.

        It writes to :attr:`shot_offsets_write_path`.  On the services box
        that is the configurator's inbox, which commits the file to the
        configs repo's main as the worker (the live checkout the worker
        reads is never written: git refuses to fast-forward over an
        uncommitted file); elsewhere it is the read path, a working-tree
        change to review and commit.

        Returns
        -------
        Path
            The file written.

        Raises
        ------
        GeecsConfigurationError
            The experiment folder does not exist — the configs root is
            misconfigured or unreachable, and creating the tree here would
            plant an experiment folder in the wrong place.
        """
        path = self.shot_offsets_write_path
        if not path.parent.is_dir():
            raise GeecsConfigurationError(
                f"cannot write shot offsets for experiment {self._experiment!r}: "
                f"no configs folder at {path.parent} (check the configs root "
                "and that the share is mounted)"
            )
        _write_yaml_atomically(path, offsets.model_dump())
        logger.info("shot offsets written to %s", path)
        return path
