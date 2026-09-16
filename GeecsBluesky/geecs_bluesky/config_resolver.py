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
import os
import re
import tempfile
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
    OptimizerConfig,
    ScanVariables,
    ScanVariableSpec,
    ShotOffsets,
    TriggerProfile,
)
from geecs_schemas.convert import convert_shot_control

logger = logging.getLogger(__name__)

#: A preset file stem: a plain name, no path separators, no leading dot.
_PRESET_STEM = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


# ---------------------------------------------------------------------------
# ConfigResolver protocol
# ---------------------------------------------------------------------------


def _write_yaml_atomically(path: Path, document: dict) -> None:
    """Write *document* as YAML to *path*: temp file beside it, fsync, ``os.replace``.

    The one write primitive of this module (``write_shot_offsets``,
    ``write_preset``): a reader — the worker reopening its environment,
    another host's resolver, the scanner listing presets — never sees a
    half-written file, a failure part-way leaves the previous document
    intact and no temporary behind, and the destination's mode is kept
    (0644 for a new file) so the shared checkout stays readable to
    whoever reviews and commits the change.
    """
    payload = yaml.safe_dump(document, sort_keys=False)
    # The destination's mode, or the default a normal umask would give:
    # NamedTemporaryFile creates 0600 and os.replace keeps the temp
    # inode's mode, so without this one write makes the file unreadable
    # to the operator who has to review and commit it, to git run as
    # anyone else, and to another host's resolver.
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = 0o644
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            # Durability before the rename: without it a crash can leave
            # a zero-length file where a calibration used to be.
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, mode)
        except OSError:
            # The configs repo usually lives on the data share, and CIFS
            # mounts reject chmod unless mounted with unix extensions.
            # A mode we could not set is cosmetic; losing the ten shots
            # this document cost is not, so never fail the write for it.
            logger.warning(
                "could not set the mode of %s (the share may not support "
                "it) — the file is written, but check it is readable by "
                "whoever has to review and commit it",
                path,
                exc_info=True,
            )
        os.replace(temporary, path)
        temporary = None
    finally:
        # Covers the write and the chmod as well as the replace: a full
        # or disconnected share fails in `handle.write`, and a leftover
        # dot-file in a git working tree is somebody's next puzzle.
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    - ``optimizer_configs/<name>.yaml`` — validated native ``OptimizerConfig``
      documents, read fresh for each request.

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
        # (mtime_ns, size) of the catalog file the cached document came from.
        self._scan_variables_cache: tuple[tuple[int, int], ScanVariables] | None = None

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

    def preset_path(self, name: str) -> Path:
        """The file preset *name* lives in (``presets/<name>.yaml`` or its ``.yml`` twin)."""
        return self._named_yaml_path(self.PRESET_FOLDER, self._strip_yaml_suffix(name))

    def write_preset(self, preset: Preset, *, overwrite: bool = False) -> Path:
        """Save *preset* as ``presets/<preset.name>.yaml``; the file round-trips through :meth:`resolve_preset`.

        The scanner's "Save as preset" writes here (the Qt console's
        ``PresetStore`` was the previous writer; this keeps the folder one
        owner).  Atomic like :meth:`write_shot_offsets`, same mode rules,
        same "the configs repo is a git checkout — committing is a human
        act" contract: the path is returned so the caller can say which
        file now differs from the tree.

        Parameters
        ----------
        preset : Preset
            The validated document.  Its ``name`` is the file stem and must
            be a plain file name: letters, digits, ``_``, ``-`` and ``.``
            (no separators, no leading dot).  A ``.yaml`` / ``.yml`` suffix
            is stripped — the stem is the preset's name in the document too.
        overwrite : bool, default False
            Replace an existing preset of that name.  Off by default so a
            typo in the name cannot silently replace a curated preset.

        Raises
        ------
        GeecsConfigurationError
            A name that is not a file stem, the experiment folder missing
            (never created here — see :meth:`write_shot_offsets`), or an
            existing preset without *overwrite*.
        """
        name = preset.name
        while name != self._strip_yaml_suffix(name):
            name = self._strip_yaml_suffix(name)
        if name != preset.name:
            # ``jet.yaml`` is the file, ``jet`` the preset: the document's
            # name must be what ``list_presets`` says and ``resolve_preset``
            # is asked for.
            preset = preset.model_copy(update={"name": name})
        if not _PRESET_STEM.fullmatch(name):
            raise GeecsConfigurationError(
                f"preset name {name!r} is not a file name: use letters, digits, "
                "'_', '-' and '.', no separators"
            )
        folder = self._root / self.PRESET_FOLDER
        if not self._root.is_dir():
            raise GeecsConfigurationError(
                f"cannot write preset {name!r} for experiment {self._experiment!r}: "
                f"no configs folder at {self._root} (check the configs root and "
                "that the share is mounted)"
            )
        path = self.preset_path(name)
        if path.exists() and not overwrite:
            raise GeecsConfigurationError(
                f"preset {name!r} already exists at {path}; pass overwrite=True "
                "to replace it"
            )
        # The presets folder itself may be absent in a fresh experiment; the
        # experiment folder above it is the thing never created here.
        folder.mkdir(exist_ok=True)
        _write_yaml_atomically(path, preset.model_dump(mode="json"))
        logger.info("preset %r written to %s", name, path)
        return path

    @property
    def analysis_config_dir(self) -> Path:
        """Analysis-config tree beside scanner_configs in the same configs repository."""
        return self._root.parents[2] / "scan_analysis_configs"

    def optimizer_config_path(self, name: str) -> Path:
        """Path used to resolve an optimizer and its relative seed dumps."""
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
        except ValueError as exc:
            raise GeecsConfigurationError(f"optimizer config {name!r}: {exc}") from exc

    def diagnostic_device(self, stem: str) -> str:
        """Resolve a diagnostic's device without importing the analysis runtime."""
        from geecs_schemas.analysis import AnalysisDiagnostic

        if not stem or stem in (".", "..") or any(c in stem for c in ("/", "\\")):
            raise GeecsConfigurationError("diagnostic must be a file stem")
        paths = [
            p
            for p in (self.analysis_config_dir / "analyzers").rglob("*")
            if p.suffix in (".yaml", ".yml") and p.stem == stem
        ]
        if len(paths) != 1:
            raise GeecsConfigurationError(
                f"diagnostic {stem!r}: expected one document, found {len(paths)}"
            )
        return AnalysisDiagnostic.model_validate(
            self._load_yaml(paths[0], "diagnostic", stem)
        ).name

    def list_optimizer_configs(self) -> list[str]:
        """Optimizer-config names (``OptimizerConfig`` documents; sorted; ``[]`` if none)."""
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
        path = self._root / self.SCAN_VARIABLES_FOLDER / "scan_variables.yaml"
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
        (plain vs pseudo).  Promoted from the private cache method for
        the console's movable panel; cached per resolver until the file's
        mtime or size changes (the other kinds are re-read on every call).

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
        ``actions.yaml`` (edited by hand in the configs repo since the
        console's editor went) must be what the next queue item runs.  One small YAML per item.
        """
        path = self._root / self.ACTION_FOLDER / "actions.yaml"
        document = self._load_yaml(path, "action library", "actions")
        if not document:
            # An empty file (a fresh experiment's placeholder) is an empty
            # library, as the former console's store read it.  ``_load_yaml``
            # maps YAML ``None`` to ``{}``, so a literal ``{}`` reads the same
            # way here (that store rejected the literal — a file no writer
            # produces; its ``save_library`` wrote ``plans: {}``).
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

    SHOT_OFFSETS_FILE = "shot_offsets.yaml"

    @property
    def shot_offsets_path(self) -> Path:
        """Where this experiment's measured drain offsets live (written or not)."""
        return self._root / self.SHOT_OFFSETS_FILE

    def resolve_shot_offsets(self) -> ShotOffsets | None:
        """Load ``<experiment>/shot_offsets.yaml``; ``None`` if never measured.

        The per-device edge-to-stamp latencies the ``measure_shot_offsets``
        calibration plan writes (``03`` §4.F).  ``None`` — the state of
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
        if not path.exists():
            return None
        document = self._load_yaml(path, "shot offsets", self.SHOT_OFFSETS_FILE)
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

        The configs repo is a **git checkout**, usually on the data share.
        This writes the working tree only: committing and pushing is a
        human act, and the caller logs the path so the operator knows there
        is an uncommitted change to review.

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
        path = self.shot_offsets_path
        if not path.parent.is_dir():
            raise GeecsConfigurationError(
                f"cannot write shot offsets for experiment {self._experiment!r}: "
                f"no configs folder at {path.parent} (check the configs root "
                "and that the share is mounted)"
            )
        _write_yaml_atomically(path, offsets.model_dump())
        logger.info("shot offsets written to %s", path)
        return path
