"""Scan-variable catalog persistence: one :class:`~geecs_schemas.ScanVariables` YAML.

Unlike presets (one file per preset), the scan-variable catalog is **one
document per experiment** — the same file :class:`geecs_bluesky.config_resolver.
ConfigsRepoResolver` reads::

    scanner_configs/experiments/<Experiment>/scan_devices/scan_variables.yaml

Loading mirrors the resolver's dialect handling: a file whose top level
carries ``schema_version`` is the new schema; when it is absent the legacy
pair (``scan_devices.yaml`` + ``composite_variables.yaml``) is converted
through :func:`geecs_schemas.convert.convert_scan_variables` — so an
experiment that has not migrated yet still opens in the editor.  Saving
always writes the new-schema file (``model_dump(mode="json",
exclude_none=True)``, which round-trips every set field through
``ScanVariables.model_validate`` and omits unset optionals exactly the way
the production catalogs are written).

Offline-safety mirrors :class:`~geecs_console.services.presets.PresetStore`:
:meth:`ScanVariableStore.load` degrades to an **empty catalog** when the
configs repo or the experiment folder is missing (the editor still opens);
save without a resolvable target raises :class:`ScanVariableStoreError` with
a message fit for the status bar.  Creating the ``scan_devices/`` directory
with ``mkdir(parents=True, exist_ok=True)`` is deliberate and fine — this is
a config directory in the configs repo, not a ``scans/ScanNNN/`` data folder,
so the repo's scan-folder creation invariant does not apply.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from geecs_console.services._experiment_name import check_experiment_name
from geecs_console.services.configs import _configs_base
from geecs_schemas import ScanVariables

logger = logging.getLogger(__name__)

SCAN_VARIABLES_FOLDER = "scan_devices"
CATALOG_FILE = "scan_variables.yaml"
LEGACY_SIMPLE_FILE = "scan_devices.yaml"
LEGACY_COMPOSITE_FILE = "composite_variables.yaml"


class ScanVariableStoreError(RuntimeError):
    """A catalog operation cannot be carried out (surfaced in the status bar)."""


def empty_catalog() -> ScanVariables:
    """Return a fresh empty catalog (what a new experiment starts from).

    Returns
    -------
    ScanVariables
        A validated catalog with no variables.
    """
    return ScanVariables(variables={})


class ScanVariableStore:
    """Load/save one experiment's scan-variable catalog.

    Parameters
    ----------
    experiment : str, optional
        Experiment folder under ``scanner_configs/experiments``.  May be
        empty at construction (before the operator picks one) — loading is
        then empty and saving raises.
    experiments_root : str or Path, optional
        Override for the experiments root (tests point this at a tmp dir);
        defaults to the production resolution
        (:func:`geecs_bluesky.scanner_configs.scanner_configs_base`, resolved
        lazily so the module stays import-safe offline).
    """

    def __init__(
        self, experiment: str = "", experiments_root: str | Path | None = None
    ) -> None:
        self._experiment = experiment
        self._experiments_root = (
            Path(experiments_root) if experiments_root is not None else None
        )

    @property
    def experiment(self) -> str:
        """The experiment this store reads and writes the catalog for."""
        return self._experiment

    def set_experiment(self, experiment: str) -> None:
        """Switch the store to *experiment*.

        Parameters
        ----------
        experiment : str
            The new experiment folder name ("" for none selected).
        """
        self._experiment = experiment

    # ------------------------------------------------------------------
    # Folder resolution
    # ------------------------------------------------------------------

    def _root(self) -> Optional[Path]:
        """Return the experiments root, or ``None`` when unresolvable."""
        if self._experiments_root is not None:
            return self._experiments_root
        return _configs_base()

    def _check_experiment(self) -> None:
        """Reject an experiment name that would escape the experiments root.

        Delegates to the shared
        :func:`~geecs_console.services._experiment_name.check_experiment_name`
        guard every config store applies before any path join (issue #513).

        Raises
        ------
        ScanVariableStoreError
            When the experiment name contains a path separator or is a
            relative-path special name.
        """
        check_experiment_name(self._experiment, ScanVariableStoreError)

    def _folder(self) -> Optional[Path]:
        """Return the ``scan_devices/`` dir, or ``None`` offline/unselected."""
        root = self._root()
        if root is None or not self._experiment:
            return None
        self._check_experiment()
        return root / self._experiment / SCAN_VARIABLES_FOLDER

    def _folder_or_raise(self) -> Path:
        """Return the ``scan_devices/`` dir, raising a clear error otherwise.

        Returns
        -------
        Path
            ``<experiments root>/<experiment>/scan_devices``.

        Raises
        ------
        ScanVariableStoreError
            When the configs repo is not found or no experiment is selected.
        """
        root = self._root()
        if root is None:
            raise ScanVariableStoreError(
                "Configs repo not found — set GEECS_SCANNER_CONFIG_DIR or "
                "config.ini [Paths] scanner_config_root_path."
            )
        if not self._experiment:
            raise ScanVariableStoreError("No experiment selected.")
        self._check_experiment()
        return root / self._experiment / SCAN_VARIABLES_FOLDER

    def catalog_path(self) -> Optional[Path]:
        """Return the new-schema catalog path, or ``None`` offline/unselected.

        Returns
        -------
        Path or None
            ``<scan_devices dir>/scan_variables.yaml`` — the file
            :meth:`save` writes, whether or not it exists yet.
        """
        folder = self._folder()
        return None if folder is None else folder / CATALOG_FILE

    # ------------------------------------------------------------------
    # The store surface
    # ------------------------------------------------------------------

    def load(self) -> ScanVariables:
        """Load the experiment's catalog (new schema, else converted legacy).

        Returns
        -------
        ScanVariables
            The validated catalog.  An unresolvable configs repo, unselected
            experiment, or absent catalog file all degrade to an **empty**
            catalog — the editor still opens offline.

        Raises
        ------
        ScanVariableStoreError
            Unparsable YAML, a document the schema rejects, or a legacy pair
            the converter rejects — always with a message fit for the status
            bar.  A *missing* file is not an error.
        """
        folder = self._folder()
        if folder is None:
            return empty_catalog()
        new_schema = folder / CATALOG_FILE
        if new_schema.exists():
            return self._load_new_schema(new_schema)
        return self._load_legacy(folder)

    def _load_new_schema(self, path: Path) -> ScanVariables:
        """Load and validate the new-schema catalog at *path*.

        Raises
        ------
        ScanVariableStoreError
            Unparsable YAML, a non-mapping document, or schema rejection.
        """
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ScanVariableStoreError(
                f"Scan-variable catalog is not valid YAML ({path}): {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise ScanVariableStoreError(
                f"Scan-variable catalog ({path}) should be a YAML mapping of "
                f"ScanVariables fields, got {type(document).__name__}."
            )
        try:
            return ScanVariables.model_validate(document)
        except ValidationError as exc:
            raise ScanVariableStoreError(
                f"Scan-variable catalog ({path}) is not a valid ScanVariables "
                f"document: {exc}"
            ) from exc

    def _load_legacy(self, folder: Path) -> ScanVariables:
        """Convert the legacy pair in *folder*; empty catalog when absent.

        Raises
        ------
        ScanVariableStoreError
            When the legacy converter rejects either file.
        """
        simple = folder / LEGACY_SIMPLE_FILE
        composites = folder / LEGACY_COMPOSITE_FILE
        if not simple.exists() and not composites.exists():
            return empty_catalog()
        # Lazy import: the converter is pydantic-only (geecs_schemas), but
        # keeping it out of module import keeps the cheap path cheap.
        from geecs_schemas.convert import convert_scan_variables

        try:
            catalog = convert_scan_variables(
                simple if simple.exists() else None,
                composites if composites.exists() else None,
            )
        except Exception as exc:
            raise ScanVariableStoreError(
                f"Legacy scan-variable files in {folder} could not be converted: {exc}"
            ) from exc
        logger.info(
            "loaded legacy scan-variable pair from %s (%d variables)",
            folder,
            len(catalog.variables),
        )
        return catalog

    def save(self, catalog: ScanVariables) -> Path:
        """Write *catalog* as the experiment's new-schema catalog file.

        Parameters
        ----------
        catalog : ScanVariables
            The catalog to persist.

        Returns
        -------
        Path
            The YAML file written (``scan_devices/scan_variables.yaml``).

        Raises
        ------
        ScanVariableStoreError
            Missing configs repo, no experiment selected, or an OS-level
            write failure.
        """
        folder = self._folder_or_raise()
        path = folder / CATALOG_FILE
        # A config dir, not a scans/ScanNNN/ data folder — parents=True is
        # explicitly fine here (the scan-folder invariant does not apply).
        folder.mkdir(parents=True, exist_ok=True)
        # exclude_none omits unset optionals (confirm, inverse) the way the
        # production catalogs are written; every set field round-trips.
        document = yaml.safe_dump(
            catalog.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        )
        try:
            path.write_text(document, encoding="utf-8")
        except OSError as exc:
            raise ScanVariableStoreError(
                f"Could not write scan-variable catalog: {exc}"
            ) from exc
        logger.info(
            "saved scan-variable catalog (%d variables) to %s",
            len(catalog.variables),
            path,
        )
        return path
