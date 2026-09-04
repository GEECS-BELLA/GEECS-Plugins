"""Scan-variable catalog persistence: one :class:`~geecs_schemas.ScanVariables` YAML.

Unlike presets (one file per preset), the scan-variable catalog is **one
document per experiment** — the same file :class:`geecs_bluesky.config_resolver.
ConfigsRepoResolver` reads::

    scanner_configs/experiments/<Experiment>/scan_devices/scan_variables.yaml

The catalog is new-schema only — there is no legacy dialect (the
``scan_devices.yaml`` + ``composite_variables.yaml`` pair and its converter
were retired 2026-09, GEECS-Plugins#779); an experiment without the file
opens as an empty catalog.  Saving writes the new-schema file (``model_dump(mode="json",
exclude_none=True)``, which round-trips every set field through
``ScanVariables.model_validate`` and omits unset optionals exactly the way
the production catalogs are written).

Offline-safety mirrors :class:`~geecs_console.services.presets.PresetStore`:
:meth:`ScanVariableStore.load` degrades to an **empty catalog** when the
configs repo or the experiment folder is missing (the editor still opens);
save without a resolvable target raises :class:`ScanVariableStoreError` with
a message fit for the status bar.  Directory creation and the
scan-folder-invariant rationale live in
:mod:`geecs_console.services.config_store`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from geecs_console.services.config_store import ExperimentConfigStore

# The base resolves the configs root through this module's namespace, so
# tests can monkeypatch ``scan_variable_store._configs_base``.
from geecs_console.services.configs import _configs_base  # noqa: F401
from geecs_schemas import ScanVariables

logger = logging.getLogger(__name__)

SCAN_VARIABLES_FOLDER = "scan_devices"
CATALOG_FILE = "scan_variables.yaml"


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


class ScanVariableStore(ExperimentConfigStore):
    """Load/save one experiment's scan-variable catalog."""

    FOLDER = SCAN_VARIABLES_FOLDER
    ERROR = ScanVariableStoreError

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
        """Load the experiment's catalog (an absent file is an empty catalog).

        Returns
        -------
        ScanVariables
            The validated catalog.  An unresolvable configs repo, unselected
            experiment, or absent catalog file all degrade to an **empty**
            catalog — the editor still opens offline.

        Raises
        ------
        ScanVariableStoreError
            Unparsable YAML or a document the schema rejects — always with a
            message fit for the status bar.  A *missing* file is not an error.
        """
        folder = self._folder()
        if folder is None:
            return empty_catalog()
        path = folder / CATALOG_FILE
        if not path.exists():
            return empty_catalog()
        return self._load_new_schema(path)

    def _load_new_schema(self, path: Path) -> ScanVariables:
        """Load and validate the new-schema catalog at *path*.

        Raises
        ------
        ScanVariableStoreError
            Unparsable YAML, a non-mapping document, or schema rejection.
        """
        document = self._read_mapping(
            path,
            describe="Scan-variable catalog",
            mapping_hint=" of ScanVariables fields",
        )
        try:
            return ScanVariables.model_validate(document)
        except ValidationError as exc:
            raise ScanVariableStoreError(
                f"Scan-variable catalog ({path}) is not a valid ScanVariables "
                f"document: {exc}"
            ) from exc

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
        # exclude_none omits unset optionals (confirm, inverse) the way the
        # production catalogs are written; every set field round-trips.
        self._write_document(
            path,
            catalog.model_dump(mode="json", exclude_none=True),
            "scan-variable catalog",
        )
        logger.info(
            "saved scan-variable catalog (%d variables) to %s",
            len(catalog.variables),
            path,
        )
        return path
