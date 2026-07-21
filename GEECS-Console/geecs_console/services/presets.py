"""Preset persistence: named :class:`~geecs_schemas.ScanRequest` YAML files.

A preset IS a saved ``ScanRequest`` — nothing more.  :class:`PresetStore`
keeps one YAML file per preset (``<name>.yaml``, a plain
``model_dump(mode="json")`` document that round-trips through
``ScanRequest.model_validate``) in the per-experiment configs area, beside
the other config kinds ``ConfigsRepoResolver`` reads::

    scanner_configs/experiments/<Experiment>/presets/<name>.yaml

Offline-safety mirrors :class:`~geecs_console.services.configs.ConsoleConfigs`:
listing degrades to empty with no configs root; ``load``/``save``/``delete``
raise :class:`PresetStoreError` with a clear message the window surfaces in
the status bar.  Directory creation and the scan-folder-invariant rationale
live in :mod:`geecs_console.services.config_store`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from geecs_console.services.config_store import NamedConfigStore

# The base resolves the configs root through this module's namespace, so
# tests can monkeypatch ``presets._configs_base`` (see config_store._root).
from geecs_console.services.configs import _configs_base  # noqa: F401
from geecs_schemas import ScanRequest

logger = logging.getLogger(__name__)

PRESET_FOLDER = "presets"


class PresetStoreError(RuntimeError):
    """A preset operation cannot be carried out (surfaced in the status bar)."""


class PresetStore(NamedConfigStore):
    """Load/save named scan-request presets for one experiment."""

    FOLDER = PRESET_FOLDER
    NOUN = "Preset"
    LABEL = "Preset"
    ERROR = PresetStoreError

    def load(self, name: str) -> ScanRequest:
        """Load preset *name* as a validated :class:`ScanRequest`.

        Parameters
        ----------
        name : str
            The preset name (file stem).

        Returns
        -------
        ScanRequest
            The schema-validated request the preset stores.

        Raises
        ------
        PresetStoreError
            Missing configs repo / experiment / file, unparsable YAML, or a
            document the schema rejects — always with a message fit for the
            status bar.
        """
        path = self._existing_path(name)
        document = self._read_mapping(
            path, describe=f"Preset {name!r}", mapping_hint=" of ScanRequest fields"
        )
        try:
            return ScanRequest.model_validate(document)
        except ValidationError as exc:
            raise PresetStoreError(
                f"Preset {name!r} ({path}) is not a valid ScanRequest: {exc}"
            ) from exc

    def save(self, name: str, request: ScanRequest) -> Path:
        """Write *request* as preset *name* (overwriting an existing one).

        Parameters
        ----------
        name : str
            The preset name (file stem).
        request : ScanRequest
            The request to persist.

        Returns
        -------
        Path
            The YAML file written.

        Raises
        ------
        PresetStoreError
            Missing configs repo, no experiment selected, a bad name, or an
            OS-level write failure.
        """
        path = self._path(name)
        self._write_document(path, request.model_dump(mode="json"), f"preset {name!r}")
        logger.info("saved preset %r to %s", name, path)
        return path
