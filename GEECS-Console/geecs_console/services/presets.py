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
the status bar.  Creating the ``presets/`` directory with
``mkdir(parents=True, exist_ok=True)`` is deliberate and fine — this is a
config directory in the configs repo, not a ``scans/ScanNNN/`` data folder,
so the repo's scan-folder creation invariant does not apply.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from geecs_console.services.configs import _configs_base
from geecs_schemas import ScanRequest

logger = logging.getLogger(__name__)

PRESET_FOLDER = "presets"


class PresetStoreError(RuntimeError):
    """A preset operation cannot be carried out (surfaced in the status bar)."""


class PresetStore:
    """Load/save named scan-request presets for one experiment.

    Parameters
    ----------
    experiment : str, optional
        Experiment folder under ``scanner_configs/experiments``.  May be
        empty at construction (before the operator picks one) — listing is
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
        """The experiment this store reads and writes presets for."""
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

    def _folder(self) -> Optional[Path]:
        """Return the experiment's presets dir, or ``None`` offline/unselected."""
        root = self._experiments_root
        if root is None:
            root = _configs_base()
        if root is None or not self._experiment:
            return None
        return root / self._experiment / PRESET_FOLDER

    def _folder_or_raise(self) -> Path:
        """Return the presets dir, raising a clear error when unresolvable.

        Returns
        -------
        Path
            ``<experiments root>/<experiment>/presets``.

        Raises
        ------
        PresetStoreError
            When the configs repo is not found or no experiment is selected.
        """
        root = self._experiments_root
        if root is None:
            root = _configs_base()
        if root is None:
            raise PresetStoreError(
                "Configs repo not found — set GEECS_SCANNER_CONFIG_DIR or "
                "config.ini [Paths] scanner_config_root_path."
            )
        if not self._experiment:
            raise PresetStoreError("No experiment selected.")
        return root / self._experiment / PRESET_FOLDER

    def _path(self, name: str) -> Path:
        """Return the YAML path for preset *name*, validating the name.

        Parameters
        ----------
        name : str
            The preset name (the file stem — no path separators).

        Returns
        -------
        Path
            ``<presets dir>/<name>.yaml`` (an existing ``.yml`` twin is
            preferred when the ``.yaml`` spelling is absent).

        Raises
        ------
        PresetStoreError
            On an empty name or one that would escape the presets dir.
        """
        cleaned = name.strip()
        if not cleaned:
            raise PresetStoreError("Preset name must not be empty.")
        if any(sep in cleaned for sep in ("/", "\\")) or cleaned in (".", ".."):
            raise PresetStoreError(
                f"Preset name {name!r} must be a plain file name (no path separators)."
            )
        folder = self._folder_or_raise()
        path = folder / f"{cleaned}.yaml"
        if not path.exists():
            twin = folder / f"{cleaned}.yml"
            if twin.exists():
                return twin
        return path

    # ------------------------------------------------------------------
    # The store surface
    # ------------------------------------------------------------------

    def list_names(self) -> list[str]:
        """List the saved preset names, sorted; never raises.

        Returns
        -------
        list of str
            YAML file stems in the presets dir — empty when the configs
            repo, experiment folder, or presets dir is missing.
        """
        folder = self._folder()
        if folder is None or not folder.is_dir():
            return []
        return sorted(
            path.stem for path in folder.iterdir() if path.suffix in (".yaml", ".yml")
        )

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
        path = self._path(name)
        if not path.exists():
            raise PresetStoreError(f"Preset {name!r} not found ({path}).")
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PresetStoreError(
                f"Preset {name!r} is not valid YAML ({path}): {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise PresetStoreError(
                f"Preset {name!r} ({path}) should be a YAML mapping of "
                f"ScanRequest fields, got {type(document).__name__}."
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
        # A config dir, not a scans/ScanNNN/ data folder — parents=True is
        # explicitly fine here (the scan-folder invariant does not apply).
        path.parent.mkdir(parents=True, exist_ok=True)
        document = yaml.safe_dump(
            request.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
        try:
            path.write_text(document, encoding="utf-8")
        except OSError as exc:
            raise PresetStoreError(f"Could not write preset {name!r}: {exc}") from exc
        logger.info("saved preset %r to %s", name, path)
        return path

    def delete(self, name: str) -> None:
        """Delete preset *name*.

        Parameters
        ----------
        name : str
            The preset name (file stem).

        Raises
        ------
        PresetStoreError
            Missing configs repo / experiment, a bad name, a preset that
            does not exist, or an OS-level delete failure.
        """
        path = self._path(name)
        if not path.exists():
            raise PresetStoreError(f"Preset {name!r} not found ({path}).")
        try:
            path.unlink()
        except OSError as exc:
            raise PresetStoreError(f"Could not delete preset {name!r}: {exc}") from exc
        logger.info("deleted preset %r (%s)", name, path)
