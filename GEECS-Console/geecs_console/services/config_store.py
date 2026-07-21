"""Shared persistence base for the per-experiment config stores.

Five stores keep the console's config kinds in the configs repo
(``scanner_configs/experiments/<Experiment>/...``): presets, save sets,
trigger profiles, the scan-variable catalog, and the action library.
:class:`ExperimentConfigStore` owns what they all share — experiment
selection, configs-root and folder resolution, the experiment-name guard,
and safe YAML read/write with status-bar-ready errors.
:class:`NamedConfigStore` adds the file-per-name surface (name validation,
``.yml`` twin fallback, sorted listing, delete) the preset / save-set /
trigger-profile trio share.  Each concrete store keeps its own error
class, folder constants, and schema-specific load/save logic in its own
module.

Directory creation policy — the canonical statement for every store:
creating a config directory with ``mkdir(parents=True, exist_ok=True)``
is deliberate and fine.  These are config directories in the configs
repo, not ``scans/ScanNNN/`` data folders, so the repo's scan-folder
creation invariant does not apply.

The production configs root is resolved through each concrete store's
module namespace: every store module imports ``_configs_base`` from
:mod:`geecs_console.services.configs`, and :meth:`ExperimentConfigStore._root`
reads that binding at call time — so tests can monkeypatch
``<store module>._configs_base`` per store, exactly as before the
extraction.  (This module deliberately does not import
:mod:`~geecs_console.services.configs`, which imports :func:`yaml_stems`
from here.)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from geecs_console.services._experiment_name import check_experiment_name

_CONFIGS_REPO_MISSING = (
    "Configs repo not found — set GEECS_SCANNER_CONFIG_DIR or "
    "config.ini [Paths] scanner_config_root_path."
)
_NO_EXPERIMENT = "No experiment selected."


def yaml_stems(folder: Path) -> list[str]:
    """List the YAML file stems in *folder*, sorted; empty when absent.

    Parameters
    ----------
    folder : Path
        The directory to scan (need not exist).

    Returns
    -------
    list of str
        Sorted stems of the ``.yaml``/``.yml`` files in *folder*.
    """
    if not folder.is_dir():
        return []
    return sorted(
        path.stem for path in folder.iterdir() if path.suffix in (".yaml", ".yml")
    )


class ExperimentConfigStore:
    """Base for the per-experiment config stores.

    Subclasses set :attr:`FOLDER` and :attr:`ERROR` and add their
    schema-specific load/save surface on top of the resolution and I/O
    helpers here.

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

    FOLDER: str
    """The store's per-experiment subfolder name (e.g. ``"presets"``)."""

    ERROR: type[RuntimeError]
    """The store's concrete error class (status-bar/inline-ready messages)."""

    def __init__(
        self, experiment: str = "", experiments_root: str | Path | None = None
    ) -> None:
        self._experiment = experiment
        self._experiments_root = (
            Path(experiments_root) if experiments_root is not None else None
        )

    @property
    def experiment(self) -> str:
        """The experiment this store reads and writes configs for."""
        return self._experiment

    def set_experiment(self, experiment: str) -> None:
        """Switch the store to *experiment*.

        Parameters
        ----------
        experiment : str
            The new experiment folder name ("" for none selected).
        """
        self._experiment = experiment

    @property
    def _logger(self) -> logging.Logger:
        """The concrete store module's logger (log lines keep their origin)."""
        return logging.getLogger(type(self).__module__)

    # ------------------------------------------------------------------
    # Folder resolution
    # ------------------------------------------------------------------

    def _root(self) -> Optional[Path]:
        """Return the experiments root, or ``None`` offline.

        The production resolution is read through the concrete store's
        module namespace (each store module imports ``_configs_base``), so
        tests can monkeypatch ``<store module>._configs_base``.
        """
        if self._experiments_root is not None:
            return self._experiments_root
        return sys.modules[type(self).__module__]._configs_base()

    def _folder(self) -> Optional[Path]:
        """Return the store's per-experiment dir, or ``None`` offline/unselected.

        The experiment name is validated before any path join by the shared
        guard in :mod:`geecs_console.services._experiment_name` (see its
        module docstring for the traversal rationale).

        Raises
        ------
        RuntimeError
            The store's :attr:`ERROR` when the experiment name would escape
            the experiments root; a merely missing root or unselected
            experiment yields ``None``, not an error.
        """
        root = self._root()
        if root is None or not self._experiment:
            return None
        check_experiment_name(self._experiment, self.ERROR)
        return root / self._experiment / self.FOLDER

    def _folder_or_raise(self) -> Path:
        """Return the store's per-experiment dir, raising when unresolvable.

        Returns
        -------
        Path
            ``<experiments root>/<experiment>/<FOLDER>``.

        Raises
        ------
        RuntimeError
            The store's :attr:`ERROR` when the configs repo is not found,
            no experiment is selected, or the experiment name would escape
            the experiments root.
        """
        root = self._root()
        if root is None:
            raise self.ERROR(_CONFIGS_REPO_MISSING)
        if not self._experiment:
            raise self.ERROR(_NO_EXPERIMENT)
        check_experiment_name(self._experiment, self.ERROR)
        return root / self._experiment / self.FOLDER

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    def _read_mapping(
        self,
        path: Path,
        *,
        describe: str,
        mapping_hint: str = "",
        empty_as: Optional[dict] = None,
    ) -> dict:
        """Read *path* as one YAML mapping, with status-bar-ready errors.

        Parameters
        ----------
        path : Path
            The YAML file to read (must exist — callers check first where a
            friendlier not-found message applies).
        describe : str
            Message subject, e.g. ``"Preset 'align'"`` — rendered as
            ``"<describe> is not valid YAML (<path>): ..."``.
        mapping_hint : str, optional
            Message tail after "should be a YAML mapping", e.g.
            ``" of ScanRequest fields"``.
        empty_as : dict, optional
            Returned as-is for an all-empty document (YAML ``None``); when
            omitted, an empty document fails the mapping check instead.

        Returns
        -------
        dict
            The parsed top-level mapping.

        Raises
        ------
        RuntimeError
            The store's :attr:`ERROR` on unparsable YAML or a non-mapping
            document.
        """
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise self.ERROR(f"{describe} is not valid YAML ({path}): {exc}") from exc
        if document is None and empty_as is not None:
            return empty_as
        if not isinstance(document, dict):
            raise self.ERROR(
                f"{describe} ({path}) should be a YAML mapping{mapping_hint}, "
                f"got {type(document).__name__}."
            )
        return document

    def _write_document(self, path: Path, document: dict, describe: str) -> None:
        """Write *document* to *path* as YAML, creating the config dir.

        Parameters
        ----------
        path : Path
            The YAML file to write.
        document : dict
            The mapping to dump.  Each store's ``model_dump`` call stays
            with the store — the dump kwargs differ between them.
        describe : str
            Message subject for a write failure, e.g. ``"preset 'align'"``.

        Raises
        ------
        RuntimeError
            The store's :attr:`ERROR` on an OS-level write failure.
        """
        # A config dir, not a scans/ScanNNN/ data folder — parents=True is
        # explicitly fine here (see the module docstring).
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise self.ERROR(f"Could not write {describe}: {exc}") from exc


class NamedConfigStore(ExperimentConfigStore):
    """Base for stores keeping one YAML file per named config.

    Adds the file-per-name surface on top of :class:`ExperimentConfigStore`;
    subclasses also set :attr:`NOUN` and :attr:`LABEL`.
    """

    NOUN: str
    """The name-validation noun (``"<NOUN> name must not be empty."``)."""

    LABEL: str
    """The message label for not-found/delete texts (e.g. ``"Trigger profile"``)."""

    def _path(self, name: str) -> Path:
        """Return the YAML path for config *name*, validating the name.

        Parameters
        ----------
        name : str
            The config name (the file stem — no path separators).

        Returns
        -------
        Path
            ``<store folder>/<name>.yaml`` (an existing ``.yml`` twin is
            preferred when the ``.yaml`` spelling is absent).

        Raises
        ------
        RuntimeError
            The store's :attr:`ERROR` on an empty name or one that would
            escape the store's folder.
        """
        cleaned = name.strip()
        if not cleaned:
            raise self.ERROR(f"{self.NOUN} name must not be empty.")
        if any(sep in cleaned for sep in ("/", "\\")) or cleaned in (".", ".."):
            raise self.ERROR(
                f"{self.NOUN} name {name!r} must be a plain file name "
                "(no path separators)."
            )
        folder = self._folder_or_raise()
        path = folder / f"{cleaned}.yaml"
        if not path.exists():
            twin = folder / f"{cleaned}.yml"
            if twin.exists():
                return twin
        return path

    def _existing_path(self, name: str) -> Path:
        """Return the YAML path for config *name*, raising when absent.

        Parameters
        ----------
        name : str
            The config name (file stem).

        Returns
        -------
        Path
            The existing YAML file for *name*.

        Raises
        ------
        RuntimeError
            The store's :attr:`ERROR` on a bad name or a config that does
            not exist.
        """
        path = self._path(name)
        if not path.exists():
            raise self.ERROR(f"{self.LABEL} {name!r} not found ({path}).")
        return path

    def list_names(self) -> list[str]:
        """List the saved config names, sorted.

        Returns
        -------
        list of str
            YAML file stems in the store's folder — empty when the configs
            repo, experiment folder, or store folder is missing.

        Raises
        ------
        RuntimeError
            The store's :attr:`ERROR` when the experiment name would escape
            the experiments root; a merely *missing* folder is not an error.
        """
        folder = self._folder()
        if folder is None:
            return []
        return yaml_stems(folder)

    def delete(self, name: str) -> None:
        """Delete config *name*.

        Parameters
        ----------
        name : str
            The config name (file stem).

        Raises
        ------
        RuntimeError
            The store's :attr:`ERROR` on a missing configs repo /
            experiment, a bad name, a config that does not exist, or an
            OS-level delete failure.
        """
        path = self._existing_path(name)
        label = self.LABEL.lower()
        try:
            path.unlink()
        except OSError as exc:
            raise self.ERROR(f"Could not delete {label} {name!r}: {exc}") from exc
        self._logger.info("deleted %s %r (%s)", label, name, path)
