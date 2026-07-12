"""Save-set persistence: named :class:`~geecs_schemas.SaveSet` YAML files.

:class:`SaveSetStore` keeps one YAML file per save set (``<name>.yaml``, a
plain ``model_dump(mode="json")`` document that round-trips through
``SaveSet.model_validate``) in the per-experiment configs area — the same
``save_devices/`` folder ``ConfigsRepoResolver`` reads::

    scanner_configs/experiments/<Experiment>/save_devices/<name>.yaml

Loading mirrors the resolver's dispatch: a document carrying
``schema_version`` is validated directly; a legacy save element
(``Devices:`` mapping) is converted through
:func:`geecs_schemas.convert.convert_save_element` — *except* when the
conversion would extract setup/closeout action plans out of the file.  Those
plan bodies live only inside the legacy document, so editing-and-saving such
an element here would silently drop them (the new-schema entries keep only
name *references*).  The store refuses to load those with a clear message
instead of losing data.

Offline-safety mirrors :class:`~geecs_console.services.presets.PresetStore`:
listing degrades to empty with no configs root; ``load``/``save``/``delete``/
``rename`` raise :class:`SaveSetStoreError` with a status-bar-ready message.
Creating the ``save_devices/`` directory with ``mkdir(parents=True,
exist_ok=True)`` is deliberate and fine — this is a config directory in the
configs repo, not a ``scans/ScanNNN/`` data folder, so the repo's scan-folder
creation invariant does not apply.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from geecs_console.services.configs import SAVE_SET_FOLDER, _configs_base
from geecs_schemas import SaveSet

logger = logging.getLogger(__name__)


class SaveSetStoreError(RuntimeError):
    """A save-set operation cannot be carried out (surfaced in the status bar)."""


class SaveSetStore:
    """Load/save named save sets for one experiment.

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
        """The experiment this store reads and writes save sets for."""
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
    # Folder resolution (mirrors PresetStore)
    # ------------------------------------------------------------------

    def _root(self) -> Optional[Path]:
        """Return the experiments root, or ``None`` offline."""
        if self._experiments_root is not None:
            return self._experiments_root
        return _configs_base()

    def _folder(self) -> Optional[Path]:
        """Return the experiment's save-set dir, or ``None`` offline/unselected."""
        root = self._root()
        if root is None or not self._experiment:
            return None
        return root / self._experiment / SAVE_SET_FOLDER

    def _folder_or_raise(self) -> Path:
        """Return the save-set dir, raising a clear error when unresolvable.

        Returns
        -------
        Path
            ``<experiments root>/<experiment>/save_devices``.

        Raises
        ------
        SaveSetStoreError
            When the configs repo is not found or no experiment is selected.
        """
        root = self._root()
        if root is None:
            raise SaveSetStoreError(
                "Configs repo not found — set GEECS_SCANNER_CONFIG_DIR or "
                "config.ini [Paths] scanner_config_root_path."
            )
        if not self._experiment:
            raise SaveSetStoreError("No experiment selected.")
        return root / self._experiment / SAVE_SET_FOLDER

    def _path(self, name: str) -> Path:
        """Return the YAML path for save set *name*, validating the name.

        Parameters
        ----------
        name : str
            The save-set name (the file stem — no path separators).

        Returns
        -------
        Path
            ``<save_devices dir>/<name>.yaml`` (an existing ``.yml`` twin is
            preferred when the ``.yaml`` spelling is absent).

        Raises
        ------
        SaveSetStoreError
            On an empty name or one that would escape the save-set dir.
        """
        cleaned = name.strip()
        if not cleaned:
            raise SaveSetStoreError("Save set name must not be empty.")
        if any(sep in cleaned for sep in ("/", "\\")) or cleaned in (".", ".."):
            raise SaveSetStoreError(
                f"Save set name {name!r} must be a plain file name "
                "(no path separators)."
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
        """List the saved save-set names, sorted; never raises.

        Returns
        -------
        list of str
            YAML file stems in the ``save_devices`` dir — empty when the
            configs repo, experiment folder, or save-set dir is missing.
        """
        folder = self._folder()
        if folder is None or not folder.is_dir():
            return []
        return sorted(
            path.stem for path in folder.iterdir() if path.suffix in (".yaml", ".yml")
        )

    def _read_document(self, name: str, path: Path) -> dict:
        """Read one YAML mapping, with status-bar-ready errors.

        Parameters
        ----------
        name : str
            The save-set name (for messages).
        path : Path
            The YAML file to read.

        Returns
        -------
        dict
            The parsed top-level mapping.

        Raises
        ------
        SaveSetStoreError
            A missing file, unparsable YAML, or a non-mapping document.
        """
        if not path.exists():
            raise SaveSetStoreError(f"Save set {name!r} not found ({path}).")
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SaveSetStoreError(
                f"Save set {name!r} is not valid YAML ({path}): {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise SaveSetStoreError(
                f"Save set {name!r} ({path}) should be a YAML mapping of "
                f"SaveSet fields, got {type(document).__name__}."
            )
        return document

    def load(self, name: str) -> SaveSet:
        """Load save set *name* as a validated :class:`SaveSet`.

        New-schema documents (carrying ``schema_version``) validate directly;
        legacy save elements are converted — unless the conversion would
        extract inline setup/closeout action plans, which saving back would
        silently drop (see the module docstring).

        Parameters
        ----------
        name : str
            The save-set name (file stem).

        Returns
        -------
        SaveSet
            The schema-validated save set.

        Raises
        ------
        SaveSetStoreError
            Missing configs repo / experiment / file, unparsable YAML, a
            document the schema rejects, or a legacy element this editor
            cannot round-trip without losing its inline actions — always
            with a message fit for the status bar.
        """
        path = self._path(name)
        document = self._read_document(name, path)
        if "schema_version" in document:
            try:
                return SaveSet.model_validate(document)
            except ValidationError as exc:
                raise SaveSetStoreError(
                    f"Save set {name!r} ({path}) is not a valid SaveSet: {exc}"
                ) from exc
        return self._load_legacy(name, path, document)

    def _load_legacy(self, name: str, path: Path, document: dict) -> SaveSet:
        """Convert a legacy save element, refusing lossy conversions.

        Parameters
        ----------
        name : str
            The save-set name (file stem).
        path : Path
            The legacy YAML file (for messages).
        document : dict
            The parsed legacy mapping (``Devices:`` dialect).

        Returns
        -------
        SaveSet
            The converted save set.

        Raises
        ------
        SaveSetStoreError
            A document the converter rejects, an action-only element, or an
            element whose inline setup/closeout actions would be lost by a
            save from this editor.
        """
        from geecs_schemas.convert import SchemaConversionError, convert_save_element

        try:
            result = convert_save_element(document, name=path.stem)
        except SchemaConversionError as exc:
            raise SaveSetStoreError(
                f"Save set {name!r} ({path}) is not a valid legacy save element: {exc}"
            ) from exc
        for note in result.notes:
            logger.info("save set %s (converted from legacy): %s", path.stem, note)
        if result.save_set is None:
            raise SaveSetStoreError(
                f"Save set {name!r} ({path}) is an action-only legacy "
                "element — it lists no devices to record."
            )
        if result.actions:
            # The plan bodies exist only inside the legacy file; the
            # converted entries carry name references.  Saving from this
            # editor would rewrite the file without the bodies — refuse
            # rather than lose the operator's rituals.
            plans = ", ".join(sorted(result.actions))
            raise SaveSetStoreError(
                f"Save set {name!r} ({path}) is a legacy element with inline "
                f"setup/closeout actions ({plans}) — migrate those to the "
                "experiment's action library before editing it here."
            )
        return result.save_set

    def save(self, name: str, save_set: SaveSet) -> Path:
        """Write *save_set* as *name* (new schema, overwriting an existing file).

        Parameters
        ----------
        name : str
            The save-set name (file stem).
        save_set : SaveSet
            The save set to persist.

        Returns
        -------
        Path
            The YAML file written.

        Raises
        ------
        SaveSetStoreError
            Missing configs repo, no experiment selected, a bad name, or an
            OS-level write failure.
        """
        path = self._path(name)
        # A config dir, not a scans/ScanNNN/ data folder — parents=True is
        # explicitly fine here (the scan-folder invariant does not apply).
        path.parent.mkdir(parents=True, exist_ok=True)
        document = yaml.safe_dump(
            save_set.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
        try:
            path.write_text(document, encoding="utf-8")
        except OSError as exc:
            raise SaveSetStoreError(
                f"Could not write save set {name!r}: {exc}"
            ) from exc
        logger.info("saved save set %r to %s", name, path)
        return path

    def delete(self, name: str) -> None:
        """Delete save set *name*.

        Parameters
        ----------
        name : str
            The save-set name (file stem).

        Raises
        ------
        SaveSetStoreError
            Missing configs repo / experiment, a bad name, a save set that
            does not exist, or an OS-level delete failure.
        """
        path = self._path(name)
        if not path.exists():
            raise SaveSetStoreError(f"Save set {name!r} not found ({path}).")
        try:
            path.unlink()
        except OSError as exc:
            raise SaveSetStoreError(
                f"Could not delete save set {name!r}: {exc}"
            ) from exc
        logger.info("deleted save set %r (%s)", name, path)

    def rename(self, old: str, new: str) -> Path:
        """Rename save set *old* to *new*, keeping the document's ``name`` in step.

        A new-schema document's ``name`` field is rewritten to *new* so the
        file stem and the in-file name never drift apart; a legacy document
        (no ``name`` field) moves untouched.

        Parameters
        ----------
        old : str
            The current save-set name (file stem).
        new : str
            The new name.  Must not collide with an existing save set.

        Returns
        -------
        Path
            The renamed YAML file.

        Raises
        ------
        SaveSetStoreError
            Missing configs repo / experiment, a bad name, a source that
            does not exist, a target that already exists, or an OS-level
            failure.
        """
        source = self._path(old)
        if not source.exists():
            raise SaveSetStoreError(f"Save set {old!r} not found ({source}).")
        target = self._path(new)
        if target.exists():
            raise SaveSetStoreError(
                f"Cannot rename {old!r} to {new!r}: a save set with that "
                "name already exists."
            )
        document = self._read_document(old, source)
        if "name" in document:
            document["name"] = new.strip()
        text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
        try:
            target.write_text(text, encoding="utf-8")
            source.unlink()
        except OSError as exc:
            raise SaveSetStoreError(
                f"Could not rename save set {old!r} to {new!r}: {exc}"
            ) from exc
        logger.info("renamed save set %r to %r (%s)", old, new, target)
        return target
