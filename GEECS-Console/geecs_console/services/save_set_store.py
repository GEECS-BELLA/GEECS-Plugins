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
Directory creation and the scan-folder-invariant rationale live in
:mod:`geecs_console.services.config_store`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from geecs_console.services.config_store import NamedConfigStore

# The base resolves the configs root through this module's namespace, so
# tests can monkeypatch ``save_set_store._configs_base``.
from geecs_console.services.configs import SAVE_SET_FOLDER, _configs_base  # noqa: F401
from geecs_schemas import SaveSet

logger = logging.getLogger(__name__)


class SaveSetStoreError(RuntimeError):
    """A save-set operation cannot be carried out (surfaced in the status bar)."""


class SaveSetStore(NamedConfigStore):
    """Load/save named save sets for one experiment."""

    FOLDER = SAVE_SET_FOLDER
    NOUN = "Save set"
    LABEL = "Save set"
    ERROR = SaveSetStoreError

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
        return self._read_mapping(
            path, describe=f"Save set {name!r}", mapping_hint=" of SaveSet fields"
        )

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
        self._write_document(
            path, save_set.model_dump(mode="json"), f"save set {name!r}"
        )
        logger.info("saved save set %r to %s", name, path)
        return path

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
        source = self._existing_path(old)
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
