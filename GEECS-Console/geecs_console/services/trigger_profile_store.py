"""Trigger-profile persistence over the per-experiment shot-control dir.

A trigger profile GATES real hardware (the DG645 that fires the machine
trigger, the gas jet it drives) at scan time, so the load/save round-trip
must be lossless: :class:`TriggerProfileStore` keeps one YAML file per
profile (``<name>.yaml``, a plain ``model_dump(mode="json")`` document that
round-trips through ``TriggerProfile.model_validate``) in the same folder
``ConfigsRepoResolver`` reads::

    scanner_configs/experiments/<Experiment>/shot_control_configurations/<name>.yaml

Legacy files (the pre-schema single-device dialect, no ``schema_version``
key) are loaded through :func:`geecs_schemas.convert.convert_shot_control` —
exactly what the resolver does — so the existing corpus opens in the editor;
saving such a profile migrates the file to the versioned schema.

Offline-safety mirrors :class:`~geecs_console.services.presets.PresetStore`:
listing degrades to empty with no configs root; ``load``/``save``/``delete``/
``rename`` raise :class:`TriggerProfileStoreError` with a message fit for
inline display.  Directory creation and the scan-folder-invariant rationale
live in :mod:`geecs_console.services.config_store`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from geecs_console.services.config_store import NamedConfigStore

# The base resolves the configs root through this module's namespace, so
# tests can monkeypatch ``trigger_profile_store._configs_base``.
from geecs_console.services.configs import TRIGGER_FOLDER, _configs_base  # noqa: F401
from geecs_schemas import TriggerProfile
from geecs_schemas.convert import SchemaConversionError, convert_shot_control

logger = logging.getLogger(__name__)


class TriggerProfileStoreError(RuntimeError):
    """A trigger-profile operation cannot be carried out (shown inline)."""


class TriggerProfileStore(NamedConfigStore):
    """Load/save named trigger profiles for one experiment."""

    FOLDER = TRIGGER_FOLDER
    NOUN = "Profile"
    LABEL = "Trigger profile"
    ERROR = TriggerProfileStoreError

    def _read_document(self, name: str) -> tuple[dict, Path]:
        """Read profile *name*'s raw YAML mapping (shared by load/is_legacy).

        Parameters
        ----------
        name : str
            The profile name (file stem).

        Returns
        -------
        tuple of (dict, Path)
            The parsed top-level mapping and the file it came from.

        Raises
        ------
        TriggerProfileStoreError
            Missing configs repo / experiment / file, unparsable YAML, or a
            document that is not a mapping.
        """
        path = self._existing_path(name)
        document = self._read_mapping(
            path, describe=f"Trigger profile {name!r}", empty_as={}
        )
        return document, path

    def exists(self, name: str) -> bool:
        """Whether a profile file called *name* is on disk.

        Parameters
        ----------
        name : str
            The profile name (file stem).

        Returns
        -------
        bool
            ``True`` when either YAML spelling exists (``False`` offline).
        """
        try:
            return self._path(name).exists()
        except TriggerProfileStoreError:
            return False

    def is_legacy(self, name: str) -> bool:
        """Whether profile *name* is a legacy (pre-schema) file on disk.

        Parameters
        ----------
        name : str
            The profile name (file stem).

        Returns
        -------
        bool
            ``True`` when the file carries no top-level ``schema_version`` —
            saving it from the editor migrates it to the versioned schema.

        Raises
        ------
        TriggerProfileStoreError
            Missing configs repo / experiment / file, or unparsable YAML.
        """
        document, _ = self._read_document(name)
        return "schema_version" not in document

    def load(self, name: str) -> TriggerProfile:
        """Load profile *name* as a validated :class:`TriggerProfile`.

        New-schema files (top-level ``schema_version``) validate directly;
        legacy shot-control files go through the same converter the resolver
        uses, so the editor opens the existing corpus unchanged.

        Parameters
        ----------
        name : str
            The profile name (file stem).

        Returns
        -------
        TriggerProfile
            The schema-validated profile the file stores.

        Raises
        ------
        TriggerProfileStoreError
            Missing configs repo / experiment / file, unparsable YAML, a
            document the schema rejects, or an empty/deviceless legacy file
            — always with a message fit for inline display.
        """
        document, path = self._read_document(name)
        if "schema_version" in document:
            try:
                return TriggerProfile.model_validate(document)
            except ValidationError as exc:
                raise TriggerProfileStoreError(
                    f"Trigger profile {name!r} ({path}) is not a valid "
                    f"TriggerProfile: {exc}"
                ) from exc
        try:
            profile = convert_shot_control(document, name=name)
        except (SchemaConversionError, ValidationError) as exc:
            raise TriggerProfileStoreError(
                f"Trigger profile {name!r} ({path}) could not be converted "
                f"from the legacy shot-control format: {exc}"
            ) from exc
        if profile is None:
            raise TriggerProfileStoreError(
                f"Trigger profile {name!r} ({path}) is empty / names no "
                "device — there is nothing to edit."
            )
        return profile

    def save(self, name: str, profile: TriggerProfile) -> Path:
        """Write *profile* as profile *name* (overwriting an existing one).

        The document is a plain ``model_dump(mode="json")`` in declared field
        order — every schema field (``schema_version``, ``name``, ``states``,
        ``variants``, ``description``) is preserved, and loading it back
        yields a model equal to *profile*.

        Parameters
        ----------
        name : str
            The profile name (file stem).
        profile : TriggerProfile
            The profile to persist.

        Returns
        -------
        Path
            The YAML file written.

        Raises
        ------
        TriggerProfileStoreError
            Missing configs repo, no experiment selected, a bad name, or an
            OS-level write failure.
        """
        path = self._path(name)
        self._write_document(
            path, profile.model_dump(mode="json"), f"trigger profile {name!r}"
        )
        logger.info("saved trigger profile %r to %s", name, path)
        return path

    def rename(self, old: str, new: str) -> Path:
        """Rename profile *old* to *new*, refusing to overwrite.

        A new-schema file also gets its stored ``name`` field updated (the
        rest of the document is carried over verbatim — no model round-trip,
        so nothing else in the file changes).  A legacy file has no name
        field and is renamed as-is.

        Parameters
        ----------
        old : str
            The current profile name (file stem).
        new : str
            The new profile name (file stem).

        Returns
        -------
        Path
            The renamed YAML file.

        Raises
        ------
        TriggerProfileStoreError
            Missing configs repo / experiment, a bad name, a source that
            does not exist, a target that already exists, or an OS-level
            failure.
        """
        source = self._existing_path(old)
        target = self._folder_or_raise() / f"{new.strip()}.yaml"
        # Validate the new name (empty / path separators) with _path, but
        # target the .yaml spelling regardless of any .yml twin.
        self._path(new)
        if target.exists() or target.with_suffix(".yml").exists():
            raise TriggerProfileStoreError(
                f"Trigger profile {new!r} already exists — delete it first "
                "or pick another name."
            )
        document, _ = self._read_document(old)
        try:
            if "schema_version" in document and "name" in document:
                document["name"] = new.strip()
                target.write_text(
                    yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                source.unlink()
            else:
                source.rename(target)
        except OSError as exc:
            raise TriggerProfileStoreError(
                f"Could not rename trigger profile {old!r} to {new!r}: {exc}"
            ) from exc
        logger.info("renamed trigger profile %r to %r (%s)", old, new, target)
        return target
