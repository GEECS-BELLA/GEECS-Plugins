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

Offline-safety mirrors :class:`~geecs_console.services.presets.PresetStore`
(the store pattern this module copies): listing degrades to empty with no
configs root; ``load``/``save``/``delete``/``rename`` raise
:class:`TriggerProfileStoreError` with a message fit for inline display.
Creating the ``shot_control_configurations/`` directory with
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

from geecs_console.services._experiment_name import check_experiment_name
from geecs_console.services.configs import TRIGGER_FOLDER, _configs_base
from geecs_schemas import TriggerProfile
from geecs_schemas.convert import SchemaConversionError, convert_shot_control

logger = logging.getLogger(__name__)


class TriggerProfileStoreError(RuntimeError):
    """A trigger-profile operation cannot be carried out (shown inline)."""


class TriggerProfileStore:
    """Load/save named trigger profiles for one experiment.

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
        """The experiment this store reads and writes trigger profiles for."""
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
        """Return the shot-control dir, or ``None`` offline/unselected.

        Raises
        ------
        TriggerProfileStoreError
            When the experiment name would escape the experiments root
            (issue #513) — checked before any path join.
        """
        root = self._experiments_root
        if root is None:
            root = _configs_base()
        if root is None or not self._experiment:
            return None
        check_experiment_name(self._experiment, TriggerProfileStoreError)
        return root / self._experiment / TRIGGER_FOLDER

    def _folder_or_raise(self) -> Path:
        """Return the shot-control dir, raising a clear error when unresolvable.

        Returns
        -------
        Path
            ``<experiments root>/<experiment>/shot_control_configurations``.

        Raises
        ------
        TriggerProfileStoreError
            When the configs repo is not found, no experiment is selected,
            or the experiment name would escape the experiments root.
        """
        root = self._experiments_root
        if root is None:
            root = _configs_base()
        if root is None:
            raise TriggerProfileStoreError(
                "Configs repo not found — set GEECS_SCANNER_CONFIG_DIR or "
                "config.ini [Paths] scanner_config_root_path."
            )
        if not self._experiment:
            raise TriggerProfileStoreError("No experiment selected.")
        check_experiment_name(self._experiment, TriggerProfileStoreError)
        return root / self._experiment / TRIGGER_FOLDER

    def _path(self, name: str) -> Path:
        """Return the YAML path for profile *name*, validating the name.

        Parameters
        ----------
        name : str
            The profile name (the file stem — no path separators).

        Returns
        -------
        Path
            ``<shot-control dir>/<name>.yaml`` (an existing ``.yml`` twin is
            preferred when the ``.yaml`` spelling is absent).

        Raises
        ------
        TriggerProfileStoreError
            On an empty name or one that would escape the shot-control dir.
        """
        cleaned = name.strip()
        if not cleaned:
            raise TriggerProfileStoreError("Profile name must not be empty.")
        if any(sep in cleaned for sep in ("/", "\\")) or cleaned in (".", ".."):
            raise TriggerProfileStoreError(
                f"Profile name {name!r} must be a plain file name (no path separators)."
            )
        folder = self._folder_or_raise()
        path = folder / f"{cleaned}.yaml"
        if not path.exists():
            twin = folder / f"{cleaned}.yml"
            if twin.exists():
                return twin
        return path

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
        path = self._path(name)
        if not path.exists():
            raise TriggerProfileStoreError(
                f"Trigger profile {name!r} not found ({path})."
            )
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TriggerProfileStoreError(
                f"Trigger profile {name!r} is not valid YAML ({path}): {exc}"
            ) from exc
        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise TriggerProfileStoreError(
                f"Trigger profile {name!r} ({path}) should be a YAML mapping, "
                f"got {type(document).__name__}."
            )
        return document, path

    # ------------------------------------------------------------------
    # The store surface
    # ------------------------------------------------------------------

    def list_names(self) -> list[str]:
        """List the saved profile names, sorted.

        Returns
        -------
        list of str
            YAML file stems in the shot-control dir — empty when the configs
            repo, experiment folder, or shot-control dir is missing.

        Raises
        ------
        TriggerProfileStoreError
            When the experiment name would escape the experiments root
            (issue #513); a merely *missing* folder is not an error.
        """
        folder = self._folder()
        if folder is None or not folder.is_dir():
            return []
        return sorted(
            path.stem for path in folder.iterdir() if path.suffix in (".yaml", ".yml")
        )

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
        # A config dir, not a scans/ScanNNN/ data folder — parents=True is
        # explicitly fine here (the scan-folder invariant does not apply).
        path.parent.mkdir(parents=True, exist_ok=True)
        document = yaml.safe_dump(
            profile.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
        try:
            path.write_text(document, encoding="utf-8")
        except OSError as exc:
            raise TriggerProfileStoreError(
                f"Could not write trigger profile {name!r}: {exc}"
            ) from exc
        logger.info("saved trigger profile %r to %s", name, path)
        return path

    def delete(self, name: str) -> None:
        """Delete profile *name*.

        Parameters
        ----------
        name : str
            The profile name (file stem).

        Raises
        ------
        TriggerProfileStoreError
            Missing configs repo / experiment, a bad name, a profile that
            does not exist, or an OS-level delete failure.
        """
        path = self._path(name)
        if not path.exists():
            raise TriggerProfileStoreError(
                f"Trigger profile {name!r} not found ({path})."
            )
        try:
            path.unlink()
        except OSError as exc:
            raise TriggerProfileStoreError(
                f"Could not delete trigger profile {name!r}: {exc}"
            ) from exc
        logger.info("deleted trigger profile %r (%s)", name, path)

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
        source = self._path(old)
        if not source.exists():
            raise TriggerProfileStoreError(
                f"Trigger profile {old!r} not found ({source})."
            )
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
