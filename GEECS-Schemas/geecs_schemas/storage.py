"""Where every registered document kind is kept: a file in the configs repo, or not a file.

:data:`KIND_STORAGE` holds one entry per :data:`geecs_schemas.SCHEMA_REGISTRY`
kind, so a registered kind is never silently missing from a tool that
lists the configs repo's documents. A :class:`FileStorage` names the root
(:data:`STORAGE_ROOT_PATHS`), the folder under it, the file pattern and
the suffixes its readers accept, and, where two kinds share a folder, the
loader that tells them apart; the tier is the model's own ``tier`` ClassVar, declared on the
schema and nowhere else. A :class:`NotAFile` says why the kind has no file
(a database row, a plan parameter, a sub-document, package data).

The folder and file names are the constants beside each kind's model; this
table only gathers them.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import ConfigDict, Field

from geecs_schemas._base import SchemaModel
from geecs_schemas.analysis.document import ANALYZERS_FOLDER
from geecs_schemas.analysis.group import GROUPS_FOLDER
from geecs_schemas.archive_policy import ARCHIVE_POLICY_FILENAME, ARCHIVER_CONFIG_FOLDER
from geecs_schemas.derived_channels import (
    DERIVED_CHANNELS_FILENAME,
    GATEWAY_CONFIG_FOLDER,
)
from geecs_schemas.epics import NATIVE_DEVICES_FILENAME
from geecs_schemas.epics.operations import OPERATOR_POLICY_FILENAME
from geecs_schemas.epics.panel_devices import PANEL_DEVICES_FILENAME
from geecs_schemas.experiment_defaults import EXPERIMENT_DEFAULTS_FILENAME
from geecs_schemas.home_layout import HOME_LAYOUT_FILENAME
from geecs_schemas.optimizer_config import OPTIMIZER_CONFIGS_FOLDER
from geecs_schemas.preset import PRESETS_FOLDER
from geecs_schemas.scan_variables import SCAN_VARIABLES_FILENAME, SCAN_VARIABLES_FOLDER
from geecs_schemas.screens import SCREENS_DIRNAME
from geecs_schemas.setup import SETUPS_FOLDER
from geecs_schemas.shot_offsets import SHOT_OFFSETS_FILENAME
from geecs_schemas.trigger_profile import TRIGGER_PROFILES_FOLDER

#: The configs repo's two document roots: ``experiment`` is one experiment's
#: scanner configs, ``analysis`` the analysis configs every experiment shares.
StorageRoot = Literal["experiment", "analysis"]

#: Each root's path inside the configs repo, the one statement of it
#: (GEECS-Core's resolver names the same ``scanner_configs/experiments``; this
#: package imports nothing in the repo). ``<experiment>`` is a placeholder.
STORAGE_ROOT_PATHS: dict[StorageRoot, str] = {
    "experiment": "scanner_configs/experiments/<experiment>",
    "analysis": "scan_analysis_configs",
}

#: The file pattern of a kind with one file per document, ``<name>`` its stem.
NAMED = "<name>.yaml"

#: The spellings a one-file-per-document reader accepts, written form first.
YAML_SUFFIXES = (".yaml", ".yml")

#: The loader that tells the analyzers tree's two kinds apart, by the
#: document's top-level ``kind`` (a ``.himg`` job has one, a recipe none).
ANALYSIS_DISPATCH = "geecs_schemas.analysis.load_analysis_document"


class FileStorage(SchemaModel):
    """A kind kept as files in the configs repo: ``<root>/<folder>/<pattern>``."""

    model_config = ConfigDict(frozen=True)

    storage: Literal["file"] = "file"
    root: StorageRoot = Field(description="Which of the configs repo's roots.")
    folder: str = Field(
        description="The folder under the root; empty for a file at its top."
    )
    pattern: str = Field(
        description="The file name, or '<name>.yaml' for one file per document."
    )
    suffixes: tuple[str, ...] = Field(
        default=(".yaml",),
        description=(
            "The suffixes its readers accept for '<name>' files, the written "
            "one first; a fixed file name is read as spelled."
        ),
    )
    nested: bool = Field(
        default=False,
        description="Files sit in subfolders below the folder (the analysis trees).",
    )
    dispatch: str | None = Field(
        default=None,
        description=(
            "When kinds share a folder: the loader (a dotted name) that reads a "
            "file as the kind it declares. A lister must go through it."
        ),
    )

    def path_pattern(self) -> str:
        """The files' path in the configs repo, ``**/`` marking nested subfolders.

        Returns
        -------
        str
            E.g. ``scanner_configs/experiments/<experiment>/presets/<name>.yaml``.
        """
        folder = f"{self.folder}/" if self.folder else ""
        nested = "**/" if self.nested else ""
        return f"{STORAGE_ROOT_PATHS[self.root]}/{folder}{nested}{self.pattern}"


class NotAFile(SchemaModel):
    """A registered kind that is not a file in the configs repo, and why."""

    model_config = ConfigDict(frozen=True)

    storage: Literal["not-a-file"] = "not-a-file"
    reason: str = Field(min_length=1, description="Why the kind has no file.")


KindStorage = Union[FileStorage, NotAFile]


def _experiment(
    folder: str, pattern: str = NAMED, suffixes: tuple[str, ...] = (".yaml",)
) -> FileStorage:
    return FileStorage(
        root="experiment", folder=folder, pattern=pattern, suffixes=suffixes
    )


def _analysis(folder: str, dispatch: str | None = None) -> FileStorage:
    return FileStorage(
        root="analysis",
        folder=folder,
        pattern=NAMED,
        suffixes=YAML_SUFFIXES,
        nested=True,
        dispatch=dispatch,
    )


#: Every SCHEMA_REGISTRY kind, exactly once (pinned by tests/test_storage.py).
KIND_STORAGE: dict[str, KindStorage] = {
    "optimizer_config": _experiment(OPTIMIZER_CONFIGS_FOLDER, suffixes=YAML_SUFFIXES),
    "preset": _experiment(PRESETS_FOLDER, suffixes=YAML_SUFFIXES),
    "scan_variables": _experiment(SCAN_VARIABLES_FOLDER, SCAN_VARIABLES_FILENAME),
    "trigger_profile": _experiment(TRIGGER_PROFILES_FOLDER, suffixes=YAML_SUFFIXES),
    "experiment_defaults": _experiment("", EXPERIMENT_DEFAULTS_FILENAME),
    "derived_channels": _experiment(GATEWAY_CONFIG_FOLDER, DERIVED_CHANNELS_FILENAME),
    "archive_policy": _experiment(ARCHIVER_CONFIG_FOLDER, ARCHIVE_POLICY_FILENAME),
    "shot_offsets": _experiment("", SHOT_OFFSETS_FILENAME),
    "native_devices": _experiment("", NATIVE_DEVICES_FILENAME),
    "panel_devices": _experiment("", PANEL_DEVICES_FILENAME),
    "operator_policy": _experiment("", OPERATOR_POLICY_FILENAME),
    "screen": _experiment(SCREENS_DIRNAME),
    "home_layout": _experiment("", HOME_LAYOUT_FILENAME),
    "setup": _experiment(SETUPS_FOLDER, suffixes=YAML_SUFFIXES),
    "analysis_recipe": _analysis(ANALYZERS_FOLDER, ANALYSIS_DISPATCH),
    "himg_job": _analysis(ANALYZERS_FOLDER, ANALYSIS_DISPATCH),
    "analysis_group": _analysis(GROUPS_FOLDER),
    "scan_request": NotAFile(
        reason="a plan parameter: built per submit from a preset, never saved"
    ),
    "journal_entry": NotAFile(
        reason="a database row: the device module's record of a write attempt"
    ),
    "log_entry": NotAFile(reason="a database row: the logbook's store"),
    "family_file": NotAFile(
        reason=(
            "package data (geecs_schemas/epics/families/): shown read-only as "
            "'code', changed by a PR to GEECS-Plugins"
        )
    ),
}
