"""Where every registered document kind is kept: a file in the configs repo, or not a file.

:data:`KIND_STORAGE` holds one entry per :data:`geecs_schemas.SCHEMA_REGISTRY`
kind, so a registered kind is never silently missing from a tool that
lists the configs repo's documents. A :class:`FileStorage` names the root
(``experiment``: ``scanner_configs/experiments/<experiment>/``;
``analysis``: ``scan_analysis_configs/``), the folder under it and the
file pattern; the tier is the model's own ``tier`` ClassVar, declared on the
schema and nowhere else. A :class:`NotAFile` says why the kind has no file
(a database row, a plan parameter, a sub-document, package data).

The folder and file names are the constants beside each kind's model; this
table only gathers them.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import ConfigDict, Field

from geecs_schemas._base import SchemaModel
from geecs_schemas.action_plan import ACTION_LIBRARY_FILENAME, ACTION_LIBRARY_FOLDER
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

#: The configs repo's two document roots: ``experiment`` is
#: ``scanner_configs/experiments/<experiment>/``, ``analysis`` is
#: ``scan_analysis_configs/`` (shared by every experiment).
StorageRoot = Literal["experiment", "analysis"]

#: The file pattern of a kind with one file per document, ``<name>`` its stem.
NAMED = "<name>.yaml"


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
    nested: bool = Field(
        default=False,
        description="Files sit in subfolders below the folder (the analysis trees).",
    )


class NotAFile(SchemaModel):
    """A registered kind that is not a file in the configs repo, and why."""

    model_config = ConfigDict(frozen=True)

    storage: Literal["not-a-file"] = "not-a-file"
    reason: str = Field(min_length=1, description="Why the kind has no file.")


KindStorage = Union[FileStorage, NotAFile]


def _experiment(folder: str, pattern: str = NAMED) -> FileStorage:
    return FileStorage(root="experiment", folder=folder, pattern=pattern)


#: Every SCHEMA_REGISTRY kind, exactly once (pinned by tests/test_storage.py).
KIND_STORAGE: dict[str, KindStorage] = {
    "optimizer_config": _experiment(OPTIMIZER_CONFIGS_FOLDER),
    "preset": _experiment(PRESETS_FOLDER),
    "scan_variables": _experiment(SCAN_VARIABLES_FOLDER, SCAN_VARIABLES_FILENAME),
    "trigger_profile": _experiment(TRIGGER_PROFILES_FOLDER),
    "action_plan_library": _experiment(ACTION_LIBRARY_FOLDER, ACTION_LIBRARY_FILENAME),
    "experiment_defaults": _experiment("", EXPERIMENT_DEFAULTS_FILENAME),
    "derived_channels": _experiment(GATEWAY_CONFIG_FOLDER, DERIVED_CHANNELS_FILENAME),
    "archive_policy": _experiment(ARCHIVER_CONFIG_FOLDER, ARCHIVE_POLICY_FILENAME),
    "shot_offsets": _experiment("", SHOT_OFFSETS_FILENAME),
    "native_devices": _experiment("", NATIVE_DEVICES_FILENAME),
    "panel_devices": _experiment("", PANEL_DEVICES_FILENAME),
    "operator_policy": _experiment("", OPERATOR_POLICY_FILENAME),
    "screen": _experiment(SCREENS_DIRNAME),
    "home_layout": _experiment("", HOME_LAYOUT_FILENAME),
    "setup": _experiment(SETUPS_FOLDER),
    "analysis_recipe": FileStorage(
        root="analysis", folder=ANALYZERS_FOLDER, pattern=NAMED, nested=True
    ),
    "himg_job": FileStorage(
        root="analysis", folder=ANALYZERS_FOLDER, pattern=NAMED, nested=True
    ),
    "analysis_group": FileStorage(
        root="analysis", folder=GROUPS_FOLDER, pattern=NAMED, nested=True
    ),
    "scan_request": NotAFile(
        reason="a plan parameter: built per submit from a preset, never saved"
    ),
    "action_plan": NotAFile(
        reason="a sub-document: one plan inside the action library's file"
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
