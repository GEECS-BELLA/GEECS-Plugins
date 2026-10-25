"""The experiment's composed panels: ``<experiment>/panels/*.yaml`` in the configs repository.

``GET /api/panels`` serves them (``Planning/epics_estate.md`` §2.4): each
a GEECS-Schemas ``ComposedPanel``, validated against the bridge's own
device → family map (the registry joined with the panel data), so a panel
naming a device the bridge does not serve, or a role its family lacks, is
reported rather than drawn half-working. ``panels/overrides/<family>.yaml``
(``PanelOverride``: hide, relabel, reorder a family's page) come with them.
Read on every request: a configs pull is live without a restart, as for
every other config the share's checkout holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import yaml
from geecs_core.configs_repo import scanner_configs_base
from geecs_schemas.epics.family import FamilyFile
from geecs_schemas.epics.panels import ComposedPanel, PanelOverride
from pydantic import BaseModel, ConfigDict, Field

#: The folder of composed panels in the experiment's configs folder.
PANELS_FOLDER = "panels"
#: Its sub-folder of family page overrides.
OVERRIDES_FOLDER = "overrides"


class PanelsReply(BaseModel):
    """``GET /api/panels``: the composed panels and family overrides that validate, and why the others do not."""

    model_config = ConfigDict(extra="forbid")

    panels: dict[str, ComposedPanel] = Field(
        default_factory=dict, description="File stem → the panel."
    )
    overrides: dict[str, PanelOverride] = Field(
        default_factory=dict,
        description="Family → its page's override for this experiment.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Files that could not be read or do not fit the devices served.",
    )


def panels_dir(experiment: str, base: Optional[Path] = None) -> Optional[Path]:
    """Where the experiment's composed panels would be; ``None`` without a configs repository."""
    base = base if base is not None else scanner_configs_base()
    return None if base is None else base / experiment / PANELS_FOLDER


def load_panels(
    folder: Optional[Path],
    devices: Mapping[str, str],
    families: Mapping[str, FamilyFile],
) -> PanelsReply:
    """Read and validate every panel and override in *folder* (an absent folder: none).

    *devices* is the bridge's device → family template map: a panel's
    device widgets, components and trend lines are checked against it.
    """
    reply = PanelsReply()
    if folder is None or not folder.is_dir():
        return reply
    context = {"devices": dict(devices), "families": families}
    for path in sorted(folder.glob("*.yaml")):
        document = _read(path, reply.errors)
        if document is None:
            continue
        try:
            reply.panels[path.stem] = ComposedPanel.model_validate(
                document, context=context
            )
        except ValueError as exc:
            reply.errors.append(f"{path.name}: {exc}")
    for path in sorted((folder / OVERRIDES_FOLDER).glob("*.yaml")):
        document = _read(path, reply.errors)
        if document is None:
            continue
        try:
            override = PanelOverride.model_validate(
                document, context={"families": families}
            )
        except ValueError as exc:
            reply.errors.append(f"{OVERRIDES_FOLDER}/{path.name}: {exc}")
            continue
        if override.family != path.stem:
            reply.errors.append(
                f"{OVERRIDES_FOLDER}/{path.name}: overrides {override.family!r}, "
                f"not the family its name says"
            )
            continue
        reply.overrides[override.family] = override
    return reply


def _read(path: Path, errors: list[str]) -> Optional[object]:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"{path.name}: {exc}")
        return None
