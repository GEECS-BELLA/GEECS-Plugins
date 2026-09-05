"""The analysis group document: the list of diagnostics that run together after a scan.

A group is what LiveWatch and the task queue schedule.  It names
diagnostics by their file stems; an entry may be disabled in place or
given a group-specific priority.  Format version 1 — the pre-0.19.0 group
YAML is identical apart from the ``schema_version`` stamp, which is added
automatically.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import Field, field_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


class AnalyzerRef(SchemaModel):
    """One diagnostic in a group, optionally disabled or re-prioritised for this group."""

    ref: str = Field(
        ..., min_length=1, description="The diagnostic's ID (its YAML file stem)."
    )
    enabled: bool = Field(
        True,
        description="False keeps the entry listed but skips it when the group runs.",
    )
    priority: Optional[int] = Field(
        None,
        ge=0,
        description="Run order within this group; unset uses the diagnostic's own scan.priority.",
    )


class AnalysisGroup(VersionedSchemaModel):
    """A named set of diagnostics to run after each scan, in priority order.

    Entries may be written as bare diagnostic IDs or as ``{ref, enabled,
    priority}`` mappings; bare IDs are stored as enabled entries with no
    priority override.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Display name, conventionally <facility>_<purpose>.",
    )
    description: Optional[str] = Field(
        None, description="Free-text note about when this group is used."
    )
    upload_to_scanlog: bool = Field(
        True,
        description="Upload the group's summary figures to the experiment scan log.",
    )
    analyzers: List[AnalyzerRef] = Field(
        default_factory=list,
        description="The diagnostics to run; a bare ID means enabled with the diagnostic's own priority.",
    )

    @field_validator("analyzers", mode="before")
    @classmethod
    def _expand_bare_ids(cls, value: Any) -> Any:
        """Accept bare diagnostic IDs alongside full entries."""
        if not isinstance(value, list):
            return value
        return [{"ref": entry} if isinstance(entry, str) else entry for entry in value]


__all__ = ["AnalysisGroup", "AnalyzerRef"]
