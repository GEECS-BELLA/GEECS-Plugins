"""
Pydantic models for the Analysis Provenance Standard v0.1.

These models match the JSON schema defined in:
standards/analysis-provenance/schema/v0.1/provenance.schema.json
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Software(BaseModel):
    """Software identification."""

    name: str = Field(..., description="Name of the software, tool, or script")
    version: str | None = Field(None, description="Version string")


class CodeVersion(BaseModel):
    """Source code identification for reproducibility."""

    repository: str | None = Field(
        None, description="Repository URL (e.g., GitHub URL)"
    )
    commit: str | None = Field(None, description="Git commit SHA (full or abbreviated)")
    branch: str | None = Field(None, description="Branch name")
    dirty: bool | None = Field(
        None, description="True if there were uncommitted changes"
    )


class AnalysisEntry(BaseModel):
    """A single analysis operation that wrote to the data file."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO 8601 datetime when analysis was performed",
    )
    columns_written: list[str] = Field(
        ..., min_length=1, description="Column names that were added or modified"
    )
    software: Software | None = Field(None, description="Software identification")
    code_version: CodeVersion | None = Field(
        None, description="Source code identification"
    )
    dependencies: dict[str, str] | None = Field(
        None, description="Key package versions as {name: version}"
    )
    config: dict[str, Any] | None = Field(
        None, description="Configuration used for analysis"
    )
    config_ref: str | None = Field(
        None, description="Path to external configuration file"
    )
    notes: str | None = Field(None, description="Human-readable context or comments")
    user: str | None = Field(
        None, description="Username or identifier of who ran the analysis"
    )

    model_config = {
        "extra": "allow"
    }  # Allow additional fields for forward compatibility


class ProvenanceFile(BaseModel):
    """Root object for a provenance file."""

    schema_version: str = Field(
        default="0.1",
        pattern=r"^0\.1$",
        description="Version of the provenance specification",
    )
    analyses: list[AnalysisEntry] = Field(
        default_factory=list,
        description="List of analysis entries, ordered chronologically",
    )

    def append_entry(self, entry: AnalysisEntry) -> None:
        """Append a new analysis entry (append-only per spec)."""
        self.analyses.append(entry)

    def get_column_provenance(self, column_name: str) -> AnalysisEntry | None:
        """Get the most recent entry that wrote to a specific column."""
        for entry in reversed(self.analyses):
            if column_name in entry.columns_written:
                return entry
        return None

    def get_columns_without_provenance(self, all_columns: list[str]) -> list[str]:
        """Return columns that don't appear in any provenance entry."""
        tracked_columns = set()
        for entry in self.analyses:
            tracked_columns.update(entry.columns_written)
        return [col for col in all_columns if col not in tracked_columns]
