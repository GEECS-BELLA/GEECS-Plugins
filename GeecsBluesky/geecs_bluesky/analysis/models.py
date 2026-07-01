"""Models for post-run analysis artifacts linked to Bluesky raw runs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ScalarValue = int | float | str | bool | None


class AnalysisStatus(str, Enum):
    """Per-input analysis status."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class AnalysisScope(str, Enum):
    """Execution scope required by an analyzer."""

    EVENT = "event"
    SCAN = "scan"


class CodeVersion(BaseModel):
    """Source code identity used to produce an analysis artifact."""

    repository: str | None = None
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None


class SoftwareEnvironment(BaseModel):
    """Runtime environment relevant for reproducing analysis."""

    python_version: str | None = None
    platform: str | None = None
    packages: dict[str, str] = Field(default_factory=dict)


class InputAssetRef(BaseModel):
    """Reference to one raw input asset or event consumed by analysis."""

    raw_run_uid: str
    event_uid: str | None = None
    scan_number: int | None = None
    scan_event_index: int | None = None
    shot_number: int | None = None
    device: str
    device_type: str | None = None
    data_key: str
    event_field: str | None = None
    datum_id: str | None = None
    resource_uid: str | None = None
    asset_spec: str | None = None
    payload_kind: str | None = None
    loader_name: str | None = None
    loader_kind: str | None = None
    resource_root: str | None = None
    resource_path: str | None = None


class DerivedAssetRef(BaseModel):
    """Reference to one file created by post-run analysis."""

    asset_id: str
    asset_spec: str
    relative_path: str
    source_datum_id: str | None = None
    source_event_uid: str | None = None
    description: str | None = None

    @field_validator("relative_path")
    @classmethod
    def relative_path_must_be_relative(cls, value: str) -> str:
        """Reject absolute paths in portable metadata."""
        if Path(value).is_absolute():
            raise ValueError(
                "relative_path must be relative to the invocation directory"
            )
        return value


class FeatureRow(BaseModel):
    """One row of per-shot or per-input analysis output."""

    model_config = ConfigDict(extra="forbid")

    raw_run_uid: str
    event_uid: str | None = None
    scan_number: int | None = None
    scan_event_index: int | None = None
    shot_number: int | None = None
    device: str
    data_key: str
    datum_id: str | None = None
    asset_spec: str | None = None
    analyzer_id: str
    status: AnalysisStatus = AnalysisStatus.SUCCESS
    error_message: str | None = None
    elapsed_s: float | None = None
    features: dict[str, ScalarValue] = Field(default_factory=dict)
    derived_assets: dict[str, str] = Field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, ScalarValue]:
        """Flatten required columns, feature scalars, and derived asset columns."""
        base: dict[str, ScalarValue] = {
            "raw_run_uid": self.raw_run_uid,
            "event_uid": self.event_uid,
            "scan_number": self.scan_number,
            "scan_event_index": self.scan_event_index,
            "shot_number": self.shot_number,
            "device": self.device,
            "data_key": self.data_key,
            "datum_id": self.datum_id,
            "asset_spec": self.asset_spec,
            "analyzer_id": self.analyzer_id,
            "status": self.status.value,
            "error_message": self.error_message,
            "elapsed_s": self.elapsed_s,
        }
        _merge_prefixed(base, "feature", self.features)
        _merge_prefixed(base, "asset", self.derived_assets)
        return base


class AnalysisResult(BaseModel):
    """Algorithm result before it is attached to raw run identity."""

    status: AnalysisStatus = AnalysisStatus.SUCCESS
    features: dict[str, ScalarValue] = Field(default_factory=dict)
    derived_assets: list[DerivedAssetRef] = Field(default_factory=list)
    error_message: str | None = None
    elapsed_s: float | None = None

    @classmethod
    def failed(cls, error: Exception) -> "AnalysisResult":
        """Build a failed result from an exception."""
        return cls(status=AnalysisStatus.FAILED, error_message=str(error))


class AnalysisInvocationMetadata(BaseModel):
    """Provenance manifest for one post-run analysis invocation."""

    model_config = ConfigDict(extra="allow")

    schema_version: Literal["0.1"] = "0.1"
    analysis_id: str
    analyzer_id: str
    analyzer_name: str
    analyzer_version: str | None = None
    analysis_scope: AnalysisScope = AnalysisScope.EVENT
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user: str | None = None

    raw_run_uid: str
    scan_number: int | None = None
    experiment: str | None = None

    inputs: list[InputAssetRef] = Field(default_factory=list)
    outputs: list[DerivedAssetRef] = Field(default_factory=list)
    analysis_root: str | None = None
    analysis_output_path: str | None = None
    feature_table: str | None = None
    derived_run_uid: str | None = None

    code_version: CodeVersion | None = None
    environment: SoftwareEnvironment | None = None
    # Analyzer configuration is intentionally free-form so existing analysis
    # configs can be recorded losslessly before a shared schema exists.
    config: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


def _merge_prefixed(
    target: dict[str, ScalarValue],
    prefix: str,
    values: dict[str, ScalarValue],
) -> None:
    for key, value in values.items():
        column = f"{prefix}:{key}"
        if column in target:
            raise ValueError(f"duplicate feature-table column {column!r}")
        target[column] = value
