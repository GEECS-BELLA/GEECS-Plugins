"""Scan-side config models — re-exported from GEECS-Schemas, plus the loader's runtime wrapper.

The ``scan:`` section (:class:`ScanRuntime`), the background-source
directive and the group document moved to
:mod:`geecs_schemas.analysis` (GEECS-Schemas 0.19.0 / ScanAnalysis 1.19.0)
so the whole diagnostic validates with pydantic alone; the historical
names ``ScanRuntimeConfig`` and ``AnalysisGroupConfig`` are kept as
aliases.  ``diag.scan`` is typed in-document now — there is no second-stage
``ScanRuntimeConfig.model_validate(diag.scan or {})`` step any more.

:class:`ResolvedDiagnosticConfig` stays here: it is what the group loader
hands the factory (a diagnostic paired with its file-stem ID and the
group's effective priority), not a document anyone writes.
"""

from __future__ import annotations

from geecs_schemas.analysis import (
    AnalysisDiagnostic,
    AnalysisGroup,
    AnalyzerRef,
    AutodetectBackgroundSpec,
    BackgroundSource,
    FromCurrentScanSpec,
    RendererOptions,
    ScanRuntime,
)
from pydantic import BaseModel, ConfigDict, Field

#: Transitional aliases — the pre-1.19.0 names of the scan-side models.
ScanRuntimeConfig = ScanRuntime
AnalysisGroupConfig = AnalysisGroup


class ResolvedDiagnosticConfig(BaseModel):
    """A diagnostic loaded from disk and resolved against a group reference.

    Produced by the analysis-group loader: pairs the on-disk
    :class:`~geecs_schemas.analysis.AnalysisDiagnostic` with its
    filename-derived ID and the group's effective priority.  This is what
    the factory consumes to build a runnable scan analyzer.

    Attributes
    ----------
    id : str
        Filename stem of the diagnostic YAML.  Used by the task queue for
        status tracking; unique within a resolved group.
    enabled : bool
        Whether to execute this diagnostic for the group (disabled
        references are dropped by the loader, so this is ``True`` in
        practice).
    priority : int
        Effective execution priority — the group's override if given,
        else the diagnostic's own ``scan.priority``.  The loader sorts
        ascending by this value.
    diagnostic : AnalysisDiagnostic
        The validated on-disk diagnostic.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    enabled: bool = True
    priority: int = Field(ge=0)
    diagnostic: AnalysisDiagnostic


__all__ = [
    "AnalysisDiagnostic",
    "AnalysisGroup",
    "AnalysisGroupConfig",
    "AnalyzerRef",
    "AutodetectBackgroundSpec",
    "BackgroundSource",
    "FromCurrentScanSpec",
    "RendererOptions",
    "ResolvedDiagnosticConfig",
    "ScanRuntime",
    "ScanRuntimeConfig",
]
