"""Analysis-config schemas: the analysis recipe, the ``.himg`` job, the group document, and their parts.

Everything the analysis stack loads from ``scan_analysis_configs/`` is
described here — pydantic only, so editors, the data portal, MCP and CI
can validate a document without the analysis stack installed.  Two
documents share the ``analyzers/`` tree: the :class:`AnalysisRecipe` (the
analysis core's native shape: input, ordered steps, a measure, the
per-frame draw, the summary kinds) and the :class:`HimgJob`, which carries
the ``.himg`` data-management jobs.  :func:`load_analysis_document`
dispatches on a top-level ``kind`` (a job), else reads a recipe.
"""

from geecs_schemas.analysis.canonical import canonical_document
from geecs_schemas.analysis.document import AnalysisDocument, load_analysis_document
from geecs_schemas.analysis.himg import HimgJob, HimgJobKind, HimgJobRuntime
from geecs_schemas.analysis.group import AnalysisGroup, AnalyzerRef
from geecs_schemas.analysis.processing_1d import Data1DLoading, Data1DType
from geecs_schemas.analysis.recipe import (
    CURRENT_RECIPE_VERSION,
    SUMMARY_KINDS,
    AnalysisRecipe,
    AverageSummary,
    CameraInput,
    FigureStyle,
    FrameInput,
    ScanStatistic,
    ImageGridSummary,
    LineInput,
    MeasureRef,
    RecipeInput,
    RecipeRuntime,
    ScalarFitSummary,
    StepRef,
    Summary,
    WaterfallSummary,
)
from geecs_schemas._base import declared_schema_version

__all__ = [
    "CURRENT_RECIPE_VERSION",
    "SUMMARY_KINDS",
    "AnalysisDocument",
    "HimgJob",
    "HimgJobKind",
    "HimgJobRuntime",
    "AnalysisGroup",
    "AnalysisRecipe",
    "AverageSummary",
    "AnalyzerRef",
    "CameraInput",
    "Data1DLoading",
    "Data1DType",
    "FigureStyle",
    "FrameInput",
    "ScanStatistic",
    "ImageGridSummary",
    "LineInput",
    "MeasureRef",
    "RecipeInput",
    "RecipeRuntime",
    "ScalarFitSummary",
    "StepRef",
    "Summary",
    "WaterfallSummary",
    "canonical_document",
    "declared_schema_version",
    "load_analysis_document",
]
