"""Analysis-config schemas: the analysis recipe, the ``.himg`` and scalar-columns jobs, the group document, and their parts.

Everything the analysis stack loads from ``scan_analysis_configs/`` is
described here — pydantic only, so editors, the data portal, MCP and CI
can validate a document without the analysis stack installed.  Three
documents share the ``analyzers/`` tree: the :class:`AnalysisRecipe` (the
analysis core's native shape: input, ordered steps, a measure, the
per-frame draw, the summary kinds), the :class:`HimgJob`, which carries
the ``.himg`` data-management jobs, and the :class:`ScalarJob` (new
s-file columns from formulas over existing ones).  :func:`load_analysis_document`
dispatches on a top-level ``kind`` (a job), else reads a recipe.
"""

from geecs_schemas.analysis.canonical import canonical_document
from geecs_schemas.analysis.document import AnalysisDocument, load_analysis_document
from geecs_schemas.analysis.himg import HimgJob, HimgJobKind, HimgJobRuntime
from geecs_schemas.analysis.group import AnalysisGroup, AnalyzerRef
from geecs_schemas.analysis.scalar import (
    SCALAR_FUNCTIONS,
    SCALAR_REDUCTIONS,
    ScalarCase,
    ScalarColumn,
    ScalarJob,
    ScalarJobRuntime,
)
from geecs_schemas.analysis.processing_1d import Data1DLoading, Data1DType
from geecs_schemas.analysis.recipe import (
    ANALYSIS_OUTPUT_PREFIX,
    CURRENT_RECIPE_VERSION,
    SUMMARY_KINDS,
    AnalysisRecipe,
    AverageSummary,
    BandSummary,
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
    is_analysis_output,
)
from geecs_schemas._base import declared_schema_version

__all__ = [
    "ANALYSIS_OUTPUT_PREFIX",
    "is_analysis_output",
    "CURRENT_RECIPE_VERSION",
    "SUMMARY_KINDS",
    "AnalysisDocument",
    "HimgJob",
    "HimgJobKind",
    "HimgJobRuntime",
    "AnalysisGroup",
    "AnalysisRecipe",
    "AverageSummary",
    "BandSummary",
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
    "SCALAR_FUNCTIONS",
    "SCALAR_REDUCTIONS",
    "ScalarCase",
    "ScalarColumn",
    "ScalarJob",
    "ScalarJobRuntime",
    "ScalarFitSummary",
    "StepRef",
    "Summary",
    "WaterfallSummary",
    "canonical_document",
    "declared_schema_version",
    "load_analysis_document",
]
