"""Public configuration API for ImageAnalysis.

The config *models* live in GEECS-Schemas (``geecs_schemas.analysis``) as of
ImageAnalysis 2.0 — the whole diagnostic document validates with pydantic
alone — and this package owns what needs the analysis stack:

* :mod:`loader` — YAML → typed model: :func:`load_diagnostic`,
  :func:`load_camera_config`, :func:`load_line_config`,
  :func:`list_diagnostics`, :func:`find_config_file`.
* :mod:`factory` — typed diagnostic → live analyzer:
  :func:`create_image_analyzer`.
* :mod:`registry` — analyzer ``kind`` → implementing class.
* :mod:`array2d_processing` / :mod:`array1d_processing` /
  :mod:`diagnostic` — re-exports of the schema models under the names
  the processing code and analyzers use.
"""

from geecs_schemas.analysis import (
    ANALYZER_SPECS,
    AnalysisDiagnostic,
    AnalysisGroup,
    AnalyzerRef,
    AnalyzerSpec,
    BackgroundSource,
    RendererOptions,
    ScanRuntime,
)

from .array1d_processing import (
    BackgroundConfig as Background1DConfig,
    BackgroundMethod as BackgroundMethod1D,
    Data1DConfig,
    Data1DLoading,
    Data1DType,
    FilteringConfig as Filtering1DConfig,
    FilterMethod,
    InterpolationConfig,
    Line1DConfig,
    PipelineStepType as PipelineStepType1D,
    ROI1DConfig,
    ThresholdingConfig as Thresholding1DConfig,
    ThresholdMethod as ThresholdMethod1D,
    to_data1d_config,
)
from .array2d_processing import (
    BackgroundConfig,
    BackgroundMethod,
    CameraConfig,
    CircularMaskConfig,
    CrosshairConfig,
    CrosshairMaskingConfig,
    FilteringConfig,
    NormalizationConfig,
    NormalizationMethod,
    ProcessingStepType,
    ROIConfig,
    ThresholdingConfig,
    ThresholdMethod,
    ThresholdMode,
    TransformConfig,
    VignetteConfig,
    VignetteMethod,
)
from .diagnostic import DiagnosticAnalysisConfig
from .factory import create_image_analyzer
from .loader import (
    find_config_file,
    list_diagnostics,
    load_camera_config,
    load_diagnostic,
    load_line_config,
)
from .registry import ANALYZER_CLASS_PATHS, analyzer_class

__all__ = [
    # ----- documents (geecs_schemas.analysis) -----
    "ANALYZER_SPECS",
    "AnalysisDiagnostic",
    "AnalysisGroup",
    "AnalyzerRef",
    "AnalyzerSpec",
    "BackgroundSource",
    "DiagnosticAnalysisConfig",
    "RendererOptions",
    "ScanRuntime",
    # ----- 2D processing models -----
    "BackgroundConfig",
    "BackgroundMethod",
    "CameraConfig",
    "CircularMaskConfig",
    "CrosshairConfig",
    "CrosshairMaskingConfig",
    "FilteringConfig",
    "NormalizationConfig",
    "NormalizationMethod",
    "ProcessingStepType",
    "ROIConfig",
    "ThresholdingConfig",
    "ThresholdMethod",
    "ThresholdMode",
    "TransformConfig",
    "VignetteConfig",
    "VignetteMethod",
    # ----- 1D processing models -----
    "Background1DConfig",
    "BackgroundMethod1D",
    "Data1DConfig",
    "Data1DLoading",
    "Data1DType",
    "Filtering1DConfig",
    "FilterMethod",
    "InterpolationConfig",
    "Line1DConfig",
    "PipelineStepType1D",
    "ROI1DConfig",
    "Thresholding1DConfig",
    "ThresholdMethod1D",
    "to_data1d_config",
    # ----- loader / factory / registry -----
    "ANALYZER_CLASS_PATHS",
    "analyzer_class",
    "create_image_analyzer",
    "find_config_file",
    "list_diagnostics",
    "load_camera_config",
    "load_diagnostic",
    "load_line_config",
]
