"""Public configuration API for ImageAnalysis.

This subpackage owns every config-related concern, organised into
one concern per file:

* :mod:`array2d_processing` — :class:`CameraConfig` and 2D processing
  sub-models (ROI, background, crosshair, vignette, filtering, …).
* :mod:`array1d_processing` — :class:`Line1DConfig` and 1D processing
  sub-models.
* :mod:`diagnostic` — diagnostic-level Pydantic models:
  :class:`DiagnosticAnalysisConfig` (the top-level unified-YAML model)
  and :class:`ImageAnalyzerSpec` (the field model for ``image_analyzer``).
* :mod:`loader` — YAML file → typed config:
  :func:`load_camera_config`, :func:`load_line_config`,
  :func:`load_diagnostic`, :func:`find_config_file`.
* :mod:`factory` — typed config → live instance:
  :func:`create_image_analyzer`.

The ``scan:`` field on a unified diagnostic is weakly typed at this
layer (``Optional[Dict[str, Any]]``) so ImageAnalysis can own the
full diagnostic schema without depending on scan-side runtime types.
ScanAnalysis validates the scan dict against
:class:`scan_analysis.config.diagnostic_models.ScanRuntimeConfig` at
build time.
"""

from .array1d_processing import (
    BackgroundConfig as Background1DConfig,
)
from .array1d_processing import (
    Data1DConfig,
    Data1DType,
    FilteringConfig as Filtering1DConfig,
    FilterMethod,
    InterpolationConfig,
    Line1DConfig,
    PipelineConfig as Pipeline1DConfig,
    PipelineStepType as PipelineStepType1D,
    ROI1DConfig,
    ThresholdingConfig as Thresholding1DConfig,
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
    PipelineConfig,
    ProcessingStepType,
    ROIConfig,
    ThresholdingConfig,
    ThresholdMethod,
    ThresholdMode,
    TransformConfig,
    VignetteConfig,
    VignetteMethod,
)
from .diagnostic import (
    DiagnosticAnalysisConfig,
    ImageAnalyzerSpec,
    resolve_image_analyzer_value,
)
from .factory import create_image_analyzer
from .loader import load_diagnostic
from .loader import (
    find_config_file,
    load_camera_config,
    load_line_config,
)

__all__ = [
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
    "PipelineConfig",
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
    "Data1DConfig",
    "Data1DType",
    "Filtering1DConfig",
    "FilterMethod",
    "InterpolationConfig",
    "Line1DConfig",
    "Pipeline1DConfig",
    "PipelineStepType1D",
    "ROI1DConfig",
    "Thresholding1DConfig",
    # ----- Loader -----
    "find_config_file",
    "load_camera_config",
    "load_line_config",
    # ----- Diagnostic (unified) -----
    "DiagnosticAnalysisConfig",
    "ImageAnalyzerSpec",
    "resolve_image_analyzer_value",
    "create_image_analyzer",
    "load_diagnostic",
]
