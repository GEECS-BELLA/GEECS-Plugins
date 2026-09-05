"""Camera (2D) processing models — re-exported from GEECS-Schemas.

The models moved to :mod:`geecs_schemas.analysis.processing_2d` in
ImageAnalysis 2.0 so the whole diagnostic validates with pydantic alone;
this module keeps every historical name importable for the processing
functions and analyzers.
"""

from geecs_schemas.analysis.processing_2d import (
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

__all__ = [
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
]
