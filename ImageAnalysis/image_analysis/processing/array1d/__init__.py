"""1D array processing module.

This module provides processing functions for 1D data such as lineouts,
spectra, and other single-dimensional array data.

Data Format Convention:
    All functions in this module expect Nx2 numpy arrays where:
    - Column 0: x-values (independent variable - wavelength, time, position, etc.)
    - Column 1: y-values (dependent variable - intensity, signal, etc.)

Main Components:
    - config_models: Pydantic configuration models
    - background: Background computation and subtraction
    - filtering: Filtering operations (Gaussian, median, bilateral)
    - thresholding: Thresholding operations
    - pipeline: Main processing pipeline orchestration
"""

from .background import (
    compute_background,
    load_background_from_file,
    save_background_to_file,
    subtract_background,
)
from .config_models import (
    BackgroundConfig,
    BackgroundMethod,
    FilteringConfig,
    FilterMethod,
    Line1DConfig,
    PipelineConfig,
    PipelineStepType,
    ROI1DConfig,
    ThresholdingConfig,
    ThresholdMethod,
)
from .filtering import (
    apply_bilateral_filter,
    apply_filtering,
    apply_gaussian_filter,
    apply_median_filter,
)
from .pipeline import apply_line_processing_pipeline, validate_pipeline_config
from .roi import apply_roi_1d
from .thresholding import apply_thresholding, find_threshold_crossings

__all__ = [
    # Configuration models
    "Line1DConfig",
    "BackgroundConfig",
    "BackgroundMethod",
    "FilteringConfig",
    "FilterMethod",
    "ROI1DConfig",
    "ThresholdingConfig",
    "ThresholdMethod",
    "PipelineConfig",
    "PipelineStepType",
    # Background operations
    "compute_background",
    "subtract_background",
    "load_background_from_file",
    "save_background_to_file",
    # Filtering operations
    "apply_filtering",
    "apply_gaussian_filter",
    "apply_median_filter",
    "apply_bilateral_filter",
    # ROI operations
    "apply_roi_1d",
    # Thresholding operations
    "apply_thresholding",
    "find_threshold_crossings",
    # Pipeline
    "apply_line_processing_pipeline",
    "validate_pipeline_config",
]
