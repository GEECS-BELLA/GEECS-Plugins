"""
Image processing functions for the ImageAnalysis library.

This module contains functions that take images as input and return processed images.
This is distinct from the algorithms module which extracts scalar measurements from images.

Modules
-------
config_models : Pydantic configuration models for processing functions
background : Background computation and subtraction methods
background_manager : Dedicated background management class
masking : Image masking and cropping operations
filtering : Image filtering and noise reduction
transforms : Geometric image transformations
thresholding : Image thresholding operations
pipeline : Unified processing pipeline leveraging full CameraConfig
"""

# Import key processing functions for easy access
from .background import (
    compute_background,
    subtract_background,
    load_background_from_file,
    save_background_to_file,
)
from .background_manager import BackgroundManager
from .masking import apply_crosshair_masking, apply_roi_cropping, apply_circular_mask
from .filtering import (
    apply_gaussian_filter,
    apply_median_filter,
    apply_filtering_config,
)
from .transforms import (
    apply_rotation,
    apply_vertical_flip,
    apply_horizontal_flip,
    apply_transform_config,
)
from .thresholding import apply_threshold
from .pipeline import (
    apply_camera_processing_pipeline,
    apply_non_background_processing,
    create_background_manager_from_config,
    get_processing_summary,
)
from .config_models import (
    CameraConfig,
    BackgroundConfig,
    FilteringConfig,
    CrosshairMaskingConfig,
    ROIConfig,
    TransformConfig,
    ThresholdingConfig,
    CircularMaskConfig,
    BackgroundMethod,
    BackgroundType,
    ThresholdMethod,
    ThresholdMode,
)

# Registry for dynamic function lookup
PROCESSING_FUNCTIONS = {
    "crosshair_masking": apply_crosshair_masking,
    "roi_cropping": apply_roi_cropping,
    "circular_masking": apply_circular_mask,
    "gaussian_filter": apply_gaussian_filter,
    "median_filter": apply_median_filter,
    "thresholding": apply_threshold,
}

__all__ = [
    # Background processing
    "compute_background",
    "subtract_background",
    "load_background_from_file",
    "save_background_to_file",
    "BackgroundManager",
    # Masking operations
    "apply_crosshair_masking",
    "apply_roi_cropping",
    "apply_circular_mask",
    # Filtering operations
    "apply_gaussian_filter",
    "apply_median_filter",
    "apply_filtering_config",
    # Transform operations
    "apply_rotation",
    "apply_horizontal_flip",
    "apply_vertical_flip",
    "apply_transform_config",
    # Thresholding operations
    "apply_threshold",
    # Unified processing pipeline
    "apply_camera_processing_pipeline",
    "create_background_manager_from_config",
    "apply_non_background_processing",
    "get_processing_summary",
    # Configuration models
    "CameraConfig",
    "BackgroundConfig",
    "FilteringConfig",
    "CrosshairMaskingConfig",
    "ROIConfig",
    "TransformConfig",
    "ThresholdingConfig",
    "CircularMaskConfig",
    "BackgroundMethod",
    "BackgroundType",
    "ThresholdMethod",
    "ThresholdMode",
    # Function registry
    "PROCESSING_FUNCTIONS",
]
