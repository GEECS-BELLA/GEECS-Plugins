"""2D array processing module for camera/image data.

This module provides all processing operations for 2D image data, including:
- Background computation and subtraction
- Masking (ROI, crosshair, circular)
- Filtering (Gaussian, median, bilateral)
- Transforms (rotation, flip, resize)
- Thresholding
- Pipeline orchestration
"""

# Config models
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
    ThresholdMethod,
    ThresholdMode,
)

# backgroudn operations
from .background import (
    compute_background,
    subtract_background,
    load_background_from_file,
    save_background_to_file,
)

from .background_manager import BackgroundManager


# Masking operations
from .masking import (
    apply_crosshair_masking,
    apply_roi_cropping,
    apply_circular_mask,
)

# Filtering operations
from .filtering import (
    apply_gaussian_filter,
    apply_median_filter,
    apply_filtering_config,
)

# Transform operations
from .transforms import (
    apply_rotation,
    apply_vertical_flip,
    apply_horizontal_flip,
    apply_transform_config,
)


# Thresholding operations
from .thresholding import apply_threshold

# Pipeline
from .pipeline import (
    apply_camera_processing_pipeline,
    apply_non_background_processing,
    create_background_manager_from_config,
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
    "ThresholdMethod",
    "ThresholdMode",
    # Function registry
    "PROCESSING_FUNCTIONS",
]
