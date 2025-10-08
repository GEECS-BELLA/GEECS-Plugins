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
    ROIConfig,
    BackgroundConfig,
    BackgroundMethod,
    DynamicBackgroundConfig,
    DynamicBackgroundMethod,
    MaskingConfig,
    FilteringConfig,
    FilterMethod,
    ThresholdingConfig,
    ThresholdMethod,
    TransformConfig,
    PipelineConfig,
    PipelineStepType,
)

# Background operations
from .background import (
    compute_background,
    subtract_background,
    load_background_from_file,
    save_background_to_file,
)

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
    apply_bilateral_filter,
)

# Transform operations
from .transforms import (
    apply_rotation,
    apply_flip,
    apply_resize,
)

# Thresholding operations
from .thresholding import (
    apply_thresholding,
)

# Pipeline
from .pipeline import (
    apply_camera_processing_pipeline,
)

# Background Manager
from .background_manager import (
    BackgroundManager,
    create_background_manager_from_config,
)

__all__ = [
    # Config models
    "CameraConfig",
    "ROIConfig",
    "BackgroundConfig",
    "BackgroundMethod",
    "DynamicBackgroundConfig",
    "DynamicBackgroundMethod",
    "MaskingConfig",
    "FilteringConfig",
    "FilterMethod",
    "ThresholdingConfig",
    "ThresholdMethod",
    "TransformConfig",
    "PipelineConfig",
    "PipelineStepType",
    # Background
    "compute_background",
    "subtract_background",
    "load_background_from_file",
    "save_background_to_file",
    # Masking
    "apply_crosshair_masking",
    "apply_roi_cropping",
    "apply_circular_mask",
    # Filtering
    "apply_gaussian_filter",
    "apply_median_filter",
    "apply_bilateral_filter",
    # Transforms
    "apply_rotation",
    "apply_flip",
    "apply_resize",
    # Thresholding
    "apply_thresholding",
    # Pipeline
    "apply_camera_processing_pipeline",
    # Background Manager
    "BackgroundManager",
    "create_background_manager_from_config",
]
