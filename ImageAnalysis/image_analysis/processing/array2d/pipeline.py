"""
Unified image processing pipeline that leverages the full CameraConfig Pydantic model.

This module provides a single, comprehensive processing pipeline function that
applies all configured processing steps in the correct order, taking full
advantage of the Pydantic model structure and validation.

The pipeline now supports configurable step ordering via the PipelineConfig model,
allowing users to customize which steps are executed and in what order.
"""

import logging
from typing import Optional, Dict, Callable

from ...types import Array2D
from .config_models import CameraConfig, ProcessingStepType
from .background_manager import BackgroundManager
from .masking import apply_crosshair_masking, apply_roi_cropping, apply_circular_mask
from .filtering import apply_filtering_config
from .transforms import apply_transform_config
from .thresholding import apply_threshold
from ...utils import ensure_float64_processing

logger = logging.getLogger(__name__)


# Individual step functions for the pipeline
def _apply_crosshair_step(image: Array2D, camera_config: CameraConfig) -> Array2D:
    """Apply crosshair masking if configured."""
    if camera_config.crosshair_masking and camera_config.crosshair_masking.enabled:
        return apply_crosshair_masking(image, camera_config.crosshair_masking)
    return image


def _apply_roi_step(image: Array2D, camera_config: CameraConfig) -> Array2D:
    """Apply ROI cropping if configured."""
    if camera_config.roi:
        return apply_roi_cropping(image, camera_config.roi)
    return image


def _apply_circular_mask_step(image: Array2D, camera_config: CameraConfig) -> Array2D:
    """Apply circular masking if configured."""
    if camera_config.circular_mask and camera_config.circular_mask.enabled:
        return apply_circular_mask(image, camera_config.circular_mask)
    return image


def _apply_thresholding_step(image: Array2D, camera_config: CameraConfig) -> Array2D:
    """Apply thresholding if configured."""
    if camera_config.thresholding and camera_config.thresholding.enabled:
        return apply_threshold(
            image,
            camera_config.thresholding.method.value,
            camera_config.thresholding.value,
            camera_config.thresholding.mode.value,
            camera_config.thresholding.invert,
        )
    return image


def _apply_filtering_step(image: Array2D, camera_config: CameraConfig) -> Array2D:
    """Apply filtering if configured."""
    if camera_config.filtering:
        return apply_filtering_config(image, camera_config.filtering)
    return image


def _apply_transforms_step(image: Array2D, camera_config: CameraConfig) -> Array2D:
    """Apply geometric transforms if configured."""
    if camera_config.transforms:
        return apply_transform_config(image, camera_config.transforms)
    return image


# Step registry mapping step types to their execution functions
STEP_REGISTRY: Dict[ProcessingStepType, Callable[[Array2D, CameraConfig], Array2D]] = {
    ProcessingStepType.CROSSHAIR_MASKING: _apply_crosshair_step,
    ProcessingStepType.ROI: _apply_roi_step,
    ProcessingStepType.CIRCULAR_MASK: _apply_circular_mask_step,
    ProcessingStepType.THRESHOLDING: _apply_thresholding_step,
    ProcessingStepType.FILTERING: _apply_filtering_step,
    ProcessingStepType.TRANSFORMS: _apply_transforms_step,
}


def apply_camera_processing_pipeline(
    image: Array2D,
    camera_config: CameraConfig,
    background_manager: Optional[BackgroundManager] = None,
) -> Array2D:
    """
    Apply the complete camera processing pipeline using the Pydantic configuration model.

    The pipeline now supports configurable step ordering via the optional PipelineConfig.
    If no pipeline configuration is provided, a default order matching the original
    hardcoded sequence is used.

    Parameters
    ----------
    image : Array2D
        Input image to process
    camera_config : CameraConfig
        Complete camera configuration including optional pipeline config
    background_manager : Optional[BackgroundManager]
        Background manager for background processing

    Returns
    -------
    Array2D
        Processed image
    """
    # Ensure float64 processing dtype
    processed = ensure_float64_processing(image)
    logger.debug("Starting processing pipeline for camera: %s", camera_config.name)

    # Get pipeline configuration (use default if not specified)
    from .config_models import PipelineConfig

    pipeline_config = camera_config.pipeline or PipelineConfig()

    # Execute steps in configured order
    for step_type in pipeline_config.steps:
        if step_type == ProcessingStepType.BACKGROUND:
            # Background step is handled specially via the background manager
            if background_manager is not None:
                processed = background_manager.process_single_image(processed)
                logger.debug("Applied background processing")
        else:
            # Get the step function from registry
            step_func = STEP_REGISTRY.get(step_type)
            if step_func:
                processed = step_func(processed, camera_config)
                logger.debug(f"Applied {step_type.value}")

    logger.debug("Completed processing pipeline. Final shape: %s", processed.shape)
    return processed


def apply_non_background_processing(
    image: Array2D, camera_config: CameraConfig
) -> Array2D:
    """
    Apply all processing steps except background processing.

    This function applies the configured pipeline without background processing,
    useful when the image is already background-corrected.

    Parameters
    ----------
    image : Array2D
        Input image (already background-processed if applicable).
    camera_config : CameraConfig
        Full camera configuration model.

    Returns
    -------
    Array2D
        Processed image.
    """
    # Use the main pipeline but without background manager
    return apply_camera_processing_pipeline(
        image, camera_config, background_manager=None
    )


def create_background_manager_from_config(
    camera_config: CameraConfig,
) -> Optional[BackgroundManager]:
    """
    Create a BackgroundManager from the camera configuration.

    This is a convenience function that creates a BackgroundManager if
    background processing is configured in the camera config.

    Parameters
    ----------
    camera_config : CameraConfig
        Complete camera configuration Pydantic model

    Returns
    -------
    Optional[BackgroundManager]
        BackgroundManager instance if background is configured, None otherwise
    """
    if camera_config.background and camera_config.background.enabled:
        return BackgroundManager(camera_config.background)
    return None
