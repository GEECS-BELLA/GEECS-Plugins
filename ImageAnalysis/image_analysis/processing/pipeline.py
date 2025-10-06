"""
Unified image processing pipeline that leverages the full CameraConfig Pydantic model.

This module provides a single, comprehensive processing pipeline function that
applies all configured processing steps in the correct order, taking full
advantage of the Pydantic model structure and validation.
"""

import logging
from typing import Optional

from ..types import Array2D
from .config_models import CameraConfig
from .background_manager import BackgroundManager
from .masking import apply_crosshair_masking, apply_roi_cropping, apply_circular_mask
from .filtering import apply_filtering_config
from .transforms import apply_transform_config
from .thresholding import apply_threshold
from ..utils import ensure_float64_processing

logger = logging.getLogger(__name__)


def _apply_non_background_steps(image: Array2D, camera_config: CameraConfig) -> Array2D:
    """
    Apply all non-background processing steps in a canonical order.

    Order:
        1) Crosshair masking
        2) ROI cropping
        3) Circular masking
        4) Thresholding
        5) Filtering
        6) Geometric transforms

    Parameters
    ----------
    image : Array2D
        Input image (already dtype-normalized and/or background-processed).
    camera_config : CameraConfig
        Full camera configuration model.

    Returns
    -------
    Array2D
        Processed image.
    """
    processed = image

    # 1) Crosshair masking
    if camera_config.crosshair_masking and camera_config.crosshair_masking.enabled:
        processed = apply_crosshair_masking(processed, camera_config.crosshair_masking)
        logger.debug("Applied crosshair masking")

    # 2) ROI cropping
    if camera_config.roi:
        processed = apply_roi_cropping(processed, camera_config.roi)
        logger.debug("Applied ROI cropping: %s", camera_config.roi)

    # 3) Circular masking
    if camera_config.circular_mask and camera_config.circular_mask.enabled:
        processed = apply_circular_mask(processed, camera_config.circular_mask)
        logger.debug("Applied circular masking")

    # 4) Thresholding
    if camera_config.thresholding and camera_config.thresholding.enabled:
        processed = apply_threshold(
            processed,
            camera_config.thresholding.method.value,
            camera_config.thresholding.value,
            camera_config.thresholding.mode.value,
            camera_config.thresholding.invert,
        )
        logger.debug("Applied thresholding: %s", camera_config.thresholding.method)

    # 5) Filtering
    if camera_config.filtering:
        processed = apply_filtering_config(processed, camera_config.filtering)
        logger.debug("Applied filtering")

    # 6) Geometric transforms
    if camera_config.transforms:
        processed = apply_transform_config(processed, camera_config.transforms)
        logger.debug("Applied geometric transforms")

    return processed


def apply_camera_processing_pipeline(
    image: Array2D,
    camera_config: CameraConfig,
    background_manager: Optional[BackgroundManager] = None,
) -> Array2D:
    """
    Apply the complete camera processing pipeline using the Pydantic configuration model.

    Processing Order:
        0) Ensure float64 processing dtype
        1) Background processing (if manager provided)
        2–7) Non-background steps via `_apply_non_background_steps`:
              crosshair → ROI → circular → threshold → filtering → transforms
    """
    # 0) dtype
    processed = ensure_float64_processing(image)
    logger.debug("Starting processing pipeline for camera: %s", camera_config.name)

    # 1) Background
    if background_manager is not None:
        processed = background_manager.process_single_image(processed)
        logger.debug("Applied background processing")

    # 2–7) Everything else
    processed = _apply_non_background_steps(processed, camera_config)

    logger.debug("Completed processing pipeline. Final shape: %s", processed.shape)
    return processed


def apply_non_background_processing(
    image: Array2D, camera_config: CameraConfig
) -> Array2D:
    """
    Apply all processing steps except background processing.

    This is a thin wrapper that forwards to the canonical non-background
    sequence used by the full pipeline. Use when the image is already
    background-corrected (or background is not used).

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
    return _apply_non_background_steps(image, camera_config)


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
