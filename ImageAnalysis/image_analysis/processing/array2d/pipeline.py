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
from image_analysis.config.array2d_processing import CameraConfig, ProcessingStepType
from .background import apply_background
from .masking import apply_crosshair_masking, apply_roi_cropping, apply_circular_mask
from .filtering import apply_filtering_config
from .transforms import apply_transform_config
from .thresholding import apply_threshold
from .normalization import apply_normalization
from .vignette import apply_vignette_config
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


def _apply_vignette_step(image: Array2D, camera_config: CameraConfig) -> Array2D:
    """Apply vignette correction if configured."""
    if camera_config.vignette and camera_config.vignette.enabled:
        return apply_vignette_config(image, camera_config.vignette)
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


def _apply_normalization_step(image: Array2D, camera_config: CameraConfig) -> Array2D:
    """Apply normalization if configured."""
    if camera_config.normalization and camera_config.normalization.enabled:
        return apply_normalization(image, camera_config.normalization)
    return image


# Step registry mapping step types to their execution functions
STEP_REGISTRY: Dict[ProcessingStepType, Callable[[Array2D, CameraConfig], Array2D]] = {
    ProcessingStepType.CROSSHAIR_MASKING: _apply_crosshair_step,
    ProcessingStepType.VIGNETTE: _apply_vignette_step,
    ProcessingStepType.ROI: _apply_roi_step,
    ProcessingStepType.CIRCULAR_MASK: _apply_circular_mask_step,
    ProcessingStepType.THRESHOLDING: _apply_thresholding_step,
    ProcessingStepType.FILTERING: _apply_filtering_step,
    ProcessingStepType.NORMALIZATION: _apply_normalization_step,
    ProcessingStepType.TRANSFORMS: _apply_transforms_step,
}


def apply_camera_processing_pipeline(
    image: Array2D,
    camera_config: CameraConfig,
    background_cache: Optional[Dict[str, Array2D]] = None,
) -> Array2D:
    """Apply the complete camera processing pipeline.

    Steps run in the order declared by ``camera_config.pipeline.steps``
    (defaulting to the canonical order when no pipeline config is set).
    The background step calls :func:`apply_background` with
    ``camera_config.background``; pass ``background_cache`` (a path-keyed
    dict held by the analyzer instance) to avoid reloading the same
    background file on every shot.

    Parameters
    ----------
    image : Array2D
        Input image.
    camera_config : CameraConfig
        Camera configuration (processing pipeline + per-step configs).
    background_cache : dict, optional
        Path-keyed cache forwarded to :func:`apply_background`.
        Suppressed (no caching) when ``None``.

    Returns
    -------
    Array2D
        Processed image.
    """
    processed = ensure_float64_processing(image)
    logger.debug("Starting processing pipeline for camera: %s", camera_config.name)

    from image_analysis.config.array2d_processing import PipelineConfig

    pipeline_config = camera_config.pipeline or PipelineConfig()

    for step_type in pipeline_config.steps:
        if step_type == ProcessingStepType.BACKGROUND:
            if camera_config.background is not None:
                processed = apply_background(
                    processed, camera_config.background, cache=background_cache
                )
                logger.debug("Applied background processing")
        else:
            step_func = STEP_REGISTRY.get(step_type)
            if step_func:
                processed = step_func(processed, camera_config)
                logger.debug(f"Applied {step_type.value}")

    logger.debug("Completed processing pipeline. Final shape: %s", processed.shape)
    return processed


def apply_non_background_processing(
    image: Array2D, camera_config: CameraConfig
) -> Array2D:
    """Apply all processing steps except background.

    Convenience wrapper over :func:`apply_camera_processing_pipeline`
    with ``background_cache=None``; the background step is still
    invoked, but only emits a no-op since the underlying function
    is itself a no-op when ``config.enabled=False`` (use this when
    the image is already background-corrected upstream).

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
    return apply_camera_processing_pipeline(image, camera_config, background_cache=None)
