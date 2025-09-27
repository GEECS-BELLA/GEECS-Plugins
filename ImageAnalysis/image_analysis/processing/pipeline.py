"""
Unified image processing pipeline that leverages the full CameraConfig Pydantic model.

This module provides a single, comprehensive processing pipeline function that
applies all configured processing steps in the correct order, taking full
advantage of the Pydantic model structure and validation.
"""

import logging
from typing import Optional, List

from ..types import Array2D
from .config_models import CameraConfig
from .background_manager import BackgroundManager
from .masking import apply_crosshair_masking, apply_roi_cropping, apply_circular_mask
from .filtering import apply_filtering_config
from .transforms import apply_transform_config
from .thresholding import apply_threshold
from ..utils import ensure_float64_processing

logger = logging.getLogger(__name__)


def apply_camera_processing_pipeline(
    image: Array2D,
    camera_config: CameraConfig,
    background_manager: Optional[BackgroundManager] = None,
) -> Array2D:
    """
    Apply the complete camera processing pipeline using the Pydantic configuration model.

    This function provides a unified processing pipeline that leverages the full
    CameraConfig structure, applying all configured processing steps in the correct
    order with proper type safety and validation.

    Processing Order:
    1. Data type conversion (ensure float64 for processing)
    2. Background processing (if background_manager provided)
    3. Crosshair masking (fiducial markers)
    4. Circular masking (if configured)
    5. ROI cropping
    6. Thresholding
    7. Filtering (noise reduction)
    8. Geometric transforms

    Parameters
    ----------
    image : Array2D
        Input image to process
    camera_config : CameraConfig
        Complete camera configuration Pydantic model
    background_manager : Optional[BackgroundManager]
        Background manager for handling background processing

    Returns
    -------
    Array2D
        Fully processed image
    """
    # Step 1: Ensure proper data type for processing
    processed_image = ensure_float64_processing(image)

    logger.debug(f"Starting processing pipeline for camera: {camera_config.name}")

    # Step 2: Background processing (if manager provided)
    if background_manager is not None:
        processed_image = background_manager.process_single_image(processed_image)
        logger.debug("Applied background processing")

    # Step 3: Crosshair masking (fiducial markers)
    if camera_config.crosshair_masking and camera_config.crosshair_masking.enabled:
        processed_image = apply_crosshair_masking(
            processed_image, camera_config.crosshair_masking
        )
        logger.debug("Applied crosshair masking")
        print("Applied crosshair masking")

    # Step 4: Circular masking
    if camera_config.circular_mask and camera_config.circular_mask.enabled:
        processed_image = apply_circular_mask(
            processed_image, camera_config.circular_mask
        )
        logger.debug("Applied circular masking")

    # Step 5: ROI cropping
    if camera_config.roi:
        processed_image = apply_roi_cropping(processed_image, camera_config.roi)
        logger.debug(f"Applied ROI cropping: {camera_config.roi}")

    # Step 6: Thresholding
    if camera_config.thresholding and camera_config.thresholding.enabled:
        processed_image = apply_threshold(
            processed_image,
            camera_config.thresholding.method.value,
            camera_config.thresholding.value,
            camera_config.thresholding.mode.value,
            camera_config.thresholding.invert,
        )
        logger.debug(f"Applied thresholding: {camera_config.thresholding.method}")

    # Step 7: Filtering (noise reduction)
    if camera_config.filtering:
        processed_image = apply_filtering_config(
            processed_image, camera_config.filtering
        )
        logger.debug("Applied filtering")

    # Step 8: Geometric transforms
    if camera_config.transforms:
        processed_image = apply_transform_config(
            processed_image, camera_config.transforms
        )
        logger.debug("Applied geometric transforms")

    logger.debug(f"Completed processing pipeline. Final shape: {processed_image.shape}")

    return processed_image


def apply_camera_processing_pipeline_batch(
    images: List[Array2D],
    camera_config: CameraConfig,
    background_manager: Optional[BackgroundManager] = None,
) -> List[Array2D]:
    """
    Apply the complete camera processing pipeline to a batch of images.

    This function handles batch processing with optimized background computation
    when a background manager is provided.

    Parameters
    ----------
    images : List[Array2D]
        List of input images to process
    camera_config : CameraConfig
        Complete camera configuration Pydantic model
    background_manager : Optional[BackgroundManager]
        Background manager for handling background processing

    Returns
    -------
    List[Array2D]
        List of fully processed images
    """
    if not images:
        return []

    logger.info(f"Starting batch processing pipeline for {len(images)} images")

    # Step 1: Ensure proper data type for all images
    float_images = [ensure_float64_processing(img) for img in images]

    # Step 2: Batch background processing (if manager provided)
    if background_manager is not None:
        processed_images = background_manager.process_image_batch(float_images)
        logger.debug("Applied batch background processing")
    else:
        processed_images = float_images

    # Step 3-8: Apply remaining processing steps to each image
    final_images = []
    for i, img in enumerate(processed_images):
        # Apply all non-background processing steps
        final_img = _apply_non_background_processing(img, camera_config)
        final_images.append(final_img)

        if i % 10 == 0:  # Log progress for large batches
            logger.debug(f"Processed {i + 1}/{len(processed_images)} images")

    logger.info(f"Completed batch processing pipeline for {len(images)} images")

    return final_images


def _apply_non_background_processing(
    image: Array2D, camera_config: CameraConfig
) -> Array2D:
    """
    Apply all processing steps except background processing.

    This is a helper function for batch processing that applies steps 3-8
    of the processing pipeline.

    Parameters
    ----------
    image : Array2D
        Input image (already background-processed)
    camera_config : CameraConfig
        Complete camera configuration Pydantic model

    Returns
    -------
    Array2D
        Processed image
    """
    processed_image = image

    # Step 3: Crosshair masking
    if camera_config.crosshair_masking and camera_config.crosshair_masking.enabled:
        processed_image = apply_crosshair_masking(
            processed_image, camera_config.crosshair_masking
        )

    # Step 4: Circular masking
    if camera_config.circular_mask and camera_config.circular_mask.enabled:
        processed_image = apply_circular_mask(
            processed_image, camera_config.circular_mask
        )

    # Step 5: ROI cropping
    if camera_config.roi:
        processed_image = apply_roi_cropping(processed_image, camera_config.roi)

    # Step 6: Thresholding
    if camera_config.thresholding and camera_config.thresholding.enabled:
        processed_image = apply_threshold(
            processed_image,
            camera_config.thresholding.method.value,
            camera_config.thresholding.value,
            camera_config.thresholding.mode.value,
            camera_config.thresholding.invert,
        )

    # Step 7: Filtering
    if camera_config.filtering:
        processed_image = apply_filtering_config(
            processed_image, camera_config.filtering
        )

    # Step 8: Geometric transforms
    if camera_config.transforms:
        processed_image = apply_transform_config(
            processed_image, camera_config.transforms
        )

    return processed_image


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


def get_processing_summary(camera_config: CameraConfig) -> dict:
    """
    Get a summary of the processing steps that will be applied.

    This function analyzes the camera configuration and returns a summary
    of which processing steps are enabled and their key parameters.

    Parameters
    ----------
    camera_config : CameraConfig
        Complete camera configuration Pydantic model

    Returns
    -------
    dict
        Summary of processing steps and their configuration
    """
    summary = {
        "camera_name": camera_config.name,
        "camera_type": camera_config.camera_type.value,
        "processing_steps": [],
    }

    # Background processing
    if camera_config.background and camera_config.background.enabled:
        summary["processing_steps"].append(
            {
                "step": "background",
                "type": camera_config.background.type.value,
                "method": camera_config.background.method.value,
                "enabled": True,
            }
        )

    # Crosshair masking
    if camera_config.crosshair_masking and camera_config.crosshair_masking.enabled:
        summary["processing_steps"].append(
            {
                "step": "crosshair_masking",
                "locations": [
                    camera_config.crosshair_masking.fiducial_cross1_location,
                    camera_config.crosshair_masking.fiducial_cross2_location,
                ],
                "mask_size": camera_config.crosshair_masking.mask_size,
                "enabled": True,
            }
        )

    # Circular masking
    if camera_config.circular_mask and camera_config.circular_mask.enabled:
        summary["processing_steps"].append(
            {
                "step": "circular_mask",
                "center": camera_config.circular_mask.center,
                "radius": camera_config.circular_mask.radius,
                "mask_outside": camera_config.circular_mask.mask_outside,
                "enabled": True,
            }
        )

    # ROI cropping
    if camera_config.roi:
        summary["processing_steps"].append(
            {
                "step": "roi_cropping",
                "roi": {
                    "x_min": camera_config.roi.x_min,
                    "x_max": camera_config.roi.x_max,
                    "y_min": camera_config.roi.y_min,
                    "y_max": camera_config.roi.y_max,
                },
                "enabled": True,
            }
        )

    # Thresholding
    if camera_config.thresholding and camera_config.thresholding.enabled:
        summary["processing_steps"].append(
            {
                "step": "thresholding",
                "method": camera_config.thresholding.method.value,
                "value": camera_config.thresholding.value,
                "mode": camera_config.thresholding.mode.value,
                "invert": camera_config.thresholding.invert,
                "enabled": True,
            }
        )

    # Filtering
    if camera_config.filtering:
        filters = {}
        if camera_config.filtering.gaussian_sigma is not None:
            filters["gaussian_sigma"] = camera_config.filtering.gaussian_sigma
        if camera_config.filtering.median_kernel_size is not None:
            filters["median_kernel_size"] = camera_config.filtering.median_kernel_size
        if camera_config.filtering.bilateral_d is not None:
            filters["bilateral"] = {
                "d": camera_config.filtering.bilateral_d,
                "sigma_color": camera_config.filtering.bilateral_sigma_color,
                "sigma_space": camera_config.filtering.bilateral_sigma_space,
            }

        if filters:
            summary["processing_steps"].append(
                {"step": "filtering", "filters": filters, "enabled": True}
            )

    # Geometric transforms
    if camera_config.transforms:
        transforms = {}
        if camera_config.transforms.rotation_angle != 0:
            transforms["rotation_angle"] = camera_config.transforms.rotation_angle
        if camera_config.transforms.flip_horizontal:
            transforms["flip_horizontal"] = True
        if camera_config.transforms.flip_vertical:
            transforms["flip_vertical"] = True
        if camera_config.transforms.distortion_correction:
            transforms["distortion_correction"] = True

        if transforms:
            summary["processing_steps"].append(
                {
                    "step": "geometric_transforms",
                    "transforms": transforms,
                    "enabled": True,
                }
            )

    return summary
