"""
Image masking and cropping operations for image processing.

This module provides functions for applying various types of masks to images
including crosshair masking, ROI cropping, and circular masking.
All functions take images as input and return processed images.
"""

import numpy as np
import cv2
import logging
from typing import Tuple
from ..types import Array2D
from .config_models import CrosshairMaskingConfig, ROIConfig, CircularMaskConfig

logger = logging.getLogger(__name__)


def apply_crosshair_masking(image: Array2D, config: CrosshairMaskingConfig) -> Array2D:
    """
    Apply crosshair masking to remove fiducial markers from image.

    This function supports both the original sophisticated crosshair masking
    with rotation and custom dimensions, and the legacy simple line masking.

    Parameters
    ----------
    image : Array2D
        Input image to process.
    config : CrosshairMaskingConfig
        Configuration specifying crosshair parameters.

    Returns
    -------
    Array2D
        Image with crosshairs masked out.

    """
    if not config.enabled:
        return image.copy()

    # Create a copy to avoid modifying the original image
    masked_image = image.copy()

    if config.is_legacy_mode():
        # Use legacy simple line masking
        logger.debug("Using legacy crosshair masking mode")
        for cross_location in [
            config.fiducial_cross1_location,
            config.fiducial_cross2_location,
        ]:
            if cross_location is not None:
                masked_image = _mask_crosshair_at_location(
                    masked_image, cross_location, config.mask_size, config.mask_value
                )
    else:
        # Use sophisticated crosshair masking with rotation support
        logger.debug(
            f"Using sophisticated crosshair masking for {len(config.crosshairs)} crosshairs"
        )
        for crosshair in config.crosshairs:
            cross_mask = create_cross_mask(
                image.shape,
                crosshair.center,
                crosshair.width,
                crosshair.height,
                crosshair.thickness,
                crosshair.angle,
            )
            # Apply mask: multiply image by (1 - mask) to zero out crosshair regions
            # Then add mask_value * mask to set crosshair regions to desired value
            masked_image = (
                masked_image * (1 - cross_mask) + config.mask_value * cross_mask
            )

    return masked_image


def create_cross_mask(
    image_shape: Tuple[int, int],
    center: Tuple[int, int],
    width: int,
    height: int,
    thickness: int,
    angle: float = 0.0,
) -> Array2D:
    """
    Create a binary mask for a crosshair pattern with rotation support.

    This function recreates the original sophisticated crosshair masking
    algorithm that was used in the EBeamProfileAnalyzer.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        Shape of the image (height, width).
    center : Tuple[int, int]
        Center coordinates (x, y) of the crosshair.
    width : int
        Width of the crosshair in pixels.
    height : int
        Height of the crosshair in pixels.
    thickness : int
        Thickness of the crosshair lines in pixels.
    angle : float, optional
        Rotation angle in degrees. Default is 0.0.

    Returns
    -------
    Array2D
        Binary mask where 1 indicates crosshair regions and 0 indicates background.

    """
    img_height, img_width = image_shape
    center_x, center_y = center

    # Create a blank mask
    mask = np.zeros((img_height, img_width), dtype=np.float64)

    # Calculate half dimensions
    half_width = width // 2
    half_height = height // 2
    half_thickness = thickness // 2

    # Create horizontal and vertical rectangles for the cross
    # Horizontal bar
    h_x1 = max(0, center_x - half_width)
    h_x2 = min(img_width, center_x + half_width)
    h_y1 = max(0, center_y - half_thickness)
    h_y2 = min(img_height, center_y + half_thickness)

    # Vertical bar
    v_x1 = max(0, center_x - half_thickness)
    v_x2 = min(img_width, center_x + half_thickness)
    v_y1 = max(0, center_y - half_height)
    v_y2 = min(img_height, center_y + half_height)

    # Draw the cross on the mask
    if h_y2 > h_y1 and h_x2 > h_x1:
        mask[h_y1:h_y2, h_x1:h_x2] = 1.0
    if v_y2 > v_y1 and v_x2 > v_x1:
        mask[v_y1:v_y2, v_x1:v_x2] = 1.0

    # Apply rotation if needed
    if abs(angle) > 1e-6:  # Only rotate if angle is significant
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

        # Apply rotation to the mask
        mask = cv2.warpAffine(
            mask,
            rotation_matrix,
            (img_width, img_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Threshold to maintain binary mask after interpolation
        mask = (mask > 0.5).astype(np.float64)

    return mask


def apply_roi_cropping(image: Array2D, config: ROIConfig) -> Array2D:
    """
    Crop image to specified region of interest.

    Parameters
    ----------
    image : Array2D
        Input image to crop.
    config : ROIConfig
        Configuration specifying the region of interest boundaries.

    Returns
    -------
    Array2D
        Cropped image containing only the specified ROI.

    Raises
    ------
    ValueError
        If ROI boundaries are outside image dimensions.

    """
    height, width = image.shape

    # Validate ROI boundaries
    if config.x_min < 0 or config.x_max > width:
        raise ValueError(
            f"X boundaries [{config.x_min}, {config.x_max}] outside image width {width}"
        )
    if config.y_min < 0 or config.y_max > height:
        raise ValueError(
            f"Y boundaries [{config.y_min}, {config.y_max}] outside image height {height}"
        )

    # Crop the image
    cropped_image = image[config.y_min : config.y_max, config.x_min : config.x_max]

    return cropped_image


def apply_circular_mask(image: Array2D, config: CircularMaskConfig) -> Array2D:
    """
    Apply circular mask to image.

    Parameters
    ----------
    image : Array2D
        Input image to mask.
    config : CircularMaskConfig
        Configuration specifying circle center, radius, and masking behavior.

    Returns
    -------
    Array2D
        Image with circular mask applied.
    """
    if not config.enabled:
        return image.copy()

    # Create a copy to avoid modifying the original image
    masked_image = image.copy()

    # Create coordinate grids
    height, width = image.shape
    y_coords, x_coords = np.ogrid[:height, :width]

    # Calculate distance from center
    center_x, center_y = config.center
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    # Create mask based on configuration
    if config.mask_outside:
        # Mask pixels outside the circle
        mask = distances > config.radius
    else:
        # Mask pixels inside the circle
        mask = distances <= config.radius

    # Apply mask
    masked_image[mask] = config.mask_value

    return masked_image


def apply_rectangular_mask(
    image: Array2D,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    mask_value: float = 0.0,
) -> Array2D:
    """
    Apply rectangular mask to specified region of image.

    Parameters
    ----------
    image : Array2D
        Input image to mask.
    x_min, x_max : int
        X boundaries of rectangular mask region.
    y_min, y_max : int
        Y boundaries of rectangular mask region.
    mask_value : float, optional
        Value to use for masked pixels. Default is 0.0.

    Returns
    -------
    Array2D
        Image with rectangular region masked.

    """
    # Create a copy to avoid modifying the original image
    masked_image = image.copy()

    # Validate boundaries
    height, width = image.shape
    x_min = max(0, min(x_min, width))
    x_max = max(0, min(x_max, width))
    y_min = max(0, min(y_min, height))
    y_max = max(0, min(y_max, height))

    # Apply mask
    masked_image[y_min:y_max, x_min:x_max] = mask_value

    return masked_image


def _mask_crosshair_at_location(
    image: Array2D, location: Tuple[int, int], mask_size: int, mask_value: float
) -> Array2D:
    """
    Mask a crosshair pattern at the specified location.

    Parameters
    ----------
    image : Array2D
        Image to modify.
    location : Tuple[int, int]
        (x, y) coordinates of crosshair center.
    mask_size : int
        Size of the crosshair mask.
    mask_value : float
        Value to use for masked pixels.

    Returns
    -------
    Array2D
        Image with crosshair masked at the specified location.
    """
    x_center, y_center = location
    height, width = image.shape

    # Calculate crosshair boundaries with bounds checking
    half_size = mask_size // 2

    # Horizontal line of crosshair
    x_start = max(0, x_center - half_size)
    x_end = min(width, x_center + half_size + 1)
    y_line = max(0, min(y_center, height - 1))

    # Vertical line of crosshair
    y_start = max(0, y_center - half_size)
    y_end = min(height, y_center + half_size + 1)
    x_line = max(0, min(x_center, width - 1))

    # Apply horizontal line mask
    if 0 <= y_line < height:
        image[y_line, x_start:x_end] = mask_value

    # Apply vertical line mask
    if 0 <= x_line < width:
        image[y_start:y_end, x_line] = mask_value

    return image


def create_mask_from_threshold(
    image: Array2D, threshold: float, mask_above: bool = True
) -> Array2D:
    """
    Create binary mask based on intensity threshold.

    Parameters
    ----------
    image : Array2D
        Input image to threshold.
    threshold : float
        Intensity threshold value.
    mask_above : bool, optional
        If True, mask pixels above threshold. If False, mask below threshold.
        Default is True.

    Returns
    -------
    Array2D
        Binary mask array (0 for masked, 1 for unmasked).

    """
    if mask_above:
        mask = (image <= threshold).astype(np.float64)
    else:
        mask = (image >= threshold).astype(np.float64)

    return mask


def apply_mask_array(image: Array2D, mask: Array2D, mask_value: float = 0.0) -> Array2D:
    """
    Apply a binary mask array to an image.

    Parameters
    ----------
    image : Array2D
        Input image to mask.
    mask : Array2D
        Binary mask array (0 for masked regions, non-zero for unmasked).
    mask_value : float, optional
        Value to use for masked pixels. Default is 0.0.

    Returns
    -------
    Array2D
        Masked image.

    Raises
    ------
    ValueError
        If image and mask have different shapes.

    """
    if image.shape != mask.shape:
        raise ValueError(f"Image shape {image.shape} != mask shape {mask.shape}")

    # Create a copy to avoid modifying the original image
    masked_image = image.copy()

    # Apply mask (mask == 0 means masked region)
    masked_image[mask == 0] = mask_value

    return masked_image
