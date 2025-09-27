"""
Image masking and cropping operations for image processing.

This module provides functions for applying various types of masks to images
including crosshair masking, ROI cropping, and circular masking.
All functions take images as input and return processed images.
"""

import numpy as np
import logging
from typing import Tuple
from ..types import Array2D
from .config_models import CrosshairMaskingConfig, ROIConfig, CircularMaskConfig

logger = logging.getLogger(__name__)


def apply_crosshair_masking(image: Array2D, config: CrosshairMaskingConfig) -> Array2D:
    """
    Apply crosshair masking to remove fiducial markers from image.

    This function masks out crosshair-shaped fiducial markers that are commonly
    present in beam imaging systems. The crosshairs are replaced with the
    specified mask value.

    Parameters
    ----------
    image : Array2D
        Input image to process.
    config : CrosshairMaskingConfig
        Configuration specifying crosshair locations, mask size, and mask value.

    Returns
    -------
    Array2D
        Image with crosshairs masked out.

    Examples
    --------
    >>> config = CrosshairMaskingConfig(
    ...     fiducial_cross1_location=(175, 100),
    ...     fiducial_cross2_location=(175, 924),
    ...     mask_size=20
    ... )
    >>> masked_image = apply_crosshair_masking(image, config)
    """
    if not config.enabled:
        return image.copy()

    # Create a copy to avoid modifying the original image
    masked_image = image.copy()

    # Apply masking for each crosshair location
    for cross_location in [
        config.fiducial_cross1_location,
        config.fiducial_cross2_location,
    ]:
        masked_image = _mask_crosshair_at_location(
            masked_image, cross_location, config.mask_size, config.mask_value
        )

    return masked_image


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

    Examples
    --------
    >>> config = ROIConfig(x_min=100, x_max=900, y_min=50, y_max=950)
    >>> cropped_image = apply_roi_cropping(image, config)
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

    Examples
    --------
    >>> masked_image = apply_rectangular_mask(image, 100, 200, 150, 250, mask_value=0.0)
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

    Examples
    --------
    >>> mask = create_mask_from_threshold(image, threshold=100, mask_above=True)
    >>> masked_image = image * mask
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

    Examples
    --------
    >>> mask = create_mask_from_threshold(image, 50)
    >>> masked_image = apply_mask_array(image, mask, mask_value=np.nan)
    """
    if image.shape != mask.shape:
        raise ValueError(f"Image shape {image.shape} != mask shape {mask.shape}")

    # Create a copy to avoid modifying the original image
    masked_image = image.copy()

    # Apply mask (mask == 0 means masked region)
    masked_image[mask == 0] = mask_value

    return masked_image
