"""
Image filtering and noise reduction operations for image processing.

This module provides functions for applying various types of filters to images
including Gaussian, median, and bilateral filtering for noise reduction and
image enhancement. All functions take images as input and return processed images.
"""

import numpy as np
import logging
from scipy.ndimage import median_filter, gaussian_filter
from ...types import Array2D
from .config_models import FilteringConfig

logger = logging.getLogger(__name__)


def apply_gaussian_filter(image: Array2D, sigma: float) -> Array2D:
    """
    Apply Gaussian filter to image for noise reduction.

    Parameters
    ----------
    image : Array2D
        Input image to filter.
    sigma : float
        Standard deviation of the Gaussian kernel. Larger values produce
        more smoothing.

    Returns
    -------
    Array2D
        Gaussian-filtered image.


    """
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")

    # Apply Gaussian filter
    filtered_image = gaussian_filter(image.astype(np.float64), sigma=sigma)

    # Convert back to original dtype
    return filtered_image.astype(image.dtype)


def apply_median_filter(image: Array2D, kernel_size: int) -> Array2D:
    """
    Apply median filter to image for noise reduction.

    Median filtering is particularly effective at removing salt-and-pepper noise
    while preserving edges.

    Parameters
    ----------
    image : Array2D
        Input image to filter.
    kernel_size : int
        Size of the median filter kernel. Must be odd and positive.

    Returns
    -------
    Array2D
        Median-filtered image.

    Raises
    ------
    ValueError
        If kernel_size is not positive or not odd.


    """
    if kernel_size <= 0:
        raise ValueError("Median kernel size must be positive")
    if kernel_size % 2 == 0:
        raise ValueError("Median kernel size must be odd")

    # Apply median filter
    filtered_image = median_filter(image, size=kernel_size)

    return filtered_image


def apply_filtering_config(image: Array2D, config: FilteringConfig) -> Array2D:
    """
    Apply filtering operations based on configuration.

    This function applies multiple filtering operations in sequence based on
    the provided configuration. Filters are applied in the order: Gaussian,
    median, bilateral.

    Parameters
    ----------
    image : Array2D
        Input image to filter.
    config : FilteringConfig
        Configuration specifying which filters to apply and their parameters.

    Returns
    -------
    Array2D
        Filtered image with all specified filters applied.


    """
    filtered_image = image.copy()

    # Apply Gaussian filter if specified
    if config.gaussian_sigma is not None:
        filtered_image = apply_gaussian_filter(filtered_image, config.gaussian_sigma)
        logger.debug(f"Applied Gaussian filter with sigma={config.gaussian_sigma}")

    # Apply median filter if specified
    if config.median_kernel_size is not None:
        filtered_image = apply_median_filter(filtered_image, config.median_kernel_size)
        logger.debug(
            f"Applied median filter with kernel_size={config.median_kernel_size}"
        )

    return filtered_image
