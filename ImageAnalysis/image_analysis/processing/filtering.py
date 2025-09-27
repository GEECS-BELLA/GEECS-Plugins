"""
Image filtering and noise reduction operations for image processing.

This module provides functions for applying various types of filters to images
including Gaussian, median, and bilateral filtering for noise reduction and
image enhancement. All functions take images as input and return processed images.
"""

import numpy as np
import logging
from typing import Optional
from scipy import ndimage
from scipy.ndimage import median_filter, gaussian_filter
from ..types import Array2D
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

    Examples
    --------
    >>> filtered_image = apply_gaussian_filter(image, sigma=1.5)
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

    Examples
    --------
    >>> filtered_image = apply_median_filter(image, kernel_size=5)
    """
    if kernel_size <= 0:
        raise ValueError("Median kernel size must be positive")
    if kernel_size % 2 == 0:
        raise ValueError("Median kernel size must be odd")

    # Apply median filter
    filtered_image = median_filter(image, size=kernel_size)

    return filtered_image


def apply_bilateral_filter(
    image: Array2D, d: int, sigma_color: float, sigma_space: float
) -> Array2D:
    """
    Apply bilateral filter to image for edge-preserving noise reduction.

    Bilateral filtering reduces noise while preserving edges by considering
    both spatial proximity and intensity similarity.

    Parameters
    ----------
    image : Array2D
        Input image to filter.
    d : int
        Diameter of each pixel neighborhood used during filtering.
    sigma_color : float
        Filter sigma in the color space. Larger values mean that farther
        colors within the pixel neighborhood will be mixed together.
    sigma_space : float
        Filter sigma in the coordinate space. Larger values mean that
        farther pixels will influence the computation.

    Returns
    -------
    Array2D
        Bilateral-filtered image.

    Notes
    -----
    This is a simplified bilateral filter implementation. For production use,
    consider using OpenCV's cv2.bilateralFilter for better performance.

    Examples
    --------
    >>> filtered_image = apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)
    """
    if d <= 0:
        raise ValueError("Bilateral filter diameter must be positive")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("Bilateral filter sigmas must be positive")

    # Simple bilateral filter implementation
    # For production, consider using cv2.bilateralFilter
    filtered_image = _bilateral_filter_simple(image, d, sigma_color, sigma_space)

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

    Examples
    --------
    >>> config = FilteringConfig(gaussian_sigma=1.0, median_kernel_size=3)
    >>> filtered_image = apply_filtering_config(image, config)
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

    # Apply bilateral filter if specified
    if config.bilateral_d is not None:
        filtered_image = apply_bilateral_filter(
            filtered_image,
            config.bilateral_d,
            config.bilateral_sigma_color,
            config.bilateral_sigma_space,
        )
        logger.debug(f"Applied bilateral filter with d={config.bilateral_d}")

    return filtered_image


def apply_unsharp_mask(
    image: Array2D, sigma: float = 1.0, strength: float = 1.0
) -> Array2D:
    """
    Apply unsharp masking for image sharpening.

    Unsharp masking enhances edges and fine details by subtracting a blurred
    version of the image from the original.

    Parameters
    ----------
    image : Array2D
        Input image to sharpen.
    sigma : float, optional
        Standard deviation for Gaussian blur. Default is 1.0.
    strength : float, optional
        Strength of the sharpening effect. Default is 1.0.

    Returns
    -------
    Array2D
        Sharpened image.

    Examples
    --------
    >>> sharpened_image = apply_unsharp_mask(image, sigma=1.5, strength=0.8)
    """
    if sigma <= 0:
        raise ValueError("Unsharp mask sigma must be positive")

    # Convert to float for processing
    image_float = image.astype(np.float64)

    # Create blurred version
    blurred = gaussian_filter(image_float, sigma=sigma)

    # Apply unsharp mask: original + strength * (original - blurred)
    sharpened = image_float + strength * (image_float - blurred)

    # Clip to valid range and convert back to original dtype
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        sharpened = np.clip(sharpened, info.min, info.max)

    return sharpened.astype(image.dtype)


def apply_wiener_filter(
    image: Array2D, noise_variance: Optional[float] = None
) -> Array2D:
    """
    Apply Wiener filter for noise reduction.

    The Wiener filter is optimal for removing additive noise when the noise
    characteristics are known.

    Parameters
    ----------
    image : Array2D
        Input image to filter.
    noise_variance : float, optional
        Estimated noise variance. If None, it will be estimated from the image.

    Returns
    -------
    Array2D
        Wiener-filtered image.

    Examples
    --------
    >>> filtered_image = apply_wiener_filter(image, noise_variance=100.0)
    """
    # Convert to float for processing
    image_float = image.astype(np.float64)

    # Estimate noise variance if not provided
    if noise_variance is None:
        # Simple noise estimation using Laplacian
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        convolved = ndimage.convolve(image_float, laplacian)
        noise_variance = np.var(convolved) * 0.5
        logger.debug(f"Estimated noise variance: {noise_variance}")

    # Apply Wiener filter in frequency domain
    # This is a simplified implementation
    fft_image = np.fft.fft2(image_float)

    # Power spectral density
    psd = np.abs(fft_image) ** 2

    # Wiener filter
    wiener_filter = psd / (psd + noise_variance)

    # Apply filter and inverse transform
    filtered_fft = fft_image * wiener_filter
    filtered_image = np.real(np.fft.ifft2(filtered_fft))

    return filtered_image.astype(image.dtype)


def _bilateral_filter_simple(
    image: Array2D, d: int, sigma_color: float, sigma_space: float
) -> Array2D:
    """
    Simple bilateral filter implementation.

    This is a basic implementation for demonstration. For production use,
    consider using optimized implementations like OpenCV's bilateralFilter.

    Parameters
    ----------
    image : Array2D
        Input image to filter.
    d : int
        Diameter of pixel neighborhood.
    sigma_color : float
        Filter sigma in color space.
    sigma_space : float
        Filter sigma in coordinate space.

    Returns
    -------
    Array2D
        Bilateral-filtered image.
    """
    # Convert to float for processing
    image_float = image.astype(np.float64)
    height, width = image_float.shape
    filtered = np.zeros_like(image_float)

    # Half window size
    half_d = d // 2

    # Precompute spatial weights
    spatial_weights = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            spatial_dist = np.sqrt((i - half_d) ** 2 + (j - half_d) ** 2)
            spatial_weights[i, j] = np.exp(-(spatial_dist**2) / (2 * sigma_space**2))

    # Apply bilateral filter
    for y in range(height):
        for x in range(width):
            # Define neighborhood bounds
            y_min = max(0, y - half_d)
            y_max = min(height, y + half_d + 1)
            x_min = max(0, x - half_d)
            x_max = min(width, x + half_d + 1)

            # Extract neighborhood
            neighborhood = image_float[y_min:y_max, x_min:x_max]

            # Calculate intensity differences
            center_intensity = image_float[y, x]
            intensity_diffs = neighborhood - center_intensity

            # Calculate intensity weights
            intensity_weights = np.exp(-(intensity_diffs**2) / (2 * sigma_color**2))

            # Get corresponding spatial weights
            spatial_subset = spatial_weights[
                y_min - y + half_d : y_max - y + half_d,
                x_min - x + half_d : x_max - x + half_d,
            ]

            # Combine weights
            combined_weights = intensity_weights * spatial_subset

            # Normalize and apply
            weight_sum = np.sum(combined_weights)
            if weight_sum > 0:
                filtered[y, x] = np.sum(neighborhood * combined_weights) / weight_sum
            else:
                filtered[y, x] = center_intensity

    return filtered.astype(image.dtype)
