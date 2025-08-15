"""General image processing utilities.

This module provides a collection of helper functions for basic image
manipulation tasks such as thresholding, hot‑pixel correction, simple
filtering pipelines, and beam property extraction. The functions are
intended for quick use in scripts and notebooks and follow NumPy‑style
docstrings.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.measure import regionprops, label


def image_signal_thresholding(
    image: np.ndarray,
    median_filter_size: int = 2,
    threshold_coeff: float = 0.1,
) -> np.ndarray:
    """Threshold an image based on a median‑filtered background estimate.

    Parameters
    ----------
    image : np.ndarray
        Input image. The original data type is preserved.
    median_filter_size : int, optional
        Size of the median filter kernel used to estimate the background.
        Default is ``2``.
    threshold_coeff : float, optional
        Multiplicative coefficient applied to the maximum of the blurred
        image to define the threshold. Pixels below ``threshold_coeff *
        max(blurred)`` are set to zero. Default is ``0.1``.

    Returns
    -------
    np.ndarray
        Thresholded image with the same dtype as the input.
    """
    # Preserve input dtype, convert to a working type
    data_type = image.dtype
    image = image.astype("float64")

    # Median filter to obtain a background estimate
    blurred_image = median_filter(image, size=median_filter_size)

    # Zero out low‑signal pixels
    image[blurred_image < blurred_image.max() * threshold_coeff] = 0

    # Restore original dtype
    return image.astype(data_type)


def hotpixel_correction(
    image: np.ndarray,
    median_filter_size: int = 2,
    threshold_factor: float = 3,
) -> np.ndarray:
    """Replace hot pixels using a median‑filter reference.

    The function identifies pixels whose deviation from a median‑filtered
    version exceeds a multiple of the standard deviation and replaces them
    with the median value.

    Parameters
    ----------
    image : np.ndarray
        Input image. The original data type is preserved.
    median_filter_size : int, optional
        Kernel size for the median filter. Default is ``2``.
    threshold_factor : float, optional
        Multiplicative factor applied to the standard deviation of the
        difference image to define the hot‑pixel threshold. Default is ``3``.

    Returns
    -------
    np.ndarray
        Image with hot pixels corrected, same dtype as the input.
    """
    data_type = image.dtype
    image = image.astype("float64")

    # Median filtering to create a reference
    blurred = median_filter(image, size=median_filter_size)
    difference = image - blurred
    threshold = threshold_factor * np.std(difference)

    # Locate hot pixels
    hot_pixels = np.nonzero(np.abs(difference) > threshold)

    # Replace hot pixels with median values
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        image[y, x] = blurred[y, x]

    return image.astype(data_type)


def filter_image(
    image: np.ndarray,
    median_filter_size: int = 2,
    median_filter_cycles: int = 0,
    gaussian_filter_size: int = 3,
    gaussian_filter_cycles: int = 0,
) -> np.ndarray:
    """Apply a configurable sequence of median and Gaussian filters.

    Parameters
    ----------
    image : np.ndarray
        Input image to be filtered.
    median_filter_size : int, optional
        Kernel size for each median‑filter iteration. Default is ``2``.
    median_filter_cycles : int, optional
        Number of times the median filter is applied. ``0`` disables median
        filtering. Default is ``0``.
    gaussian_filter_size : int, optional
        Sigma value for each Gaussian‑filter iteration. Default is ``3``.
    gaussian_filter_cycles : int, optional
        Number of times the Gaussian filter is applied. ``0`` disables Gaussian
        filtering. Default is ``0``.

    Returns
    -------
    np.ndarray
        Filtered image with the same dtype as the input.
    """
    # Preserve dtype, work in float32 for speed
    data_type = image.dtype
    processed_image = image.astype(np.float32)

    for _ in range(median_filter_cycles):
        processed_image = median_filter(processed_image, size=median_filter_size)

    for _ in range(gaussian_filter_cycles):
        processed_image = gaussian_filter(processed_image, sigma=gaussian_filter_size)

    # Cast back to original dtype
    return processed_image.astype(data_type)


def find_beam_properties(image: np.ndarray) -> dict[str, np.ndarray]:
    """Extract basic beam properties from a binary mask.

    The function thresholds the image to a binary mask, labels connected
    components, and returns the centroid of the largest region.

    Parameters
    ----------
    image : np.ndarray
        Input image containing a beam signal.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with a single key ``'centroid'`` mapping to the weighted
        centroid coordinates of the largest detected region.
    """
    beam_properties: dict[str, np.ndarray] = {}

    # Create a binary mask (non‑zero → 1)
    image_binary = image.copy()
    image_binary[image_binary > 0] = 1
    image_binary = image_binary.astype(int)

    # Label connected components
    image_label = label(image_binary)

    # Extract region properties and keep the largest area
    props = regionprops(image_label, image)
    areas = [prop.area for prop in props]
    largest = props[areas.index(max(areas))]

    # Store the weighted centroid
    beam_properties["centroid"] = largest.centroid_weighted

    return beam_properties
