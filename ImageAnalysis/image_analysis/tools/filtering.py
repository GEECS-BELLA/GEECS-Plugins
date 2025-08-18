"""Image filtering utilities.

Provides functions for hot‑pixel clipping, wavelet denoising, and a combined
basic filtering pipeline. The module follows NumPy docstring conventions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.restoration import denoise_wavelet, cycle_spin
from scipy.ndimage import median_filter
import scipy.ndimage as simg


def clip_hot_pixels(
    image: np.ndarray,
    median_filter_size: int = 2,
    threshold_factor: float = 3,
) -> np.ndarray:
    """Clip hot pixels in an image using median filtering.

    Parameters
    ----------
    image : np.ndarray
        Input image to be processed. The function preserves the original data
        type.
    median_filter_size : int, optional
        Size of the median filter kernel. A value of ``2`` is the default and
        provides a small neighbourhood for estimating the local background.
    threshold_factor : float, optional
        Multiplicative factor applied to the standard deviation of the
        difference between the original image and the median‑filtered image to
        determine a hot‑pixel threshold.

    Returns
    -------
    np.ndarray
        Image with hot pixels replaced by the median filter values, cast back to
        the original data type.
    """
    data_type = image.dtype
    image = image.astype("float64")

    # Median filtering to obtain a smoothed version of the image
    blurred = median_filter(image, size=median_filter_size)
    difference = image - blurred
    threshold = threshold_factor * np.std(difference)
    hot_pixels = np.nonzero(np.abs(difference) > threshold)

    # Replace hot pixels with the median‐filtered values
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        image[y, x] = blurred[y, x]

    return image.astype(data_type)


def denoise(image: np.ndarray, max_shifts: int = 3) -> np.ndarray:
    """Apply wavelet denoising with cyclic shifting.

    Parameters
    ----------
    image : np.ndarray
        Input image to be denoised.
    max_shifts : int, optional
        Number of cyclic shifts applied in each dimension before denoising.
        Higher values improve noise reduction at the cost of additional
        computation.

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    return cycle_spin(image, func=denoise_wavelet, max_shifts=max_shifts)


def basic_filter(
    image: np.ndarray,
    hp_median: int = 2,
    hp_threshold: float = 3.0,
    denoise_cycles: int = 3,
    gauss_filter: float = 5.0,
    com_threshold: float = 0.5,
) -> dict[str, Any]:
    """Apply a basic sequence of image filters.

    This convenience function runs hot‑pixel clipping, wavelet denoising,
    Gaussian blurring, and (optionally) thresholding based on a fraction of the
    maximum pixel value. The results are returned in a dictionary containing
    intermediate images and diagnostic information.

    Parameters
    ----------
    image : np.ndarray
        Raw input image.
    hp_median : int, optional
        Median filter size for hot‑pixel clipping. ``0`` disables this step.
    hp_threshold : float, optional
        Threshold factor for hot‑pixel detection.
    denoise_cycles : int, optional
        Number of cyclic shifts for the wavelet denoising step. ``0``
        disables denoising.
    gauss_filter : float, optional
        Standard deviation for the Gaussian blur. ``0`` disables this step.
    com_threshold : float, optional
        Fraction of the maximum pixel value used to create a binary mask for
        centre‑of‑mass (COM) calculations. ``0`` disables thresholding.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        ``filter_pars`` : dict
            The parameters used for each filter stage.

        ``positions`` : dict
            ``max_ij`` – coordinates of the global maximum pixel;
            ``com_ij`` – centre‑of‑mass of the thresholded image;
            ``long_names`` – descriptive names;
            ``short_names`` – abbreviated names.

        ``arrays`` : dict
            ``raw_roi`` – original image;
            ``denoised`` – after wavelet denoising;
            ``blurred`` – after Gaussian blur;
            ``thresholded`` – after thresholding (or a copy of ``blurred`` if
            disabled).
    """
    # Clip hot pixels
    if hp_median > 0:
        image_clipped = clip_hot_pixels(
            image, median_filter_size=hp_median, threshold_factor=hp_threshold
        )
    else:
        image_clipped = image.copy()

    # Denoise
    if denoise_cycles > 0:
        image_denoised = denoise(image_clipped, max_shifts=denoise_cycles)
    else:
        image_denoised = image_clipped.copy()

    # Gaussian blur
    if gauss_filter > 0:
        image_blurred = simg.gaussian_filter(image, sigma=gauss_filter)
    else:
        image_blurred = image_denoised.copy()

    # Thresholding and centre‑of‑mass
    if com_threshold > 0:
        i_max, j_max = np.where(image_blurred == image_blurred.max(initial=0))
        i_max, j_max = i_max[0].item(), j_max[0].item()
        val_max = image_blurred[i_max, j_max]
        binary = image_blurred > (com_threshold * val_max)
        image_thresholded = image_blurred * binary.astype(float)
        i_com, j_com = simg.center_of_mass(image_thresholded)
    else:
        i_max, j_max, i_com, j_com = -1, -1, -1, -1
        image_thresholded = image_blurred.copy()

    filters = {
        "hp_median": hp_median,
        "hp_threshold": hp_threshold,
        "denoise_cycles": denoise_cycles,
        "gauss_filter": gauss_filter,
        "com_threshold": com_threshold,
    }

    positions = {
        "max_ij": (int(i_max), int(j_max)),
        "com_ij": (int(i_com), int(j_com)),
        "long_names": ["maximum", "center of mass"],
        "short_names": ["max", "com"],
    }

    arrays = {
        "raw_roi": image,
        "denoised": image_denoised,
        "blurred": image_blurred,
        "thresholded": image_thresholded,
    }

    return {"filter_pars": filters, "positions": positions, "arrays": arrays}
