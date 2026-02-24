"""Filtering operations for 1D array data.

This module provides filtering functions for 1D data (lineouts, spectra, etc.).

Data Format:
    All functions expect Nx2 numpy arrays where:
    - Column 0: x-values (independent variable)
    - Column 1: y-values (dependent variable)

    Filtering is applied only to the y-values (column 1).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage, signal

from .config_models import FilteringConfig, FilterMethod

logger = logging.getLogger(__name__)


def apply_butterworth_filter(
    data: np.ndarray,
    order: int = 1,
    crit_f: float = 0.125,
    filt_type: str = "low",
) -> np.ndarray:
    """Apply forward-backward Butterworth filter to 1D data.

    Uses scipy.signal.filtfilt for zero-phase filtering (forward and backward
    propagation) to avoid phase distortion. This is a general-purpose filtering
    function suitable for any 1D data analysis.

    Parameters
    ----------
    data : np.ndarray
        Input signal array (1D)
    order : int, default=1
        Filter order (number of poles)
    crit_f : float, default=0.125
        Normalized critical frequency (0 < crit_f < 1)
        where 1.0 corresponds to Nyquist frequency
    filt_type : str, default='low'
        Filter type: 'low', 'high', 'band', 'bandstop'

    Returns
    -------
    np.ndarray
        Filtered signal with same shape as input

    Raises
    ------
    ValueError
        If filter parameters are invalid
    """
    try:
        # Design Butterworth filter
        b, a = signal.butter(order, crit_f, filt_type)

        # Apply filter in forward and backward direction (zero-phase)
        filtered_data = signal.filtfilt(b, a, data)

        logger.debug(
            f"Applied Butterworth filter: order={order}, crit_f={crit_f}, type={filt_type}"
        )
        return filtered_data
    except Exception as e:
        logger.error(f"Butterworth filter failed: {e}")
        raise ValueError(f"Butterworth filter failed: {e}") from e


def apply_gaussian_filter(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian filter to 1D data.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    sigma : float
        Standard deviation for Gaussian kernel

    Returns
    -------
    np.ndarray
        Filtered data in Nx2 format

    Raises
    ------
    ValueError
        If data format is invalid
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    result = data.copy()
    result[:, 1] = ndimage.gaussian_filter1d(data[:, 1], sigma=sigma)

    logger.debug(f"Applied Gaussian filter with sigma={sigma}")
    return result


def apply_median_filter(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filter to 1D data.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    kernel_size : int
        Size of the median filter kernel (must be odd)

    Returns
    -------
    np.ndarray
        Filtered data in Nx2 format

    Raises
    ------
    ValueError
        If data format is invalid or kernel_size is even
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    result = data.copy()
    result[:, 1] = ndimage.median_filter(data[:, 1], size=kernel_size)

    logger.debug(f"Applied median filter with kernel_size={kernel_size}")
    return result


def apply_bilateral_filter(
    data: np.ndarray, sigma_spatial: float = 1.0, sigma_range: Optional[float] = None
) -> np.ndarray:
    """Apply bilateral filter to 1D data.

    The bilateral filter is an edge-preserving smoothing filter that
    considers both spatial proximity and value similarity.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    sigma_spatial : float
        Standard deviation for spatial Gaussian (in index space)
    sigma_range : float, optional
        Standard deviation for range Gaussian (in value space).
        If None, uses standard deviation of the data.

    Returns
    -------
    np.ndarray
        Filtered data in Nx2 format

    Raises
    ------
    ValueError
        If data format is invalid

    Notes
    -----
    This is a simplified 1D bilateral filter implementation.
    For each point, it computes a weighted average of nearby points,
    where weights depend on both spatial distance and value difference.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    y_values = data[:, 1]

    # Use data std if sigma_range not provided
    if sigma_range is None:
        sigma_range = np.std(y_values)
        if sigma_range == 0:
            sigma_range = 1.0  # Avoid division by zero

    # Determine kernel size based on sigma_spatial
    kernel_radius = int(np.ceil(3 * sigma_spatial))

    filtered = np.zeros_like(y_values)

    for i in range(len(y_values)):
        # Define window around current point
        i_min = max(0, i - kernel_radius)
        i_max = min(len(y_values), i + kernel_radius + 1)

        # Get values in window
        window_indices = np.arange(i_min, i_max)
        window_values = y_values[window_indices]

        # Compute spatial weights (Gaussian based on distance)
        spatial_dist = np.abs(window_indices - i)
        spatial_weights = np.exp(-(spatial_dist**2) / (2 * sigma_spatial**2))

        # Compute range weights (Gaussian based on value difference)
        value_diff = np.abs(window_values - y_values[i])
        range_weights = np.exp(-(value_diff**2) / (2 * sigma_range**2))

        # Combined weights
        weights = spatial_weights * range_weights
        weights /= np.sum(weights)  # Normalize

        # Weighted average
        filtered[i] = np.sum(weights * window_values)

    result = data.copy()
    result[:, 1] = filtered

    logger.debug(
        f"Applied bilateral filter with sigma_spatial={sigma_spatial}, "
        f"sigma_range={sigma_range}"
    )
    return result


def apply_filtering(data: np.ndarray, config: FilteringConfig) -> np.ndarray:
    """Apply filtering to 1D data based on configuration.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    config : FilteringConfig
        Filtering configuration

    Returns
    -------
    np.ndarray
        Filtered data in Nx2 format

    Raises
    ------
    ValueError
        If data format is invalid or filter method is unsupported
    """
    if config.method == FilterMethod.NONE:
        return data.copy()

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    if config.method == FilterMethod.GAUSSIAN:
        return apply_gaussian_filter(data, sigma=config.sigma)

    elif config.method == FilterMethod.MEDIAN:
        return apply_median_filter(data, kernel_size=config.kernel_size)

    elif config.method == FilterMethod.BILATERAL:
        # For bilateral, use sigma as spatial sigma
        return apply_bilateral_filter(data, sigma_spatial=config.sigma)

    else:
        raise ValueError(f"Unsupported filter method: {config.method}")
