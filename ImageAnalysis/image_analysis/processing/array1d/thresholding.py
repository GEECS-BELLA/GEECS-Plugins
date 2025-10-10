"""Thresholding operations for 1D array data.

This module provides thresholding functions for 1D data (lineouts, spectra, etc.).

Data Format:
    All functions expect Nx2 numpy arrays where:
    - Column 0: x-values (independent variable)
    - Column 1: y-values (dependent variable)

    Thresholding is applied only to the y-values (column 1).
"""

from __future__ import annotations

import logging

import numpy as np

from .config_models import ThresholdingConfig, ThresholdMethod

logger = logging.getLogger(__name__)


def apply_thresholding(data: np.ndarray, config: ThresholdingConfig) -> np.ndarray:
    """Apply thresholding to 1D data based on configuration.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    config : ThresholdingConfig
        Thresholding configuration

    Returns
    -------
    np.ndarray
        Thresholded data in Nx2 format

    Raises
    ------
    ValueError
        If data format is invalid or threshold method is unsupported
    """
    if config.method == ThresholdMethod.NONE:
        return data.copy()

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    # Determine threshold value
    if config.method == ThresholdMethod.ABSOLUTE:
        threshold = config.threshold_value
    elif config.method == ThresholdMethod.PERCENTILE:
        threshold = np.percentile(data[:, 1], config.percentile)
    else:
        raise ValueError(f"Unsupported threshold method: {config.method}")

    # Apply threshold
    result = data.copy()
    if config.clip_below:
        # Clip values below threshold (set to threshold)
        result[:, 1] = np.maximum(data[:, 1], threshold)
        logger.debug(f"Clipped values below threshold={threshold}")
    else:
        # Clip values above threshold (set to threshold)
        result[:, 1] = np.minimum(data[:, 1], threshold)
        logger.debug(f"Clipped values above threshold={threshold}")

    return result


def find_threshold_crossings(
    data: np.ndarray, threshold: float, direction: str = "both"
) -> np.ndarray:
    """Find indices where data crosses a threshold.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    threshold : float
        Threshold value to detect crossings
    direction : str
        Direction of crossing to detect:
        - "rising": detect rising edge (below to above threshold)
        - "falling": detect falling edge (above to below threshold)
        - "both": detect both rising and falling edges

    Returns
    -------
    np.ndarray
        Array of indices where crossings occur

    Raises
    ------
    ValueError
        If data format is invalid or direction is unsupported
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    if direction not in ["rising", "falling", "both"]:
        raise ValueError(
            f"Invalid direction: {direction}. Use 'rising', 'falling', or 'both'"
        )

    y_values = data[:, 1]

    # Find where values are above threshold
    above = y_values > threshold

    # Find transitions
    if direction == "rising":
        # Rising edge: False to True
        crossings = np.where(np.diff(above.astype(int)) == 1)[0] + 1
    elif direction == "falling":
        # Falling edge: True to False
        crossings = np.where(np.diff(above.astype(int)) == -1)[0] + 1
    else:  # both
        # Any transition
        crossings = np.where(np.diff(above.astype(int)) != 0)[0] + 1

    logger.debug(
        f"Found {len(crossings)} {direction} crossings at threshold={threshold}"
    )
    return crossings
