"""ROI (Region of Interest) processing for 1D data.

This module provides functions for applying ROI to 1D data based on x-axis values,
unlike 2D ROIs which use pixel indices.
"""

import logging
import numpy as np
from numpy.typing import NDArray

from geecs_schemas.analysis.processing_1d import LineROIConfig

logger = logging.getLogger(__name__)


def build_roi_mask_1d(x_data: NDArray, roi_config: LineROIConfig) -> NDArray:
    """Build a boolean mask for x-values inside the configured ROI."""
    mask = np.ones(len(x_data), dtype=bool)

    if roi_config.x_min is not None:
        mask &= x_data >= roi_config.x_min
        logger.debug("Applied x_min=%s, %s points remain", roi_config.x_min, mask.sum())

    if roi_config.x_max is not None:
        mask &= x_data <= roi_config.x_max
        logger.debug("Applied x_max=%s, %s points remain", roi_config.x_max, mask.sum())

    return mask


def apply_roi_1d(data: NDArray, roi_config: LineROIConfig) -> NDArray:
    """Apply ROI to 1D data based on x-axis values.

    This function filters the data to keep only points where the x-values
    fall within the specified range. Unlike 2D ROIs which use pixel indices,
    this operates on the actual x-axis values, allowing for physically
    meaningful ROI specifications (e.g., wavelength range, time window).

    Parameters
    ----------
    data : NDArray
        Nx2 array where column 0 is x-values and column 1 is y-values
    roi_config : LineROIConfig
        ROI configuration specifying x_min and/or x_max

    Returns
    -------
    NDArray
        Filtered Nx2 array containing only points within the ROI.
        If no points fall within the ROI, returns an empty Nx2 array.

    Examples
    --------
    Apply time window to scope trace::

        roi = LineROIConfig(x_min=0.0e-6, x_max=10.0e-6)
        filtered_data = apply_roi_1d(scope_data, roi)

    Apply wavelength range to spectrum::

        roi = LineROIConfig(x_min=400, x_max=700)
        filtered_spectrum = apply_roi_1d(spectrum_data, roi)

    Apply only lower bound::

        roi = LineROIConfig(x_min=0.0)  # Keep only positive x values
        filtered_data = apply_roi_1d(data, roi)
    """
    if data.shape[0] == 0:
        logger.warning("Empty data array provided to apply_roi_1d")
        return data

    # Extract x-values
    x_data = data[:, 0]

    mask = build_roi_mask_1d(x_data, roi_config)

    # Filter data
    filtered_data = data[mask]

    # Log results
    n_original = len(data)
    n_filtered = len(filtered_data)
    if n_filtered == 0:
        logger.warning(
            "ROI filtering removed all %s points. ROI range: [%s, %s], Data x-range: [%.3e, %.3e]",
            n_original,
            roi_config.x_min,
            roi_config.x_max,
            x_data.min(),
            x_data.max(),
        )
    else:
        logger.info(
            "ROI filtering: kept %s/%s points (%.1f%%)",
            n_filtered,
            n_original,
            100 * n_filtered / n_original,
        )

    return filtered_data
