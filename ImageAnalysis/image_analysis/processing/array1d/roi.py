"""ROI (Region of Interest) processing for 1D data.

This module provides functions for applying ROI to 1D data based on x-axis values,
unlike 2D ROIs which use pixel indices.
"""

import logging
import numpy as np
from numpy.typing import NDArray

from image_analysis.processing.array1d.config_models import ROI1DConfig

logger = logging.getLogger(__name__)


def apply_roi_1d(data: NDArray, roi_config: ROI1DConfig) -> NDArray:
    """Apply ROI to 1D data based on x-axis values.

    This function filters the data to keep only points where the x-values
    fall within the specified range. Unlike 2D ROIs which use pixel indices,
    this operates on the actual x-axis values, allowing for physically
    meaningful ROI specifications (e.g., wavelength range, time window).

    Parameters
    ----------
    data : NDArray
        Nx2 array where column 0 is x-values and column 1 is y-values
    roi_config : ROI1DConfig
        ROI configuration specifying x_min and/or x_max

    Returns
    -------
    NDArray
        Filtered Nx2 array containing only points within the ROI.
        If no points fall within the ROI, returns an empty Nx2 array.

    Examples
    --------
    Apply time window to scope trace::

        roi = ROI1DConfig(x_min=0.0e-6, x_max=10.0e-6)
        filtered_data = apply_roi_1d(scope_data, roi)

    Apply wavelength range to spectrum::

        roi = ROI1DConfig(x_min=400, x_max=700)
        filtered_spectrum = apply_roi_1d(spectrum_data, roi)

    Apply only lower bound::

        roi = ROI1DConfig(x_min=0.0)  # Keep only positive x values
        filtered_data = apply_roi_1d(data, roi)
    """
    if data.shape[0] == 0:
        logger.warning("Empty data array provided to apply_roi_1d")
        return data

    # Extract x-values
    x_data = data[:, 0]

    # Create boolean mask
    mask = np.ones(len(x_data), dtype=bool)

    # Apply lower bound if specified
    if roi_config.x_min is not None:
        mask &= x_data >= roi_config.x_min
        logger.debug(f"Applied x_min={roi_config.x_min}, {mask.sum()} points remain")

    # Apply upper bound if specified
    if roi_config.x_max is not None:
        mask &= x_data <= roi_config.x_max
        logger.debug(f"Applied x_max={roi_config.x_max}, {mask.sum()} points remain")

    # Filter data
    filtered_data = data[mask]

    # Log results
    n_original = len(data)
    n_filtered = len(filtered_data)
    if n_filtered == 0:
        logger.warning(
            f"ROI filtering removed all {n_original} points. "
            f"ROI range: [{roi_config.x_min}, {roi_config.x_max}], "
            f"Data x-range: [{x_data.min():.3e}, {x_data.max():.3e}]"
        )
    else:
        logger.info(
            f"ROI filtering: kept {n_filtered}/{n_original} points "
            f"({100 * n_filtered / n_original:.1f}%)"
        )

    return filtered_data
