"""Interpolation operations for 1D array data.

This module provides interpolation functions to map unevenly-spaced 1D data
onto a uniform grid. This is particularly useful for waterfall plot rendering
where evenly-spaced x-axes are expected.

Data Format:
    All functions expect Nx2 numpy arrays where:
    - Column 0: x-values (independent variable)
    - Column 1: y-values (dependent variable)
"""

from __future__ import annotations

import logging

import numpy as np

from .config_models import InterpolationConfig

logger = logging.getLogger(__name__)


def apply_interpolation(data: np.ndarray, config: InterpolationConfig) -> np.ndarray:
    """Apply interpolation to map 1D data onto a uniform x-axis grid.

    This function takes unevenly-spaced 1D data and interpolates it onto
    an evenly-spaced grid. This is useful for creating consistent waterfall
    plots where all traces should share the same x-axis.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    config : InterpolationConfig
        Interpolation configuration

    Returns
    -------
    np.ndarray
        Interpolated data in Mx2 format where M = config.num_points

    Raises
    ------
    ValueError
        If data format is invalid

    Notes
    -----
    - If config.enabled is False, returns a copy of the original data
    - Uses linear interpolation via np.interp
    - Automatically determines x_min/x_max from data if not specified
    - Returns original data (with warning) if interpolation fails
    """
    if not config.enabled:
        return data.copy()

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    if len(data) == 0:
        logger.warning("Cannot interpolate empty data array")
        return data.copy()

    try:
        x_values = data[:, 0]
        y_values = data[:, 1]

        # Determine interpolation range
        x_min = config.x_min if config.x_min is not None else np.min(x_values)
        x_max = config.x_max if config.x_max is not None else np.max(x_values)

        # Create uniform x-axis
        uniform_x = np.linspace(x_min, x_max, config.num_points)

        # Interpolate y-values onto uniform grid
        interpolated_y = np.interp(uniform_x, x_values, y_values)

        # Combine into Nx2 array
        result = np.column_stack([uniform_x, interpolated_y])

        logger.debug(
            f"Interpolated data from {len(data)} to {config.num_points} points "
            f"over range [{x_min:.3e}, {x_max:.3e}]"
        )

        return result

    except Exception as e:
        logger.warning(
            f"Interpolation failed: {e}. Returning original data.",
            exc_info=True,
        )
        return data.copy()


def check_spacing_uniformity(
    data: np.ndarray, tolerance: float = 1e-6
) -> tuple[bool, float]:
    """Check if x-values in data are uniformly spaced.

    Parameters
    ----------
    data : np.ndarray
        Input data in Nx2 format (x, y)
    tolerance : float, default=1e-6
        Relative tolerance for determining uniformity

    Returns
    -------
    is_uniform : bool
        True if spacing is uniform within tolerance
    max_deviation : float
        Maximum relative deviation from mean spacing

    Notes
    -----
    This utility function can be used to determine if interpolation
    is actually needed for a given dataset.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array, got shape {data.shape}")

    if len(data) < 3:
        return True, 0.0  # Too few points to determine non-uniformity

    x_values = data[:, 0]
    spacings = np.diff(x_values)

    if len(spacings) == 0:
        return True, 0.0

    mean_spacing = np.mean(spacings)

    if mean_spacing == 0:
        # All x-values are the same
        return True, 0.0

    relative_deviations = np.abs(spacings - mean_spacing) / mean_spacing
    max_deviation = np.max(relative_deviations)

    is_uniform = max_deviation <= tolerance

    return is_uniform, max_deviation
