"""
Axis interpolation utilities for remapping image coordinates.

Useful for energy-dispersed spectra, wavelength-dispersed data, or any
case where pixel coordinates need to be mapped to non-uniform physical units.
"""

import numpy as np
import logging
from typing import Tuple, Optional
from ..types import Array2D

logger = logging.getLogger(__name__)


def interpolate_image_axis(
    image: Array2D,
    pixel_to_physical: np.ndarray,
    axis: int = 1,
    num_points: int = 1500,
    physical_min: Optional[float] = None,
    physical_max: Optional[float] = None,
) -> Tuple[Array2D, np.ndarray]:
    """
    Interpolate image along one axis to uniform physical coordinates.

    Performs row-by-row (or column-by-column) interpolation to remap pixel
    indices to a uniformly-spaced physical coordinate grid.

    Parameters
    ----------
    image : Array2D
        Input image to interpolate.
    pixel_to_physical : np.ndarray
        1D array mapping pixel indices to physical coordinates.
        Length must match image dimension along specified axis.
        Example: pixel_to_physical[0] = energy at pixel 0
    axis : int, default=1
        Axis to interpolate along (0=vertical/rows, 1=horizontal/cols).
    num_points : int, default=1500
        Number of points in output uniform grid.
    physical_min : float, optional
        Minimum physical coordinate. Auto-detected if None.
    physical_max : float, optional
        Maximum physical coordinate. Auto-detected if None.

    Returns
    -------
    tuple of (Array2D, np.ndarray)
        - Interpolated image with shape (height, num_points) for axis=1
          or (num_points, width) for axis=0
        - Uniform physical coordinate axis (1D array of length num_points)

    Notes
    -----
    This function uses numpy.interp for linear interpolation, which is fast
    and efficient for this use case. For higher-order interpolation, consider
    using scipy.interpolate functions.
    """
    # Validate inputs
    expected_length = image.shape[axis]
    if len(pixel_to_physical) != expected_length:
        raise ValueError(
            f"pixel_to_physical length ({len(pixel_to_physical)}) must match "
            f"image dimension along axis {axis} ({expected_length})"
        )

    # Get calibration range
    calib_min = np.min(pixel_to_physical)
    calib_max = np.max(pixel_to_physical)

    # Determine physical range
    if physical_min is None:
        physical_min = calib_min
    if physical_max is None:
        physical_max = calib_max

    # Clip to calibration range and warn if clipping occurred
    if physical_min < calib_min:
        logger.warning(
            f"Requested physical_min ({physical_min:.3f}) below calibration range "
            f"({calib_min:.3f}). Clipping to calibration minimum."
        )
        physical_min = calib_min

    if physical_max > calib_max:
        logger.warning(
            f"Requested physical_max ({physical_max:.3f}) above calibration range "
            f"({calib_max:.3f}). Clipping to calibration maximum."
        )
        physical_max = calib_max

    # Create uniform physical axis
    uniform_axis = np.linspace(physical_min, physical_max, num_points)

    # Interpolate
    if axis == 1:  # Horizontal (typical for energy dispersion)
        # Interpolate each row
        output = np.zeros((image.shape[0], num_points), dtype=image.dtype)
        for i in range(image.shape[0]):
            output[i] = np.interp(uniform_axis, pixel_to_physical, image[i, :])
        logger.debug(
            f"Interpolated {image.shape[0]} rows from {image.shape[1]} to {num_points} points"
        )
    else:  # axis == 0, Vertical
        # Interpolate each column
        output = np.zeros((num_points, image.shape[1]), dtype=image.dtype)
        for j in range(image.shape[1]):
            output[:, j] = np.interp(uniform_axis, pixel_to_physical, image[:, j])
        logger.debug(
            f"Interpolated {image.shape[1]} columns from {image.shape[0]} to {num_points} points"
        )

    return output, uniform_axis
