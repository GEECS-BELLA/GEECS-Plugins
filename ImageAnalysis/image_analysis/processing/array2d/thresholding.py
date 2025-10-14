"""
Thresholding operations for image processing.

This module provides functions for applying various thresholding operations
to images, including constant value thresholds and percentage-based thresholds.
"""

import numpy as np
import logging
from typing import Literal
from ...types import Array2D

logger = logging.getLogger(__name__)

ThresholdMethod = Literal["constant", "percentage_max"]
ThresholdMode = Literal["binary", "to_zero", "truncate", "to_zero_inv", "truncate_inv"]


def apply_constant_threshold(
    image: Array2D,
    threshold_value: float,
    mode: ThresholdMode = "binary",
    invert: bool = False,
) -> Array2D:
    """
    Apply constant value threshold to image.

    Parameters
    ----------
    image : Array2D
        Input image to threshold.
    threshold_value : float
        Constant threshold value.
    mode : ThresholdMode
        Thresholding mode to apply.
    invert : bool
        Whether to invert the threshold operation.

    Returns
    -------
    Array2D
        Thresholded image.


    """
    image = np.asarray(image, dtype=np.float64)

    # Determine max value for binary operations
    if image.dtype == np.uint8:
        max_val = 255.0
    elif image.dtype == np.uint16:
        max_val = 65535.0
    else:
        max_val = np.max(image) if np.max(image) > 0 else 1.0

    if mode == "binary":
        # Binary threshold: pixels >= threshold become max_val, others become 0
        result = np.where(image >= threshold_value, max_val, 0.0)

    elif mode == "to_zero":
        # To zero: pixels < threshold become 0, others unchanged
        result = np.where(image >= threshold_value, image, 0.0)

    elif mode == "truncate":
        # Truncate: pixels > threshold become threshold, others unchanged
        result = np.where(image > threshold_value, threshold_value, image)

    elif mode == "to_zero_inv":
        # Inverse to zero: pixels >= threshold become 0, others unchanged
        result = np.where(image >= threshold_value, 0.0, image)

    elif mode == "truncate_inv":
        # Inverse truncate: pixels <= threshold become threshold, others unchanged
        result = np.where(image <= threshold_value, threshold_value, image)

    else:
        raise ValueError(f"Unknown threshold mode: {mode}")

    # Apply inversion if requested
    if invert:
        if mode == "binary":
            result = max_val - result
        else:
            # For non-binary modes, invert by subtracting from max
            result = max_val - result

    logger.debug(
        f"Applied constant threshold {threshold_value} with mode '{mode}', invert={invert}"
    )

    return result


def apply_percentage_threshold(
    image: Array2D,
    percentage: float,
    mode: ThresholdMode = "binary",
    invert: bool = False,
) -> Array2D:
    """
    Apply percentage-based threshold to image.

    The threshold value is calculated as a percentage of the maximum pixel value
    in the image.

    Parameters
    ----------
    image : Array2D
        Input image to threshold.
    percentage : float
        Percentage of maximum value to use as threshold (0-100).
    mode : ThresholdMode
        Thresholding mode to apply.
    invert : bool
        Whether to invert the threshold operation.

    Returns
    -------
    Array2D
        Thresholded image.


    """
    image = np.asarray(image, dtype=np.float64)

    if not 0 <= percentage <= 100:
        raise ValueError(f"Percentage must be between 0 and 100, got {percentage}")

    # Calculate threshold as percentage of max value
    max_value = np.max(image)
    threshold_value = (percentage / 100.0) * max_value

    logger.debug(
        f"Calculated threshold {threshold_value} from {percentage}% of max value {max_value}"
    )

    return apply_constant_threshold(image, threshold_value, mode, invert)


def apply_threshold(
    image: Array2D,
    method: ThresholdMethod,
    value: float,
    mode: ThresholdMode = "binary",
    invert: bool = False,
) -> Array2D:
    """
    Apply threshold to image using specified method.

    This is the main thresholding function that dispatches to the appropriate
    method-specific function.

    Parameters
    ----------
    image : Array2D
        Input image to threshold.
    method : ThresholdMethod
        Thresholding method ("constant" or "percentage_max").
    value : float
        Threshold value (absolute for constant, percentage for percentage_max).
    mode : ThresholdMode
        Thresholding mode to apply.
    invert : bool
        Whether to invert the threshold operation.

    Returns
    -------
    Array2D
        Thresholded image.

    Raises
    ------
    ValueError
        If method is not recognized.


    """
    if method == "constant":
        return apply_constant_threshold(image, value, mode, invert)

    elif method == "percentage_max":
        return apply_percentage_threshold(image, value, mode, invert)

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def get_threshold_value(image: Array2D, method: ThresholdMethod, value: float) -> float:
    """
    Calculate the actual threshold value that would be applied.

    This function is useful for debugging or displaying what threshold
    value will be used without actually applying the threshold.

    Parameters
    ----------
    image : Array2D
        Input image.
    method : ThresholdMethod
        Thresholding method.
    value : float
        Threshold parameter.

    Returns
    -------
    float
        Actual threshold value that would be applied.


    """
    image = np.asarray(image, dtype=np.float64)

    if method == "constant":
        return float(value)

    elif method == "percentage_max":
        if not 0 <= value <= 100:
            raise ValueError(f"Percentage must be between 0 and 100, got {value}")
        max_value = np.max(image)
        return (value / 100.0) * max_value

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def validate_threshold_parameters(
    method: ThresholdMethod, value: float, mode: ThresholdMode
) -> None:
    """
    Validate threshold parameters.

    Parameters
    ----------
    method : ThresholdMethod
        Thresholding method.
    value : float
        Threshold value.
    mode : ThresholdMode
        Thresholding mode.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    valid_methods = ["constant", "percentage_max"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")

    valid_modes = ["binary", "to_zero", "truncate", "to_zero_inv", "truncate_inv"]
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}, got '{mode}'")

    if method == "percentage_max":
        if not 0 <= value <= 100:
            raise ValueError(f"Percentage value must be between 0 and 100, got {value}")

    elif method == "constant":
        if value < 0:
            raise ValueError(
                f"Constant threshold value must be non-negative, got {value}"
            )
