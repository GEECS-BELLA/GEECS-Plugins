"""Image normalization utilities.

Provides functions to normalize images by various methods including total intensity,
peak value, and constant divisor normalization. Each normalization method is
implemented as a separate function for clarity and testability.
"""

import logging
import numpy as np
from ...types import Array2D
from .config_models import NormalizationConfig, NormalizationMethod

logger = logging.getLogger(__name__)


def apply_image_total_normalization(image: Array2D) -> Array2D:
    """Normalize image by dividing by total intensity (sum of all pixels).

    Parameters
    ----------
    image : Array2D
        Input image to normalize (should be float64)

    Returns
    -------
    Array2D
        Normalized image. Returns original image if total intensity is zero.

    Notes
    -----
    This method divides the image by the sum of all pixel values, resulting
    in an image where the total intensity equals 1.0.
    """
    total = image.sum()
    if total > 0:
        logger.debug(f"Normalizing by total intensity: {total:.6e}")
        return image / total
    else:
        logger.warning(
            "Image has zero total intensity, skipping normalization"
        )
        return image


def apply_image_max_normalization(image: Array2D) -> Array2D:
    """Normalize image by dividing by peak value (maximum pixel).

    Parameters
    ----------
    image : Array2D
        Input image to normalize (should be float64)

    Returns
    -------
    Array2D
        Normalized image. Returns original image if peak value is zero.

    Notes
    -----
    This method divides the image by the maximum pixel value, resulting
    in an image where the peak intensity equals 1.0.
    """
    peak = image.max()
    if peak > 0:
        logger.debug(f"Normalizing by peak value: {peak:.6e}")
        return image / peak
    else:
        logger.warning("Image has zero peak value, skipping normalization")
        return image


def apply_constant_normalization(
    image: Array2D, constant_value: float
) -> Array2D:
    """Normalize image by dividing by a constant value.

    Parameters
    ----------
    image : Array2D
        Input image to normalize (should be float64)
    constant_value : float
        Divisor value. Must be non-zero.

    Returns
    -------
    Array2D
        Normalized image. Returns original image if constant_value is zero.

    Notes
    -----
    This method divides the image by a user-specified constant value.
    Useful for normalizing to a known reference intensity.
    """
    if constant_value == 0:
        logger.warning(
            "Constant normalization requires non-zero value, skipping"
        )
        return image
    logger.debug(f"Normalizing by constant: {constant_value}")
    return image / constant_value


def apply_normalization(
    image: Array2D, config: NormalizationConfig
) -> Array2D:
    """Apply normalization to image based on configuration.

    This is the main dispatcher function that routes to the appropriate
    normalization method based on the configuration.

    Parameters
    ----------
    image : Array2D
        Input image to normalize (should be float64)
    config : NormalizationConfig
        Normalization configuration specifying method and parameters

    Returns
    -------
    Array2D
        Normalized image with same shape and dtype as input.
        Returns original image if normalization cannot be performed.

    Notes
    -----
    Supported methods:
    - IMAGE_TOTAL: Divides by sum of all pixel values
    - IMAGE_MAX: Divides by maximum pixel value
    - CONSTANT: Divides by specified constant value
    """
    if config.method == NormalizationMethod.IMAGE_TOTAL:
        return apply_image_total_normalization(image)

    elif config.method == NormalizationMethod.IMAGE_MAX:
        return apply_image_max_normalization(image)

    elif config.method == NormalizationMethod.CONSTANT:
        return apply_constant_normalization(image, config.constant_value)

    else:
        logger.warning(f"Unknown normalization method: {config.method}")
        return image
