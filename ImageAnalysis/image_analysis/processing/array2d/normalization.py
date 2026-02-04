"""Image normalization utilities.

Provides functions to normalize images by various methods including total intensity,
peak value, and constant divisor normalization.
"""

import logging
import numpy as np
from ...types import Array2D
from .config_models import NormalizationConfig

logger = logging.getLogger(__name__)


def apply_normalization(
    image: Array2D, config: NormalizationConfig
) -> Array2D:
    """Apply normalization to image based on configuration.

    Normalizes the image by dividing by a normalization factor determined by
    the configured method. Handles edge cases gracefully by returning the
    original image if normalization cannot be performed.

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
    - "total_intensity": Divides by sum of all pixel values
    - "peak_value": Divides by maximum pixel value
    - "constant": Divides by specified constant value
    """
    if config.method == "total_intensity":
        total = image.sum()
        if total > 0:
            logger.debug(f"Normalizing by total intensity: {total:.6e}")
            return image / total
        else:
            logger.warning(
                "Image has zero total intensity, skipping normalization"
            )
            return image

    elif config.method == "peak_value":
        peak = image.max()
        if peak > 0:
            logger.debug(f"Normalizing by peak value: {peak:.6e}")
            return image / peak
        else:
            logger.warning("Image has zero peak value, skipping normalization")
            return image

    elif config.method == "constant":
        if config.constant_value is None or config.constant_value == 0:
            logger.warning(
                "Constant normalization requires non-zero constant_value, skipping"
            )
            return image
        logger.debug(f"Normalizing by constant: {config.constant_value}")
        return image / config.constant_value

    else:
        logger.warning(f"Unknown normalization method: {config.method}")
        return image
