"""
Geometric image transformation operations for image processing.

This module provides functions for applying geometric transformations to images
including rotation, flipping, and distortion correction.
All functions take images as input and return processed images.
"""

import numpy as np
import logging
from typing import Optional
from scipy.ndimage import rotate
from ..types import Array2D
from .config_models import TransformConfig

logger = logging.getLogger(__name__)


def apply_rotation(
    image: Array2D, angle: float, reshape: bool = False, fill_value: float = 0.0
) -> Array2D:
    """
    Rotate image by specified angle.

    Parameters
    ----------
    image : Array2D
        Input image to rotate.
    angle : float
        Rotation angle in degrees. Positive values rotate counterclockwise.
    reshape : bool, optional
        If True, expand output to fit entire rotated image. If False,
        keep original image size. Default is False.
    fill_value : float, optional
        Value to use for pixels outside the original image boundary.
        Default is 0.0.

    Returns
    -------
    Array2D
        Rotated image.


    """
    if angle == 0.0:
        return image.copy()

    # Apply rotation using scipy
    rotated_image = rotate(
        image,
        angle,
        reshape=reshape,
        cval=fill_value,
        prefilter=False,  # Avoid additional smoothing
    )

    return rotated_image.astype(image.dtype)


def apply_horizontal_flip(image: Array2D) -> Array2D:
    """
    Flip image horizontally (left-right).

    Parameters
    ----------
    image : Array2D
        Input image to flip.

    Returns
    -------
    Array2D
        Horizontally flipped image.


    """
    return np.fliplr(image)


def apply_vertical_flip(image: Array2D) -> Array2D:
    """
    Flip image vertically (up-down).

    Parameters
    ----------
    image : Array2D
        Input image to flip.

    Returns
    -------
    Array2D
        Vertically flipped image.


    """
    return np.flipud(image)


def apply_transform_config(image: Array2D, config: TransformConfig) -> Array2D:
    """
    Apply geometric transformations based on configuration.

    This function applies multiple transformations in sequence based on
    the provided configuration. Transformations are applied in the order:
    rotation, horizontal flip, vertical flip, distortion correction.

    Parameters
    ----------
    image : Array2D
        Input image to transform.
    config : TransformConfig
        Configuration specifying which transformations to apply.

    Returns
    -------
    Array2D
        Transformed image with all specified transformations applied.


    """
    transformed_image = image.copy()

    # Apply rotation if specified
    if config.rotation_angle != 0.0:
        transformed_image = apply_rotation(transformed_image, config.rotation_angle)
        logger.debug(f"Applied rotation: {config.rotation_angle} degrees")

    # Apply horizontal flip if specified
    if config.flip_horizontal:
        transformed_image = apply_horizontal_flip(transformed_image)
        logger.debug("Applied horizontal flip")

    # Apply vertical flip if specified
    if config.flip_vertical:
        transformed_image = apply_vertical_flip(transformed_image)
        logger.debug("Applied vertical flip")

    return transformed_image


def create_rotation_matrix(angle: float, center: Optional[tuple] = None) -> np.ndarray:
    """
    Create 2D rotation matrix for specified angle and center.

    Parameters
    ----------
    angle : float
        Rotation angle in degrees.
    center : tuple, optional
        Center of rotation (x, y). If None, assumes origin (0, 0).

    Returns
    -------
    np.ndarray
        3x3 transformation matrix for rotation about specified center.


    """
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    if center is None:
        # Simple rotation about origin
        return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    else:
        # Rotation about specified center
        cx, cy = center
        return np.array(
            [
                [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
                [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
                [0, 0, 1],
            ]
        )


def create_translation_matrix(tx: float, ty: float) -> np.ndarray:
    """
    Create 2D translation matrix.

    Parameters
    ----------
    tx : float
        Translation in X direction.
    ty : float
        Translation in Y direction.

    Returns
    -------
    np.ndarray
        3x3 transformation matrix for translation.


    """
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
