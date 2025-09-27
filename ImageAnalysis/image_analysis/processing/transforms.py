"""
Geometric image transformation operations for image processing.

This module provides functions for applying geometric transformations to images
including rotation, flipping, and distortion correction.
All functions take images as input and return processed images.
"""

import numpy as np
import logging
from typing import Optional, List
from scipy import ndimage
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


def apply_distortion_correction(
    image: Array2D,
    distortion_coeffs: List[float],
    camera_matrix: Optional[np.ndarray] = None,
) -> Array2D:
    """
    Apply distortion correction to image.

    This function corrects lens distortion using radial and tangential
    distortion coefficients. This is a simplified implementation.

    Parameters
    ----------
    image : Array2D
        Input image to correct.
    distortion_coeffs : List[float]
        Distortion coefficients [k1, k2, p1, p2, k3] where:
        - k1, k2, k3: radial distortion coefficients
        - p1, p2: tangential distortion coefficients
    camera_matrix : np.ndarray, optional
        Camera intrinsic matrix. If None, assumes centered principal point
        and unit focal length.

    Returns
    -------
    Array2D
        Distortion-corrected image.

    Notes
    -----
    This is a simplified distortion correction. For production use,
    consider using OpenCV's cv2.undistort for better performance and accuracy.


    """
    if not distortion_coeffs:
        return image.copy()

    height, width = image.shape

    # Default camera matrix if not provided
    if camera_matrix is None:
        fx = fy = min(width, height) / 2  # Approximate focal length
        cx, cy = width / 2, height / 2  # Principal point at center
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Extract camera parameters
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Extract distortion coefficients
    k1 = distortion_coeffs[0] if len(distortion_coeffs) > 0 else 0.0
    k2 = distortion_coeffs[1] if len(distortion_coeffs) > 1 else 0.0
    p1 = distortion_coeffs[2] if len(distortion_coeffs) > 2 else 0.0
    p2 = distortion_coeffs[3] if len(distortion_coeffs) > 3 else 0.0
    k3 = distortion_coeffs[4] if len(distortion_coeffs) > 4 else 0.0

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Normalize coordinates
    x_norm = (x_coords - cx) / fx
    y_norm = (y_coords - cy) / fy

    # Calculate radial distance
    r_squared = x_norm**2 + y_norm**2
    r_fourth = r_squared**2
    r_sixth = r_squared * r_fourth

    # Radial distortion factor
    radial_factor = 1 + k1 * r_squared + k2 * r_fourth + k3 * r_sixth

    # Tangential distortion
    tangential_x = 2 * p1 * x_norm * y_norm + p2 * (r_squared + 2 * x_norm**2)
    tangential_y = p1 * (r_squared + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm

    # Apply distortion correction
    x_corrected = x_norm * radial_factor + tangential_x
    y_corrected = y_norm * radial_factor + tangential_y

    # Convert back to pixel coordinates
    x_pixel = x_corrected * fx + cx
    y_pixel = y_corrected * fy + cy

    # Interpolate corrected image
    corrected_image = ndimage.map_coordinates(
        image.astype(np.float64),
        [y_pixel, x_pixel],
        order=1,  # Linear interpolation
        cval=0.0,
        prefilter=False,
    )

    return corrected_image.astype(image.dtype)


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

    # Apply distortion correction if specified
    if config.distortion_correction and config.distortion_coeffs:
        transformed_image = apply_distortion_correction(
            transformed_image, config.distortion_coeffs
        )
        logger.debug("Applied distortion correction")

    return transformed_image


def apply_affine_transform(
    image: Array2D, transform_matrix: np.ndarray, output_shape: Optional[tuple] = None
) -> Array2D:
    """
    Apply affine transformation to image.

    Parameters
    ----------
    image : Array2D
        Input image to transform.
    transform_matrix : np.ndarray
        2x3 or 3x3 affine transformation matrix.
    output_shape : tuple, optional
        Shape of output image (height, width). If None, uses input shape.

    Returns
    -------
    Array2D
        Affine-transformed image.


    """
    if output_shape is None:
        output_shape = image.shape

    # Ensure we have a 2x3 matrix
    if transform_matrix.shape == (3, 3):
        transform_matrix = transform_matrix[:2, :]
    elif transform_matrix.shape != (2, 3):
        raise ValueError("Transform matrix must be 2x3 or 3x3")

    # Apply affine transformation
    transformed_image = ndimage.affine_transform(
        image.astype(np.float64),
        transform_matrix[:2, :2],  # Linear part
        offset=transform_matrix[:2, 2],  # Translation part
        output_shape=output_shape,
        cval=0.0,
        prefilter=False,
    )

    return transformed_image.astype(image.dtype)


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


def create_scaling_matrix(
    scale_x: float, scale_y: Optional[float] = None
) -> np.ndarray:
    """
    Create 2D scaling matrix.

    Parameters
    ----------
    scale_x : float
        Scaling factor in X direction.
    scale_y : float, optional
        Scaling factor in Y direction. If None, uses scale_x (uniform scaling).

    Returns
    -------
    np.ndarray
        3x3 transformation matrix for scaling.


    """
    if scale_y is None:
        scale_y = scale_x

    return np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])


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
