"""Vignette correction utilities for 2D image processing."""

from pathlib import Path
import numpy as np

from ...types import Array2D
from .config_models import VignetteConfig, VignetteMethod


def build_radial_vignette_map(
    image_shape: tuple[int, int],
    full_width: int,
    full_height: int,
    x_offset: int = 0,
    y_offset: int = 0,
    vgnt4: float = 0.0,
    vgnt2: float = 0.0,
    vgnt0: float = 1.0,
    min_model_value: float = 1e-9,
) -> Array2D:
    """
    Build a MATLAB-compatible radial vignette correction map.

    The coordinate convention mirrors the original MATLAB implementation:
    centered coordinate = pixel_index_1_based - full_size/2 + 0.5

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of the saved image `(height, width)`.
    full_width : int
        Full sensor width in pixels.
    full_height : int
        Full sensor height in pixels.
    x_offset : int, default=0
        Saved image x-offset in full-sensor pixels.
    y_offset : int, default=0
        Saved image y-offset in full-sensor pixels.
    vgnt4 : float, default=0.0
        4th-order radial coefficient.
    vgnt2 : float, default=0.0
        2nd-order radial coefficient.
    vgnt0 : float, default=1.0
        0th-order radial coefficient.
    min_model_value : float, default=1e-9
        Lower bound on attenuation model before inversion.

    Returns
    -------
    Array2D
        Multiplicative correction map with shape `image_shape`.
    """
    height, width = image_shape

    x_full_1_based = x_offset + np.arange(width, dtype=float) + 1.0
    y_full_1_based = y_offset + np.arange(height, dtype=float) + 1.0

    x_centered = x_full_1_based - (full_width / 2.0) + 0.5
    y_centered = y_full_1_based - (full_height / 2.0) + 0.5
    xx, yy = np.meshgrid(x_centered, y_centered)
    radius = np.sqrt(xx**2 + yy**2)

    attenuation = vgnt4 * radius**4 + vgnt2 * radius**2 + vgnt0
    attenuation = np.maximum(attenuation, min_model_value)
    return 1.0 / attenuation


def _resolve_vignette_map_path(map_file_path: str) -> Path:
    """Resolve vignette map path, with package-root fallback for relatives."""
    path = Path(map_file_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(__file__).resolve().parents[2] / path


def load_vignette_map_from_file(
    map_file_path: str,
    image_shape: tuple[int, int],
    full_width: int | None = None,
    full_height: int | None = None,
    x_offset: int = 0,
    y_offset: int = 0,
) -> Array2D:
    """
    Load a vignette map from file and adapt it to image shape.

    Supports either:
    - direct map matching saved image shape, or
    - full-sensor map cropped using full dimensions and offsets.
    """
    map_path = _resolve_vignette_map_path(map_file_path)
    vignette_map = np.load(map_path)

    if vignette_map.ndim != 2:
        raise ValueError(
            f"Vignette map must be 2D, got shape {vignette_map.shape} from {map_path}"
        )

    image_h, image_w = image_shape
    if vignette_map.shape == (image_h, image_w):
        return vignette_map

    if (
        full_width is not None
        and full_height is not None
        and vignette_map.shape == (full_height, full_width)
    ):
        y_end = y_offset + image_h
        x_end = x_offset + image_w
        if y_end > full_height or x_end > full_width:
            raise ValueError(
                "Saved image offsets + shape exceed full-sensor map dimensions"
            )
        return vignette_map[y_offset:y_end, x_offset:x_end]

    raise ValueError(
        "Vignette map shape mismatch. "
        f"Map shape={vignette_map.shape}, image shape={(image_h, image_w)}. "
        "Provide either an image-shaped map, or a full-sensor map with "
        "full_width/full_height and offsets."
    )


def apply_vignette_map(image: Array2D, vignette_map: Array2D) -> Array2D:
    """Apply a multiplicative vignette correction map."""
    if image.shape != vignette_map.shape:
        raise ValueError(
            f"Vignette map shape {vignette_map.shape} does not match image shape {image.shape}"
        )
    return image * vignette_map


def apply_vignette_config(image: Array2D, config: VignetteConfig) -> Array2D:
    """
    Apply vignette correction according to configuration.

    Returns the original image unchanged when correction is disabled.
    """
    if not config.enabled:
        return image

    if config.method == VignetteMethod.RADIAL_POLYNOMIAL:
        vignette_map = build_radial_vignette_map(
            image_shape=image.shape,
            full_width=int(config.full_width),
            full_height=int(config.full_height),
            x_offset=config.x_offset,
            y_offset=config.y_offset,
            vgnt4=config.vgnt4,
            vgnt2=config.vgnt2,
            vgnt0=config.vgnt0,
            min_model_value=config.min_model_value,
        )
    elif config.method == VignetteMethod.MAP_FILE:
        vignette_map = load_vignette_map_from_file(
            map_file_path=str(config.map_file_path),
            image_shape=image.shape,
            full_width=config.full_width,
            full_height=config.full_height,
            x_offset=config.x_offset,
            y_offset=config.y_offset,
        )
    else:
        raise ValueError(f"Unsupported vignette method: {config.method}")

    return apply_vignette_map(image, vignette_map)
