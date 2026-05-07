"""Synthetic data generators for testing and benchmarking image analysis algorithms.

Each generator returns a numpy array that mimics a class of real experimental
data with controllable ground-truth parameters, making it possible to write
tests that assert algorithm outputs against known inputs.
"""

import numpy as np
from scipy.ndimage import rotate
from typing import Tuple, Optional


def gaussian_beam_2d(
    shape: Tuple[int, int] = (256, 256),
    center: Tuple[float, float] = (128.0, 128.0),
    sigma: Tuple[float, float] = (20.0, 20.0),
    peak_value: float = 3000.0,
    background_level: float = 0.0,
    noise_level: float = 0.0,
    bit_depth_max: int = 65535,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a synthetic 2D Gaussian beam image.

    Produces a clean, axis-aligned Gaussian spot at a known position with
    controllable size, background, and noise.  Useful for testing any analyzer
    that reports beam centroid, width, or total-intensity scalars.

    Parameters
    ----------
    shape : tuple of int
        Image shape ``(height, width)``.
    center : tuple of float
        Beam centre ``(row, col)`` in pixel coordinates.
    sigma : tuple of float
        Gaussian standard deviation ``(sigma_row, sigma_col)`` in pixels.
    peak_value : float
        Peak intensity of the Gaussian before background is added.
    background_level : float
        Uniform background offset added to every pixel.
    noise_level : float
        Standard deviation of Gaussian noise added after background.
    bit_depth_max : int
        Output values are clipped to ``[0, bit_depth_max]``.
    seed : int, optional
        Random seed for reproducible noise.

    Returns
    -------
    np.ndarray
        Synthetic image, dtype ``uint16``.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    h, w = shape
    cy, cx = center
    sy, sx = sigma

    ys = np.arange(h, dtype=float)
    xs = np.arange(w, dtype=float)
    xx, yy = np.meshgrid(xs, ys)

    img = peak_value * np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    img += background_level

    if noise_level > 0:
        img += noise_level * rng.standard_normal(img.shape)

    return np.clip(img, 0, bit_depth_max).astype(np.uint16)


def gaussian_peak_1d(
    x: np.ndarray,
    center: float,
    sigma: float,
    amplitude: float = 1.0,
    background_level: float = 0.0,
    noise_level: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a synthetic 1D Gaussian peak on a given x-axis.

    Useful for testing line analyzers (LineAnalyzer, ICT1DAnalyzer, etc.)
    against known centroid, width, and amplitude values.

    Parameters
    ----------
    x : np.ndarray
        Independent-axis values (1D array).
    center : float
        Centre of the Gaussian peak, in the same units as ``x``.
    sigma : float
        Standard deviation of the Gaussian, in the same units as ``x``.
    amplitude : float
        Peak amplitude above background.
    background_level : float
        Uniform baseline offset.
    noise_level : float
        Standard deviation of Gaussian noise added to the output.
    seed : int, optional
        Random seed for reproducible noise.

    Returns
    -------
    np.ndarray
        1D array of y-values, dtype ``float64``.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    y = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background_level

    if noise_level > 0:
        y += noise_level * rng.standard_normal(y.shape)

    return y.astype(np.float64)


def generate_bowtie_image(
    shape: Tuple[int, int] = (256, 1256),
    min_sigma: float = 5.0,
    divergence: float = 0.05,
    angle_deg: float = 0.0,
    vertical_offset: int = 0,
    total_charge: float = 1.0,
    noise_level: float = 50.0,
    bit_depth_max: int = 4095,
    energy_spread: float = 100.0,
    background_level: float = 40,
    energy_center: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a synthetic bowtie beam image.

    The horizontal envelope represents total per-column charge.  Primarily
    used for testing and benchmarking the ``BowtieFitAlgorithm``.

    Parameters
    ----------
    shape : tuple of int
        Image shape ``(height, width)``.
    min_sigma : float
        Minimum vertical beam size (pixels).
    divergence : float
        Horizontal divergence angle (radians).
    angle_deg : float
        Optional rotation applied to the finished image.
    vertical_offset : int
        Offset of beam centre from image midpoint (pixels).
    total_charge : float
        Total integrated signal across the image.
    noise_level : float
        Amplitude of added Gaussian noise.
    bit_depth_max : int
        Output clipped to ``[0, bit_depth_max]``.
    energy_spread : float
        Width of the horizontal energy envelope (pixels).
    background_level : float
        Uniform background offset.
    energy_center : float, optional
        Centre of horizontal energy envelope; defaults to image centre.
    seed : int, optional
        Random seed for reproducible noise generation.

    Returns
    -------
    np.ndarray
        Synthetic image, dtype ``uint16``.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    center_x = w // 2
    center_y = h // 2 + vertical_offset
    energy_center = energy_center or center_x

    ys = np.arange(h)
    x_vals = np.arange(w)

    energy_profile = np.exp(-((x_vals - energy_center) ** 2) / (2 * energy_spread**2))
    energy_profile /= energy_profile.sum()
    energy_profile *= total_charge

    img = np.zeros((h, w), dtype=np.float32)

    for x in range(w):
        dx = x - center_x
        sigma = min_sigma * np.sqrt(1 + (divergence * dx / min_sigma) ** 2)
        profile = np.exp(-((ys - center_y) ** 2) / (2 * sigma**2))
        profile /= profile.sum()
        img[:, x] = profile * energy_profile[x]

    img = rotate(img, angle=angle_deg, reshape=False, order=1, mode="constant")
    img = img / np.max(img) * bit_depth_max
    img += background_level

    if noise_level > 0:
        img += noise_level * rng.standard_normal(img.shape)

    return np.clip(img, 0, None).astype(np.uint16)
