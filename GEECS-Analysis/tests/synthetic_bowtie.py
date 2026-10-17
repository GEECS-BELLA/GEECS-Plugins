"""The synthetic bow-tie frame the HiResMagCam tests feed the core.

One copy for two packages' tests: ``GEECS-Analysis/tests/test_hi_res_mag_cam.py``
and ``ScanAnalysis/tests/test_core_hi_res_mag_cam.py`` load this file by path
(test directories are not importable packages). Moved from ImageAnalysis's
``tools/synthetic_generators.py`` (#1039); the stored expected values were
generated from these frames, so the arithmetic must not change.
"""

import numpy as np
from scipy.ndimage import rotate
from typing import Tuple, Optional


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
