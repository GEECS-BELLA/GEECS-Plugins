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

) -> np.ndarray:
    """
    Generate a synthetic bowtie beam image where the horizontal envelope represents total per-column charge.

    This is primarily used for testing and benchmarking the BowtieFitAlgorithm.
    It generates a horizontally-elongated beam with vertical size that varies
    along the horizontal axis according to Gaussian beam divergence.

    Args:
        shape: Image shape (height, width).
        min_sigma: Minimum vertical beam size.
        divergence: Horizontal divergence (theta).
        angle_deg: Optional rotation to apply to the image.
        vertical_offset: Offset beam center vertically.
        total_charge: Total integrated signal across image.
        noise_level: Amplitude of added Gaussian noise.
        bit_depth_max: Maximum value for intensity.
        energy_center: Center of horizontal envelope.
        energy_spread: Width of horizontal envelope.

    Returns:
        np.ndarray: Synthetic image (dtype: uint16).
    """
    h, w = shape
    center_x = w // 2
    center_y = h // 2 + vertical_offset
    energy_center = energy_center or center_x

    ys = np.arange(h)
    x_vals = np.arange(w)

    # Normalized horizontal charge envelope (used as total intensity per column)
    energy_profile = np.exp(-((x_vals - energy_center) ** 2) / (2 * energy_spread ** 2))
    energy_profile /= energy_profile.sum()  # normalize to 1
    energy_profile *= total_charge

    img = np.zeros((h, w), dtype=np.float32)

    for x in range(w):
        dx = x - center_x
        sigma = min_sigma * np.sqrt(1 + (divergence * dx / min_sigma) ** 2)
        profile = np.exp(-((ys - center_y) ** 2) / (2 * sigma ** 2))
        profile /= profile.sum()  # normalize to unit area
        img[:, x] = profile * energy_profile[x]  # apply charge envelope as total column energy

    # Optional image rotation
    img = rotate(img, angle=angle_deg, reshape=False, order=1, mode='constant')
    img = img / np.max(img) * bit_depth_max

    img += background_level

    if noise_level > 0:
        img += noise_level * np.random.randn(*img.shape)

    return np.clip(img, 0, None).astype(np.uint16)