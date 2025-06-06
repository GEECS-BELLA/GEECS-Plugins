import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_centroid(profile: np.ndarray) -> float:
    """
    Compute centroid (center of mass) of a 1D profile.

    Returns 0.0 and logs a warning if profile has non-positive total intensity.
    """
    profile = np.asarray(profile, dtype=float)
    total = profile.sum()
    if total <= 0:
        logger.warning("compute_centroid: Profile has non-positive total intensity. Returning 0.0.")
        return 0.0
    coords = np.arange(profile.size)
    return np.sum(coords * profile) / total


def compute_rms(profile: np.ndarray) -> float:
    """
    Compute RMS width of a 1D profile.

    Returns 0.0 and logs a warning if profile has non-positive total intensity.
    """
    profile = np.asarray(profile, dtype=float)
    total = profile.sum()
    if total <= 0:
        logger.warning("compute_rms: Profile has non-positive total intensity. Returning 0.0.")
        return 0.0
    coords = np.arange(profile.size)
    centroid = compute_centroid(profile)
    return np.sqrt(np.sum((coords - centroid) ** 2 * profile) / total)


def compute_fwhm(profile: np.ndarray) -> float:
    """
    Compute Full Width at Half Maximum (FWHM) of a 1D profile using linear interpolation.

    Returns 0.0 and logs a warning if total intensity is non-positive.
    """
    profile = np.asarray(profile, dtype=float)
    if profile.sum() <= 0:
        logger.warning("compute_fwhm: Profile has non-positive total intensity. Returning 0.0.")
        return 0.0

    profile -= profile.min()
    max_val = profile.max()
    if max_val <= 0:
        logger.warning("compute_fwhm: Profile has non-positive peak after baseline shift. Returning 0.0.")
        return 0.0

    half_max = max_val / 2
    indices = np.where(profile >= half_max)[0]
    if len(indices) < 2:
        return 0.0

    left, right = indices[0], indices[-1]

    def interp_edge(i1, i2):
        y1, y2 = profile[i1], profile[i2]
        if y2 == y1:
            return float(i1)
        return i1 + (half_max - y1) / (y2 - y1)

    left_edge = interp_edge(left - 1, left) if left > 0 else float(left)
    right_edge = interp_edge(right, right + 1) if right < len(profile) - 1 else float(right)

    return right_edge - left_edge



def compute_peak_location(profile: np.ndarray) -> int:
    """
    Compute index of peak value in a 1D profile.

    Returns 0 and logs a warning if profile is empty.
    """
    profile = np.asarray(profile, dtype=float)
    if profile.size == 0:
        logger.warning("compute_peak_location: Profile is empty. Returning 0.")
        return 0
    return int(np.argmax(profile))


def compute_2d_centroids(img: np.ndarray) -> tuple[float, float]:
    """
    Compute centroids along x and y axes from a 2D image.

    Returns (0.0, 0.0) and logs a warning if total image intensity is non-positive.
    """
    img = np.asarray(img, dtype=float)
    if img.sum() <= 0:
        logger.warning("compute_2d_centroids: Image has non-positive total intensity. Returning (0.0, 0.0).")
        return 0.0, 0.0
    return compute_centroid(img.sum(axis=0)), compute_centroid(img.sum(axis=1))


def compute_2d_rms(img: np.ndarray) -> tuple[float, float]:
    """
    Compute RMS widths along x and y axes from a 2D image.

    Returns (0.0, 0.0) and logs a warning if total image intensity is non-positive.
    """
    img = np.asarray(img, dtype=float)
    if img.sum() <= 0:
        logger.warning("compute_2d_rms: Image has non-positive total intensity. Returning (0.0, 0.0).")
        return 0.0, 0.0
    return compute_rms(img.sum(axis=0)), compute_rms(img.sum(axis=1))


def compute_2d_fwhm(img: np.ndarray) -> tuple[float, float]:
    """
    Compute FWHM along x and y axes from a 2D image.

    Returns (0.0, 0.0) and logs a warning if total image intensity is non-positive.
    """
    img = np.asarray(img, dtype=float)
    if img.sum() <= 0:
        logger.warning("compute_2d_fwhm: Image has non-positive total intensity. Returning (0.0, 0.0).")
        return 0.0, 0.0
    return compute_fwhm(img.sum(axis=0)), compute_fwhm(img.sum(axis=1))


def compute_2d_peak_locations(img: np.ndarray) -> tuple[int, int]:
    """
    Compute peak locations along x and y axes from a 2D image.

    Returns (0, 0) and logs a warning if total image intensity is non-positive.
    """
    img = np.asarray(img, dtype=float)
    if img.sum() <= 0:
        logger.warning("compute_2d_peak_locations: Image has non-positive total intensity. Returning (0, 0).")
        return 0, 0
    return compute_peak_location(img.sum(axis=0)), compute_peak_location(img.sum(axis=1))


def beam_profile_stats(img: np.ndarray, prefix: str = "") -> dict[str, float]:
    """
    Compute beam profile statistics (centroid, RMS, FWHM, peak) from 2D image.

    Parameters:
        img (np.ndarray): 2D image array.
        prefix (str): Optional prefix for dictionary keys.

    Returns:
        dict[str, float]: Dictionary of computed stats with optional prefixed keys.
    """
    img = np.asarray(img, dtype=float)
    if img.sum() <= 0:
        logger.warning("beam_profile_stats: Image has non-positive total intensity. Returning all 0.0 values.")
        prefix = f"{prefix}_" if prefix else ""
        return {
            f"{prefix}x_mean": 0.0,
            f"{prefix}x_rms": 0.0,
            f"{prefix}x_fwhm": 0.0,
            f"{prefix}x_peak": 0.0,
            f"{prefix}y_mean": 0.0,
            f"{prefix}y_rms": 0.0,
            f"{prefix}y_fwhm": 0.0,
            f"{prefix}y_peak": 0.0,
        }

    x_proj = img.sum(axis=0)
    y_proj = img.sum(axis=1)

    x_centroid = compute_centroid(x_proj)
    x_rms = compute_rms(x_proj)
    x_fwhm = compute_fwhm(x_proj)
    x_peak = compute_peak_location(x_proj)

    y_centroid = compute_centroid(y_proj)
    y_rms = compute_rms(y_proj)
    y_fwhm = compute_fwhm(y_proj)
    y_peak = compute_peak_location(y_proj)

    prefix = f"{prefix}_" if prefix else ""

    return {
        f"{prefix}x_mean": x_centroid,
        f"{prefix}x_rms": x_rms,
        f"{prefix}x_fwhm": x_fwhm,
        f"{prefix}x_peak": x_peak,
        f"{prefix}y_mean": y_centroid,
        f"{prefix}y_rms": y_rms,
        f"{prefix}y_fwhm": y_fwhm,
        f"{prefix}y_peak": y_peak,
    }
