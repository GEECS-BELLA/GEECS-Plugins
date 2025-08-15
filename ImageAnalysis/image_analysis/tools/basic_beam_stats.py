from __future__ import annotations
from typing import NamedTuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ProjectionStats(NamedTuple):
    CoM: float
    rms: float
    fwhm: float
    peak_location: float

class ImageStats(NamedTuple):
    total: float
    peak_value: float

class BeamStats(NamedTuple):
    image: ImageStats
    x: ProjectionStats
    y: ProjectionStats

def compute_center_of_mass(profile: np.ndarray) -> float:
    """
    Compute center of mass of a 1D profile.

    Returns 0.0 and logs a warning if profile has non-positive total intensity.
    """
    profile = np.asarray(profile, dtype=float)
    total = profile.sum()
    if total <= 0:
        logger.warning("compute_center_of_mass: Profile has non-positive total intensity. Returning 0.0.")
        return np.nan
    coords = np.arange(profile.size)
    return np.sum(coords * profile) / total

def compute_rms(profile: np.ndarray) -> float:
    """
    Compute RMS width of a 1D profile.

    Returns 0.0 and logs a warning if profile has non-positive total intensity.
    """
    profile = np.asarray(profile, dtype=float)
    total = profile.sum()
    profile[profile < 0] = 0  # Set all negative values to zero
    if total <= 0:
        logger.warning("compute_rms: Profile has non-positive total intensity. Returning 0.0.")
        return np.nan
    coords = np.arange(profile.size)
    com = compute_center_of_mass(profile)
    return np.sqrt(np.sum((coords - com) ** 2 * profile) / total)

def compute_fwhm(profile: np.ndarray) -> float:
    """
    Compute Full Width at Half Maximum (FWHM) of a 1D profile using linear interpolation.

    Returns 0.0 and logs a warning if total intensity is non-positive.
    """
    profile = np.asarray(profile, dtype=float)
    if profile.sum() <= 0:
        logger.warning("compute_fwhm: Profile has non-positive total intensity. Returning 0.0.")
        return np.nan

    profile -= profile.min()
    max_val = profile.max()
    if max_val <= 0:
        logger.warning("compute_fwhm: Profile has non-positive peak after baseline shift. Returning 0.0.")
        return np.nan

    half_max = max_val / 2
    indices = np.where(profile >= half_max)[0]
    if len(indices) < 2:
        return np.nan

    left, right = indices[0], indices[-1]

    def interp_edge(i1, i2):
        y1, y2 = profile[i1], profile[i2]
        if y2 == y1:
            return float(i1)
        return i1 + (half_max - y1) / (y2 - y1)

    left_edge = interp_edge(left - 1, left) if left > 0 else float(left)
    right_edge = interp_edge(right, right + 1) if right < len(profile) - 1 else float(right)

    return right_edge - left_edge

def compute_peak_location(profile: np.ndarray) -> float:
    """
    Compute index of peak value in a 1D profile.

    Returns 0 and logs a warning if profile is empty.
    """
    profile = np.asarray(profile, dtype=float)
    if profile.size == 0:
        logger.warning("compute_peak_location: Profile is empty. Returning 0.")
        return np.nan
    return int(np.argmax(profile))

def beam_profile_stats(img: np.ndarray) -> BeamStats:
    """
    Compute beam profile statistics (global and per-axis) from a 2D image.

    Parameters
    ----------
    img : np.ndarray
        2D image array.

    Returns
    -------
    BeamStats
        Named tuple with:
          - image: ImageStats for overall image metrics
          - x, y: ProjectionStats for horizontal and vertical projections
    """
    img = np.asarray(img, dtype=float)
    total_counts = img.sum()

    if total_counts <= 0:
        logger.warning("beam_profile_stats: Image has non-positive total intensity. Returning NaNs.")
        nan_proj = ProjectionStats(np.nan, np.nan, np.nan, np.nan)
        nan_img = ImageStats(total=total_counts, peak_value=np.nan)
        return BeamStats(image=nan_img, x=nan_proj, y=nan_proj)

    # projections
    x_proj = img.sum(axis=0)
    y_proj = img.sum(axis=1)

    return BeamStats(
        image=ImageStats(
            total=total_counts,
            peak_value=np.max(img)
        ),
        x=ProjectionStats(
            CoM=compute_center_of_mass(x_proj),
            rms=compute_rms(x_proj),
            fwhm=compute_fwhm(x_proj),
            peak_location=compute_peak_location(x_proj),
        ),
        y=ProjectionStats(
            CoM=compute_center_of_mass(y_proj),
            rms=compute_rms(y_proj),
            fwhm=compute_fwhm(y_proj),
            peak_location=compute_peak_location(y_proj),
        ),
    )

def flatten_beam_stats(stats: BeamStats, prefix: str | None = None) -> dict[str, float]:
    """
    Flatten a BeamStats NamedTuple into a flat dictionary with optional prefix.

    Parameters
    ----------
    stats : BeamStats
        Structured beam profile statistics.
    prefix : str, optional
        Optional prefix for keys.

    Returns
    -------
    dict[str, float]
        Flat dictionary with keys like '<prefix>_image_total', '<prefix>_x_CoM', etc.
    """

    flat: dict[str, float] = {}
    for field in stats._fields:
        nested = getattr(stats, field)
        for k, v in nested._asdict().items():
            key = f"{prefix}_{field}_{k}" if prefix else f"{field}_{k}"
            flat[key] = v
    return flat
