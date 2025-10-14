"""Beam statistics utilities.

Provides data structures and functions for computing beam profile
statistics from images. The module follows NumPy docstring conventions.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ProjectionStats(NamedTuple):
    """Statistics of a 1‑D projection of a beam image.

    Attributes
    ----------
    CoM : float
        Center‑of‑mass of the projection.
    rms : float
        Root‑mean‑square width of the projection.
    fwhm : float
        Full‑width at half‑maximum of the projection.
    peak_location : float
        Index of the maximum value in the projection.
    """

    CoM: float
    rms: float
    fwhm: float
    peak_location: float


class ImageStats(NamedTuple):
    """Overall statistics of a 2‑D beam image.

    Attributes
    ----------
    total : float
        Sum of all pixel values (total intensity).
    peak_value : float
        Maximum pixel value in the image.
    """

    total: float
    peak_value: float


class BeamStats(NamedTuple):
    """Container for beam statistics of an image.

    Attributes
    ----------
    image : ImageStats
        Global image statistics.
    x : ProjectionStats
        Statistics of the horizontal (x‑axis) projection.
    y : ProjectionStats
        Statistics of the vertical (y‑axis) projection.
    """

    image: ImageStats
    x: ProjectionStats
    y: ProjectionStats


def compute_center_of_mass(profile: np.ndarray) -> float:
    """Compute the center of mass of a 1‑D profile.

    Parameters
    ----------
    profile : np.ndarray
        1‑D array containing intensity values.

    Returns
    -------
    float
        Center of mass value. Returns ``np.nan`` and logs a warning if the
        total intensity is non‑positive.
    """
    profile = np.asarray(profile, dtype=float)
    total = profile.sum()
    if total <= 0:
        logger.warning(
            "compute_center_of_mass: Profile has non-positive total intensity. Returning np.nan."
        )
        return np.nan
    coords = np.arange(profile.size)
    return np.sum(coords * profile) / total


def compute_rms(profile: np.ndarray) -> float:
    """Compute the RMS width of a 1‑D profile.

    Parameters
    ----------
    profile : np.ndarray
        1‑D array containing intensity values.

    Returns
    -------
    float
        RMS width. Returns ``np.nan`` and logs a warning if the total intensity
        is non‑positive.
    """
    profile = np.asarray(profile, dtype=float)
    total = profile.sum()
    profile[profile < 0] = 0
    if total <= 0:
        logger.warning(
            "compute_rms: Profile has non-positive total intensity. Returning np.nan."
        )
        return np.nan
    coords = np.arange(profile.size)
    com = compute_center_of_mass(profile)
    return np.sqrt(np.sum((coords - com) ** 2 * profile) / total)


def compute_fwhm(profile: np.ndarray) -> float:
    """Compute the full width at half maximum (FWHM) of a 1‑D profile.

    Parameters
    ----------
    profile : np.ndarray
        1‑D array containing intensity values.

    Returns
    -------
    float
        FWHM value. Returns ``np.nan`` and logs a warning if the profile has
        non‑positive total intensity or cannot determine a half‑maximum.
    """
    profile = np.asarray(profile, dtype=float)
    if profile.sum() <= 0:
        logger.warning(
            "compute_fwhm: Profile has non-positive total intensity. Returning np.nan."
        )
        return np.nan

    profile -= profile.min()
    max_val = profile.max()
    if max_val <= 0:
        logger.warning(
            "compute_fwhm: Profile has non-positive peak after baseline shift. Returning np.nan."
        )
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
    right_edge = (
        interp_edge(right, right + 1) if right < len(profile) - 1 else float(right)
    )

    return right_edge - left_edge


def compute_peak_location(profile: np.ndarray) -> float:
    """Return the index of the peak value in a 1‑D profile.

    Parameters
    ----------
    profile : np.ndarray
        1‑D array containing intensity values.

    Returns
    -------
    float
        Index of the maximum value. Returns ``np.nan`` and logs a warning if the
        profile is empty.
    """
    profile = np.asarray(profile, dtype=float)
    if profile.size == 0:
        logger.warning("compute_peak_location: Profile is empty. Returning np.nan.")
        return np.nan
    return int(np.argmax(profile))


def beam_profile_stats(img: np.ndarray) -> BeamStats:
    """Compute beam profile statistics from a 2‑D image.

    Parameters
    ----------
    img : np.ndarray
        2‑D array representing the beam image.

    Returns
    -------
    BeamStats
        Named tuple containing global image statistics and per‑axis projection
        statistics. If the image has non‑positive total intensity, the returned
        fields contain ``np.nan``.
    """
    img = np.asarray(img, dtype=float)
    total_counts = img.sum()

    if total_counts <= 0:
        logger.warning(
            "beam_profile_stats: Image has non-positive total intensity. Returning NaNs."
        )
        nan_proj = ProjectionStats(np.nan, np.nan, np.nan, np.nan)
        nan_img = ImageStats(total=total_counts, peak_value=np.nan)
        return BeamStats(image=nan_img, x=nan_proj, y=nan_proj)

    x_proj = img.sum(axis=0)
    y_proj = img.sum(axis=1)

    return BeamStats(
        image=ImageStats(total=total_counts, peak_value=np.max(img)),
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


def flatten_beam_stats(
    stats: BeamStats, prefix: Optional[Union[str, None]] = None
) -> dict[str, float]:
    """Flatten a :class:`BeamStats` instance into a dictionary.

    Parameters
    ----------
    stats : BeamStats
        The beam statistics to flatten.
    prefix : str, None
        Optional prefix to prepend to each key.

    Returns
    -------
    dict[str, float]
        Dictionary mapping field names to values. Keys are of the form
        ``"{prefix}_{section}_{field}"`` when ``prefix`` is provided,
        otherwise ``"{section}_{field}"``.
    """
    flat: dict[str, float] = {}
    for field in stats._fields:
        nested = getattr(stats, field)
        for k, v in nested._asdict().items():
            key = f"{prefix}_{field}_{k}" if prefix else f"{field}_{k}"
            flat[key] = v
    return flat
