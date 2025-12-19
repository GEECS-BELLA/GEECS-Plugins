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
        Statistics of the horizontal (x-axis) projection.
    y : ProjectionStats
        Statistics of the vertical (y-axis) projection.
    x_45 : ProjectionStats
        Statistics of the +45° “column-after-rotation” projection
        (implemented via NW–SE diagonal sums with no resampling).
    y_45 : ProjectionStats
        Statistics of the +45° “row-after-rotation” projection
        (implemented via NE–SW anti-diagonal sums with no resampling).
    """

    image: ImageStats
    x: ProjectionStats
    y: ProjectionStats
    x_45: ProjectionStats
    y_45: ProjectionStats


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


def _diag_projection(img: np.ndarray) -> np.ndarray:
    """NW–SE diagonal sums (equivalent to column projection after +45° rotate)."""
    img = np.asarray(img, dtype=float)
    h, w = img.shape
    return np.array([np.diag(img, k=k).sum() for k in range(-(h - 1), w)])


def _antidiag_projection(img: np.ndarray) -> np.ndarray:
    """NE–SW anti-diagonal sums (equivalent to row projection after +45° rotate)."""
    img = np.asarray(img, dtype=float)
    flipped = np.fliplr(img)
    h, w = flipped.shape
    return np.array([np.diag(flipped, k=k).sum() for k in range(-(h - 1), w)])


def beam_profile_stats(img: np.ndarray) -> BeamStats:
    """Compute beam profile statistics from a 2-D image."""
    img = np.asarray(img, dtype=float)
    total_counts = img.sum()

    if total_counts <= 0:
        logger.warning(
            "beam_profile_stats: Image has non-positive total intensity. Returning NaNs."
        )
        nan_proj = ProjectionStats(np.nan, np.nan, np.nan, np.nan)
        nan_img = ImageStats(total=total_counts, peak_value=np.nan)
        return BeamStats(
            image=nan_img, x=nan_proj, y=nan_proj, x_45=nan_proj, y_45=nan_proj
        )

    # Base projections
    x_proj = img.sum(axis=0)
    y_proj = img.sum(axis=1)

    # Exact 45° projections via diagonal sums (no padding bias, no interpolation)
    x45_proj = _diag_projection(img)  # NW–SE
    y45_proj = _antidiag_projection(img)  # NE–SW

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
        x_45=ProjectionStats(
            CoM=compute_center_of_mass(x45_proj),
            rms=compute_rms(x45_proj),
            fwhm=compute_fwhm(x45_proj),
            peak_location=compute_peak_location(x45_proj),
        ),
        y_45=ProjectionStats(
            CoM=compute_center_of_mass(y45_proj),
            rms=compute_rms(y45_proj),
            fwhm=compute_fwhm(y45_proj),
            peak_location=compute_peak_location(y45_proj),
        ),
    )


def flatten_beam_stats(
    stats: BeamStats,
    prefix: Optional[Union[str, None]] = None,
    suffix: Optional[Union[str, None]] = None,
) -> dict[str, float]:
    """Flatten a :class:`BeamStats` instance into a dictionary.

    Parameters
    ----------
    stats : BeamStats
        The beam statistics to flatten.
    prefix : str, None
        Optional prefix to prepend to each key.
    suffix : str, None
        Optional suffix to append to each key (underscore is auto-prepended).
        Useful for distinguishing multiple analysis variations (e.g., "curtis"
        becomes "_curtis" in the key).

    Returns
    -------
    dict[str, float]
        Dictionary mapping field names to values. Keys are of the form
        ``"{prefix}_{section}_{field}{suffix}"`` when both are provided,
        ``"{prefix}_{section}_{field}"`` when only prefix is provided,
        ``"{section}_{field}{suffix}"`` when only suffix is provided,
        or ``"{section}_{field}"`` when neither is provided.
    """
    flat: dict[str, float] = {}
    suffix_str = f"_{suffix}" if suffix else ""
    for field in stats._fields:
        nested = getattr(stats, field)
        for k, v in nested._asdict().items():
            key = f"{prefix}_{field}_{k}" if prefix else f"{field}_{k}"
            key = f"{key}{suffix_str}"
            flat[key] = v
    return flat
