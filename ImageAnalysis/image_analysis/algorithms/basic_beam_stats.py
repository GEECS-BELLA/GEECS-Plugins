"""Basic beam statistics utilities.

Provides data structures and functions for computing fundamental beam profile
statistics from images: projection-based stats (CoM, rms, fwhm, peak_location)
along x, y, and ±45° axes, plus image-level totals.

For advanced/optional algorithms (e.g., slope metrics), see separate modules
such as :mod:`image_analysis.algorithms.beam_slopes`.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union
import logging

import numpy as np

from image_analysis.algorithms.basic_line_stats import LineBasicStats

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
        Statistics of the +45° "column-after-rotation" projection
        (implemented via NW–SE diagonal sums with no resampling).
    y_45 : ProjectionStats
        Statistics of the +45° "row-after-rotation" projection
        (implemented via NE–SW anti-diagonal sums with no resampling).
    """

    image: ImageStats
    x: ProjectionStats
    y: ProjectionStats
    x_45: ProjectionStats
    y_45: ProjectionStats


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


def _projection_to_line_data(projection: np.ndarray) -> np.ndarray:
    """Convert a 1D projection array to Nx2 format for LineBasicStats.

    Parameters
    ----------
    projection : np.ndarray
        1D projection array

    Returns
    -------
    np.ndarray
        Nx2 array where column 0 is index coordinates and column 1 is projection values
    """
    x_coords = np.arange(len(projection))
    return np.column_stack([x_coords, projection])


def _line_stats_to_projection_stats(line_stats: LineBasicStats) -> ProjectionStats:
    """Extract ProjectionStats fields from LineBasicStats.

    Parameters
    ----------
    line_stats : LineBasicStats
        Complete line statistics

    Returns
    -------
    ProjectionStats
        Projection statistics with 4 fields (subset of LineBasicStats)
    """
    return ProjectionStats(
        CoM=line_stats.CoM,
        rms=line_stats.rms,
        fwhm=line_stats.fwhm,
        peak_location=line_stats.peak_location,
    )


def beam_profile_stats(img: np.ndarray) -> BeamStats:
    """Compute basic beam profile statistics from a 2-D image.

    Computes projection statistics (CoM, rms, fwhm, peak_location) along
    four axes (x, y, x_45, y_45) and image-level totals (total, peak_value).

    Parameters
    ----------
    img : np.ndarray
        2D image array

    Returns
    -------
    BeamStats
        Beam statistics including image-level stats and 4 projection stats.
    """
    img = np.asarray(img, dtype=float)
    total_counts = img.sum()
    nan_proj = ProjectionStats(np.nan, np.nan, np.nan, np.nan)

    if total_counts <= 0:
        logger.warning(
            "beam_profile_stats: Image has non-positive total intensity. Returning NaNs."
        )
        nan_img = ImageStats(total=total_counts, peak_value=np.nan)
        return BeamStats(
            image=nan_img, x=nan_proj, y=nan_proj, x_45=nan_proj, y_45=nan_proj
        )

    # Standard x/y projections
    x_stats = _line_stats_to_projection_stats(
        LineBasicStats(line_data=_projection_to_line_data(img.sum(axis=0)))
    )
    y_stats = _line_stats_to_projection_stats(
        LineBasicStats(line_data=_projection_to_line_data(img.sum(axis=1)))
    )

    # 45° projections
    x45_stats = _line_stats_to_projection_stats(
        LineBasicStats(line_data=_projection_to_line_data(_diag_projection(img)))
    )
    y45_stats = _line_stats_to_projection_stats(
        LineBasicStats(line_data=_projection_to_line_data(_antidiag_projection(img)))
    )

    return BeamStats(
        image=ImageStats(total=total_counts, peak_value=float(np.max(img))),
        x=x_stats,
        y=y_stats,
        x_45=x45_stats,
        y_45=y45_stats,
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
            fragment = f"{field}_{k}"
            key = f"{prefix}_{fragment}" if prefix else fragment
            key = f"{key}{suffix_str}"
            flat[key] = v
    return flat


# ---------------------------------------------------------------------------
# Backward compatibility: re-export functions moved to basic_line_stats
# ---------------------------------------------------------------------------
_MOVED_TO_BASIC_LINE_STATS = {
    "compute_center_of_mass",
    "compute_rms",
    "compute_fwhm",
    "compute_peak_location",
}


def __getattr__(name):
    """Lazy re-exports for backward compatibility."""
    if name in _MOVED_TO_BASIC_LINE_STATS:
        import warnings

        from image_analysis.algorithms import basic_line_stats

        warnings.warn(
            f"{name} has been moved to image_analysis.algorithms.basic_line_stats. "
            f"Please update your imports. This backward-compatible re-export will be "
            f"removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(basic_line_stats, name)
