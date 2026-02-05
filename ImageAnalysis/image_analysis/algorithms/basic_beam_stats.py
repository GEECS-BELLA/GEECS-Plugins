"""Beam statistics utilities.

Provides data structures and functions for computing beam profile
statistics from images.
"""

from __future__ import annotations
from typing import Callable, Literal, NamedTuple, Optional, Union
import numpy as np
import logging

from image_analysis.algorithms.basic_line_stats import (
    LineBasicStats,
    compute_center_of_mass,
    compute_peak_location,
)

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
    com_slope_x : float
        Weighted RMS residual of column-by-column CoM from linear fit.
        Lower = straighter beam (less tilt/shear).
    com_slope_y : float
        Weighted RMS residual of row-by-row CoM from linear fit.
    peak_slope_x : float
        Weighted RMS residual of column-by-column peak location from linear fit.
    peak_slope_y : float
        Weighted RMS residual of row-by-row peak location from linear fit.
    """

    total: float
    peak_value: float
    com_slope_x: float
    com_slope_y: float
    peak_slope_x: float
    peak_slope_y: float


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


class LineByLineResult(NamedTuple):
    """Result of extracting statistics from each line of an image.

    Attributes
    ----------
    positions : np.ndarray
        Position of each line (indices).
    values : np.ndarray
        Computed statistic for each line.
    weights : np.ndarray
        Total counts per line (for weighted fitting).
    """

    positions: np.ndarray
    values: np.ndarray
    weights: np.ndarray


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
    """Compute beam profile statistics from a 2-D image.

    This function computes statistics for 4 projections (x, y, x_45, y_45)
    using LineBasicStats as the foundation for all 1D calculations.

    Parameters
    ----------
    img : np.ndarray
        2D image array

    Returns
    -------
    BeamStats
        Beam statistics including image-level stats and 4 projection stats
    """
    img = np.asarray(img, dtype=float)
    total_counts = img.sum()

    if total_counts <= 0:
        logger.warning(
            "beam_profile_stats: Image has non-positive total intensity. Returning NaNs."
        )
        nan_proj = ProjectionStats(np.nan, np.nan, np.nan, np.nan)
        nan_img = ImageStats(
            total=total_counts,
            peak_value=np.nan,
            com_slope_x=np.nan,
            com_slope_y=np.nan,
            peak_slope_x=np.nan,
            peak_slope_y=np.nan,
        )
        return BeamStats(
            image=nan_img, x=nan_proj, y=nan_proj, x_45=nan_proj, y_45=nan_proj
        )

    # Create 4 projections
    x_proj = img.sum(axis=0)
    y_proj = img.sum(axis=1)
    x45_proj = _diag_projection(img)  # NW–SE
    y45_proj = _antidiag_projection(img)  # NE–SW

    # Compute stats using LineBasicStats (single source of truth)
    x_line_stats = LineBasicStats(line_data=_projection_to_line_data(x_proj))
    y_line_stats = LineBasicStats(line_data=_projection_to_line_data(y_proj))
    x45_line_stats = LineBasicStats(line_data=_projection_to_line_data(x45_proj))
    y45_line_stats = LineBasicStats(line_data=_projection_to_line_data(y45_proj))

    # Compute straightness metrics
    com_x_data = extract_line_stats(img, compute_center_of_mass, axis="x")
    com_y_data = extract_line_stats(img, compute_center_of_mass, axis="y")
    peak_x_data = extract_line_stats(img, compute_peak_location, axis="x")
    peak_y_data = extract_line_stats(img, compute_peak_location, axis="y")

    # Convert to ProjectionStats (extract 4 of 7 fields)
    return BeamStats(
        image=ImageStats(
            total=total_counts,
            peak_value=np.max(img),
            com_slope_x=compute_slope(*com_x_data),
            com_slope_y=compute_slope(*com_y_data),
            peak_slope_x=compute_slope(*peak_x_data),
            peak_slope_y=compute_slope(*peak_y_data),
        ),
        x=_line_stats_to_projection_stats(x_line_stats),
        y=_line_stats_to_projection_stats(y_line_stats),
        x_45=_line_stats_to_projection_stats(x45_line_stats),
        y_45=_line_stats_to_projection_stats(y45_line_stats),
    )


def extract_line_stats(
    img: np.ndarray,
    stat_func: Callable[[np.ndarray], float],
    axis: Literal["x", "y"] = "x",
) -> LineByLineResult:
    """Apply a statistic function to each line (row or column) of an image.

    Parameters
    ----------
    img : np.ndarray
        2D image array.
    stat_func : Callable[[np.ndarray], float]
        Function that takes a 1D array (one line) and returns a scalar.
        Can use functions from basic_line_stats (e.g., compute_center_of_mass,
        compute_rms) or any custom function.
    axis : {'x', 'y'}
        Which axis to iterate over:
        - 'x': Apply stat_func to each column (positions are column indices)
        - 'y': Apply stat_func to each row (positions are row indices)

    Returns
    -------
    x_values : np.ndarray
        Position of each line (indices).
    y_values : np.ndarray
        Computed statistic for each line.
    weights : np.ndarray
        Total counts (sum) for each line, useful for weighted fitting.

    Examples
    --------
    >>> from image_analysis.algorithms.basic_line_stats import compute_center_of_mass
    >>> x, y, w = extract_line_stats(image, compute_center_of_mass, axis='x')
    >>> # x = [0, 1, 2, ...] column indices
    >>> # y = CoM for each column
    >>> # w = total counts per column
    >>>
    >>> # For beam tilt detection, fit a line:
    >>> mask = np.isfinite(y) & (w > 100)
    >>> slope, intercept = fit_weighted_line(x[mask], y[mask], w[mask])
    """
    img = np.asarray(img, dtype=float)

    # axis=0 applies func to each column, axis=1 applies to each row
    apply_axis = 0 if axis == "x" else 1

    y_values = np.apply_along_axis(stat_func, apply_axis, img)
    weights = img.sum(axis=apply_axis)
    x_values = np.arange(len(y_values), dtype=float)

    return LineByLineResult(positions=x_values, values=y_values, weights=weights)


def compute_slope(
    positions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted linear slope fitting.

    Fits a line y = mx + b, then returns the slope.

    Parameters
    ----------
    positions : np.ndarray
        X values (e.g., column indices)
    values : np.ndarray
        Y values (e.g., CoM per column)
    weights : np.ndarray
        Weights for each point (e.g., total counts)

    Returns
    -------
    float
        slope of fitted line.
    """
    mask = np.isfinite(values) & (weights > 0)
    if mask.sum() < 2:
        return np.nan

    x, y, w = positions[mask], values[mask], weights[mask]

    # Weighted linear fit
    coeffs = np.polyfit(x, y, 1, w=np.sqrt(w))  # [slope, intercept]

    return float(coeffs[0])


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
