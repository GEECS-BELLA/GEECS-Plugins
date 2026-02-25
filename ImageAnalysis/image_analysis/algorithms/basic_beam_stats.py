"""Beam statistics utilities.

Provides data structures and functions for computing beam profile
statistics from images.
"""

from __future__ import annotations
from enum import Enum
from typing import Callable, Literal, NamedTuple, Optional, Set, Union
import logging

import numpy as np
from pydantic import BaseModel, Field

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


def beam_profile_stats(
    img: np.ndarray,
    enabled_stats: Optional[Set[BeamStatType]] = None,
    include_45: bool = True,
) -> BeamStats:
    """Compute beam profile statistics from a 2-D image.

    This function computes statistics for up to 4 projections (x, y, x_45, y_45)
    using LineBasicStats as the foundation for all 1D calculations.  Expensive
    computations (slopes, 45° projections) are skipped when not requested.

    Parameters
    ----------
    img : np.ndarray
        2D image array
    enabled_stats : set of BeamStatType, optional
        Statistics to compute.  ``None`` (the default) resolves to
        :data:`DEFAULT_BEAM_STATS`.  Slope metrics are only computed when
        at least one slope member is present in this set.
    include_45 : bool
        Whether to compute the ±45° diagonal projections.  Default ``True``.

    Returns
    -------
    BeamStats
        Beam statistics including image-level stats and projection stats.
        Fields that were skipped are filled with ``NaN``.
    """
    effective = enabled_stats if enabled_stats is not None else DEFAULT_BEAM_STATS
    need_slopes = bool(effective & _SLOPE_STATS)

    img = np.asarray(img, dtype=float)
    total_counts = img.sum()
    nan_proj = ProjectionStats(np.nan, np.nan, np.nan, np.nan)

    if total_counts <= 0:
        logger.warning(
            "beam_profile_stats: Image has non-positive total intensity. Returning NaNs."
        )
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

    # --- Standard x/y projections (always computed) ---
    x_proj = img.sum(axis=0)
    y_proj = img.sum(axis=1)
    x_stats = _line_stats_to_projection_stats(
        LineBasicStats(line_data=_projection_to_line_data(x_proj))
    )
    y_stats = _line_stats_to_projection_stats(
        LineBasicStats(line_data=_projection_to_line_data(y_proj))
    )

    # --- 45° projections (skipped when not requested) ---
    if include_45:
        x45_stats = _line_stats_to_projection_stats(
            LineBasicStats(line_data=_projection_to_line_data(_diag_projection(img)))
        )
        y45_stats = _line_stats_to_projection_stats(
            LineBasicStats(
                line_data=_projection_to_line_data(_antidiag_projection(img))
            )
        )
    else:
        x45_stats = nan_proj
        y45_stats = nan_proj

    # --- Slope metrics (skipped when not requested) ---
    if need_slopes:
        com_slope_x = compute_slope(
            *extract_line_stats(img, compute_center_of_mass, axis="x")
        )
        com_slope_y = compute_slope(
            *extract_line_stats(img, compute_center_of_mass, axis="y")
        )
        peak_slope_x = compute_slope(
            *extract_line_stats(img, compute_peak_location, axis="x")
        )
        peak_slope_y = compute_slope(
            *extract_line_stats(img, compute_peak_location, axis="y")
        )
    else:
        com_slope_x = com_slope_y = peak_slope_x = peak_slope_y = np.nan

    return BeamStats(
        image=ImageStats(
            total=total_counts,
            peak_value=np.max(img),
            com_slope_x=com_slope_x,
            com_slope_y=com_slope_y,
            peak_slope_x=peak_slope_x,
            peak_slope_y=peak_slope_y,
        ),
        x=x_stats,
        y=y_stats,
        x_45=x45_stats,
        y_45=y45_stats,
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


class BeamStatType(str, Enum):
    """Identifiers for individual beam statistics.

    Each value corresponds to a key fragment produced by :func:`flatten_beam_stats`.
    The full key is ``{section}_{field}`` (e.g., ``"x_rms"`` or ``"image_total"``).
    """

    # Image-level
    IMAGE_TOTAL = "image_total"
    IMAGE_PEAK_VALUE = "image_peak_value"
    IMAGE_COM_SLOPE_X = "image_com_slope_x"
    IMAGE_COM_SLOPE_Y = "image_com_slope_y"
    IMAGE_PEAK_SLOPE_X = "image_peak_slope_x"
    IMAGE_PEAK_SLOPE_Y = "image_peak_slope_y"

    # Per-axis projection stats  (x, y, x_45, y_45)
    COM = "CoM"
    RMS = "rms"
    FWHM = "fwhm"
    PEAK_LOCATION = "peak_location"


#: The default set of beam statistics reported when no explicit selection is
#: provided.  Excludes the four slope metrics (com_slope_x/y, peak_slope_x/y)
#: which are expensive and rarely needed.  Per-axis members (COM, RMS, FWHM,
#: PEAK_LOCATION) act as wildcards and apply to all four axes (x, y, x_45, y_45).
DEFAULT_BEAM_STATS: Set[BeamStatType] = {
    BeamStatType.IMAGE_TOTAL,
    BeamStatType.IMAGE_PEAK_VALUE,
    BeamStatType.COM,
    BeamStatType.RMS,
    BeamStatType.FWHM,
    BeamStatType.PEAK_LOCATION,
}


#: The set of slope-related stat types.  Used by :func:`beam_profile_stats` to
#: decide whether to run the expensive line-by-line slope computation.
_SLOPE_STATS: Set[BeamStatType] = {
    BeamStatType.IMAGE_COM_SLOPE_X,
    BeamStatType.IMAGE_COM_SLOPE_Y,
    BeamStatType.IMAGE_PEAK_SLOPE_X,
    BeamStatType.IMAGE_PEAK_SLOPE_Y,
}


class BeamAnalysisConfig(BaseModel):
    """Typed configuration for :class:`BeamAnalyzer`.

    This model is validated from ``camera_config.analysis`` at analyzer init
    time, giving users IDE autocompletion and config-file validation.

    Attributes
    ----------
    enabled_stats : set of BeamStatType, optional
        Subset of beam statistics to include in the flattened output.
        ``None`` (the default) uses :data:`DEFAULT_BEAM_STATS` which excludes
        slope metrics.
    include_45_projections : bool
        Whether to compute the ±45° diagonal projections (x_45, y_45).
        Default is ``True``.  Set to ``False`` to skip the diagonal sums,
        which saves computation when only x/y stats are needed.
    """

    enabled_stats: Optional[Set[BeamStatType]] = Field(
        default=None,
        description=(
            "Subset of beam statistics to report. "
            "None (default) uses DEFAULT_BEAM_STATS which excludes slope metrics."
        ),
    )
    include_45_projections: bool = Field(
        default=True,
        description=(
            "Whether to compute ±45° diagonal projections. "
            "Set to False to skip diagonal sums when only x/y stats are needed."
        ),
    )


def flatten_beam_stats(
    stats: BeamStats,
    prefix: Optional[Union[str, None]] = None,
    suffix: Optional[Union[str, None]] = None,
    enabled_stats: Optional[Set[BeamStatType]] = None,
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
    enabled_stats : set of BeamStatType, optional
        If provided, only keys whose ``{section}_{field}`` fragment matches a
        member of the set are included.  ``None`` (the default) includes all.

    Returns
    -------
    dict[str, float]
        Dictionary mapping field names to values. Keys are of the form
        ``"{prefix}_{section}_{field}{suffix}"`` when both are provided,
        ``"{prefix}_{section}_{field}"`` when only prefix is provided,
        ``"{section}_{field}{suffix}"`` when only suffix is provided,
        or ``"{section}_{field}"`` when neither is provided.
    """
    # Resolve None → DEFAULT_BEAM_STATS
    effective = enabled_stats if enabled_stats is not None else DEFAULT_BEAM_STATS

    # Build set of allowed key fragments for fast lookup.
    # Image-level members have values like "image_total" which match the
    # full fragment directly.  Per-axis members have values like "CoM" which
    # should match *any* axis prefix (x_CoM, y_rms, x_45_fwhm, …).
    allowed_full: Set[str] = set()  # matches full "field_k" fragments
    allowed_field: Set[str] = set()  # matches just the "k" portion
    for s in effective:
        val = s.value
        if val.startswith("image_"):
            allowed_full.add(val)
        else:
            allowed_field.add(val)

    flat: dict[str, float] = {}
    suffix_str = f"_{suffix}" if suffix else ""
    for field in stats._fields:
        nested = getattr(stats, field)
        for k, v in nested._asdict().items():
            fragment = f"{field}_{k}"
            # Allow if the full fragment matches OR the field name matches
            if fragment not in allowed_full and k not in allowed_field:
                continue

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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
