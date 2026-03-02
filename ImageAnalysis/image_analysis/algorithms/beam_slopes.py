"""Beam slope (straightness) metrics.

Computes line-by-line statistics across rows or columns of a 2-D beam image
and fits a weighted linear slope.  These metrics quantify how "straight" a
beam is — lower slopes indicate less tilt or shear.

This is an optional algorithm that can be composed into
:class:`~image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer`
via its ``BeamAnalysisConfig.compute_slopes`` flag.
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple, Optional, Union
import logging

import numpy as np

from image_analysis.algorithms.basic_line_stats import (
    compute_center_of_mass,
    compute_peak_location,
)

logger = logging.getLogger(__name__)


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
    LineByLineResult
        positions, values, and weights arrays.

    Examples
    --------
    >>> from image_analysis.algorithms.basic_line_stats import compute_center_of_mass
    >>> result = extract_line_stats(image, compute_center_of_mass, axis='x')
    >>> # result.positions = [0, 1, 2, ...] column indices
    >>> # result.values = CoM for each column
    >>> # result.weights = total counts per column
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
        Slope of fitted line.
    """
    mask = np.isfinite(values) & (weights > 0)
    if mask.sum() < 2:
        return np.nan

    x, y, w = positions[mask], values[mask], weights[mask]

    # Weighted linear fit
    coeffs = np.polyfit(x, y, 1, w=np.sqrt(w))  # [slope, intercept]

    return float(coeffs[0])


def compute_beam_slopes(
    img: np.ndarray,
    prefix: Optional[Union[str, None]] = None,
    suffix: Optional[Union[str, None]] = None,
) -> dict[str, float]:
    """Compute beam slope metrics for a 2-D image.

    Measures how the center-of-mass and peak location vary across rows and
    columns, fitting a weighted linear slope to each.  These metrics quantify
    beam straightness — lower values indicate less tilt or shear.

    Parameters
    ----------
    img : np.ndarray
        2D image array.
    prefix : str, optional
        Prefix to prepend to each key (e.g., camera name).
    suffix : str, optional
        Suffix to append to each key (underscore is auto-prepended).

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        ``image_com_slope_x``, ``image_com_slope_y``,
        ``image_peak_slope_x``, ``image_peak_slope_y``
        (with prefix/suffix applied).
    """
    img = np.asarray(img, dtype=float)

    if img.sum() <= 0:
        logger.warning(
            "compute_beam_slopes: Image has non-positive total intensity. "
            "Returning NaN slopes."
        )
        raw = {
            "image_com_slope_x": np.nan,
            "image_com_slope_y": np.nan,
            "image_peak_slope_x": np.nan,
            "image_peak_slope_y": np.nan,
        }
    else:
        raw = {
            "image_com_slope_x": compute_slope(
                *extract_line_stats(img, compute_center_of_mass, axis="x")
            ),
            "image_com_slope_y": compute_slope(
                *extract_line_stats(img, compute_center_of_mass, axis="y")
            ),
            "image_peak_slope_x": compute_slope(
                *extract_line_stats(img, compute_peak_location, axis="x")
            ),
            "image_peak_slope_y": compute_slope(
                *extract_line_stats(img, compute_peak_location, axis="y")
            ),
        }

    # Apply prefix/suffix
    suffix_str = f"_{suffix}" if suffix else ""
    result: dict[str, float] = {}
    for k, v in raw.items():
        key = f"{prefix}_{k}" if prefix else k
        key = f"{key}{suffix_str}"
        result[key] = v
    return result
