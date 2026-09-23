"""Basic beam statistics utilities.

Provides data structures and functions for computing fundamental beam profile
statistics from images: projection-based stats (CoM, rms, fwhm, peak_location)
along x, y, and ±45° axes, plus image-level totals.

For advanced/optional algorithms (e.g., slope metrics), see separate modules
such as :mod:`geecs_analysis.algorithms.beam_slopes`.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Set
import logging

import numpy as np

from geecs_analysis.algorithms.basic_line_stats import LineBasicStats

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


def _projection_to_line_data(
    projection: np.ndarray, coordinates: np.ndarray | None = None
) -> np.ndarray:
    """Pair samples with supplied coordinates, or local indices for diagonals."""
    if coordinates is None:
        coordinates = np.arange(len(projection))
    return np.column_stack([coordinates, projection])


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
    axes: tuple[np.ndarray, np.ndarray] | None = None,
) -> BeamStats:
    """Compute legacy beam statistics with explicit `(y, x)` coordinates.

    Coordinates replace the implicit ROI offset. Diagonal projections remain
    in local index space. Numerical conventions, including index-space moments
    followed by coordinate conversion, are preserved from ImageAnalysis.
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

    y_coordinates, x_coordinates = axes if axes is not None else (None, None)

    # Standard x/y projections use the frame's calibrated global coordinates.
    x_stats = _line_stats_to_projection_stats(
        LineBasicStats(
            line_data=_projection_to_line_data(
                img.sum(axis=0), coordinates=x_coordinates
            )
        )
    )
    y_stats = _line_stats_to_projection_stats(
        LineBasicStats(
            line_data=_projection_to_line_data(
                img.sum(axis=1), coordinates=y_coordinates
            )
        )
    )

    # 45° projections — diagonal index space; no simple global offset applies.
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
    include: Optional[Set[str]] = None,
) -> dict[str, float]:
    """Flatten a :class:`BeamStats` instance into a dictionary of bare-key scalars.

    Emits keys of the form ``"{section}_{field}"`` (e.g. ``"image_total"``,
    ``"x_CoM"``, ``"y_fwhm"``) with no prefix or suffix. Naming/disambiguation
    across analyzers is ScanAnalysis's responsibility per issue #412 — the
    scan-side wrapper applies ``metric_prefix`` and ``metric_suffix`` when
    storing per-shot results.

    Parameters
    ----------
    stats : BeamStats
        The beam statistics to flatten.
    include : set of str, optional
        If provided, only emit entries whose key is in this set. ``None``
        (the default) emits all entries.

    Returns
    -------
    dict[str, float]
        Dictionary mapping bare field names to values.
    """
    flat: dict[str, float] = {}
    for field in stats._fields:
        nested = getattr(stats, field)
        for k, v in nested._asdict().items():
            fragment = f"{field}_{k}"
            if include is not None and fragment not in include:
                continue
            flat[fragment] = v
    return flat
