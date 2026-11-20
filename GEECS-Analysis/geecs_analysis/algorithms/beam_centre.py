"""A beam's centre above a threshold, and the image shifted to put it in the middle."""

from __future__ import annotations

from typing import Optional

import numpy as np

from geecs_analysis.algorithms.basic_line_stats import compute_center_of_mass


def threshold_level(image: np.ndarray, threshold: float) -> float:
    """The pixel value a threshold names: counts at 1 or more, else a fraction of the peak."""
    if threshold >= 1.0:
        return float(threshold)
    return float(threshold) * float(np.nanmax(image))


def beam_centre(image: np.ndarray, threshold: float) -> Optional[tuple[float, float]]:
    """The centre of mass ``(row, column)`` of the pixels above *threshold*.

    Each pixel above the level weighs its value minus the level, so the
    background under the beam does not drag the centre towards the
    image's middle. The centre is the centre of mass
    (:func:`~geecs_analysis.algorithms.basic_line_stats.compute_center_of_mass`)
    of those weights' two projections, in pixel indices.

    Parameters
    ----------
    image : numpy.ndarray
        A 2-D image.
    threshold : float
        Counts when 1 or more; below 1, a fraction of the image's peak
        (0.5: half the brightest pixel).

    Returns
    -------
    tuple of float or None
        ``None`` when no pixel is above the level.
    """
    data = np.asarray(image, dtype=float)
    weights = data - threshold_level(data, threshold)
    weights[~(weights > 0)] = 0.0  # below the level, and NaN
    if not float(weights.sum()) > 0.0:
        return None
    row = float(compute_center_of_mass(weights.sum(axis=1)))
    column = float(compute_center_of_mass(weights.sum(axis=0)))
    return row, column


def centre_on(image: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
    """*image* shifted by whole pixels so *centre* lands at its middle.

    Pixels shifted in from past the edge take the image's median (its
    background, for a beam image), so they neither darken nor brighten
    the levels. The shape is kept.
    """
    data = np.asarray(image, dtype=float)
    height, width = data.shape[:2]
    dy = int(round((height - 1) / 2 - centre[0]))
    dx = int(round((width - 1) / 2 - centre[1]))
    out = np.full_like(data, float(np.nanmedian(data)))
    if abs(dy) >= height or abs(dx) >= width:
        return out
    src_y = slice(max(0, -dy), height - max(0, dy))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    out[dst_y, dst_x] = data[src_y, src_x]
    return out
