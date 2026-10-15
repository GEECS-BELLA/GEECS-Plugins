"""Plasma downramp shock metrics of a phase map.

Ported from ImageAnalysis' ``DownrampPhaseAnalyzer`` with its numerical
conventions unchanged — the separable Gaussian mask and its binarisation,
the laser-axis rotation (``scipy.ndimage.rotate``, linear, on a reshaped
canvas), the sliding-window steepest-slope search, the row-wise shock angle
fit, the column-sum gradient and the plateau window — so the
``downramp_phase`` measure reproduces the legacy class bit for bit (pinned
by a differential test that uses ImageAnalysis as the oracle). The only
changes are shape: the plotting is gone and the geometry it drew is
returned instead, and a profile with no steepest window reports an
undefined plateau rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.special import cosdg, sindg

from geecs_analysis.algorithms.beam_slopes import compute_beam_slopes


def apply_gaussian_mask(
    image: np.ndarray,
    center_x: Optional[int] = None,
    center_y: Optional[int] = None,
    sigma_x: float = 100,
    sigma_y: float = 20,
    binarize: bool = False,
    threshold_factor: float = 0.01,
) -> np.ndarray:
    """Apply a separable 2D Gaussian mask and optionally binarize.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image.
    center_x : int, optional
        X-coordinate of Gaussian center; defaults to image center.
    center_y : int, optional
        Y-coordinate of Gaussian center; defaults to image center.
    sigma_x : float, default=100
        Standard deviation of Gaussian along x.
    sigma_y : float, default=20
        Standard deviation of Gaussian along y.
    binarize : bool, default=False
        If True, return ``binary_mask * image`` using `threshold_factor`.
    threshold_factor : float, default=0.01
        Fraction of the max masked value used as a binarization threshold.

    Returns
    -------
    numpy.ndarray
        Masked image if `binarize` is False, else ``binary_mask * image``.
    """
    rows, cols = image.shape
    if center_x is None:
        center_x = cols // 2
    if center_y is None:
        center_y = rows // 2

    y, x = np.mgrid[0:rows, 0:cols]
    gaussian_mask = np.exp(
        -(
            ((x - center_x) ** 2) / (2 * sigma_x**2)
            + ((y - center_y) ** 2) / (2 * sigma_y**2)
        )
    )
    masked_image = image * gaussian_mask

    if not binarize:
        return masked_image
    threshold = threshold_factor * np.nanmax(masked_image)
    binary_mask = (masked_image > threshold).astype(np.float32)
    return binary_mask * image


@dataclass(frozen=True)
class LocalSlope:
    """The steepest window of a profile: NaN slope and no centre when there is none."""

    slope: float
    center: Optional[int]


def find_max_local_slope(profile: np.ndarray, window_size: int = 20) -> LocalSlope:
    """Find the maximum local slope and its position in a 1D array using a sliding window.

    A profile whose total is not positive, or in which no window has a
    positive total and a nonzero slope, has no steepest window.
    """
    if np.nansum(profile) <= 0:
        return LocalSlope(np.nan, None)

    x = np.arange(len(profile))
    max_slope = 0.0
    best_center = None

    for start in range(0, len(profile) - window_size + 1):
        end = start + window_size
        x_window = x[start:end]
        y_window = profile[start:end]

        # Skip window if it contains only non-positive values
        if np.nansum(y_window) <= 0:
            continue

        coeffs = np.polyfit(x_window, y_window, 1)
        slope = coeffs[0]
        center = (start + end) // 2

        if abs(slope) > abs(max_slope):
            max_slope = slope
            best_center = center

    if best_center is None:
        return LocalSlope(np.nan, None)
    return LocalSlope(max_slope, best_center)


@dataclass(frozen=True)
class ShockAngle:
    """The row-wise steepest positions fitted by a line: the angle and the line.

    ``slope`` is columns per row and ``intercept`` the column at row 0, over
    ``first_row``..``last_row`` (the rows with a steepest window). Fewer
    than two such rows fit no line: the angle is NaN and the rows ``None``.
    """

    angle_rad: float
    slope: float
    intercept: float
    first_row: Optional[int]
    last_row: Optional[int]


def shock_angle(image: np.ndarray, window_size: int = 20) -> ShockAngle:
    """Estimate the shock angle by finding the local slope per row and fitting a line.

    Returns the angle in radians, with the weighted line (weights are the
    row sums) the legacy figure drew.
    """
    H, W = image.shape
    max_slope_positions = []

    for row_idx in range(H):
        row = image[row_idx, :]
        max_slope_positions.append(find_max_local_slope(row, window_size).center)

    max_slope_positions = np.array(max_slope_positions, dtype=float)
    rows = np.arange(H)

    # Mask invalid entries
    valid = np.isfinite(max_slope_positions)
    rows_valid = rows[valid]
    positions_valid = max_slope_positions[valid]
    weights = np.nansum(image[valid, :], axis=1)  # only valid rows

    # Fit a line to get shock angle using only valid rows
    if len(rows_valid) >= 2:
        coeffs = np.polyfit(rows_valid, positions_valid, 1, w=weights)
        slope_line = coeffs[0]
        angle_rad = np.arctan(slope_line)
        return ShockAngle(
            angle_rad,
            float(slope_line),
            float(coeffs[1]),
            int(rows_valid[0]),
            int(rows_valid[-1]),
        )
    return ShockAngle(np.nan, np.nan, np.nan, None, None)


@dataclass(frozen=True)
class Plateau:
    """The plateau window of the summed profile and the two values read from it.

    ``start``/``stop`` are the window's columns as Python resolved the
    legacy slice (``[start, stop)``; empty when ``stop <= start``, and the
    value is then NaN).
    """

    value: float
    delta: float
    start: int
    stop: int


def delta_plateau(
    profile: np.ndarray, best_center: int, window_size: int = 20
) -> Plateau:
    """Compute the plateau average and the peak-to-plateau delta of a summed profile.

    The plateau window lies upstream of the steepest window: from four
    windows before its centre to one window before it, clipped to the
    profile (the legacy arithmetic, negative stops included).
    """
    overall_max = np.max(profile)

    start_index = best_center - (1 + 3) * window_size
    end_index = start_index + 3 * window_size
    start_index = max(start_index, 0)
    end_index = min(end_index, len(profile))

    plateau_region = profile[start_index:end_index]
    plateau_value = np.mean(plateau_region) if plateau_region.size else np.nan
    delta = overall_max - plateau_value
    window = range(len(profile))[start_index:end_index]
    return Plateau(plateau_value, delta, window.start, window.stop)


@dataclass(frozen=True)
class ShockAnalysis:
    """Everything the legacy ``compile_shock_analysis`` computed, without the figure.

    ``rotated`` is the map on the laser axis (``None`` when the axis is
    undefined: no signal after masking — then nothing else is computed);
    ``profile`` its column sums. ``plateau`` is ``None`` when ``gradient``
    found no steepest window.
    """

    slopes: dict[str, float]
    laser_angle_deg: float
    rotated: Optional[np.ndarray]
    angle: Optional[ShockAngle]
    gradient: Optional[LocalSlope]
    plateau: Optional[Plateau]
    profile: Optional[np.ndarray]


def analyze_shock(phase_array: np.ndarray, window_size: int = 20) -> ShockAnalysis:
    """Rotate the masked phase onto the laser axis and measure the shock on it.

    The laser axis is the tilt of the column-wise centroids
    (``compute_beam_slopes``, whose four slopes the legacy merged into its
    results); the map is rotated by that angle on a reshaped canvas, so
    every position below is a column of the rotated canvas.
    """
    slopes = compute_beam_slopes(phase_array)
    laser_angle = np.arctan(slopes["image_com_slope_x"])
    laser_angle_deg = np.degrees(laser_angle)
    if not np.isfinite(laser_angle_deg):
        return ShockAnalysis(
            slopes, float(laser_angle_deg), None, None, None, None, None
        )
    rotated_phase = ndimage.rotate(
        phase_array, angle=laser_angle_deg, reshape=True, order=1
    )
    angle = shock_angle(rotated_phase, window_size)
    profile = np.sum(rotated_phase, axis=0)
    gradient = find_max_local_slope(profile, window_size)
    plateau = (
        None
        if gradient.center is None
        else delta_plateau(profile, gradient.center, window_size)
    )
    return ShockAnalysis(
        slopes, float(laser_angle_deg), rotated_phase, angle, gradient, plateau, profile
    )


def unrotate(
    points: np.ndarray,
    angle_deg: float,
    input_shape: tuple[int, int],
    rotated_shape: tuple[int, int],
) -> np.ndarray:
    """Map ``(row, col)`` points on the rotated canvas back onto the input map.

    The inverse of :func:`scipy.ndimage.rotate` with ``reshape=True``: the
    output point ``o`` samples the input at ``R @ o + offset``, both
    canvases rotating about their centres. ``points`` is ``(n, 2)``.
    """
    c, s = cosdg(angle_deg), sindg(angle_deg)
    rot_matrix = np.array([[c, s], [-s, c]])
    in_center = (np.asarray(input_shape, dtype=float) - 1) / 2
    out_center = rot_matrix @ ((np.asarray(rotated_shape, dtype=float) - 1) / 2)
    offset = in_center - out_center
    return (rot_matrix @ np.asarray(points, dtype=float).T).T + offset
