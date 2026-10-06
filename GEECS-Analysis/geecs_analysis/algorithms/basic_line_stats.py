"""Line statistics utilities for 1D data analysis.

Provides Pydantic models for computing statistics from 1D line profiles
with optional unit tracking. This module serves as the foundation for both
direct 1D analysis and 2D projection analysis.

Features include:
- Unit tracking for x and y axes
- Pydantic model structure for validation and serialization
- Flexible dictionary export with prefix/suffix support
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
import logging

logger = logging.getLogger(__name__)


#: Relative tolerance, of the median step, within which every step of an axis
#: must fall for the axis to count as evenly spaced. Measured against the
#: step, not the axis's magnitude, so an offset cannot hide a gap. Wide
#: enough for an axis stored in single precision (a 4096-sample float32
#: axis's steps scatter by ~2e-4 of the step); far below the unevenness of a
#: stitched trace (tens of percent between cameras) or a nonlinear
#: calibration. An axis within it keeps the legacy width arithmetic, which
#: then differs from the moments over x by at most that fraction.
EVEN_SPACING_RTOL = 1e-3


def is_evenly_spaced(coordinates: np.ndarray) -> bool:
    """Whether every step of ``coordinates`` is within ``EVEN_SPACING_RTOL`` of the median step.

    Index-space widths times one spacing are exact only on such an axis
    (#1029); :class:`LineBasicStats` keeps that legacy arithmetic there, bit
    for bit, and takes its moments over the coordinates everywhere else.
    """
    x = np.asarray(coordinates, dtype=float)
    if x.size < 3:
        return True
    steps = np.diff(x)
    step = np.median(steps)
    if step == 0:
        return bool(np.all(steps == 0))
    return bool(np.all(np.abs(steps - step) <= EVEN_SPACING_RTOL * abs(step)))


def _coordinates(profile: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """The caller's one-per-sample coordinates, validated against the profile."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != profile.shape:
        raise ValueError(
            f"coordinates {coordinates.shape} must match the profile {profile.shape}"
        )
    return coordinates


def _interval_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoid weights over sorted ``x``: ``sum(w * f)`` is the integral of ``f``.

    Each sample owns half the interval to each neighbour, the ends half of
    their one interval; on an evenly spaced axis every interior weight is
    the step. Two cameras' samples interleaved over an overlap share its
    length instead of counting it twice.
    """
    if x.size < 2:
        return np.ones_like(x)
    w = np.empty_like(x)
    w[1:-1] = (x[2:] - x[:-2]) / 2
    w[0] = (x[1] - x[0]) / 2
    w[-1] = (x[-1] - x[-2]) / 2
    return w


def _over_x(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(order, x, w)``: the sample order that sorts ``coords``, the sorted coordinates, their weights.

    Sorted so the weights are non-negative whatever the trace's direction;
    the caller signs its result by that direction.
    """
    order = np.argsort(coords, kind="stable")
    x = coords[order]
    return order, x, _interval_weights(x)


def compute_center_of_mass(
    profile: np.ndarray, coordinates: np.ndarray | None = None
) -> float:
    """Compute the center of mass of a 1‑D profile.

    In sample-index units — the intensity-weighted mean index — or, over
    ``coordinates`` (one per sample), in axis units as the Δx-weighted first
    moment: the integral of ``x · profile`` over x divided by the integral
    of ``profile`` (trapezoid weights), which does not depend on how densely
    the trace is sampled (#1029).
    """
    profile = np.asarray(profile, dtype=float)
    if coordinates is None:
        total = profile.sum()
        if total <= 0:
            logger.warning(
                "compute_center_of_mass: Profile has non-positive total intensity. Returning np.nan."
            )
            return np.nan
        return np.sum(np.arange(profile.size) * profile) / total
    order, x, w = _over_x(_coordinates(profile, coordinates))
    weighted = w * profile[order]
    total = weighted.sum()
    if total <= 0:
        logger.warning(
            "compute_center_of_mass: Profile has non-positive integral over x. Returning np.nan."
        )
        return np.nan
    return np.sum(weighted * x) / total


def compute_rms(profile: np.ndarray, coordinates: np.ndarray | None = None) -> float:
    """Compute the RMS width of a 1‑D profile.

    The intensity-weighted second moment about the centroid. In sample-index
    units it is the legacy per-sample mean: negatives clipped (in place),
    divided by the unclipped total. Over ``coordinates`` (one per sample) it
    is, in axis units, the Δx-weighted moment — the integral over x of
    ``(x - centroid)² · profile`` with negatives clipped, divided by the
    integral of the unclipped profile, trapezoid weights — so it does not
    depend on how densely the trace is sampled or on how many samples the
    overlaps of a stitched trace contribute (#1029). On an evenly spaced
    axis the two agree to rounding; on an uneven one only the coordinate
    form is a width in axis units. Over coordinates the width is signed by
    the axis direction, as the legacy spacing conversion was.
    """
    profile = np.asarray(profile, dtype=float)
    total = profile.sum()
    coords = None if coordinates is None else _coordinates(profile, coordinates)
    if coords is not None:
        order, x, w = _over_x(coords)
        # The unclipped trace's integral over x: legacy's unclipped total.
        total_x = np.sum(w * profile[order])
    profile[profile < 0] = 0
    if total <= 0:
        logger.warning(
            "compute_rms: Profile has non-positive total intensity. Returning np.nan."
        )
        return np.nan
    if coords is None:
        com = compute_center_of_mass(profile)
        return np.sqrt(np.sum((np.arange(profile.size) - com) ** 2 * profile) / total)
    if total_x <= 0:
        logger.warning(
            "compute_rms: Profile has non-positive integral over x. Returning np.nan."
        )
        return np.nan
    com = compute_center_of_mass(profile, coords)
    width = np.sqrt(np.sum(w * profile[order] * (x - com) ** 2) / total_x)
    # Signed by the axis direction, as the legacy spacing conversion was.
    return width if coords[-1] >= coords[0] else -width


def compute_fwhm(profile: np.ndarray, coordinates: np.ndarray | None = None) -> float:
    """Compute the full width at half maximum (FWHM) of a 1‑D profile.

    The distance between the outermost half-maximum crossings, each
    interpolated linearly between the two samples it falls between — in
    sample-index units, or in axis units over ``coordinates`` (one per
    sample) when given (#1029). Signed by the direction of the coordinates
    between the two crossings, as the legacy spacing conversion was.
    """
    profile = np.asarray(profile, dtype=float)
    coords = (
        np.arange(profile.size)
        if coordinates is None
        else _coordinates(profile, coordinates)
    )
    if profile.sum() <= 0:
        logger.warning(
            "compute_fwhm: Profile has non-positive total intensity. Returning np.nan."
        )
        return np.nan

    profile = profile - profile.min()
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
            return coords[i1]
        return coords[i1] + (half_max - y1) / (y2 - y1) * (coords[i2] - coords[i1])

    left_edge = interp_edge(left - 1, left) if left > 0 else coords[left]
    right_edge = (
        interp_edge(right, right + 1) if right < len(profile) - 1 else coords[right]
    )

    return right_edge - left_edge


def compute_peak_location(profile: np.ndarray) -> float:
    """Return the index of the peak value in a 1‑D profile."""
    profile = np.asarray(profile, dtype=float)
    if profile.size == 0:
        logger.warning("compute_peak_location: Profile is empty. Returning np.nan.")
        return np.nan
    return float(np.argmax(profile))


class LineBasicStats(BaseModel):
    """Basic statistics of a 1D line profile with optional unit tracking.

    This model computes standard statistical measures (CoM, RMS, FWHM, etc.)
    from a 1D line profile. It automatically computes statistics upon creation
    and can track physical units for both x and y axes.

    The model works with Nx2 arrays where column 0 contains x-coordinates
    (which could be indices, wavelengths, energies, times, etc.) and column 1
    contains the corresponding y-values (intensity, counts, voltage, etc.).

    All spatial/spectral statistics (CoM, RMS, FWHM, peak_location) are
    returned in x-coordinate units. The integrated_intensity is the legacy
    sample sum, not a quadrature. On an evenly spaced axis the widths are the
    legacy index-space widths times the spacing at the centroid (kept bit for
    bit; signed on a descending axis); on any other axis — a trace stitched
    from several cameras, a nonlinear calibration — they are measured over
    the x coordinates themselves: ``rms`` as the Δx-weighted (trapezoid)
    moment, an integral over x that does not depend on the sampling
    density, ``fwhm`` from the half-maximum crossings interpolated in x
    (#1029).

    Attributes
    ----------
    line_data : NDArray
        Nx2 array where column 0 is x-coordinates, column 1 is y-values
    x_units : Optional[str]
        Physical units for x-axis (e.g., "nm", "μm", "eV", "s").
        None indicates dimensionless/uncalibrated data.
    y_units : Optional[str]
        Physical units for y-axis (e.g., "a.u.", "counts", "V", "W").
        None indicates unknown units.
    CoM : Optional[float]
        Center of mass in x-coordinates
    rms : Optional[float]
        RMS width in x-coordinates
    fwhm : Optional[float]
        Full-width half-maximum in x-coordinates
    peak_location : Optional[float]
        Location of peak value in x-coordinates
    integrated_intensity : Optional[float]
        Legacy sum of samples after the RMS negative-value clipping
    peak_value : Optional[float]
        Maximum y-value in the profile
    """

    # Input data
    line_data: NDArray
    x_units: Optional[str] = None
    y_units: Optional[str] = None

    # Computed statistics (set during model_post_init)
    CoM: Optional[float] = None
    rms: Optional[float] = None
    fwhm: Optional[float] = None
    peak_location: Optional[float] = None
    integrated_intensity: Optional[float] = None
    peak_value: Optional[float] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow numpy arrays
        frozen=False,  # Allow setting computed fields
    )

    @field_validator("line_data")
    @classmethod
    def validate_line_data(cls, v: NDArray) -> NDArray:
        """Ensure line_data is a valid Nx2 array."""
        # Own this scratch array: the legacy RMS routine clips negatives in place.
        v = np.array(v, dtype=float, copy=True)
        if v.ndim != 2:
            raise ValueError(f"line_data must be 2D array, got shape {v.shape}")
        if v.shape[1] != 2:
            raise ValueError(
                f"line_data must have 2 columns [x, y], got {v.shape[1]} columns"
            )
        if v.shape[0] < 2:
            raise ValueError(f"line_data must have at least 2 points, got {v.shape[0]}")
        return v

    def model_post_init(self, __context: object) -> None:
        """Compute statistics when model is created."""
        if self.CoM is None:  # Only compute if not explicitly provided
            self._compute()

    def _compute(self) -> None:
        """Compute all statistics from line_data."""
        x = self.line_data[:, 0]
        y = self.line_data[:, 1]

        # Check if x is index-based (x = [0, 1, 2, ...])
        is_index_based = np.allclose(x, np.arange(len(x)), rtol=1e-9, atol=1e-9)
        # Widths: index-space moments times one spacing are exact only on an
        # evenly spaced axis, where that legacy arithmetic is kept bit for
        # bit. Anywhere else the moments are taken over x itself, Δx-weighted
        # (#1029).
        evenly_spaced = is_index_based or is_evenly_spaced(x)
        width_coordinates = None if evenly_spaced else x

        # Centroid and peak in index space; widths in index space on an
        # evenly spaced axis (converted below), else already in x units.
        com_idx = compute_center_of_mass(y)
        rms = compute_rms(y, width_coordinates)
        fwhm = compute_fwhm(y, width_coordinates)
        peak_idx = compute_peak_location(y)

        # Peak value - use numpy indexing which handles float indices
        if not np.isnan(peak_idx):
            self.peak_value = y[int(peak_idx)]
        else:
            self.peak_value = np.nan

        # Integrated intensity is the sum of y-values
        self.integrated_intensity = y.sum()

        if is_index_based:
            # No conversion needed - values are already in the same space as x
            self.CoM = com_idx
            self.rms = rms
            self.fwhm = fwhm
            self.peak_location = peak_idx
        else:
            # Map from index space to x-coordinate space
            if not np.isnan(com_idx):
                self.CoM = np.interp(com_idx, np.arange(len(x)), x)
            else:
                self.CoM = np.nan

            if np.isnan(com_idx):
                self.rms = np.nan
                self.fwhm = np.nan
            elif evenly_spaced:
                # Legacy: index-space widths scaled by dx at the CoM location
                idx = int(np.clip(com_idx, 0, len(x) - 2))
                dx = x[idx + 1] - x[idx]
                self.rms = rms * dx
                self.fwhm = fwhm * dx
            else:
                self.rms = rms
                self.fwhm = fwhm

            if not np.isnan(peak_idx):
                self.peak_location = x[int(peak_idx)]
            else:
                self.peak_location = np.nan

    def to_dict(self) -> dict[str, float]:
        """Flatten statistics to a bare-keyed dict.

        Returns keys ``CoM``, ``rms``, ``fwhm``, ``peak_location``,
        ``integrated_intensity``, ``peak_value``. Naming/disambiguation
        across analyzers is ScanAnalysis's responsibility per #412.
        """
        fields = [
            "CoM",
            "rms",
            "fwhm",
            "peak_location",
            "integrated_intensity",
            "peak_value",
        ]
        return {field: getattr(self, field) for field in fields}
