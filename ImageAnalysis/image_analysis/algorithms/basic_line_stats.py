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


# Core computation functions (copied from basic_beam_stats.py)
def compute_center_of_mass(profile: np.ndarray) -> float:
    """Compute the center of mass of a 1‑D profile."""
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
    """Compute the RMS width of a 1‑D profile."""
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
    """Compute the full width at half maximum (FWHM) of a 1‑D profile."""
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
    returned in x-coordinate units. The integrated_intensity accounts for
    the actual x-spacing using trapezoidal integration.

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
        Total area under curve (trapezoidal integration over x)
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
    def validate_line_data(cls, v):
        """Ensure line_data is a valid Nx2 array."""
        v = np.asarray(v, dtype=float)
        if v.ndim != 2:
            raise ValueError(f"line_data must be 2D array, got shape {v.shape}")
        if v.shape[1] != 2:
            raise ValueError(
                f"line_data must have 2 columns [x, y], got {v.shape[1]} columns"
            )
        if v.shape[0] < 2:
            raise ValueError(f"line_data must have at least 2 points, got {v.shape[0]}")
        return v

    def model_post_init(self, __context):
        """Compute statistics when model is created."""
        if self.CoM is None:  # Only compute if not explicitly provided
            self._compute()

    def _compute(self):
        """Compute all statistics from line_data using existing basic_beam_stats functions."""
        x = self.line_data[:, 0]
        y = self.line_data[:, 1]

        # Use existing functions from basic_beam_stats (they work on 1D arrays)
        # These return values in index space
        com_idx = compute_center_of_mass(y)
        rms_idx = compute_rms(y)
        fwhm_idx = compute_fwhm(y)
        peak_idx = compute_peak_location(y)

        # Peak value - use numpy indexing which handles float indices
        if not np.isnan(peak_idx):
            self.peak_value = y[int(peak_idx)]
        else:
            self.peak_value = np.nan

        # Integrated intensity is the sum of y-values
        self.integrated_intensity = y.sum()

        # Check if x is index-based (x = [0, 1, 2, ...])
        is_index_based = np.allclose(x, np.arange(len(x)), rtol=1e-9, atol=1e-9)

        if is_index_based:
            # No conversion needed - values are already in the same space as x
            self.CoM = com_idx
            self.rms = rms_idx
            self.fwhm = fwhm_idx
            self.peak_location = peak_idx
        else:
            # Map from index space to x-coordinate space
            if not np.isnan(com_idx):
                self.CoM = np.interp(com_idx, np.arange(len(x)), x)
            else:
                self.CoM = np.nan

            if not np.isnan(rms_idx) and not np.isnan(com_idx):
                # For RMS and FWHM, we need to scale by dx
                # Use dx at the CoM location
                idx = int(np.clip(com_idx, 0, len(x) - 2))
                dx = x[idx + 1] - x[idx]
                self.rms = rms_idx * dx
            else:
                self.rms = np.nan

            if not np.isnan(fwhm_idx) and not np.isnan(com_idx):
                idx = int(np.clip(com_idx, 0, len(x) - 2))
                dx = x[idx + 1] - x[idx]
                self.fwhm = fwhm_idx * dx
            else:
                self.fwhm = np.nan

            if not np.isnan(peak_idx):
                self.peak_location = x[int(peak_idx)]
            else:
                self.peak_location = np.nan

    def to_dict(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> dict[str, float]:
        """Flatten statistics to a dictionary for scalar output."""
        fields = [
            "CoM",
            "rms",
            "fwhm",
            "peak_location",
            "integrated_intensity",
            "peak_value",
        ]

        scalars = {}
        suffix_str = f"_{suffix}" if suffix else ""

        for field in fields:
            value = getattr(self, field)

            # Build key: [prefix_]field[_suffix]
            if prefix:
                key = f"{prefix}_{field}{suffix_str}"
            else:
                key = f"{field}{suffix_str}"

            scalars[key] = value

        return scalars
