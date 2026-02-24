"""Tests for basic_line_stats module."""

import numpy as np
import pytest
from image_analysis.algorithms.basic_line_stats import LineBasicStats


class TestLineBasicStats:
    """Test suite for LineBasicStats Pydantic model."""

    def test_gaussian_profile_with_units(self):
        """Test statistics on a Gaussian profile with wavelength units."""
        # Create a Gaussian spectrum centered at 550 nm with 20 nm sigma
        wavelengths = np.linspace(400, 700, 1000)
        center = 550.0
        sigma = 20.0
        intensities = np.exp(-((wavelengths - center) ** 2) / (2 * sigma**2))
        line_data = np.column_stack([wavelengths, intensities])

        stats = LineBasicStats(line_data=line_data, x_units="nm", y_units="a.u.")

        # Check CoM is near the center
        assert abs(stats.CoM - center) < 0.5

        # Check RMS is close to sigma
        assert abs(stats.rms - sigma) < 1.0

        # Check FWHM is approximately 2.355 * sigma for Gaussian
        expected_fwhm = 2.355 * sigma
        assert abs(stats.fwhm - expected_fwhm) < 2.0

        # Check peak location
        assert abs(stats.peak_location - center) < 1.0

        # Check units are stored
        assert stats.x_units == "nm"
        assert stats.y_units == "a.u."

    def test_uncalibrated_data(self):
        """Test with index-based coordinates (no units)."""
        # Simple peak at index 50
        x = np.arange(100)
        y = np.exp(-((x - 50) ** 2) / (2 * 10**2))
        line_data = np.column_stack([x, y])

        stats = LineBasicStats(line_data=line_data)

        assert abs(stats.CoM - 50) < 0.5
        assert stats.x_units is None
        assert stats.y_units is None

    def test_to_dict_with_prefix_and_suffix(self):
        """Test flattening to dict with prefix and suffix."""
        x = np.arange(100)
        y = np.exp(-((x - 50) ** 2) / (2 * 10**2))
        line_data = np.column_stack([x, y])

        stats = LineBasicStats(line_data=line_data, x_units="μm")

        # Test with prefix only
        result = stats.to_dict(prefix="beam")
        assert "beam_CoM" in result
        assert "beam_fwhm" in result

        # Test with suffix only
        result = stats.to_dict(suffix="calibrated")
        assert "CoM_calibrated" in result
        assert "fwhm_calibrated" in result

        # Test with both
        result = stats.to_dict(prefix="beam", suffix="var")
        assert "beam_CoM_var" in result
        assert "beam_fwhm_var" in result

    def test_nonuniform_spacing(self):
        """Test with non-uniformly spaced x-coordinates."""
        # Quadratic spacing
        x = np.linspace(0, 10, 100) ** 2
        center_x = 50.0
        y = np.exp(-((x - center_x) ** 2) / (2 * 10**2))
        line_data = np.column_stack([x, y])

        stats = LineBasicStats(line_data=line_data)

        # Should still compute correctly with trapz integration
        assert abs(stats.CoM - center_x) < 2.0
        assert stats.integrated_intensity > 0

    def test_negative_values_handled(self):
        """Test that negative values are handled (set to zero)."""
        x = np.arange(100)
        y = np.sin(x / 5.0)  # Has negative values
        line_data = np.column_stack([x, y])

        stats = LineBasicStats(line_data=line_data)

        # Should not crash and should return valid stats
        assert not np.isnan(stats.CoM)
        assert stats.integrated_intensity >= 0

    def test_zero_profile_returns_nan(self):
        """Test that zero/negative profile returns NaN statistics."""
        x = np.arange(100)
        y = np.zeros(100)
        line_data = np.column_stack([x, y])

        stats = LineBasicStats(line_data=line_data)

        assert np.isnan(stats.CoM)
        assert np.isnan(stats.rms)
        assert np.isnan(stats.fwhm)
        assert stats.integrated_intensity == 0.0

    def test_validation_rejects_invalid_shape(self):
        """Test that invalid array shapes are rejected."""
        # Wrong number of columns
        with pytest.raises(ValueError, match="2 columns"):
            LineBasicStats(line_data=np.array([[1, 2, 3], [4, 5, 6]]))

        # 1D array
        with pytest.raises(ValueError, match="2D array"):
            LineBasicStats(line_data=np.array([1, 2, 3]))

        # Too few points
        with pytest.raises(ValueError, match="at least 2 points"):
            LineBasicStats(line_data=np.array([[1, 2]]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
