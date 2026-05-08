"""Unit tests for image_analysis.processing.array2d.thresholding."""

import numpy as np
import pytest

from image_analysis.processing.array2d.thresholding import (
    apply_constant_threshold,
    apply_percentage_threshold,
    get_threshold_value,
    validate_threshold_parameters,
)


class TestConstantThreshold:
    """Tests for apply_constant_threshold()."""

    def test_to_zero_mode_zeros_below_threshold(self):
        image = np.array([[10, 50, 200]], dtype=np.float64)
        result = apply_constant_threshold(image, threshold_value=100.0, mode="to_zero")
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert result[0, 2] == 200.0

    def test_binary_mode_zeros_below_threshold(self):
        """Binary mode zeros values below threshold and preserves values above."""
        image = np.array([[10.0, 200.0]], dtype=np.float64)
        result = apply_constant_threshold(image, threshold_value=100.0, mode="binary")
        assert result[0, 0] == 0.0
        assert result[0, 1] == 200.0

    def test_truncate_mode_clips_above_threshold(self):
        image = np.array([[50.0, 200.0, 300.0]])
        result = apply_constant_threshold(image, threshold_value=150.0, mode="truncate")
        assert result[0, 1] == 150.0
        assert result[0, 2] == 150.0
        assert result[0, 0] == 50.0

    def test_uniform_image_below_threshold_zeroed(self):
        image = np.full((5, 5), 30.0)
        result = apply_constant_threshold(image, threshold_value=50.0, mode="to_zero")
        assert np.all(result == 0.0)

    def test_uniform_image_above_threshold_unchanged(self):
        image = np.full((5, 5), 80.0)
        result = apply_constant_threshold(image, threshold_value=50.0, mode="to_zero")
        assert np.allclose(result, 80.0)


class TestPercentageThreshold:
    """Tests for apply_percentage_threshold()."""

    def test_50_percent_zeroes_lower_half(self):
        image = np.array([[0.0, 25.0, 50.0, 75.0, 100.0]])
        result = apply_percentage_threshold(image, percentage=50.0, mode="to_zero")
        # threshold = 50% of max (100) = 50 → values strictly < 50 become 0
        assert result[0, 0] == 0.0  # 0 < 50 → zeroed
        assert result[0, 1] == 0.0  # 25 < 50 → zeroed
        assert result[0, 2] == 50.0  # 50 >= 50 → preserved
        assert result[0, 4] == 100.0

    def test_zero_percent_passes_all(self):
        image = np.array([[1.0, 2.0, 3.0]])
        result = apply_percentage_threshold(image, percentage=0.0, mode="to_zero")
        assert np.allclose(result, image)

    def test_invalid_percentage_raises(self):
        image = np.ones((4, 4))
        with pytest.raises(ValueError):
            apply_percentage_threshold(image, percentage=101.0, mode="to_zero")
        with pytest.raises(ValueError):
            apply_percentage_threshold(image, percentage=-1.0, mode="to_zero")


class TestGetThresholdValue:
    """Tests for get_threshold_value()."""

    def test_constant_method_returns_value_unchanged(self):
        image = np.ones((4, 4)) * 200
        assert get_threshold_value(image, method="constant", value=75.0) == 75.0

    def test_percentage_method_scales_by_max(self):
        image = np.full((4, 4), 200.0)
        result = get_threshold_value(image, method="percentage_max", value=50.0)
        assert abs(result - 100.0) < 1e-6


class TestValidateThresholdParameters:
    """Tests for validate_threshold_parameters()."""

    def test_valid_constant_passes(self):
        validate_threshold_parameters("constant", 50.0, "to_zero")

    def test_valid_percentage_passes(self):
        validate_threshold_parameters("percentage_max", 75.0, "binary")

    def test_negative_constant_raises(self):
        with pytest.raises(ValueError):
            validate_threshold_parameters("constant", -1.0, "to_zero")

    def test_percentage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            validate_threshold_parameters("percentage_max", 110.0, "to_zero")
