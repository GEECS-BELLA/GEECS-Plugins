"""Unit tests for image_analysis.processing.array2d.background."""

import numpy as np
import pytest

from image_analysis.processing.array2d.background import (
    _compute_constant_background,
    _compute_median_background,
    _compute_percentile_background,
    subtract_background,
)


class TestSubtractBackground:
    """Tests for subtract_background()."""

    def test_subtracts_constant_correctly(self):
        image = np.full((10, 10), 100, dtype=np.uint16)
        bg = np.full((10, 10), 40.0, dtype=np.float64)
        result = subtract_background(image, bg)
        assert np.allclose(result, 60.0)

    def test_preserves_input_dtype(self):
        image = np.ones((5, 5), dtype=np.float32) * 50
        bg = np.ones((5, 5), dtype=np.float64) * 10
        result = subtract_background(image, bg)
        assert result.dtype == np.float32

    def test_allows_negative_values(self):
        """Subtraction should not clip — negative values are preserved."""
        image = np.full((4, 4), 5, dtype=np.float64)
        bg = np.full((4, 4), 10.0, dtype=np.float64)
        result = subtract_background(image, bg)
        assert np.all(result < 0)

    def test_shape_mismatch_raises(self):
        image = np.ones((5, 5))
        bg = np.ones((4, 4))
        with pytest.raises(ValueError):
            subtract_background(image, bg)


class TestConstantBackground:
    """Tests for _compute_constant_background()."""

    def test_returns_correct_shape(self):
        bg = _compute_constant_background((8, 12), level=50.0)
        assert bg.shape == (8, 12)

    def test_returns_correct_value(self):
        bg = _compute_constant_background((4, 4), level=123.0)
        assert np.all(bg == 123.0)

    def test_dtype_is_float64(self):
        bg = _compute_constant_background((4, 4), level=0.0)
        assert bg.dtype == np.float64

    def test_zero_level(self):
        bg = _compute_constant_background((3, 3), level=0.0)
        assert np.all(bg == 0.0)


class TestPercentileBackground:
    """Tests for _compute_percentile_background()."""

    def test_uniform_stack_returns_that_value(self):
        images = [np.full((4, 4), 42.0)] * 10
        bg = _compute_percentile_background(images, percentile=5.0)
        assert np.allclose(bg, 42.0)

    def test_low_percentile_less_than_high_percentile(self):
        """Lower percentile should produce a smaller background than higher."""
        images = [np.full((4, 4), float(v)) for v in range(1, 11)]
        bg_low = _compute_percentile_background(images, percentile=10.0)
        bg_high = _compute_percentile_background(images, percentile=90.0)
        assert np.all(bg_low <= bg_high)

    def test_50th_percentile_is_median(self):
        images = [np.full((4, 4), float(v)) for v in range(1, 6)]
        bg = _compute_percentile_background(images, percentile=50.0)
        assert np.allclose(bg, 3.0)

    def test_output_shape_matches_input(self):
        images = [np.ones((6, 8)) * i for i in range(5)]
        bg = _compute_percentile_background(images, percentile=50.0)
        assert bg.shape == (6, 8)


class TestMedianBackground:
    """Tests for _compute_median_background()."""

    def test_median_of_uniform_stack(self):
        images = [np.full((4, 4), float(v)) for v in [1, 5, 3, 7, 2]]
        bg = _compute_median_background(images)
        assert np.allclose(bg, 3.0)

    def test_output_shape_matches_input(self):
        images = [np.ones((6, 8)) * i for i in range(5)]
        bg = _compute_median_background(images)
        assert bg.shape == (6, 8)

    def test_single_image_returns_that_image(self):
        img = np.arange(16, dtype=float).reshape(4, 4)
        bg = _compute_median_background([img])
        assert np.allclose(bg, img)
