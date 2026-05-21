"""Unit tests for image_analysis.processing.array2d.filtering."""

import numpy as np
import pytest

from image_analysis.processing.array2d.filtering import (
    apply_gaussian_filter,
    apply_median_filter,
)


class TestGaussianFilter:
    """Tests for apply_gaussian_filter()."""

    def test_smooths_sharp_spike(self):
        """A single bright pixel should spread into neighbours after filtering."""
        image = np.zeros((20, 20), dtype=np.float64)
        image[10, 10] = 1000.0
        result = apply_gaussian_filter(image, sigma=2.0)
        # Peak moves to neighbours — centre value should be less than original
        assert result[10, 10] < 1000.0
        # Total energy is approximately conserved
        assert abs(result.sum() - image.sum()) / image.sum() < 0.01

    def test_uniform_image_unchanged(self):
        """A uniform image is not changed by a Gaussian filter."""
        image = np.full((10, 10), 50.0, dtype=np.float64)
        result = apply_gaussian_filter(image, sigma=3.0)
        assert np.allclose(result, 50.0, atol=1e-6)

    def test_preserves_dtype(self):
        image = np.ones((8, 8), dtype=np.uint16) * 100
        result = apply_gaussian_filter(image, sigma=1.0)
        assert result.dtype == np.uint16

    def test_invalid_sigma_raises(self):
        image = np.ones((5, 5))
        with pytest.raises(ValueError):
            apply_gaussian_filter(image, sigma=0.0)
        with pytest.raises(ValueError):
            apply_gaussian_filter(image, sigma=-1.0)

    def test_larger_sigma_more_blurred(self):
        """Higher sigma should produce a flatter result around a peak."""
        image = np.zeros((30, 30), dtype=np.float64)
        image[15, 15] = 1000.0
        r1 = apply_gaussian_filter(image, sigma=1.0)
        r3 = apply_gaussian_filter(image, sigma=3.0)
        assert r3[15, 15] < r1[15, 15]


class TestMedianFilter:
    """Tests for apply_median_filter()."""

    def test_removes_salt_noise(self):
        """Isolated bright pixels (salt noise) should be suppressed."""
        image = np.zeros((10, 10), dtype=np.uint16)
        image[5, 5] = 65535
        result = apply_median_filter(image, kernel_size=3)
        assert result[5, 5] < 65535

    def test_uniform_image_unchanged(self):
        image = np.full((8, 8), 42, dtype=np.uint16)
        result = apply_median_filter(image, kernel_size=3)
        assert np.all(result == 42)

    def test_even_kernel_raises(self):
        image = np.ones((5, 5))
        with pytest.raises(ValueError):
            apply_median_filter(image, kernel_size=2)

    def test_zero_kernel_raises(self):
        image = np.ones((5, 5))
        with pytest.raises(ValueError):
            apply_median_filter(image, kernel_size=0)

    def test_kernel_size_1_is_identity(self):
        image = np.arange(25, dtype=np.float64).reshape(5, 5)
        result = apply_median_filter(image, kernel_size=1)
        assert np.allclose(result, image)
