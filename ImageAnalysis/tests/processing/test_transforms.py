"""Unit tests for image_analysis.processing.array2d.transforms."""

import numpy as np

from image_analysis.processing.array2d.transforms import (
    apply_horizontal_flip,
    apply_rotation,
    apply_vertical_flip,
)


class TestRotation:
    """Tests for apply_rotation()."""

    def test_zero_angle_returns_copy(self):
        image = np.arange(25, dtype=np.float64).reshape(5, 5)
        result = apply_rotation(image, angle=0.0)
        assert np.array_equal(result, image)
        assert result is not image  # should be a copy

    def test_180_degree_rotation_of_constant_image(self):
        """Rotating a constant image by any angle leaves it unchanged."""
        image = np.full((20, 20), 50.0, dtype=np.float64)
        rotated = apply_rotation(image, angle=180.0)
        # Edges may differ due to fill_value, so check the interior
        assert np.allclose(rotated[5:15, 5:15], 50.0, atol=1e-6)

    def test_360_degree_rotation_of_constant_image(self):
        """Full rotation of a constant image returns the same constant."""
        image = np.full((20, 20), 75.0, dtype=np.float64)
        result = apply_rotation(image, angle=360.0)
        assert np.allclose(result[5:15, 5:15], 75.0, atol=1e-6)

    def test_preserves_dtype(self):
        image = np.ones((8, 8), dtype=np.uint16) * 500
        result = apply_rotation(image, angle=45.0)
        assert result.dtype == np.uint16

    def test_90_degree_rotation_transposes_shape(self):
        """A non-square image rotated 90° with reshape=True changes shape."""
        image = np.ones((10, 20), dtype=np.float64)
        result = apply_rotation(image, angle=90.0, reshape=True)
        assert result.shape == (20, 10)


class TestHorizontalFlip:
    """Tests for apply_horizontal_flip()."""

    def test_reverses_columns(self):
        image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = apply_horizontal_flip(image)
        expected = np.array([[3, 2, 1], [6, 5, 4]], dtype=np.float64)
        assert np.array_equal(result, expected)

    def test_double_flip_is_identity(self):
        image = np.arange(15, dtype=np.float64).reshape(3, 5)
        assert np.array_equal(
            apply_horizontal_flip(apply_horizontal_flip(image)), image
        )

    def test_symmetric_image_unchanged(self):
        image = np.array([[1, 2, 1], [3, 4, 3]], dtype=np.float64)
        assert np.array_equal(apply_horizontal_flip(image), image)


class TestVerticalFlip:
    """Tests for apply_vertical_flip()."""

    def test_reverses_rows(self):
        image = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        result = apply_vertical_flip(image)
        expected = np.array([[5, 6], [3, 4], [1, 2]], dtype=np.float64)
        assert np.array_equal(result, expected)

    def test_double_flip_is_identity(self):
        image = np.arange(15, dtype=np.float64).reshape(5, 3)
        assert np.array_equal(apply_vertical_flip(apply_vertical_flip(image)), image)

    def test_vertical_flip_reverses_rows(self):
        """Applying vertical flip twice recovers the original."""
        image = np.arange(15, dtype=np.float64).reshape(5, 3)
        assert np.array_equal(apply_vertical_flip(apply_vertical_flip(image)), image)
