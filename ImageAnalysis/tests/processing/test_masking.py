"""Unit tests for image_analysis.processing.array2d.masking."""

import numpy as np
import pytest

from image_analysis.processing.array2d.masking import (
    apply_circular_mask,
    apply_mask_array,
    apply_rectangular_mask,
    apply_roi_cropping,
    create_mask_from_threshold,
)
from image_analysis.processing.array2d.config_models import (
    CircularMaskConfig,
    ROIConfig,
)


class TestROICropping:
    """Tests for apply_roi_cropping()."""

    def test_crops_to_correct_shape(self):
        image = np.ones((100, 100), dtype=np.uint16)
        config = ROIConfig(x_min=10, x_max=40, y_min=20, y_max=60)
        result = apply_roi_cropping(image, config)
        assert result.shape == (40, 30)  # (y_max - y_min, x_max - x_min)

    def test_crops_correct_region(self):
        image = np.zeros((10, 10), dtype=np.float64)
        image[2:5, 3:7] = 99.0
        config = ROIConfig(x_min=3, x_max=7, y_min=2, y_max=5)
        result = apply_roi_cropping(image, config)
        assert np.all(result == 99.0)

    def test_full_image_roi_returns_same_data(self):
        image = np.arange(25, dtype=np.float64).reshape(5, 5)
        config = ROIConfig(x_min=0, x_max=5, y_min=0, y_max=5)
        result = apply_roi_cropping(image, config)
        assert np.array_equal(result, image)

    # ------------------------------------------------------------------
    # Clamping: ROI larger than actual image
    # ------------------------------------------------------------------

    def test_clamps_x_max_to_image_width(self):
        """ROI x_max beyond image width is clamped; no exception raised."""
        image = np.ones((10, 10), dtype=np.float64)
        # x_max=20 exceeds width=10
        config = ROIConfig(x_min=0, x_max=20, y_min=0, y_max=5)
        result = apply_roi_cropping(image, config)
        assert result.shape == (5, 10)  # clamped to full width

    def test_clamps_y_max_to_image_height(self):
        """ROI y_max beyond image height is clamped; no exception raised."""
        image = np.ones((10, 10), dtype=np.float64)
        config = ROIConfig(x_min=0, x_max=5, y_min=0, y_max=20)
        result = apply_roi_cropping(image, config)
        assert result.shape == (10, 5)  # clamped to full height

    def test_clamped_result_contains_correct_pixels(self):
        """After clamping the returned data matches the valid sub-region."""
        image = np.arange(100, dtype=np.float64).reshape(10, 10)
        # x_max=15 clamped to 10, y_max=15 clamped to 10
        config = ROIConfig(x_min=2, x_max=15, y_min=3, y_max=15)
        result = apply_roi_cropping(image, config)
        expected = image[3:10, 2:10]
        assert np.array_equal(result, expected)

    def test_clamp_emits_warning(self, caplog):
        """A logger warning is emitted when the ROI is clamped."""
        import logging

        image = np.ones((10, 10), dtype=np.float64)
        config = ROIConfig(x_min=0, x_max=20, y_min=0, y_max=5)
        with caplog.at_level(
            logging.WARNING, logger="image_analysis.processing.array2d.masking"
        ):
            apply_roi_cropping(image, config)
        assert any("clamped" in record.message.lower() for record in caplog.records)

    def test_zero_area_after_clamp_returns_full_image(self):
        """When clamping leaves a zero-area ROI, the full image is returned."""
        image = np.arange(100, dtype=np.float64).reshape(10, 10)
        # x_min=15 clamped to 15, x_max=20 clamped to 10 → x_min >= x_max → zero area
        config = ROIConfig(x_min=15, x_max=20, y_min=0, y_max=5)
        result = apply_roi_cropping(image, config)
        assert result.shape == image.shape


class TestCircularMask:
    """Tests for apply_circular_mask().

    CircularMaskConfig uses ``center=(col, row)`` (x, y) convention and
    ``mask_outside=True`` means pixels outside the circle are masked.
    """

    def test_mask_outside_zeroes_corners(self):
        image = np.ones((50, 50), dtype=np.float64) * 100.0
        config = CircularMaskConfig(
            enabled=True,
            center=(25, 25),  # (col, row)
            radius=10,
            mask_outside=True,
            mask_value=0.0,
        )
        result = apply_circular_mask(image, config)
        # Corner pixels are outside the circle → zeroed
        assert result[0, 0] == 0.0
        assert result[49, 49] == 0.0
        # Centre should be untouched
        assert result[25, 25] == 100.0

    def test_mask_inside_zeroes_centre(self):
        image = np.ones((50, 50), dtype=np.float64) * 100.0
        config = CircularMaskConfig(
            enabled=True,
            center=(25, 25),
            radius=5,
            mask_outside=False,
            mask_value=0.0,
        )
        result = apply_circular_mask(image, config)
        assert result[25, 25] == 0.0
        assert result[0, 0] == 100.0

    def test_disabled_returns_copy_unchanged(self):
        image = np.ones((10, 10), dtype=np.float64) * 42.0
        config = CircularMaskConfig(enabled=False, center=(5, 5), radius=3)
        result = apply_circular_mask(image, config)
        assert np.all(result == 42.0)


class TestRectangularMask:
    """Tests for apply_rectangular_mask()."""

    def test_zeroes_specified_region(self):
        image = np.ones((10, 10), dtype=np.float64) * 5.0
        result = apply_rectangular_mask(
            image, x_min=2, x_max=5, y_min=3, y_max=7, mask_value=0.0
        )
        assert np.all(result[3:7, 2:5] == 0.0)
        assert result[0, 0] == 5.0

    def test_custom_mask_value(self):
        image = np.zeros((8, 8), dtype=np.float64)
        result = apply_rectangular_mask(
            image, x_min=1, x_max=4, y_min=1, y_max=4, mask_value=99.0
        )
        assert np.all(result[1:4, 1:4] == 99.0)


class TestMaskArray:
    """Tests for apply_mask_array()."""

    def test_zeros_where_mask_is_0(self):
        image = np.ones((4, 4), dtype=np.float64) * 10.0
        mask = np.ones((4, 4), dtype=np.float64)
        mask[1:3, 1:3] = 0.0
        result = apply_mask_array(image, mask)
        assert np.all(result[1:3, 1:3] == 0.0)
        assert np.all(result[0, :] == 10.0)

    def test_shape_mismatch_raises(self):
        image = np.ones((5, 5))
        mask = np.ones((4, 4))
        with pytest.raises(ValueError):
            apply_mask_array(image, mask)


class TestCreateMaskFromThreshold:
    """Tests for create_mask_from_threshold()."""

    def test_mask_above_threshold(self):
        image = np.array([[10.0, 50.0, 200.0]])
        mask = create_mask_from_threshold(image, threshold=100.0, mask_above=True)
        # Above threshold → masked (0); at/below → unmasked (1)
        assert mask[0, 0] == 1  # 10 <= 100 → unmasked
        assert mask[0, 2] == 0  # 200 > 100 → masked

    def test_mask_below_threshold(self):
        image = np.array([[10.0, 50.0, 200.0]])
        mask = create_mask_from_threshold(image, threshold=100.0, mask_above=False)
        assert mask[0, 0] == 0  # 10 < 100 → masked
        assert mask[0, 2] == 1  # 200 >= 100 → unmasked
