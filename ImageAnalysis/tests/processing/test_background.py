"""Unit tests for image_analysis.processing.array2d.background."""

import numpy as np
import pytest

from image_analysis.processing.array2d.background import (
    _compute_constant_background,
    _compute_median_background,
    _compute_percentile_background,
    compute_and_cache_scan_background,
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


# ---------------------------------------------------------------------------
# compute_and_cache_scan_background
# ---------------------------------------------------------------------------


def _write_image(path, value):
    """Write a tiny .npy as a stand-in for a real image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((4, 4), float(value)))


def _npy_loader(path):
    return np.load(path)


class TestComputeAndCacheScanBackground:
    """End-to-end discover → load → aggregate → cache behavior."""

    def test_median_aggregation_writes_npy(self, tmp_path):
        img_dir = tmp_path / "images"
        for i, v in enumerate([1.0, 2.0, 3.0], start=1):
            _write_image(img_dir / f"shot_{i:03d}.npy", v)

        output = tmp_path / "analysis" / "bg.npy"
        result = compute_and_cache_scan_background(
            image_dir=img_dir,
            file_tail=".npy",
            image_loader=_npy_loader,
            output_path=output,
            method="median",
        )

        assert result == output
        loaded = np.load(output)
        assert np.allclose(loaded, 2.0)  # median of [1, 2, 3]

    def test_mean_aggregation_matches_np_mean(self, tmp_path):
        img_dir = tmp_path / "images"
        for i, v in enumerate([1.0, 2.0, 3.0], start=1):
            _write_image(img_dir / f"shot_{i:03d}.npy", v)

        output = tmp_path / "analysis" / "bg.npy"
        compute_and_cache_scan_background(
            image_dir=img_dir,
            file_tail=".npy",
            image_loader=_npy_loader,
            output_path=output,
            method="mean",
        )
        assert np.allclose(np.load(output), 2.0)

    def test_percentile_aggregation_requires_value(self, tmp_path):
        img_dir = tmp_path / "images"
        _write_image(img_dir / "shot_001.npy", 1.0)

        with pytest.raises(ValueError, match="percentile aggregation requires"):
            compute_and_cache_scan_background(
                image_dir=img_dir,
                file_tail=".npy",
                image_loader=_npy_loader,
                output_path=tmp_path / "bg.npy",
                method="percentile",
            )

    def test_percentile_value_used(self, tmp_path):
        img_dir = tmp_path / "images"
        for i, v in enumerate([1.0, 2.0, 3.0, 4.0], start=1):
            _write_image(img_dir / f"shot_{i:03d}.npy", v)

        output = tmp_path / "bg.npy"
        compute_and_cache_scan_background(
            image_dir=img_dir,
            file_tail=".npy",
            image_loader=_npy_loader,
            output_path=output,
            method="percentile",
            percentile=0,  # min
        )
        assert np.allclose(np.load(output), 1.0)

    def test_cache_hit_skips_recompute(self, tmp_path):
        img_dir = tmp_path / "images"
        _write_image(img_dir / "shot_001.npy", 1.0)
        output = tmp_path / "bg.npy"

        # Pre-write a sentinel; the function should NOT overwrite it.
        np.save(output, np.full((4, 4), 999.0))

        result = compute_and_cache_scan_background(
            image_dir=img_dir,
            file_tail=".npy",
            image_loader=_npy_loader,
            output_path=output,
            method="median",
        )

        assert result == output
        assert np.allclose(np.load(output), 999.0), "cache was overwritten"

    def test_missing_files_raises(self, tmp_path):
        img_dir = tmp_path / "empty"
        img_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No background source images"):
            compute_and_cache_scan_background(
                image_dir=img_dir,
                file_tail=".png",
                image_loader=_npy_loader,
                output_path=tmp_path / "bg.npy",
            )

    def test_loader_failures_skipped(self, tmp_path):
        img_dir = tmp_path / "images"
        _write_image(img_dir / "good_001.npy", 5.0)
        # Empty file — loader should fail and the function should skip it.
        (img_dir / "bad_002.npy").touch()

        result = compute_and_cache_scan_background(
            image_dir=img_dir,
            file_tail=".npy",
            image_loader=_npy_loader,
            output_path=tmp_path / "bg.npy",
            method="mean",
        )
        assert np.allclose(np.load(result), 5.0)

    def test_all_loads_fail_raises(self, tmp_path):
        img_dir = tmp_path / "images"
        (img_dir).mkdir()
        (img_dir / "bad_001.npy").touch()

        with pytest.raises(ValueError, match="No usable images"):
            compute_and_cache_scan_background(
                image_dir=img_dir,
                file_tail=".npy",
                image_loader=_npy_loader,
                output_path=tmp_path / "bg.npy",
            )

    def test_unknown_method_raises(self, tmp_path):
        img_dir = tmp_path / "images"
        _write_image(img_dir / "shot_001.npy", 1.0)

        with pytest.raises(ValueError, match="Unknown background aggregation method"):
            compute_and_cache_scan_background(
                image_dir=img_dir,
                file_tail=".npy",
                image_loader=_npy_loader,
                output_path=tmp_path / "bg.npy",
                method="bogus",
            )

    def test_output_parent_directory_is_created(self, tmp_path):
        img_dir = tmp_path / "images"
        _write_image(img_dir / "shot_001.npy", 1.0)

        output = tmp_path / "deeply" / "nested" / "path" / "bg.npy"
        result = compute_and_cache_scan_background(
            image_dir=img_dir,
            file_tail=".npy",
            image_loader=_npy_loader,
            output_path=output,
            method="median",
        )
        assert result.exists()
        assert result.parent.is_dir()
