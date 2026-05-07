import numpy as np
import pytest
import logging

from image_analysis.algorithms import basic_beam_stats as bbs

# Ensure warnings get captured during testing
logging.basicConfig(level=logging.WARNING)


# Test the main beam_profile_stats function and its output structure


def test_beam_profile_stats_structure():
    img = np.zeros((3, 3))
    img[1, 1] = 10
    result = bbs.beam_profile_stats(img)

    # Check top-level type
    assert isinstance(result, bbs.BeamStats)
    assert isinstance(result.image, bbs.ImageStats)
    assert isinstance(result.x, bbs.ProjectionStats)
    assert isinstance(result.y, bbs.ProjectionStats)

    # Check fields exist and are floats
    for nt in (result.image, result.x, result.y):
        for value in nt:
            assert isinstance(value, (float, int, np.floating))


def test_image_stats_fields():
    """ImageStats has exactly total and peak_value."""
    img = np.zeros((3, 3))
    img[1, 1] = 10
    result = bbs.beam_profile_stats(img)

    assert result.image._fields == ("total", "peak_value")
    assert result.image.total == 10.0
    assert result.image.peak_value == 10.0


def test_zero_intensity_image():
    img = np.zeros((5, 5))
    result = bbs.beam_profile_stats(img)

    # Image total should be 0.0, projections should be NaN
    assert result.image.total == 0.0
    assert np.isnan(result.image.peak_value)
    assert all(np.isnan(v) for v in result.x)
    assert all(np.isnan(v) for v in result.y)


def test_negative_total_intensity_handling():
    """Test that images with negative total intensity return NaN stats."""
    img = np.array([[-1, -2], [-3, -4]])
    result = bbs.beam_profile_stats(img)

    # Should return NaN for all projection stats
    assert np.isnan(result.x.CoM)
    assert np.isnan(result.x.rms)
    assert np.isnan(result.x.fwhm)
    assert np.isnan(result.y.CoM)


def test_flatten_beam_stats():
    img = np.zeros((3, 3))
    img[1, 1] = 5
    stats = bbs.beam_profile_stats(img)
    flat = bbs.flatten_beam_stats(stats, prefix="cam")

    # Check all keys are prefixed and values are floats
    assert all(k.startswith("cam_") for k in flat.keys())
    assert all(isinstance(v, (float, int, np.floating)) for v in flat.values())

    # Ensure expected fields exist (no filtering — all fields present)
    expected_keys = {
        "cam_image_total",
        "cam_image_peak_value",
        "cam_x_CoM",
        "cam_y_CoM",
        "cam_x_45_CoM",
        "cam_y_45_CoM",
    }
    assert expected_keys.issubset(flat.keys())

    # Total count: image (2 fields) + 4 projections × 4 fields = 18
    assert len(flat) == 18


def test_flatten_beam_stats_no_prefix():
    """Flatten without prefix produces undecorated keys."""
    img = np.zeros((3, 3))
    img[1, 1] = 5
    stats = bbs.beam_profile_stats(img)
    flat = bbs.flatten_beam_stats(stats)

    assert "image_total" in flat
    assert "x_CoM" in flat
    assert "y_45_fwhm" in flat


def test_flatten_beam_stats_include_filter():
    """Only requested fragments appear when include is provided."""
    img = np.zeros((3, 3))
    img[1, 1] = 5
    stats = bbs.beam_profile_stats(img)

    # Request only two fragments
    flat = bbs.flatten_beam_stats(stats, prefix="cam", include={"image_total", "x_CoM"})

    assert set(flat.keys()) == {"cam_image_total", "cam_x_CoM"}
    assert flat["cam_image_total"] == 5.0


def test_flatten_beam_stats_include_empty_set():
    """An empty include set produces an empty dict."""
    img = np.zeros((3, 3))
    img[1, 1] = 5
    stats = bbs.beam_profile_stats(img)
    flat = bbs.flatten_beam_stats(stats, include=set())
    assert flat == {}


# ---------------------------------------------------------------------------
# roi_offset parameter in beam_profile_stats
# ---------------------------------------------------------------------------


class TestBeamProfileStatsROIOffset:
    """Position stats shift by the ROI offset; width stats are unaffected."""

    def _single_pixel_image(self, shape=(20, 30), hot_row=10, hot_col=15):
        """Image with a single bright pixel at (hot_row, hot_col)."""
        img = np.zeros(shape, dtype=float)
        img[hot_row, hot_col] = 1.0
        return img

    def test_default_offset_is_zero(self):
        """beam_profile_stats() with no roi_offset equals (0, 0)."""
        img = self._single_pixel_image(hot_row=5, hot_col=8)
        stats_default = bbs.beam_profile_stats(img)
        stats_zero = bbs.beam_profile_stats(img, roi_offset=(0, 0))
        assert stats_default.x.CoM == pytest.approx(stats_zero.x.CoM)
        assert stats_default.y.CoM == pytest.approx(stats_zero.y.CoM)

    def test_x_offset_shifts_x_com(self):
        """x_CoM increases by x_offset when roi_offset is applied."""
        img = self._single_pixel_image(hot_row=5, hot_col=8)
        x_offset = 100
        stats_local = bbs.beam_profile_stats(img)
        stats_global = bbs.beam_profile_stats(img, roi_offset=(x_offset, 0))
        assert stats_global.x.CoM == pytest.approx(stats_local.x.CoM + x_offset)

    def test_y_offset_shifts_y_com(self):
        """y_CoM increases by y_offset when roi_offset is applied."""
        img = self._single_pixel_image(hot_row=5, hot_col=8)
        y_offset = 50
        stats_local = bbs.beam_profile_stats(img)
        stats_global = bbs.beam_profile_stats(img, roi_offset=(0, y_offset))
        assert stats_global.y.CoM == pytest.approx(stats_local.y.CoM + y_offset)

    def test_x_offset_shifts_peak_location(self):
        """x peak_location is also shifted by x_offset."""
        img = self._single_pixel_image(hot_row=5, hot_col=8)
        x_offset = 37
        stats_local = bbs.beam_profile_stats(img)
        stats_global = bbs.beam_profile_stats(img, roi_offset=(x_offset, 0))
        assert stats_global.x.peak_location == pytest.approx(
            stats_local.x.peak_location + x_offset
        )

    def test_offset_does_not_affect_rms(self):
        """rms (beam width) is independent of ROI position offset."""
        img = np.zeros((40, 60), dtype=float)
        # Broad Gaussian-like profile — use a real 2D image
        from image_analysis.tools.synthetic_generators import gaussian_beam_2d

        img = gaussian_beam_2d(shape=(40, 60), center=(20.0, 30.0), seed=7)
        stats_no_offset = bbs.beam_profile_stats(img)
        stats_offset = bbs.beam_profile_stats(img, roi_offset=(200, 300))
        assert stats_offset.x.rms == pytest.approx(stats_no_offset.x.rms, rel=1e-9)
        assert stats_offset.y.rms == pytest.approx(stats_no_offset.y.rms, rel=1e-9)

    def test_offset_does_not_affect_fwhm(self):
        """fwhm (beam width) is independent of ROI position offset."""
        from image_analysis.tools.synthetic_generators import gaussian_beam_2d

        img = gaussian_beam_2d(shape=(40, 60), center=(20.0, 30.0), seed=8)
        stats_no_offset = bbs.beam_profile_stats(img)
        stats_offset = bbs.beam_profile_stats(img, roi_offset=(500, 100))
        assert stats_offset.x.fwhm == pytest.approx(stats_no_offset.x.fwhm, rel=1e-9)
        assert stats_offset.y.fwhm == pytest.approx(stats_no_offset.y.fwhm, rel=1e-9)

    def test_image_stats_unaffected_by_offset(self):
        """Total intensity and peak value are not changed by roi_offset."""
        img = self._single_pixel_image(hot_row=3, hot_col=12)
        stats_local = bbs.beam_profile_stats(img)
        stats_global = bbs.beam_profile_stats(img, roi_offset=(99, 77))
        assert stats_global.image.total == pytest.approx(stats_local.image.total)
        assert stats_global.image.peak_value == pytest.approx(
            stats_local.image.peak_value
        )


if __name__ == "__main__":
    pytest.main()
