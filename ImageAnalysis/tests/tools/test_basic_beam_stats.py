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
    # Ensure expected fields exist
    expected_fields = {"image_total", "image_peak_value", "x_CoM", "y_CoM"}
    assert any(k.endswith(f) for f in expected_fields for k in flat.keys())


if __name__ == "__main__":
    pytest.main()
