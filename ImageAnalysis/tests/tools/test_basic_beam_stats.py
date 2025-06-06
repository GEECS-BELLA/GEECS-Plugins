import numpy as np
import pytest
import logging

from image_analysis.tools import basic_beam_stats as bbs

# Ensure warnings get captured during testing
logging.basicConfig(level=logging.WARNING)

def test_compute_centroid_valid():
    profile = np.array([0, 1, 2, 3, 4])
    expected = (1*1 + 2*2 + 3*3 + 4*4) / (1+2+3+4)
    assert np.isclose(bbs.compute_centroid(profile), expected)


def test_compute_rms_valid():
    profile = np.array([0, 0, 1, 0, 0])
    assert np.isclose(bbs.compute_rms(profile), 0.0)


def test_compute_fwhm_symmetric_peak():
    profile = np.array([0, 1, 2, 3, 2, 1, 0])
    fwhm = bbs.compute_fwhm(profile)
    assert fwhm > 0
    assert isinstance(fwhm, float)


def test_compute_peak_location():
    profile = np.array([1, 3, 7, 2])
    assert bbs.compute_peak_location(profile) == 2


def test_compute_2d_centroids_and_rms():
    img = np.zeros((5, 5))
    img[2, 3] = 10
    x_c, y_c = bbs.compute_2d_centroids(img)
    x_rms, y_rms = bbs.compute_2d_rms(img)
    assert np.isclose(x_c, 3.0)
    assert np.isclose(y_c, 2.0)
    assert np.isclose(x_rms, 0.0)
    assert np.isclose(y_rms, 0.0)


def test_compute_2d_fwhm_and_peak():
    img = np.zeros((5, 5))
    img[2, 1:4] = [1, 2, 1]
    x_fwhm, y_fwhm = bbs.compute_2d_fwhm(img)
    x_peak, y_peak = bbs.compute_2d_peak_locations(img)
    assert x_fwhm > 0
    assert y_fwhm == 0.0
    assert x_peak == 2
    assert y_peak == 2


def test_beam_profile_stats_keys():
    img = np.zeros((3, 3))
    img[1, 1] = 10
    result = bbs.beam_profile_stats(img, prefix="Test")
    expected_keys = [
        "Test_x_mean", "Test_x_rms", "Test_x_fwhm", "Test_x_peak",
        "Test_y_mean", "Test_y_rms", "Test_y_fwhm", "Test_y_peak"
    ]
    assert set(result.keys()) == set(expected_keys)


def test_zero_intensity_image():
    img = np.zeros((5, 5))
    result = bbs.beam_profile_stats(img, prefix="Zero")
    for v in result.values():
        assert v == 0.0


def test_negative_values_handling():
    profile = np.array([-1, -2, -3])
    centroid = bbs.compute_centroid(profile)
    rms = bbs.compute_rms(profile)
    fwhm = bbs.compute_fwhm(profile)

    assert centroid == 0.0
    assert rms == 0.0
    assert fwhm == 0.0

if __name__ == "__main__":
    pytest.main()