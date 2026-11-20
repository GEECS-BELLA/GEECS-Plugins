"""The image clean-ups: X-ray hits out, the beam's centre, centring."""

from __future__ import annotations

import numpy as np
import pytest

from geecs_analysis.algorithms.beam_centre import (
    beam_centre,
    centre_on,
    threshold_level,
)
from geecs_analysis.algorithms.hot_pixels import remove_hot_pixels
from geecs_analysis.image_prep import ImagePrep, remove_hits


def _beam(row: float, col: float, shape=(41, 61), background=10.0) -> np.ndarray:
    y, x = np.indices(shape)
    return background + 100.0 * np.exp(-((y - row) ** 2 + (x - col) ** 2) / 8.0)


def test_a_threshold_below_one_is_a_fraction_of_the_peak_above_is_counts() -> None:
    image = np.array([[0.0, 200.0]])
    assert threshold_level(image, 0.5) == 100.0
    assert threshold_level(image, 50) == 50.0
    assert threshold_level(image, 1) == 1.0


def test_the_centre_ignores_the_background_under_the_threshold() -> None:
    row, col = beam_centre(_beam(12, 45), 0.5)
    assert row == pytest.approx(12, abs=0.05) and col == pytest.approx(45, abs=0.05)
    assert beam_centre(_beam(12, 45), 1e6) is None


def test_centring_puts_the_beam_in_the_middle_and_fills_with_the_background() -> None:
    out = centre_on(_beam(5, 10), (5, 10))
    assert out.shape == (41, 61)
    assert np.unravel_index(out.argmax(), out.shape) == (20, 30)
    assert out[0, 0] == pytest.approx(10.0)


def test_one_hot_pixel_is_removed_and_the_rest_left_alone() -> None:
    image = _beam(20, 30)
    hit = image.copy()
    hit[3, 3] = 5000.0
    for method in ("median3", "median5", "hot"):
        assert remove_hits(hit, method)[3, 3] < 20
    hot = remove_hits(hit, "hot")
    untouched = np.ones_like(image, dtype=bool)
    untouched[3, 3] = False
    assert np.array_equal(hot[untouched], image[untouched])
    assert remove_hits(hit, "none") is hit
    with pytest.raises(ValueError):
        remove_hits(hit, "gaussian")


def test_a_hit_does_not_pull_the_centre_once_filtered() -> None:
    image = _beam(20, 30)
    image[2, 2] = 1e5
    prep = ImagePrep(hits="hot", centre=0.5)
    out = prep.apply(image)
    assert np.unravel_index(out.argmax(), out.shape) == (20, 30)
    assert not ImagePrep().active and prep.active


def test_a_hit_on_the_edge_or_corner_is_found() -> None:
    image = np.full((4, 6), 10.0)
    image[0, 0] = image[2, 5] = 1000.0
    out = remove_hits(image, "hot")
    assert out[0, 0] == 10.0 and out[2, 5] == 10.0


@pytest.mark.parametrize(("sigma", "offset"), [(0.5, 0.0), (0.6, 0.1), (0.8, 0.0)])
def test_a_sharp_focal_spot_is_not_taken_for_a_hit(sigma: float, offset: float) -> None:
    """A spot of about 1-2 px FWHM keeps its peak: the optics light its neighbours."""
    y, x = np.indices((41, 41))
    rng = np.random.default_rng(1)
    r2 = (y - 20 - offset) ** 2 + (x - 20 - offset) ** 2
    image = 10 + 1000 * np.exp(-r2 / (2 * sigma**2)) + rng.normal(0, 1, (41, 41))
    out = remove_hits(image, "hot")
    np.testing.assert_array_equal(out, image)


def test_hits_on_a_broad_beam_are_removed_and_only_they_change() -> None:
    """A hit on a gentle beam's peak or flank goes; the beam is left alone."""
    y, x = np.indices((41, 41))
    rng = np.random.default_rng(2)
    beam = 10 + 100 * np.exp(-((y - 20) ** 2 + (x - 20) ** 2) / (2 * 3.0**2))
    image = beam + rng.normal(0, 1, (41, 41))
    hit = image.copy()
    spots = [(20, 20), (20, 26), (17, 20), (5, 5), (0, 0)]
    for spot in spots:
        hit[spot] += 500
    out = remove_hits(hit, "hot")
    for spot in spots:
        assert abs(out[spot] - image[spot]) < 10
    assert int((out != hit).sum()) == len(spots)


def test_the_medians_are_the_median_step_and_the_result_is_the_callers() -> None:
    """``median3``/``median5`` are the ``median`` step; the array may be written."""
    from scipy.ndimage import median_filter

    rng = np.random.default_rng(3)
    image = rng.integers(0, 4096, (12, 15)).astype(np.uint16)
    for kernel in (3, 5):
        out = remove_hits(image, f"median{kernel}")
        expected = median_filter(image.astype(float), size=kernel, mode="reflect")
        np.testing.assert_array_equal(out, expected)
        assert out.dtype == np.float64 and out.flags.writeable


def _dark(rng: np.random.Generator) -> np.ndarray:
    """A dark frame: a 100-count pedestal with a single count on one pixel in twenty."""
    return 100 + rng.poisson(0.05, (200, 200)).astype(np.uint16)


def test_single_counts_on_a_dark_frame_are_not_hits() -> None:
    """More than half the pixels on one value: the measured noise is zero, the floor holds."""
    dark = _dark(np.random.default_rng(7))
    assert int((dark != 100).sum()) > 1000  # plenty of lone counts to mistake
    np.testing.assert_array_equal(remove_hits(dark, "hot"), dark)


def test_a_dark_frame_less_its_averaged_background_keeps_its_counts() -> None:
    """The background's own scatter (far under a count) is not the noise either."""
    rng = np.random.default_rng(7)
    dark = _dark(rng)
    image = dark - (100.03 + rng.normal(0, 0.02, dark.shape))
    np.testing.assert_array_equal(remove_hits(image, "hot"), image)


def test_noise_rounded_to_whole_counts_is_not_a_hit() -> None:
    rng = np.random.default_rng(7)
    image = np.rint(100 + rng.normal(0, 0.3, (200, 200)))
    np.testing.assert_array_equal(remove_hits(image, "hot"), image)


def test_a_real_hit_on_a_dark_frame_still_goes() -> None:
    dark = _dark(np.random.default_rng(7)).astype(float)
    hit = dark.copy()
    hit[50, 60] += 40
    out = remove_hits(hit, "hot")
    assert out[50, 60] <= 101
    assert int((out != hit).sum()) == 1


def test_the_noise_floor_is_in_the_images_units_and_can_be_set() -> None:
    """A normalised image names its own floor; zero is the measured noise alone."""
    dark = _dark(np.random.default_rng(7))
    assert int((remove_hot_pixels(dark, noise_floor=0.0) != dark).sum()) > 1000
    scaled = dark / 4096.0
    hit = scaled.copy()
    hit[50, 60] += 40 / 4096.0
    np.testing.assert_array_equal(remove_hot_pixels(hit), hit)  # under one "count"
    out = remove_hot_pixels(hit, noise_floor=1 / 4096.0)
    assert int((out != hit).sum()) == 1 and out[50, 60] < hit[50, 60]
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="noise floor"):
            remove_hot_pixels(dark, noise_floor=bad)
