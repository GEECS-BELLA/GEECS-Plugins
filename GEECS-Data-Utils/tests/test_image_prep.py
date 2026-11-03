"""The image clean-ups: X-ray hits out, the beam's centre, centring."""

from __future__ import annotations

import numpy as np
import pytest

from geecs_data_utils.data.image_prep import (
    ImagePrep,
    beam_centre,
    centre_on,
    remove_hits,
    threshold_level,
)


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
