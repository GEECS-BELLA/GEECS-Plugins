"""Tests for the generic file readers in ``geecs_data_utils.io.images``."""

from __future__ import annotations

from itertools import chain, cycle
from pathlib import Path

import h5py
import numpy as np

from geecs_data_utils.io.images import (
    load_image_from_h5,
    read_imaq_image,
    read_tsv_file,
)

DATA_DIR = Path(__file__).parent / "data"


def test_regular_png_image():
    """A basic PNG decodes to the expected pixel pattern.

    Fixture from http://www.schaik.com/pngsuite/
    """
    img = read_imaq_image(DATA_DIR / "basn0g08.png")

    def generate_ref_image():
        it = cycle(chain(range(0, 256), range(254, 0, -1)))
        for _ in range(32 * 32):
            yield next(it)

    ref_img = np.reshape(np.fromiter(generate_ref_image(), int), (32, 32))
    assert (img == ref_img).all()


def test_read_tsv_file_round_trip(tmp_path):
    """``read_tsv_file`` reads back a tab-delimited 2D float array."""
    expected = np.array([[1.0, 2.5, 3.0], [4.0, 5.0, 6.5]])
    tsv_path = tmp_path / "sample.tsv"
    np.savetxt(tsv_path, expected, delimiter="\t")

    loaded = read_tsv_file(tsv_path)

    assert loaded.dtype == np.float64
    np.testing.assert_allclose(loaded, expected)


def test_load_image_from_h5_round_trip(tmp_path):
    """``load_image_from_h5`` returns the array stored under the ``image`` key."""
    expected = np.arange(12, dtype=np.uint16).reshape(3, 4)
    h5_path = tmp_path / "sample.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("image", data=expected)

    loaded = load_image_from_h5(h5_path)

    np.testing.assert_array_equal(loaded, expected)


def test_read_imaq_image_dispatches_by_extension(tmp_path):
    """``read_imaq_image`` routes ``.npy``, ``.tsv`` and ``.h5`` to the right reader."""
    arr = np.array([[7.0, 8.0], [9.0, 10.0]])

    npy_path = tmp_path / "a.npy"
    np.save(npy_path, arr)
    np.testing.assert_array_equal(read_imaq_image(npy_path), arr)

    tsv_path = tmp_path / "a.tsv"
    np.savetxt(tsv_path, arr, delimiter="\t")
    np.testing.assert_allclose(read_imaq_image(tsv_path), arr)

    h5_path = tmp_path / "a.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("image", data=arr)
    np.testing.assert_array_equal(read_imaq_image(h5_path), arr)
