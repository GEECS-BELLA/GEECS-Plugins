"""Tests for the generic file readers in ``geecs_data_utils.io.images``."""

from __future__ import annotations

import struct
from itertools import chain, cycle
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pytest

from geecs_data_utils.io.images import (
    _IMAQ_CLASS_MARKER,
    decode_imaq_image_string,
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


# --- decode_imaq_image_string -------------------------------------------------

NAME = b"UC_TestCam"


def _wrap_flatten(payload_body: bytes, struct_bytes: bytes = b"") -> bytes:
    """Assemble a minimal NI-IMAQ flattened-image byte stream.

    Mirrors the real wire layout closely enough for the decoder: a big-endian
    length-prefixed device name, the class marker + 8 wrapper bytes + IMAQ
    struct, then the device name repeated immediately before the payload.
    """
    prefixed_name = struct.pack(">I", len(NAME)) + NAME
    wrapper = b"nivissvc.*" + _IMAQ_CLASS_MARKER + b"\x00" * 8 + struct_bytes
    return prefixed_name + wrapper + prefixed_name + payload_body


def _imaq_struct(width: int, height: int, border: int) -> bytes:
    """IMAQ struct with width/height/border at the confirmed little-endian offsets."""
    sb = bytearray(64)
    struct.pack_into("<i", sb, 40, width)
    struct.pack_into("<i", sb, 48, height)
    struct.pack_into("<i", sb, 56, border)
    return bytes(sb)


def test_decode_uncompressed_16bit_with_border_and_stride():
    """Raw 16-bit frame: border and row-stride padding are cropped out."""
    width, height, border = 6, 4, 1
    rows = height + 2 * border  # 6 allocated rows
    stride_px = 10  # padded beyond cols (= width + 2*border = 8)

    img = np.arange(1, height * width + 1, dtype="<u2").reshape(height, width)
    alloc = np.zeros((rows, stride_px), dtype="<u2")
    alloc[border : border + height, border : border + width] = img

    blob = _wrap_flatten(alloc.tobytes(), _imaq_struct(width, height, border))
    out = decode_imaq_image_string(blob)

    assert out.shape == (height, width)
    assert out.dtype == np.dtype("<u2")
    np.testing.assert_array_equal(out, img)


def test_decode_uncompressed_accepts_str_input():
    """A latin-1 ``str`` (as stored in device.state) decodes identically to bytes."""
    width, height, border = 4, 3, 0
    img = np.arange(width * height, dtype="<u1").reshape(height, width)
    blob = _wrap_flatten(img.tobytes(), _imaq_struct(width, height, border))

    out = decode_imaq_image_string(blob.decode("latin-1"))
    np.testing.assert_array_equal(out, img)
    assert out.dtype == np.dtype("<u1")


def test_decode_compressed_jpeg_roundtrip():
    """A JPEG payload is detected and decoded to its native dimensions."""
    height, width = 8, 16
    img = (np.arange(height * width) % 256).astype(np.uint8).reshape(height, width)
    jpeg = iio.imwrite("<bytes>", img, extension=".jpeg")
    assert jpeg[:3] == b"\xff\xd8\xff"  # sanity: real JPEG SOI

    out = decode_imaq_image_string(_wrap_flatten(jpeg))

    assert out.shape == (height, width)  # JPEG carries its own geometry
    assert out.dtype == np.uint8


def test_decode_uncompressed_bad_size_raises():
    """A payload size inconsistent with the parsed header is rejected loudly."""
    blob = _wrap_flatten(b"\x00" * 25, _imaq_struct(width=6, height=4, border=1))
    with pytest.raises(ValueError, match="payload"):
        decode_imaq_image_string(blob)
