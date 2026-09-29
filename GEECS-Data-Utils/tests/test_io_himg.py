"""The SDK-free ``.himg`` codec (io/himg.py): parse, rebuild, refuse."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from geecs_data_utils.io.himg import (
    HIMG_FIXED_HEADER_LENGTH,
    HimgFormatError,
    HimgHeader,
    himg_bytes,
    parse_himg,
    parse_himg_header,
    read_himg,
    write_himg,
)


def make_header(width: int, height: int, blob: bytes = b"\x07" * 30, version=2):
    """A header of the verified layout: 0x00, four LE uint32, the blob."""
    return b"\x00" + struct.pack("<4I", version, width, height, len(blob)) + blob


def test_round_trip_is_byte_identical(tmp_path) -> None:
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 256, (6, 8), dtype=np.uint16)
    header = make_header(8, 6)
    data = himg_bytes(header, pixels)
    assert len(data) == HIMG_FIXED_HEADER_LENGTH + 30 + 2 * 6 * 8

    parsed_header, parsed = parse_himg(data)
    assert parsed_header == header
    assert parsed.dtype == np.uint16 and parsed.shape == (6, 8)
    np.testing.assert_array_equal(parsed, pixels)
    assert himg_bytes(parsed_header, parsed) == data

    path = write_himg(tmp_path / "shot.himg", header, pixels)
    assert path.read_bytes() == data
    assert read_himg(path)[0] == header
    np.testing.assert_array_equal(read_himg(path)[1], pixels)


def test_pixels_are_little_endian_row_major() -> None:
    header = make_header(2, 2, blob=b"")
    data = header + bytes([0x02, 0x01, 0xFF, 0x00, 0x00, 0x00, 0x34, 0x12])
    _, pixels = parse_himg(data)
    assert pixels.tolist() == [[0x0102, 0x00FF], [0x0000, 0x1234]]
    # Rebuilding from a big-endian or wider frame still writes the on-disk layout.
    assert himg_bytes(header, pixels.astype(">u2")) == data
    assert himg_bytes(header, pixels.astype(np.int64)) == data


def test_parsed_pixels_own_their_memory() -> None:
    header = make_header(3, 2, blob=b"")
    _, pixels = parse_himg(header + bytes(12))
    assert pixels.flags.writeable and pixels.flags.owndata
    pixels[0, 0] = 9  # a copy: the caller may process in place


def test_header_fields_of_a_haso4_lift_file() -> None:
    fields = parse_himg_header(make_header(4096, 3000, b"x" * 4482))
    assert fields == HimgHeader(version=2, width=4096, height=3000, blob_length=4482)
    assert fields.header_length == 4499
    assert fields.pixel_bytes == 2 * 4096 * 3000
    assert fields.file_length == 24_580_499  # the real file's size


@pytest.mark.parametrize(
    "data, message",
    [
        (b"\x00" + b"\x01" * 10, "shorter than"),
        (b"\x01" + struct.pack("<4I", 2, 2, 2, 0) + bytes(8), "lead byte"),
        (b"\x00" + struct.pack("<4I", 2, 0, 2, 0), "zero frame dimension"),
        (make_header(2, 2, b"") + bytes(7), "implies"),  # truncated
        (make_header(2, 2, b"") + bytes(9), "implies"),  # padded
    ],
)
def test_malformed_bytes_are_refused(data, message) -> None:
    with pytest.raises(HimgFormatError, match=message):
        parse_himg(data)


def test_rebuild_refuses_a_disagreeing_header_or_frame() -> None:
    header = make_header(2, 2)
    with pytest.raises(HimgFormatError, match="header is"):
        himg_bytes(header + b"extra", np.zeros((2, 2), np.uint16))
    with pytest.raises(HimgFormatError, match="frame shape"):
        himg_bytes(header, np.zeros((2, 3), np.uint16))
