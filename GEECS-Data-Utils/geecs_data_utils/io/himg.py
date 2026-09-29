"""The HASO ``.himg`` container, read and rebuilt without the vendor SDK.

A HASO wavefront sensor (Imagine Optic; the WaveKit SDK) saves one
``.himg`` per acquisition.  The file is an opaque, encrypted metadata
blob followed by the raw camera frame, and its outer layout is plain
enough to parse without the SDK (verified byte for byte on three
sensors' files, 2026-09-28)::

    byte 0            0x00
    bytes 1..16       four little-endian uint32: version (2), width, height,
                      blob_length
    bytes 17..        the metadata blob (``blob_length`` bytes; opaque —
                      the same cipher as the sensor's ``.dat`` licence file)
    then              ``width * height`` little-endian uint16 pixels, row-major

Everything before the pixels is *the header*: :data:`HIMG_FIXED_HEADER_LENGTH`
plus the blob.  Header + pixels rebuild the file byte-identically
(:func:`himg_bytes`), which is what lets the per-scan frame stack
(:mod:`geecs_data_utils.io.himg_stack`) hold a ``.himg`` losslessly: the
pixels as one frame, the header beside it.  WaveKit needs only the pixels
and *any* valid header of the same sensor to compute slopes — the
per-shot header carries a timestamp and nothing the analysis reads — so
the header travels as provenance, never as an input.

This module is the codec alone: bytes in, ``(header, pixels)`` out, and
back.  It never decodes the blob and never touches a scan folder.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "HIMG_FIXED_HEADER_LENGTH",
    "HIMG_PIXEL_DTYPE",
    "HimgFormatError",
    "HimgHeader",
    "himg_bytes",
    "parse_himg",
    "parse_himg_header",
    "read_himg",
    "write_himg",
]

#: The fixed lead of every ``.himg``: the zero byte plus four uint32 fields.
HIMG_FIXED_HEADER_LENGTH = 17
#: The on-disk pixel type: little-endian uint16, whatever the sensor's bit depth.
HIMG_PIXEL_DTYPE = np.dtype("<u2")
_FIXED_FIELDS = struct.Struct("<4I")


class HimgFormatError(ValueError):
    """The bytes are not a ``.himg`` of the layout this codec knows."""


@dataclass(frozen=True)
class HimgHeader:
    """The four fixed fields of a ``.himg`` and where its parts lie.

    Attributes
    ----------
    version : int
        The container version (2 on every file seen so far).
    width, height : int
        The frame's pixel dimensions; the frame array is ``(height, width)``.
    blob_length : int
        The opaque metadata blob's length in bytes.
    """

    version: int
    width: int
    height: int
    blob_length: int

    @property
    def header_length(self) -> int:
        """Bytes before the pixels: the fixed lead plus the blob."""
        return HIMG_FIXED_HEADER_LENGTH + self.blob_length

    @property
    def pixel_bytes(self) -> int:
        """Bytes of pixel data that follow the header."""
        return self.width * self.height * HIMG_PIXEL_DTYPE.itemsize

    @property
    def file_length(self) -> int:
        """The whole file's length for these fields."""
        return self.header_length + self.pixel_bytes


def parse_himg_header(data: bytes) -> HimgHeader:
    """Read the fixed fields off the start of a ``.himg``'s bytes.

    Parameters
    ----------
    data : bytes
        At least the first :data:`HIMG_FIXED_HEADER_LENGTH` bytes of the file.

    Raises
    ------
    HimgFormatError
        Too short, a non-zero lead byte, or zero dimensions.
    """
    if len(data) < HIMG_FIXED_HEADER_LENGTH:
        raise HimgFormatError(
            f"{len(data)} bytes is shorter than the {HIMG_FIXED_HEADER_LENGTH}-byte lead"
        )
    if data[0] != 0:
        raise HimgFormatError(f"lead byte is {data[0]:#04x}, not 0x00")
    version, width, height, blob_length = _FIXED_FIELDS.unpack_from(data, 1)
    if width == 0 or height == 0:
        raise HimgFormatError(f"zero frame dimension: width={width} height={height}")
    return HimgHeader(version, width, height, blob_length)


def parse_himg(data: bytes) -> tuple[bytes, np.ndarray]:
    """Split a whole ``.himg`` into its header bytes and its pixels.

    Parameters
    ----------
    data : bytes
        The complete file.

    Returns
    -------
    tuple of (bytes, numpy.ndarray)
        The header (everything before the pixels — :func:`himg_bytes` takes
        it back verbatim) and the ``(height, width)`` ``uint16`` frame, a
        copy that owns its memory.

    Raises
    ------
    HimgFormatError
        The bytes are not a ``.himg``, or their length disagrees with the
        header's dimensions (a truncated or padded file).
    """
    fields = parse_himg_header(data)
    if len(data) != fields.file_length:
        raise HimgFormatError(
            f"{len(data)} bytes, but the header ({fields.width} x {fields.height}, "
            f"{fields.blob_length}-byte blob) implies {fields.file_length}"
        )
    header = bytes(data[: fields.header_length])
    pixels = np.frombuffer(data, dtype=HIMG_PIXEL_DTYPE, offset=fields.header_length)
    return header, pixels.reshape(fields.height, fields.width).copy()


def read_himg(path: str | Path) -> tuple[bytes, np.ndarray]:
    """Read one ``.himg`` file — :func:`parse_himg` over its bytes."""
    return parse_himg(Path(path).read_bytes())


def himg_bytes(header: bytes, pixels: np.ndarray) -> bytes:
    """Rebuild a ``.himg`` from its header and pixels — byte-identical to the original.

    Parameters
    ----------
    header : bytes
        The header :func:`parse_himg` returned (or any header of the same
        sensor, for a frame that never had one).
    pixels : numpy.ndarray
        A ``(height, width)`` integer frame whose shape matches the header's
        fields; converted to the on-disk little-endian ``uint16``.

    Raises
    ------
    HimgFormatError
        The header is malformed or the frame's shape disagrees with it.
    """
    fields = parse_himg_header(header)
    if len(header) != fields.header_length:
        raise HimgFormatError(
            f"header is {len(header)} bytes; its fields say {fields.header_length}"
        )
    frame = np.asarray(pixels)
    if frame.shape != (fields.height, fields.width):
        raise HimgFormatError(
            f"frame shape {frame.shape} does not match the header's "
            f"({fields.height}, {fields.width})"
        )
    return header + np.ascontiguousarray(frame, dtype=HIMG_PIXEL_DTYPE).tobytes()


def write_himg(path: str | Path, header: bytes, pixels: np.ndarray) -> Path:
    """Write :func:`himg_bytes` to *path* (the parent directory must exist)."""
    target = Path(path)
    target.write_bytes(himg_bytes(header, pixels))
    return target
