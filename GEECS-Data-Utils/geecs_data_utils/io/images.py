"""Readers and decoders for images written/streamed by GEECS / NI IMAQ.

Most functions here map a file path to a NumPy array and contain no analysis
logic; higher-level packages (ImageAnalysis, ScanAnalysis, Bluesky
external-asset handlers) wrap them. :func:`decode_imaq_image_string` is the
exception: it decodes an in-memory NI IMAQ "Flatten Image to String" payload
(as received live over the device TCP stream), not a file on disk.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import png
from imageio.v3 import imread

# Marker string embedded by LabVIEW in the flattened-image wrapper, just before
# the IMAQ image cluster. Used to locate the (little-endian) IMAQ struct header.
_IMAQ_CLASS_MARKER = b"LV_ImageDTClassInfo"


def read_imaq_png_image(file_path: Union[Path, str]) -> np.ndarray:
    """
    Read a PNG file saved by NI IMAQ.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to the PNG file.

    Returns
    -------
    np.ndarray
        Image data as a NumPy array with appropriate bit depth handling.
    """
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        png_reader = png.Reader(f)

        # read operations returns rows as a generator. it also adds png headers
        # as attributes to png_reader, including sbit
        width, height, rows, info = png_reader.read()
        significant_bits = png_reader.sbit

        # NI IMAQ images use 16 bits per pixel (uncompressed) but often only
        # the left 12 bits for the data, which is given in the sbit header.
        # PNG readers don't account for this, so we right shift manually.
        bitdepth = info["bitdepth"]
        image = np.array(list(rows), f"uint{bitdepth:d}")

    if significant_bits is None:
        return image
    else:
        significant_bits = ord(significant_bits)
        return np.right_shift(image, bitdepth - significant_bits)


def read_tsv_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load a .tsv file as a 2D NumPy array of floats.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the .tsv file.

    Returns
    -------
    np.ndarray
        2D float64 array with data.
    """
    file_path = Path(file_path)
    try:
        data = np.genfromtxt(file_path, delimiter="\t")
    except Exception as e:
        raise RuntimeError(f"Failed to load .tsv file {file_path}: {e}")

    return data.astype(np.float64)


def read_imaq_image(file_path: Union[Path, str]) -> np.ndarray:
    """
    Read a BELLA camera image, handling various file formats.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to the image file. Supported extensions: .png, .npy, .tsv, .h5, others.

    Returns
    -------
    np.ndarray
        Loaded image data as a NumPy array.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".png":
        return read_imaq_png_image(file_path)
    elif file_path.suffix.lower() == ".npy":
        return np.load(file_path)
    elif file_path.suffix.lower() == ".tsv":
        return read_tsv_file(file_path)
    elif file_path.suffix.lower() == ".h5":
        return load_image_from_h5(h5_path=file_path)
    else:
        return imread(file_path)


def load_image_from_h5(h5_path: Path | str) -> np.ndarray:
    """
    Load an image stored in an HDF5 file.

    Parameters
    ----------
    h5_path : Path | str
        Path to the .h5 file containing an ``image`` dataset.

    Returns
    -------
    np.ndarray
        The image data extracted from the HDF5 file.
    """
    with h5py.File(h5_path, "r") as f:
        image = f["image"][()]  # Load the dataset into a NumPy array
    return image


def _flatten_string_to_bytes(blob: Union[str, bytes]) -> bytes:
    """Return raw bytes for a flattened-image payload.

    Device ``state`` values arrive as ``str`` (the TCP layer decodes the wire
    bytes with ``latin-1``); recover the original bytes losslessly with the same
    codec. ``bytes`` input is passed through unchanged.
    """
    return blob.encode("latin-1") if isinstance(blob, str) else bytes(blob)


def decode_imaq_image_string(blob: Union[str, bytes]) -> np.ndarray:
    """Decode an NI IMAQ "Flatten Image to String" payload to a 2-D array.

    This decodes the in-memory byte stream produced by the LabVIEW *IMAQ Flatten
    Image to String* VI (as received live over the GEECS device TCP stream and
    stored in ``device.state["image"]``). It is **not** a file reader — for
    images saved to disk use :func:`read_imaq_image` instead.

    Both device transmission modes are handled, for any camera/ROI:

    - **Compressed:** the payload embeds a standard JFIF JPEG, which carries its
      own dimensions and is decoded directly. JPEG is always 8-bit and lossy.
    - **Uncompressed:** the payload is the raw pixel buffer. Width, height, IMAQ
      border, row-stride padding and bytes-per-pixel are all derived from the
      message — the IMAQ struct stores width/height/border as little-endian
      ints (a native Windows memory dump), and stride/depth are inferred from the
      payload length. The IMAQ border and row padding are cropped out.

    Parameters
    ----------
    blob : str or bytes
        The flattened image payload, e.g. ``device.state["image"]``. ``str`` is
        recovered to bytes with ``latin-1`` (see :func:`_flatten_string_to_bytes`).

    Returns
    -------
    np.ndarray
        The image as a 2-D array (``(height, width)``). Uncompressed frames keep
        their native dtype (``uint8`` / ``uint16``); compressed frames are
        ``uint8``.

    Raises
    ------
    ValueError
        If the uncompressed IMAQ header cannot be parsed consistently with the
        payload size (e.g. a malformed or unexpected message).
    """
    data = _flatten_string_to_bytes(blob)

    # The flattened wrapper opens with a big-endian length-prefixed device name,
    # and repeats that name immediately before the payload; anchoring on the last
    # occurrence is independent of camera, ROI and pixel type.
    name_len = struct.unpack_from(">I", data, 0)[0]
    name = data[4 : 4 + name_len]
    payload = data[data.rfind(name) + len(name) :]

    # Compressed: embedded JFIF JPEG carries its own geometry. Require the JFIF
    # marker too, so a raw frame whose first bytes happen to be 0xFFD8FF is not
    # misread as a JPEG.
    if payload[:3] == b"\xff\xd8\xff" and b"JFIF" in payload[:16]:
        return np.asarray(imread(payload))

    # Uncompressed: read geometry from the IMAQ struct (little-endian). The
    # struct begins 8 bytes after the class marker (LabVIEW refnum + cluster
    # size). Offsets 40/48/56 confirmed against square and non-square frames.
    base = data.index(_IMAQ_CLASS_MARKER) + len(_IMAQ_CLASS_MARKER) + 8
    width = struct.unpack_from("<i", data, base + 40)[0]
    height = struct.unpack_from("<i", data, base + 48)[0]
    border = struct.unpack_from("<i", data, base + 56)[0]

    rows = height + 2 * border  # allocated rows incl. border
    if rows <= 0 or len(payload) % rows:
        raise ValueError(
            f"IMAQ header parse failed (w={width} h={height} border={border}); "
            f"payload {len(payload)} not divisible by {rows} rows"
        )
    stride = len(payload) // rows  # padded bytes per row
    cols = width + 2 * border

    # Largest of {1, 2, 4} bytes/pixel that still fits the measured row stride.
    for bpp in (4, 2, 1):
        if cols * bpp <= stride and stride % bpp == 0:
            break
    else:
        raise ValueError(f"width {width}+border doesn't fit stride {stride}")

    dtype = {1: "<u1", 2: "<u2", 4: "<u4"}[bpp]
    full = np.frombuffer(payload, dtype=dtype, count=rows * (stride // bpp))
    full = full.reshape(rows, stride // bpp)
    return full[border : border + height, border : border + width]
