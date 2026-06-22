"""Generic readers for image and array files written by GEECS / NI IMAQ.

These functions map a file path to a NumPy array. They contain no
analysis logic; higher-level packages (ImageAnalysis, ScanAnalysis,
Bluesky external-asset handlers) wrap them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import png
from imageio.v3 import imread


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
