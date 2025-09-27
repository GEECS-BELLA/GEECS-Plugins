"""Utility functions for interfacing with images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .types import Array2D

from warnings import warn

import numpy as np
import h5py
import png
from imageio.v3 import imread

import re


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


def extract_shot_number(filename):
    """
    Extract the shot number from the filename.

    Parameters
    ----------
    filename : str
        The filename from which to extract the shot number.

    Returns
    -------
    int
        The extracted shot number, or None if the format is incorrect.
    """
    # Match the last number before the .png extension
    match = re.search(r"_(\d+)\.png$", filename)
    if match:
        return int(match.group(1))
    return None


def ensure_float64_processing(image: "Array2D") -> "Array2D":
    """
    Ensure image is in float64 format for processing.

    This function converts images to float64 to handle:
    - 16-bit images without loss of precision
    - Negative values from background subtraction
    - Proper arithmetic operations

    Parameters
    ----------
    image : Array2D
        Input image of any numeric dtype.

    Returns
    -------
    Array2D
        Image converted to float64.

    """
    return image.astype(np.float64)


class ROI:
    """
    Specify a region of interest for an ImageAnalyzer to crop with.

    This is given a class so that there can be no confusion about the order
    of indices.

    Initialized with top, bottom, left, right indices. Cropping is done with
    slices containing these indices, which means:
        * None is valid, meaning go to the edge
        * Negative integers are valid, which means up to edge - value.
        * The top and left indices are inclusive, and bottom and right indices
          are exclusive.

    The convention of (0, 0) at the top left corner is used, which means
    top should be less than bottom.

    If not given as kwargs, the coordinates are in order that they would be
    used in slicing:
        image[i0:i1, i2:i3]
    """

    def __init__(
        self,
        top: Optional[int] = None,
        bottom: Optional[int] = None,
        left: Optional[int] = None,
        right: Optional[int] = None,
        bad_index_order="raise",
    ):
        """
        Initialize the ROI with top, bottom, left, right indices.

        Parameters
        ----------
        top, bottom, left, right : Optional[int]
            Indices of ROI. Cropping is done with slices containing these indices,
            which means:
                * None is valid, meaning go to the edge
                * Negative integers are valid, which means up to edge - value.
                * The top and left indices are inclusive, and bottom and right indices
                  are exclusive.

            The convention of (0, 0) at the top left corner is used, which means
            top should be less than bottom.

        bad_index_order : one of 'raise', 'invert', 'invert_warn'
            What to do if top > bottom, or right > left
                'raise': raise ValueError
                'invert': silently switch top/bottom, or left/right indices
                'invert_warn': switch top/bottom or left/right indices with warning.
        """

        def check_index_order(low_name, low_index, high_name, high_index):
            """Checks whether low_index < high_index, and takes appropriate action.

            Returns
            -------
            low_index, high_index : int
                Possibly inverted indices.
            """
            if low_index is None or high_index is None:
                return low_index, high_index

            if low_index > high_index:
                if bad_index_order == "raise":
                    raise ValueError(
                        f"{low_index} should be less than {high_index} ((0, 0) is at the top left corner)"
                    )
                elif bad_index_order == "invert":
                    low_index, high_index = high_index, low_index
                elif bad_index_order == "invert_warn":
                    low_index, high_index = high_index, low_index
                    warn(f"Inverting {low_index} and {high_index}.")
                else:
                    raise ValueError(
                        f"Unknown action for bad_index_order: {bad_index_order}"
                    )
            return low_index, high_index

        top, bottom = check_index_order("top", top, "bottom", bottom)
        left, right = check_index_order("left", left, "right", right)

        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        Crop an image to the region of interest.

        Parameters
        ----------
        image : np.ndarray
            The image array to be cropped.

        Returns
        -------
        np.ndarray
            The cropped image.
        """
        return image[self.top : self.bottom, self.left : self.right]

    def __str__(self) -> str:
        """
        Return the official string representation of the ROI.

        Returns
        -------
        str
            The string representation produced by ``repr(self)``.
        """
        return repr(self)

    def __repr__(self) -> str:
        """
        Return a string that recreates the ROI.

        Returns
        -------
        str
            Representation of the ROI in the form ``ROI(top, bottom, left, right)``.
        """
        return f"ROI({self.top}, {self.bottom}, {self.left}, {self.right})"


class NotAPath(Path().__class__):
    """A Path instance that evaluates to false in, for example, if statements."""

    def __bool__(self) -> bool:
        """
        Always evaluate to ``False``.

        Returns
        -------
        bool
            ``False`` indicating the path does not exist.
        """
        return False

    def is_file(self) -> bool:
        """
        Indicates the path is not a file.

        Returns
        -------
        bool
            ``False``.
        """
        return False

    def is_dir(self) -> bool:
        """
        Indicates the path is not a directory.

        Returns
        -------
        bool
            ``False``.
        """
        return False

    def exists(self) -> bool:
        """
        Check existence of the path.

        Returns
        -------
        bool
            ``False``.
        """
        return False
