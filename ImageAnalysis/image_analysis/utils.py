"""Utility functions for interfacing with images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .types import Array2D

from warnings import warn

import numpy as np

import re

# The generic file readers (``read_imaq_image``, ``read_imaq_png_image``,
# ``read_tsv_file``, ``load_image_from_h5``) moved to
# ``geecs_data_utils.io.images``. Importing them from here still works via the
# module-level ``__getattr__`` shim below, which emits a DeprecationWarning.
# This shim is temporary and will be removed in a future ImageAnalysis release;
# import the readers from ``geecs_data_utils.io.images`` directly.
_MOVED_READERS = frozenset(
    {
        "read_imaq_image",
        "read_imaq_png_image",
        "read_tsv_file",
        "load_image_from_h5",
    }
)


def __getattr__(name: str):
    """Re-export relocated readers with a deprecation warning (PEP 562)."""
    if name in _MOVED_READERS:
        warn(
            f"image_analysis.utils.{name} has moved to "
            f"geecs_data_utils.io.images.{name}. Importing it from "
            f"image_analysis.utils is deprecated and the shim will be removed "
            f"in a future release; update your import.",
            DeprecationWarning,
            stacklevel=2,
        )
        from geecs_data_utils.io import images

        return getattr(images, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
