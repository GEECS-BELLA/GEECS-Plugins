from __future__ import annotations

from math import floor, ceil
from typing import TYPE_CHECKING, Any, Optional
from ..types import Array2D
if TYPE_CHECKING:
    import numpy as np
from warnings import warn

class ROI:
    """ Specify a region of interest for an ImageAnalyzer to crop with.
    
    This is given a class so that there can be no confusion about the order
    of indices.

    Initialzed with top, bottom, left, right indices. Cropping is done with 
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
    def __init__(self, top: Optional[int] = None, 
                       bottom: Optional[int] = None, 
                       left: Optional[int] = None, 
                       right: Optional[int] = None, 

                       bad_index_order = 'raise',
                ):

        """
        Parameters
        ----------
        top, bottom, left, right : Optional[int]
            indices of ROI. Cropping is done with slices containing these indices,
            which means: 
                * None is valid, meaning go to the edge
                * Negative integers are valid, which means up to edge - value.
                * The top and left indices are inclusive, and bottom and right indices
                  are exclusive.

            The convention of (0, 0) at the top left corner is used, which means 
            top should be less than bottom. 

        bad_index_order : one of 'raise', 'invert', 'invert_warn'
            what to do if top > bottom, or right > left
                'raise': raise ValueError
                'invert': silently switch top/bottom, or left/right indices
                'invert_warn': switch top/bottom or left/right indices with warning.

        """

        def check_index_order(low_name, low_index, high_name, high_index):
            """ Checks whether low_index < high_index, and takes appropriate action.

            Returns
            -------
            low_index, high_index : int
                possibly inverted.

            """
            if low_index is None or high_index is None:
                return low_index, high_index

            if low_index > high_index:
                if bad_index_order == 'raise': 
                    raise ValueError(f"{low_index} should be less than {high_index} ((0, 0) is at the top left corner)")
                elif bad_index_order == 'invert':
                    low_index, high_index = high_index, low_index
                elif bad_index_order == 'invert_warn':
                    low_index, high_index = high_index, low_index
                    warn(f"Inverting {low_index} and {high_index}.")
                else:
                    raise ValueError(f"Unknown action for bad_index_order: {bad_index_order}")

            return low_index, high_index

        top, bottom = check_index_order('top', top, 'bottom', bottom)
        left, right = check_index_order('left', left, 'right', right)

        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def crop(self, image: np.ndarray):
        return image[self.top:self.bottom, self.left:self.right]

