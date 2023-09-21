from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

import numpy as np
import png
from imageio.v3 import imread

def read_imaq_png_image(file_path: Union[Path, str]) -> np.ndarray:
    """ Read PNG file as output by NI IMAQ, which uses an uncommon format.
    """
    png_reader = png.Reader(file_path.open('rb'))

    # read operations returns rows as a generator. it also adds png headers
    # as attributes to png_reader, including sbit
    width, height, rows, info = png_reader.read()

    # NI IMAQ images use 16 bits per pixel (uncompressed) but often only 
    # the left 12 bits for the data, which is given in the sbit header. 
    # PNG readers don't account for this, so we right shift manually.
    bitdepth = info['bitdepth']
    image = np.array(list(rows), f'uint{bitdepth:d}')

    if png_reader.sbit is None:
        return image
    else:
        significant_bits = ord(png_reader.sbit)
        return np.right_shift(image, bitdepth - significant_bits)

def read_imaq_image(file_path: Union[Path, str]) -> np.ndarray:
    """ Read BELLA camera image, in particular handle NI PNG files correctly.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.png':
        return read_imaq_png_image(file_path)
    else:
        return imread(file_path)



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

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"ROI({self.top}, {self.bottom}, {self.left}, {self.right})"
    

class NotAPath(Path().__class__):
    """ A Path instance that evaluates to false in, for example, if statements.
    """
    def __bool__(self):
        return False

    def is_file(self):
        return False

    def is_dir(self):
        return False

    def exists(self):
        return False
