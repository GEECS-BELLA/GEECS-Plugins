from __future__ import annotations

from pathlib import Path
from typing import Union, Optional
from warnings import warn

import numpy as np
import png
from imageio.v3 import imread

import struct


def read_imaq_png_image(file_path: Union[Path, str]) -> np.ndarray:
    """ Read PNG file as output by NI IMAQ, which uses an uncommon format.
    """
    with file_path.open('rb') as f:
        png_reader = png.Reader(f)

        # read operations returns rows as a generator. it also adds png headers
        # as attributes to png_reader, including sbit
        width, height, rows, info = png_reader.read()
        significant_bits = png_reader.sbit

        # NI IMAQ images use 16 bits per pixel (uncompressed) but often only 
        # the left 12 bits for the data, which is given in the sbit header. 
        # PNG readers don't account for this, so we right shift manually.
        bitdepth = info['bitdepth']
        image = np.array(list(rows), f'uint{bitdepth:d}')

    if significant_bits is None:
        return image
    else:
        significant_bits = ord(significant_bits)
        return np.right_shift(image, bitdepth - significant_bits)

def read_imaq_image(file_path: Union[Path, str]) -> np.ndarray:
    """ Read BELLA camera image, in particular handle NI PNG files correctly.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.png':
        return read_imaq_png_image(file_path)
    elif file_path.suffix.lower() == '.npy':
        return np.load(file_path)
    else:
        return imread(file_path)

def get_imaq_timestamp_from_png(file_path):
    """
    Extracts the timestamp from the PNG file's text (tEXt) chunks.
    The function looks for the 'IMAQdxTimestampHigh' and 'IMAQdxTimestampLow' keys
    and calculates the timestamp.  
    
    Args:
        file_path (str): The path to the PNG file.
    
    Returns:
        float: The timestamp in seconds if found, otherwise 0.
    """
    timestamp_high_key = "IMAQdxTimestampHigh"
    timestamp_low_key = "IMAQdxTimestampLow"
    high_chunk = low_chunk = None

    with open(file_path, 'rb') as f:
        # Read the PNG signature (8 bytes)
        signature = f.read(8)

        while True:
            # Read chunk length (4 bytes, big-endian)
            chunk_len = struct.unpack('>I', f.read(4))[0]
            
            # Read chunk type (4 bytes)
            chunk_type = f.read(4)

            if chunk_type == b'IDAT':  # Stop at image data chunk
                break

            # Read the chunk data based on the length
            chunk_data = f.read(chunk_len)
            
            # Read the CRC (4 bytes, skip it)
            f.read(4)

            if chunk_type == b'tEXt':
                # Find the null character to separate keyword and content
                null_pos = chunk_data.find(b'\x00')
                if null_pos != -1:
                    chunk_keyword = chunk_data[:null_pos].decode('utf-8')
                    chunk_content = chunk_data[null_pos+1:]
                    
                    if chunk_keyword == timestamp_high_key:
                        high_chunk = chunk_content
                    elif chunk_keyword == timestamp_low_key:
                        low_chunk = chunk_content

            # Stop searching if both timestamp chunks are found
            if high_chunk and low_chunk:
                break

    if high_chunk and low_chunk:
        # Extract last 4 bytes for high and low parts of the timestamp
        high_bytes = high_chunk[-4:]
        low_bytes = low_chunk[-4:]

        # Convert bytes to integer values
        high_value = int.from_bytes(high_bytes, 'big')
        low_value = int.from_bytes(low_bytes, 'big')

        # Combine high and low to get the full timestamp in nanoseconds
        timestamp = (high_value << 32) + low_value

        # Convert nanoseconds to seconds
        timestamp_seconds = timestamp * 8e-9
        return timestamp_seconds

    return 0  # Return 0 if the timestamp wasn't found


class ROI:
    """ Specify a region of interest for an ImageAnalyzer to crop with.
    
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
