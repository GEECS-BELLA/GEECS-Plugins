# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:24:29 2019

@author: ATAP
"""

import os
import sys
import png  # this package does low level i/o on pngs
import numpy as np
import cv2
import imageio as im
import numpy.typing as npt
from typing import Optional, Union
from pathlib import Path

sys.path.append('..')


def nBitPNG(f_name):
    """
    This reads in a png and scales pixel values to compensate for variable bit 
    depth. This is an issue for 12 bit pngs created by NI Vision, which get 
    saved with varying numbers of significant bits. This scaling is required 
    if numerical pixel values are to be read out correctly, and no image 
    analysis package seems to do it natively.
    Based on Kei's "f12bitPngOpnV04" MATLAB function.
    INPUTS:
    f_name: <str> path to png
    OUTPUTS:
    scaled_image: <2-D float numpy array> image scaled according to significant bits
    """
    
    # get significant bits (try exception handling here)
    try:
        sBIT = chunkBytes(f_name, b'sBIT')
        sig_bits = int.from_bytes(sBIT, 'little')

    except Exception:
        print('Problem reading chunks in ' + f_name)
        sig_bits = 16
    
    # read png and scale
    raw_png = im.imread(f_name, as_gray=True)
    scaled_image = raw_png/(2**(16-sig_bits))
    
    return scaled_image


def chunkBytes(f_name, chunk_name):
    """
    Looks for a named chunk in a png and returns its value if found. There is 
    no exception handling if the named chunk isn't found, in which case an 
    error is thrown.\n
    INPUTS\n
    f_name: <str> path to png\n
    chunk_name: <bytes>(e.g. b'sBIT') chunk name\n
    OUTPUTS\n
    chunk_val: <bytes>(e.g. b'sBIT') value of named chunk, if found\n
    """

    chunk_reader = png.Reader(filename=f_name)
    
    curr_name = b'turd'
    while curr_name != chunk_name:
        if curr_name == b'IEND':
            chunk_val = b'\x00'
            break
        else:
            curr_name, chunk_val = chunk_reader.chunk()
    
    return chunk_val


def read_imaq_image(file_path: Union[str, bytes, os.PathLike]) -> Optional[npt.ArrayLike]:
    if not os.path.isfile(file_path):
        return None

    try:
        png_header = read_png_header(file_path)
        image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

        if Path(file_path).suffix.lower() == '.png':
            data_bytes: int = image.dtype.itemsize
            return np.right_shift(image, data_bytes * 8 - ord(png_header[b'sBIT']))
        else:
            return image

    except Exception:
        return None


def read_png_header(file_path: Union[str, bytes, os.PathLike]) -> dict[bytes, bytes]:
    try:
        png_reader = png.Reader(file_path)
        return {key: val for key, val in png_reader.chunks() if key != b'IDAT'}

    except Exception:
        return {}
