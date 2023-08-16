from __future__ import annotations

from pathlib import Path
from typing import Union

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
    image = np.fromiter(rows, (np.dtype(f'u{bitdepth // 8}'), width))

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
