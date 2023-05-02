import os
import png
import cv2
import numpy as np
import numpy.typing as npt
from typing import Optional
from pathlib import Path
from geecs_api.api_defs import SysPath


def read_imaq_image(file_path: SysPath) -> Optional[npt.ArrayLike]:
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


def read_png_header(file_path: SysPath) -> dict[bytes, bytes]:
    try:
        png_reader = png.Reader(file_path)
        return {key: val for key, val in png_reader.chunks() if key != b'IDAT'}

    except Exception:
        return {}
