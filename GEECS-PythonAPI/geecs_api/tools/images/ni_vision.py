import png
import cv2
import numpy as np
from typing import Optional, Union
from pathlib import Path


def read_imaq_image(file_path: Union[Path, str]) -> Optional[np.ndarray]:
    file_path = Path(file_path)
    if not file_path.is_file():
        return None

    try:
        png_header = read_png_header(file_path)
        image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)

        if file_path.suffix.lower() == '.png':
            data_bytes: int = image.dtype.itemsize
            return np.right_shift(image, data_bytes * 8 - ord(png_header[b'sBIT']))
        else:
            return image

    except Exception:
        return None


def read_png_header(file_path: Path) -> dict[bytes, bytes]:
    try:
        png_reader = png.Reader(str(file_path))
        return {key: val for key, val in png_reader.chunks() if key != b'IDAT'}

    except Exception:
        return {}
