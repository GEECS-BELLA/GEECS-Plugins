import os
import re
import cv2
import numpy.typing as npt
from typing import Optional
from geecs_api.api_defs import SysPath


def read_imaq_image(file_path: SysPath, data_type: Optional[str] = None) -> tuple[Optional[npt.ArrayLike], str]:
    if not os.path.isfile(file_path):
        return None, ''

    if not data_type:
        image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)  # retrieves bit-depth automatically
        data_type: str = image.dtype.name

    n_bytes: int = int(float(re.search('[0-9]+$', data_type)[0]) / 8)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) / n_bytes

    return image.astype(data_type), data_type
