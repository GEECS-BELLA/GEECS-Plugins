import os
import glob
import cv2
import numpy as np
from progressbar import ProgressBar
from typing import Optional, Any
from geecs_api.api_defs import SysPath
from geecs_api.devices.geecs_device import api_error
import geecs_api.tools.images.ni_vision as ni
from geecs_api.tools.images.spot_analysis import spot_analysis


def average_images(images_folder: SysPath, n_images: int = 0, file_extension: str = '.png') -> Optional[np.ndarray]:
    images = list_images(images_folder, n_images, file_extension)

    # run averaging
    if images:
        try:
            avg_image = ni.read_imaq_image(images[0])
            data_type: str = avg_image.dtype.name
            avg_image = avg_image.astype('float64')

            if len(images) > 1:
                for it, image_path in enumerate(images[1:]):
                    image_data = ni.read_imaq_image(image_path)
                    image_data = image_data.astype('float64')
                    alpha = 1.0 / (it + 2)
                    beta = 1.0 - alpha
                    avg_image = cv2.addWeighted(image_data, alpha, avg_image, beta, 0.0)

            return avg_image.round().astype(data_type)

        except Exception as ex:
            api_error.error(str(ex), 'Failed to calculate average image')
            return None


def analyze_images(images_folder: SysPath, n_images: int = 0, file_extension: str = '.png', rotate_deg: int = 0) \
        -> list[dict[str, Any]]:
    paths = list_images(images_folder, n_images, file_extension)
    analyses: list[dict[str, Any]] = []

    # run averaging
    if paths:
        try:
            with ProgressBar(max_value=len(paths)) as pb:
                for image_path in paths:
                    image = np.rot90(ni.read_imaq_image(image_path), int(rotate_deg / 90))
                    analyses.append(spot_analysis(image.astype('float64')))
                    pb.increment()

        except Exception as ex:
            api_error.error(str(ex), 'Failed to analyze image')
            pass

    return analyses


def list_images(images_folder: SysPath, n_images: int = 0, file_extension: str = '.png') -> Optional[list[SysPath]]:
    # file extension
    if not file_extension:
        return None

    file_extension = file_extension.lower().strip()
    if file_extension.startswith('.'):
        file_extension = file_extension[1:]

    if file_extension not in ['png', 'jpg', 'jpeg', 'tiff', 'tif']:
        return None

    # list images
    images = sorted(glob.glob(os.path.join(images_folder, f'*.{file_extension}')),
                    key=lambda x: x[0].split('_')[-1][:-4])
    if n_images > 0:
        images = images[-n_images:]

    return images
