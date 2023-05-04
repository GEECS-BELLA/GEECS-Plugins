import os
import glob
import cv2
from typing import Optional
import numpy.typing as npt
from geecs_api.api_defs import SysPath
from geecs_api.devices.geecs_device import api_error
import geecs_api.tools.images.ni_vision as ni


def average_images(images_folder: SysPath, n_images: int = 0, file_extension: str = '.png') -> Optional[npt.ArrayLike]:
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
