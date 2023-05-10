import os
import glob
import cv2
import numpy as np
import numpy.typing as npt
from progressbar import ProgressBar
from typing import Optional, Any, Union
from geecs_api.api_defs import SysPath
from geecs_api.devices.geecs_device import api_error
import geecs_api.tools.images.ni_vision as ni
from geecs_api.tools.images.spot_analysis import spot_analysis, fwhm


def average_images(images_folder: SysPath, n_images: int = 0, file_extension: str = '.png') -> Optional[np.ndarray]:
    images = list_images(images_folder, n_images, file_extension)

    # run averaging
    if images:
        try:
            with ProgressBar(max_value=len(images)) as pb:
                avg_image = ni.read_imaq_image(images[0])
                # data_type: str = avg_image.dtype.name
                avg_image = avg_image.astype('float64')
                pb.increment()

                if len(images) > 1:
                    for it, image_path in enumerate(images[1:]):
                        image_data = ni.read_imaq_image(image_path)
                        image_data = image_data.astype('float64')
                        alpha = 1.0 / (it + 2)
                        beta = 1.0 - alpha
                        avg_image = cv2.addWeighted(image_data, alpha, avg_image, beta, 0.0)
                        pb.increment()

            # return avg_image.round().astype(data_type)
            return avg_image

        except Exception as ex:
            api_error.error(str(ex), 'Failed to calculate average image')
            return None


def analyze_images(images_folder: SysPath, n_images: int = 0, file_extension: str = '.png', rotate_deg: int = 0,
                   hp_median: int = 2, hp_threshold: float = 3., denoise_cycles: int = 0,
                   gauss_filter: float = 5., com_threshold: float = 0.5) \
        -> tuple[list[dict[str, Any]], Optional[np.ndarray]]:
    paths = list_images(images_folder, n_images, file_extension)
    analyses: list[dict[str, Any]] = []
    avg_image: Optional[np.ndarray] = None
    rot_90deg: int = int(round(rotate_deg / 90.))

    # run averaging
    if paths:
        try:
            with ProgressBar(max_value=len(paths)) as pb:
                avg_image = np.rot90(ni.read_imaq_image(paths[0]), rot_90deg)
                avg_image = avg_image.astype('float64')
                analyses.append(spot_analysis(avg_image, hp_median, hp_threshold,
                                              denoise_cycles, gauss_filter, com_threshold))
                pb.increment()

                if len(paths) > 1:
                    for it, image_path in enumerate(paths[1:]):
                        image_data = np.rot90(ni.read_imaq_image(image_path), rot_90deg)
                        image_data = image_data.astype('float64')
                        analyses.append(spot_analysis(image_data, hp_median, hp_threshold,
                                                      denoise_cycles, gauss_filter, com_threshold))
                        alpha = 1.0 / (it + 2)
                        beta = 1.0 - alpha
                        avg_image = cv2.addWeighted(image_data, alpha, avg_image, beta, 0.0)
                        pb.increment()

        except Exception as ex:
            api_error.error(str(ex), 'Failed to analyze image')
            pass

    return analyses, avg_image


def summarize_image_analyses(analyses: list[dict[str, Any]]) -> dict[str, Union[float, npt.ArrayLike]]:
    scan_pos_max = np.array([analysis['position_max'] for analysis in analyses])  # handle None
    scan_pos_max_fwhm_x = np.array([fwhm(analysis['opt_x_max'][3]) for analysis in analyses])
    scan_pos_max_fwhm_y = np.array([fwhm(analysis['opt_y_max'][3]) for analysis in analyses])

    scan_pos_com = np.array([analysis['position_com'] for analysis in analyses])
    scan_pos_com_fwhm_x = np.array([fwhm(analysis['opt_x_com'][3]) for analysis in analyses])
    scan_pos_com_fwhm_y = np.array([fwhm(analysis['opt_y_com'][3]) for analysis in analyses])

    mean_pos_max = np.mean(scan_pos_max, axis=0)
    mean_pos_max_fwhm_x = np.mean(scan_pos_max_fwhm_x)
    mean_pos_max_fwhm_y = np.mean(scan_pos_max_fwhm_y)
    std_pos_max = np.std(scan_pos_max, axis=0)
    std_pos_max_fwhm_x = np.std(scan_pos_max_fwhm_x)
    std_pos_max_fwhm_y = np.std(scan_pos_max_fwhm_y)

    mean_pos_com = np.mean(scan_pos_com, axis=0)
    mean_pos_com_fwhm_x = np.mean(scan_pos_com_fwhm_x)
    mean_pos_com_fwhm_y = np.mean(scan_pos_com_fwhm_y)
    std_pos_com = np.std(scan_pos_com, axis=0)
    std_pos_com_fwhm_x = np.std(scan_pos_com_fwhm_x)
    std_pos_com_fwhm_y = np.std(scan_pos_com_fwhm_y)

    return {'scan_pos_max': scan_pos_max,
            'scan_pos_max_fwhm_x': scan_pos_max_fwhm_x,
            'scan_pos_max_fwhm_y': scan_pos_max_fwhm_y,
            'scan_pos_com': scan_pos_com,
            'scan_pos_com_fwhm_x': scan_pos_com_fwhm_x,
            'scan_pos_com_fwhm_y': scan_pos_com_fwhm_y,
            'mean_pos_max': mean_pos_max,
            'mean_pos_max_fwhm_x': mean_pos_max_fwhm_x,
            'mean_pos_max_fwhm_y': mean_pos_max_fwhm_y,
            'std_pos_max': std_pos_max,
            'std_pos_max_fwhm_x': std_pos_max_fwhm_x,
            'std_pos_max_fwhm_y': std_pos_max_fwhm_y,
            'mean_pos_com': mean_pos_com,
            'mean_pos_com_fwhm_x': mean_pos_com_fwhm_x,
            'mean_pos_com_fwhm_y': mean_pos_com_fwhm_y,
            'std_pos_com': std_pos_com,
            'std_pos_com_fwhm_x': std_pos_com_fwhm_x,
            'std_pos_com_fwhm_y': std_pos_com_fwhm_y}


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
