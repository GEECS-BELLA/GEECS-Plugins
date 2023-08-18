from __future__ import annotations

from typing import Any

import numpy as np
from skimage.restoration import denoise_wavelet, cycle_spin
from scipy.ndimage import median_filter
import scipy.ndimage as simg

def clip_hot_pixels(image: np.ndarray, median_filter_size: int = 2, threshold_factor: float = 3) -> np.ndarray:
    data_type = image.dtype
    image = image.astype('float64')

    # median filtering and hot pixels list
    blurred = median_filter(image, size=median_filter_size)
    difference = image - blurred
    threshold = threshold_factor * np.std(difference)
    hot_pixels = np.nonzero(np.abs(difference) > threshold)

    # fix image with median values
    for (y, x) in zip(hot_pixels[0], hot_pixels[1]):
        image[y, x] = blurred[y, x]

    return image.astype(data_type)


def denoise(image: np.ndarray, max_shifts: int = 3):
    return cycle_spin(image, func=denoise_wavelet, max_shifts=max_shifts)


def basic_filter(image: np.ndarray,
                 hp_median: int = 2, hp_threshold: float = 3.0, denoise_cycles: int = 3,
                 gauss_filter: float = 5., com_threshold: float = 0.5) -> dict[str, Any]:
    """
    Applies a basic set of filters to an image

    hp_median:      hot pixels median filter size
    hp_threshold:   hot pixels threshold in # of sta. dev. of the median deviation
    gauss_filter:   gaussian filter size
    com_threshold:  image threshold for center-of-mass calculation
    """
    # clip hot pixels
    if hp_median > 0:
        image_clipped = clip_hot_pixels(image, median_filter_size=hp_median, threshold_factor=hp_threshold)
    else:
        image_clipped: np.ndarray = image.copy()

    # denoise
    if denoise_cycles > 0:
        image_denoised = denoise(image_clipped, max_shifts=denoise_cycles)
    else:
        image_denoised: np.ndarray = image_clipped.copy()

    # gaussian filter
    if gauss_filter > 0:
        image_blurred = simg.gaussian_filter(image, sigma=gauss_filter)
    else:
        image_blurred: np.ndarray = image_denoised.copy()

    # thresholding
    if com_threshold > 0:
        # peak location
        i_max, j_max = np.where(image_blurred == image_blurred.max(initial=0))
        i_max, j_max = i_max[0].item(), j_max[0].item()

        # center of mass of thresholded image
        val_max = image_blurred[i_max, j_max]
        binary = image_blurred > (com_threshold * val_max)
        image_thresholded = image_blurred * binary.astype(float)
        i_com, j_com = simg.center_of_mass(image_thresholded)
    else:
        i_max, j_max, i_com, j_com = -1
        image_thresholded: np.ndarray = image_blurred.copy()

    filters = {'hp_median': hp_median,
               'hp_threshold': hp_threshold,
               'denoise_cycles': denoise_cycles,
               'gauss_filter': gauss_filter,
               'com_threshold': com_threshold}

    positions = {'max_ij': (int(i_max), int(j_max)),
                 'com_ij': (int(i_com), int(j_com)),
                 'long_names': ['maximum', 'center of mass'],
                 'short_names': ['max', 'com']}

    arrays = {'raw_roi': image,
              'denoised': image_denoised,
              'blurred': image_blurred,
              'thresholded': image_thresholded}

    return {'filter_pars': filters,
            'positions': positions,
            'arrays': arrays,
           }

