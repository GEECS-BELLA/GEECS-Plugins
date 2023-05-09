import numpy as np
from scipy.ndimage import median_filter
from skimage.restoration import denoise_wavelet, cycle_spin


def mad(x):
    """ Median absolute deviation
    """
    return np.median(np.abs(x - np.median(x)))


def get_mad_threshold(x: np.ndarray, mad_multiplier: float = 5.2):
    return np.median(x) + mad_multiplier * mad(x)


def clip_outliers(image: np.ndarray,
                  mad_multiplier: float = 5.2,
                  nonzero_only: bool = True):
    """ Clips image values above a MAD-derived threshold.

    The threshold is median + mad_multiplier * MAD.

    if nonzero_only (default True), the median and MAD only take nonzero values
    into account, because otherwise in many pointing images the median and MAD are
    simply 0.

    """

    image_clipped = image.copy()
    thresh = (get_mad_threshold(image[image > 1e-9], mad_multiplier) if nonzero_only
              else get_mad_threshold(image, mad_multiplier))
    image_clipped[image_clipped > thresh] = thresh

    return image_clipped


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
