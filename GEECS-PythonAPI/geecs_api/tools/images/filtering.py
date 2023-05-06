import numpy as np


def mad(x):
    """ Median absolute deviation
    """
    return np.median(np.abs(x - np.median(x)))


def get_mad_threshold(X: np.ndarray, mad_multiplier: float = 5.2):
    return np.median(X) + mad_multiplier * mad(X)


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
