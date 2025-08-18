"""

General Image Processing Tools

Contributors:
Kyle Jensen, kjensen@lbl.gov (kjensen11)
Guillaume Plateau, grplateau@lbl.gov

Note (11/13/2024):
Intended to be a temporary location for methods as we build around the evolving
data acquisition and analysis framework. This will likely be reorganized and
restructured in coming days/weeks.
"""
# =============================================================================
# %% imports
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.measure import regionprops, label

# =============================================================================
# %% functions

def image_signal_thresholding(image,
                              median_filter_size=2, threshold_coeff=0.1):

    # preserve input image data type, convert to standard working data type
    data_type = image.dtype
    image = image.astype('float64')

    # perform median filtering
    blurred_image = median_filter(image, size=median_filter_size)

    # threshold with respect to the blurred image max
    image[blurred_image < blurred_image.max() * threshold_coeff] = 0

    # convert image back to input data type
    image = image.astype(data_type)

    return image

def hotpixel_correction(image,
                        median_filter_size=2, threshold_factor=3):

    # preserve input image data type, convert to standard working data type
    data_type = image.dtype
    image = image.astype('float64')

    # median filtering to establish hot pixel list
    blurred = median_filter(image, size=median_filter_size)
    difference = image - blurred
    threshold = threshold_factor * np.std(difference)
    hot_pixels = np.nonzero(np.abs(difference) > threshold)

    # replace hotpixels with median values
    for (y, x) in zip(hot_pixels[0], hot_pixels[1]):
        image[y, x] = blurred[y, x]

    # convert image back to input data type
    image = image.astype(data_type)

    return image

def filter_image(image,
                 median_filter_size=2, median_filter_cycles=0,
                 gaussian_filter_size=3, gaussian_filter_cycles=0):
    """
    This function uses predefined analysis_settings to filter an image.

    Args:
    - image: loaded image to be processed.
    - analysis_settings: settings for filtering.

    Returns:
    - Processed image with specified filters applied.
    """

    # preserve input image data type, convert to standard working data type
    data_type = image.dtype
    processed_image = image.astype(np.float32)

    for _ in range(median_filter_cycles):
        processed_image = median_filter(processed_image, size=median_filter_size)

    for _ in range(gaussian_filter_cycles):
        processed_image = gaussian_filter(processed_image, sigma=gaussian_filter_size)

    # convert image back to input data type
    processed_image.astype(data_type)

    return processed_image

def find_beam_properties(image):

    # initialize beam properties dict
    beam_properties = {}

    # construct binary and label images
    image_binary = image.copy()
    image_binary[image_binary > 0] = 1
    image_binary = image_binary.astype(int)
    image_label = label(image_binary)

    # get beam properties and reduce to the largest region
    props = regionprops(image_label, image)
    areas = [i.area for i in props]
    props = props[areas.index(max(areas))]

    # extract centroid
    beam_properties['centroid'] = props.centroid_weighted

    return beam_properties
# =============================================================================
# %% executable

def testing():
    pass

if __name__=="__main__":
    testing()
