"""
3-25-2024

Module to contain functions common to more than one Analyzer class, but not necessarily ubiquitous enough to include
in the parent ImageAnalyzer class itself.

-Chris
"""

import numpy as np


def threshold_reduction(image, threshold):
    return_image = np.copy(image) - threshold
    return_image[np.where(return_image < 0)] = 0
    return return_image


def find_max(image):
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    max_value = image[y_max, x_max]
    return x_max, y_max, max_value


def saturation_check(image, saturation_value):
    return len(np.where(image > saturation_value)[0])


def calculate_fwhm(projection_array, threshold=None):
    maxloc = np.argmax(projection_array)
    halfmax = projection_array[maxloc]/2
    if threshold is not None:
        halfmax = halfmax + 0.5*threshold

    peak_region = np.where(projection_array >= halfmax)[0]
    return peak_region[-1] - peak_region[0]


def calculate_centroid_center_of_mass(image, total_counts=None):
    if total_counts is None:
        total_counts = np.sum(image)
    centroid_x = np.sum(np.arange(image.shape[1]) * image) / total_counts
    centroid_y = np.sum(np.arange(image.shape[0]) * np.transpose(image)) / total_counts
    return centroid_x, centroid_y


def calculate_standard_deviation(amplitude_array, axis_array, axis_average=None):
    if axis_average is None:
        axis_average = calculate_axis_average(amplitude_array, axis_array)
    return np.sqrt(np.average((axis_array - axis_average) ** 2, weights=amplitude_array))


def calculate_axis_average(amplitude_array, axis_array):
    return np.average(axis_array, weights=amplitude_array)
