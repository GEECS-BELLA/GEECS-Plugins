"""
Thursday 9-7-2023

Module containing the functions to analyze a spectrometer camera for a laser.  Currently used with just
UC_UndulatorExitCam, but can perhaps generalize this analysis to work with any spectrometer image.

Q: Some functions, like the threshold reduction and saturation check, are equivalent with the magspec versions.  Should
I keep them separate or make a new module file in this same online_analysis_modules folder to house generic functions?

@Chris
"""

import numpy as np
import cv2


def rotate_image(image, rotation_degrees):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_degrees, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return rotated_image


def get_spectrum_lineouts(image, calibration_factor, zeroth_order_position):
    spectra_projection = np.sum(image, axis=0)
    pixel_axis = np.arange(0, np.shape(image)[1])
    wavelength_axis = (pixel_axis - zeroth_order_position) * -calibration_factor
    return wavelength_axis, spectra_projection


def crop_spectrum(wavelength_array, spectrum_array, minimum_wavelength):
    crop_bounds = np.where(wavelength_array > minimum_wavelength)[0]
    crop_wavelength_array = wavelength_array[crop_bounds]
    crop_spectrum_array = spectrum_array[crop_bounds]
    return crop_wavelength_array, crop_spectrum_array


def get_peak_wavelength(wavelength_array, spectrum_array):
    return wavelength_array[np.argmax(spectrum_array)]


def get_average_wavelength(wavelength_array, spectrum_array):
    return np.average(wavelength_array, weights=spectrum_array)


def get_wavelength_spread(wavelength_array, spectrum_array, average_wavelength=None):
    if average_wavelength is None:
        average_wavelength = get_average_wavelength(wavelength_array, spectrum_array)
    return np.sqrt(np.average((wavelength_array - average_wavelength) ** 2, weights=spectrum_array))


def gaussian(x, amp, sigma, x0):
    return amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def calculate_optimization_factor(wavelength_array, spectrum_array, target_wavelength, target_bandwidth):
    gaussian_weight_function = gaussian(wavelength_array, 1.0, target_bandwidth, target_wavelength)
    optimization_factor = np.sum(spectrum_array * gaussian_weight_function)
    return optimization_factor
