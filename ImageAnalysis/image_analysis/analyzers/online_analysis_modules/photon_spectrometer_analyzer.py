"""
Thursday 9-7-2023

Module containing the functions to analyze a spectrometer camera for a laser.  Currently used with just
UC_UndulatorExitCam, but can perhaps generalize this analysis to work with any spectrometer image.

Q: Some functions, like the threshold reduction and saturation check, are equivalent with the magspec versions.  Should
I keep them separate or make a new module file in this same online_analysis_modules folder to house generic functions?

@Chris
"""

import numpy as np
import time
from scipy import ndimage
import cv2
# import matplotlib.pyplot as plt


def print_time(label, start_time, do_print=False):
    if do_print:
        print(label, time.perf_counter() - start_time)
    return time.perf_counter()


def analyze_image(input_image, input_params, do_print=False):
    current_time = time.perf_counter()

    num_pixel_crop = input_params["edge_crop_pixels"]
    if num_pixel_crop > 0:
        roi_image = input_image[num_pixel_crop: -num_pixel_crop, num_pixel_crop: -num_pixel_crop]
    else:
        roi_image = input_image

    saturation_value = input_params["saturation_value_int"]
    saturation_number = saturation_check(roi_image, saturation_value)
    current_time = print_time(" Saturation Check:", current_time, do_print=do_print)

    threshold = input_params["noise_threshold_int"]
    image = threshold_reduction(roi_image, threshold)
    current_time = print_time(" Threshold Subtraction", current_time, do_print=do_print)

    total_photons = np.sum(image)
    current_time = print_time(" Total Photon Counts", current_time, do_print=do_print)

    tilt_angle = input_params["image_tilt_calibration_degrees"]
    # rotated_image = ndimage.rotate(image, tilt_angle, reshape=False)
    rotated_image = rotate_image(image, tilt_angle)
    # rotated_image = image
    current_time = print_time(" Image Rotation", current_time, do_print=do_print)

    calibration_factor = input_params["wavelength_calibration_nm/pixel"]
    zeroth_order_position = input_params["zeroth_order_calibration_pixel"]
    wavelength_array, spectrum_array = get_spectrum_lineouts(rotated_image, calibration_factor, zeroth_order_position)
    current_time = print_time(" Spectrum Lineouts", current_time, do_print=do_print)

    minimum_wavelength = input_params["minimum_wavelength_nm"]
    crop_wavelength_array, crop_spectrum_array = crop_spectrum(wavelength_array, spectrum_array, minimum_wavelength)

    if np.sum(crop_spectrum_array) == 0:
        peak_wavelength = 0
        average_wavelength = 0
        wavelength_spread = 0
        optimization_factor = 0

    else:
        peak_wavelength = get_peak_wavelength(crop_wavelength_array, crop_spectrum_array)
        average_wavelength = get_average_wavelength(crop_wavelength_array, crop_spectrum_array)
        wavelength_spread = get_wavelength_spread(crop_wavelength_array, crop_spectrum_array, average_wavelength=average_wavelength)
        current_time = print_time(" Spectrum Stats", current_time, do_print=do_print)

        target_wavelength = input_params["optimization_central_wavelength_nm"]
        target_bandwidth = input_params["optimization_bandwidth_wavelength_nm"]
        optimization_factor = calculate_optimization_factor(crop_wavelength_array, crop_spectrum_array, target_wavelength, target_bandwidth)
        print_time(" Optimization Factor", current_time, do_print=do_print)

    exit_cam_dict = {
        "camera_saturation_counts": saturation_number,
        "camera_total_intensity_counts": total_photons,
        "peak_wavelength_nm": peak_wavelength,
        "average_wavelength_nm": average_wavelength,
        "wavelength_spread_weighted_rms_nm": wavelength_spread,
        "optimization_factor": optimization_factor,
    }
    return rotated_image, exit_cam_dict, np.vstack((wavelength_array, spectrum_array))


def saturation_check(image, saturation_value):
    return len(np.where(image > saturation_value)[0])


def threshold_reduction(image, threshold):
    return_image = np.copy(image) - threshold
    return_image[np.where(return_image < 0)] = 0
    return return_image


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