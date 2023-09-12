# -*- coding: utf-8 -*-
"""
Created on Mon Aug 8 2023

Module for performing various analyses on MagSpec Images.  Originally copied from the old hi_res_mag

This updated version is cleaner and streamlined to work on LabView. Most notably by not opening TDMS file or
the interpSpec to parse any additional information. Just make sure some of the constants are correct.

Author: Chris
"""

import numpy as np
import time
from scipy import optimize

from . import mag_spec_energy_axis as energy_axis_lookup


def print_time(label, start_time, do_print=False):
    if do_print:
        print(label, time.perf_counter() - start_time)
    return time.perf_counter()


def analyze_image(input_image, input_params, do_print=False):
    current_time = time.perf_counter()

    num_pixel_crop = input_params["Pixel-Crop"]
    if num_pixel_crop > 0:
        roi_image = input_image[num_pixel_crop: -num_pixel_crop, num_pixel_crop: -num_pixel_crop]
    else:
        roi_image = input_image

    saturation_value = input_params["Saturation-Value"]
    saturation_number = saturation_check(roi_image, saturation_value)
    current_time = print_time(" Saturation Check:", current_time, do_print=do_print)

    threshold = input_params["Threshold-Value"]
    image = threshold_reduction(roi_image, threshold)
    current_time = print_time(" Threshold Subtraction", current_time, do_print=do_print)

    normalization_factor = input_params["Normalization-Factor"]
    image = normalize_image(image, normalization_factor)
    current_time = print_time(" Normalize Image:", current_time, do_print=do_print)

    image = np.copy(image[::-1, ::-1])
    current_time = print_time(" Rotate Image", current_time, do_print=do_print)

    mag_spec_name = input_params["Mag-Spec-Name"]
    image_width = np.shape(image)[1]
    pixel_arr = np.linspace(0, image_width, image_width)
    energy_arr = energy_axis_lookup.return_energy_axis(pixel_arr, mag_spec_name)
    current_time = print_time(" Calculated Energy Axis:", current_time, do_print=do_print)

    charge_on_camera = np.sum(image)
    current_time = print_time(" Charge on Camera", current_time, do_print=do_print)

    clipped_percentage = calculate_clipped_percentage(image)
    current_time = print_time(" Calculate Clipped Percentage", current_time, do_print=do_print)

    calibration_factor = input_params["Transverse-Calibration"]
    charge_arr = calculate_charge_density_distribution(image, calibration_factor)
    current_time = print_time(" Charge Projection:", current_time, do_print=do_print)

    if np.sum(charge_arr) == 0:
        peak_charge = 0.0
        average_energy = -1.0
        energy_spread = 0.0
        peak_charge_energy = 0.0
        average_beam_size = 0.0
        beam_angle = 0.0
        beam_intercept = 0.0
        projected_beam_size = 0.0
        optimization_factor = 0.0
    else:
        peak_charge = calculate_maximum_charge(charge_arr)
        current_time = print_time(" Peak Charge:", current_time, do_print=do_print)

        average_energy = calculate_average_energy(charge_arr, energy_arr)
        current_time = print_time(" Average Energy:", current_time, do_print=do_print)

        energy_spread = calculate_standard_deviation_energy(charge_arr, energy_arr, average_energy)
        current_time = print_time(" Energy Spread:", current_time, do_print=do_print)

        peak_charge_energy = calculate_peak_energy(charge_arr, energy_arr)
        current_time = print_time(" Energy at Peak Charge:", current_time, do_print=do_print)

        central_energy = input_params["Optimization-Central-Energy"]
        bandwidth_energy = input_params["Optimization-Bandwidth-Energy"]
        optimization_factor = calculate_optimization_factor(
            charge_arr, energy_arr, central_energy, bandwidth_energy
        )
        current_time = print_time(" Optimization Factor:", current_time, do_print=do_print)

        do_transverse = input_params["Do-Transverse-Calculation"]
        if do_transverse:
            binsize = input_params["Transverse-Slice-Binsize"]
            slice_threshold = input_params["Transverse-Slice-Threshold"]
            sigma_arr, x0_arr, amp_arr, err_arr = transverse_slice_loop(
                image, calibration_factor=calibration_factor, threshold=slice_threshold, binsize=binsize
            )
            current_time = print_time(" Gaussian Fits for each Slice:", current_time, do_print=do_print)

            average_beam_size = calculate_average_size(sigma_arr, amp_arr)
            current_time = print_time(" Average Beam Size:", current_time, do_print=do_print)

            linear_fit = fit_beam_angle(x0_arr, amp_arr, energy_arr)
            current_time = print_time(" Beam Angle Fit:", current_time, do_print=do_print)

            beam_angle = linear_fit[0]
            beam_intercept = linear_fit[1]
            projected_axis, projected_arr, projected_beam_size = calculate_projected_beam_size(image, calibration_factor)
            projected_beam_size = projected_beam_size * calibration_factor
            print_time(" Projected Size:", current_time, do_print=do_print)
        else:
            average_beam_size = 0.0
            projected_beam_size = 0.0
            beam_angle = 0.0
            beam_intercept = 0.0

    mag_spec_dict = {
        "Clipped-Percentage": clipped_percentage,
        "Saturation-Counts": saturation_number,
        "Charge-On-Camera": charge_on_camera,
        "Peak-Charge": peak_charge,
        "Peak-Charge-Energy": peak_charge_energy,
        "Average-Energy": average_energy,
        "Energy-Spread": energy_spread,
        "Energy-Spread-Percent": energy_spread / average_energy,
        "Average-Beam-Size": average_beam_size,
        "Projected-Beam-Size": projected_beam_size,
        "Beam-Tilt": beam_angle,
        "Beam-Intercept": beam_intercept,
        "Beam-Intercept-100MeV": 100 * beam_angle + beam_intercept,
        "Optimization-Factor": optimization_factor,
    }
    return image, mag_spec_dict, np.vstack((energy_arr, charge_arr))


def normalize_image(image, normalization_factor):
    return np.copy(image) * normalization_factor


def threshold_reduction(image, threshold):
    return_image = np.copy(image) - threshold
    return_image[np.where(return_image < 0)] = 0
    return return_image


def calculate_clipped_percentage(image):
    clip_check = np.append(np.append(np.append(image[0, :], image[:, 0]), image[-1, :]), image[:, -1])
    max_val = np.max(image)
    if max_val != 0:
        return np.max(clip_check) / max_val
    else:
        return 1.1


def calculate_projected_beam_size(image, calibration_factor):
    proj_arr = np.sum(image, axis=1)

    axis_arr = np.linspace(0, len(proj_arr) * calibration_factor, len(proj_arr))
    axis_arr = np.flip(axis_arr)
    fit = fit_data_something( proj_arr, axis_arr, gaussian,
                              guess=[max(proj_arr), 20 * calibration_factor, axis_arr[np.argmax(proj_arr)]])
    beam_size = fit[1]
    return axis_arr, proj_arr, beam_size


def calculate_charge_density_distribution(image, calibration_factor):
    charge_arr = np.sum(image, axis=0) * calibration_factor
    return charge_arr


def calculate_maximum_charge(charge_arr):
    return np.max(charge_arr)


def calculate_average_energy(charge_arr, energy_arr):
    return np.average(energy_arr, weights=charge_arr)


def calculate_standard_deviation_energy(charge_arr, energy_arr, average_energy=None):
    if average_energy is None:
        average_energy = calculate_average_energy(charge_arr, energy_arr)
    return np.sqrt(np.average((energy_arr - average_energy) ** 2, weights=charge_arr))


def calculate_peak_energy(charge_arr, energy_arr):
    return energy_arr[np.argmax(charge_arr)]


def calculate_average_size(sigma_arr, amp_arr):
    return np.average(sigma_arr, weights=amp_arr)


def fit_beam_angle(x0_arr, amp_arr, energy_arr):
    linear_fit = np.polyfit(energy_arr, x0_arr, deg=1, w=np.power(amp_arr, 2))
    return linear_fit


def find_max(image):
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    max_value = image[y_max, x_max]
    return x_max, y_max, max_value


def saturation_check(image, saturation_value):
    return len(np.where(image > saturation_value)[0])


def gaussian(x, amp, sigma, x0):
    return amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def fit_data_something(data, axis, function, guess=[0.0, 0.0, 0.0]):
    err_func = lambda p, x, y: function(x, *p) - y
    p0 = guess
    p1, success = optimize.leastsq(err_func, p0[:], args=(axis, data))
    return p1


def transverse_slice_loop(image, calibration_factor=1, threshold=0.01, binsize=1, statistic_algorithm=True):
    ny, nx = np.shape(image)
    x_loc, y_loc, max_val = find_max(image)

    sigma_arr = np.zeros(nx)
    err_arr = np.zeros(nx)
    x0_arr = np.zeros(nx)
    amp_arr = np.zeros(nx)

    for i in range(int(nx / binsize)):
        if binsize == 1:
            slice_arr = image[:, i]
        else:
            slice_arr = np.average(image[:, binsize * i: (binsize * i) + binsize - 1], axis=1)
        if np.max(slice_arr) > threshold * max_val:
            axis_arr = np.linspace(0, len(slice_arr), len(slice_arr)) * calibration_factor
            axis_arr = np.flip(axis_arr)

            if statistic_algorithm:
                slice_sigma, slice_x0, slice_amp, slice_err = get_transverse_stat_slices(axis_arr, slice_arr)
            else:
                slice_sigma, slice_x0, slice_amp, slice_err = fit_transverse_gaussian_slices(axis_arr, slice_arr)

        else:
            slice_sigma = 0
            slice_x0 = 0
            slice_amp = 0
            slice_err = 0
        sigma_arr[binsize * i: binsize * (i + 1)] = slice_sigma
        x0_arr[binsize * i: binsize * (i + 1)] = slice_x0
        amp_arr[binsize * i: binsize * (i + 1)] = slice_amp
        err_arr[binsize * i: binsize * (i + 1)] = slice_err
    return sigma_arr, x0_arr, amp_arr, err_arr


def get_transverse_stat_slices(axis_arr, slice_arr):
    amp_slice = np.average(slice_arr)
    x0_slice = np.average(np.average(axis_arr, weights=slice_arr))
    sigma_slice = np.sqrt(np.average(np.power(axis_arr - x0_slice, 2), weights=slice_arr))
    err_slice = 0
    return sigma_slice, x0_slice, amp_slice, err_slice


def fit_transverse_gaussian_slices(axis_arr, slice_arr):
    fit = fit_data_something(
        slice_arr,
        axis_arr,
        gaussian,
        guess=[
            max(slice_arr),
            5 * 43,  # calibrationFactor,
            axis_arr[np.argmax(slice_arr)],
        ],
    )
    amp_fit, sigma_fit, x0_fit = fit

    if x0_fit < axis_arr[-1] or x0_fit > axis_arr[0]:
        sigma_fit = 0
        x0_fit = 0
        amp_fit = 0
        err_fit = 0
    else:
        func = gaussian(axis_arr, *fit)
        error = np.sum(np.square(slice_arr - func))
        err_fit = np.sqrt(error) * 1e3
    return sigma_fit, x0_fit, amp_fit, err_fit


def calculate_optimization_factor(charge_arr, energy_arr, central_energy, bandwidth_energy):
    gaussian_weight_function = gaussian(energy_arr, 1.0, bandwidth_energy, central_energy)
    optimization_factor = np.sum(charge_arr * gaussian_weight_function)
    return optimization_factor


if __name__ == "__main__":
    pass
