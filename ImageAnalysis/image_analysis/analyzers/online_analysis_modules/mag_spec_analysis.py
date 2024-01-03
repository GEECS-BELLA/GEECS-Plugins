# -*- coding: utf-8 -*-
"""
Created on Mon Aug 8 2023

Module for performing various analyses on MagSpec Images.  Originally copied from the old hi_res_mag

This updated version is cleaner and streamlined to work on LabView. Most notably by not opening TDMS file or
the interpSpec to parse any additional information. Just make sure some of the constants are correct.

Author: Chris
"""

import numpy as np
from scipy import optimize


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
        return 1.


def calculate_projected_beam_size(image, calibration_factor):
    proj_arr = np.sum(image, axis=1)

    axis_arr = np.linspace(0, len(proj_arr) * calibration_factor, len(proj_arr))
    axis_arr = np.flip(axis_arr)
    fit = fit_data_something(proj_arr, axis_arr, gaussian,
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
