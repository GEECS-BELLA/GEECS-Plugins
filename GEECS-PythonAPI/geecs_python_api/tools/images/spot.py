""" @author: Guillaume Plateau, TAU Systems """

import os
import math
import numpy as np
from typing import Optional, Any
import scipy.ndimage as simg
from scipy.signal import savgol_filter
from geecs_python_api.controls.interface import api_error
from geecs_python_api.tools.distributions.fit_utility import fit_distribution, gaussian_fit

from image_analysis.tools.filtering import basic_filter
from image_analysis.tools.spot import n_sigma_window, fwhm


def spot_analysis(image: np.ndarray, positions: list[tuple[int, int, str]],
                  x_window: Optional[tuple[int, int]] = None,
                  y_window: Optional[tuple[int, int]] = None) -> Optional[dict[str, Any]]:
    """
    Runs Gaussian fits at a given list of positions (e.g. peak vs. center-of-mass)

    image: numpy array
    positions: list of tuples of (i, j, label)
    """
    analysis: Optional[dict[str, Any]] = {}

    try:
        for pos_i, pos_j, name in positions:
            analysis[name] = {'x': {}, 'y': {}}

            if not np.isnan([pos_i, pos_j]).any():
                if x_window:
                    x_window = (max(0, x_window[0]), min(x_window[1], image.shape[1] - 1))
                    axis_x = np.arange(x_window[0], x_window[1]+1)
                    data_x = image[pos_i, x_window[0]:x_window[1]+1]
                    opt_x, err_x, fit_x = profile_fit(axis_x, data_x, guess_center=pos_j)
                else:
                    axis_x = np.arange(image.shape[1])
                    data_x = image[pos_i, :]
                    opt_x, err_x, fit_x = profile_fit(axis_x, data_x, guess_center=pos_j)

                analysis[name]['x']['axis'] = axis_x
                analysis[name]['x']['data'] = data_x
                analysis[name]['x']['opt'] = opt_x
                analysis[name]['x']['err'] = err_x
                analysis[name]['x']['fit'] = fit_x

                if y_window:
                    y_window = (max(0, y_window[0]), min(y_window[1], image.shape[0] - 1))
                    axis_y = np.arange(y_window[0], y_window[1]+1)
                    data_y = image[y_window[0]:y_window[1]+1, pos_j]
                    opt_y, err_y, fit_y = profile_fit(axis_y, data_y, guess_center=pos_i)
                else:
                    axis_y = np.arange(image.shape[0])
                    data_y = image[:, pos_j]
                    opt_y, err_y, fit_y = profile_fit(axis_y, data_y, guess_center=pos_i)

                analysis[name]['y']['axis'] = axis_y
                analysis[name]['y']['data'] = data_y
                analysis[name]['y']['opt'] = opt_y
                analysis[name]['y']['err'] = err_y
                analysis[name]['y']['fit'] = fit_y

    except Exception as ex:
        api_error.error(str(ex), 'Spot analysis failed.')
        analysis = None
        pass

    return analysis


def profile_fit(x_data: np.ndarray, y_data: np.ndarray,
                guess_center: Optional[float] = None,
                guess_fwhm: Optional[float] = None,
                guess_amplitude: Optional[float] = None,
                guess_background: Optional[float] = None,
                smoothing_window: Optional[int] = None,
                crop_sigma_radius: Optional[float] = None):
    if smoothing_window is None:
        smoothed = savgol_filter(y_data, max(int(len(y_data) / 6), 4), 3)
    else:
        smoothed = savgol_filter(y_data, smoothing_window, 3)

    if not guess_center:
        guess_center = x_data[0] + \
                       (simg.center_of_mass(smoothed)[0] + np.where(smoothed == np.max(smoothed))[0][0]) / 2.

    if not guess_fwhm:
        guess_fwhm = n_sigma_window(smoothed, fwhm(0.5))
        guess_fwhm = guess_fwhm[1] - guess_fwhm[0]

    guess_std = guess_fwhm / (2 * math.sqrt(2 * math.log(2.)))

    if guess_amplitude is None:
        guess_amplitude = np.max(y_data) - np.min(y_data)

    if guess_background is None:
        guess_background = np.min(y_data)

    if isinstance(crop_sigma_radius, float):
        radius: int = round(crop_sigma_radius * guess_std)
        pos_guess_center = np.argmin(np.abs(x_data - guess_center))
        window = [max(0, pos_guess_center - radius), min(x_data.size - 1, pos_guess_center + radius)]
        x_to_fit = x_data[round(window[0]):round(window[1] + 1)]
        y_to_fit = y_data[round(window[0]):round(window[1] + 1)]
        pos_guess_center -= window[0]
        guess_center = x_to_fit[pos_guess_center]
    else:
        window = [0, x_data.size - 1]
        x_to_fit = x_data
        y_to_fit = y_data

    guess = [guess_background, guess_amplitude, guess_center, guess_std]
    bd_bkg = guess_background - 2 * np.abs(guess_background)

    # bounds = (np.array([bd_bkg, 0.5 * guess_amplitude, x_data[0], 0.1 * guess_std]),
    #           np.array([np.max(y_data), 2 * guess_amplitude, x_data[-1], 10 * guess_std]))
    bounds = (np.array([bd_bkg, 0.5 * guess_amplitude, x_to_fit[0], 0.1 * guess_std]),
              np.array([np.max(y_to_fit), 2 * guess_amplitude, x_to_fit[-1], 10 * guess_std]))

    # noinspection PyTypeChecker
    opt, err, fit = fit_distribution(x_to_fit, y_to_fit, fit_type='gaussian', guess=guess, bounds=bounds)
    opt[2] += window[0]
    fit = gaussian_fit(x_data, *opt)
    return opt, err, fit


def find_spot(image: np.ndarray,
              hp_median: int = 2, hp_threshold: float = 3., denoise_cycles: int = 3,
              gauss_filter: float = 5., com_threshold: float = 0.5) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Finds spot in an image

    hp_median:      hot pixels median filter size
    hp_threshold:   hot pixels threshold in # of sta. dev. of the median deviation
    gauss_filter:   gaussian filter size
    com_threshold:  image threshold for center-of-mass calculation
    """
    filter_dict = basic_filter(image, hp_median, hp_threshold, denoise_cycles, gauss_filter, com_threshold)

    return filter_dict['position_com'], filter_dict['position_max']


if __name__ == "__main__":
    f_path = os.path.normpath(r'C:\Users\gplateau\Box\SEN\SEN14.Prop_21_045_SF001770 Beta Unit Restart 2021'
                              + r'\Design-VerificationAndValidation\Data\210528 BS1 ENG07 Power Supply'
                              + r'\1327 70kV Max Emission\1404_70kV_e_0p65mA_f_1p570A_X_70mA_Y_-120mA')

    # img, _ = avg_tiff(img_dir=f_path, min_imgs=1)
    # x_opt, y_opt, x_fit, y_fit = spot_analysis(img)
    # x_lim = [round((x_opt[2] - 2*fwhm(x_opt[3])) / 10) * 10, round((x_opt[2] + 2*fwhm(x_opt[3])) / 10) * 10]
    # y_lim = [round((y_opt[2] - 2*fwhm(y_opt[3])) / 10) * 10, round((y_opt[2] + 2*fwhm(y_opt[3])) / 10) * 10]
    #
    # fig = plt.figure()
    # fig.add_subplot(121)
    # plt.plot(x_fit[0], x_fit[1], 'b-', label='data')
    # plt.plot(x_fit[0], x_fit[2], 'r-', label='fit (FWHM = %.1f)' % fwhm(x_opt[3]))
    # plt.gca().set_xlim(x_lim)
    # plt.xlabel('X-axis')
    # plt.ylabel('Amplitude')
    # plt.legend(loc='upper left')
    #
    # fig.add_subplot(122)
    # plt.plot(y_fit[0], y_fit[1], 'b-', label='data')
    # plt.plot(y_fit[0], y_fit[2], 'r-', label='fit (FWHM = %.1f)' % fwhm(y_opt[3]))
    # plt.gca().set_xlim(y_lim)
    # plt.xlabel('Y-axis')
    # plt.ylabel('Amplitude')
    # plt.legend(loc='upper right')
    #
    # plt.savefig(os.path.join(r'C:\Users\gplateau\Documents\Data\Tmp', 'spot_fits.png'))
    # plt.show(block=True)
