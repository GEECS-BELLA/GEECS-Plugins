""" @author: Guillaume Plateau, TAU Systems """

import os
import numpy as np
from typing import Optional, Any
import scipy.ndimage as simg
from geecs_api.tools.images.filtering import filter_image
from geecs_api.tools.distributions.fit_utility import fit_distribution


def fwhm(sd):
    return 2 * np.sqrt(2 * np.log(2)) * sd


def spot_analysis(image: np.ndarray, positions: list[tuple[int, int, str]]) -> Optional[dict[str, Any]]:
    """
    Runs Gaussian fits at a given list of positions (e.g. peak vs. center-of-mass)

    image: numpy array
    positions: list of tuples of (i, j, label)
    """
    analysis: Optional[dict[str, Any]]

    try:
        analysis = {}

        for pos_i, pos_j, name in positions:
            axis_x = np.arange(image.shape[1])
            data_x = image[pos_i, :]
            opt_x, fit_x = profile_fit(axis_x, data_x, pos_i, image[pos_i, pos_j])

            analysis[f'axis_x_{name}'] = axis_x
            analysis[f'data_x_{name}'] = data_x
            analysis[f'opt_x_{name}'] = opt_x[0]
            analysis[f'fit_x_{name}'] = fit_x

            axis_y = np.arange(image.shape[0])
            data_y = image[:, pos_j]
            opt_y, fit_y = profile_fit(axis_y, data_y, pos_j, image[pos_i, pos_j])

            analysis[f'axis_y_{name}'] = axis_y
            analysis[f'data_y_{name}'] = data_y
            analysis[f'opt_y_{name}'] = opt_y[0]
            analysis[f'fit_y_{name}'] = fit_y

    except Exception:
        analysis = None
        pass

    return analysis


def profile_fit(x_data: np.ndarray, y_data: np.ndarray,
                guess_center: Optional[int] = None,
                guess_amplitude: Optional[float] = None):
    guess_com = simg.center_of_mass(y_data)[0]
    if guess_center:
        guess_std = abs(guess_center - guess_com)
    else:
        guess_std = (x_data[-1] - x_data[0]) / 10.
    if guess_amplitude is None:
        guess_amplitude = np.max(y_data) - np.min(y_data)
    guess = [0, guess_amplitude, guess_com, guess_std]

    # noinspection PyTypeChecker
    return fit_distribution(x_data, y_data, fit_type='gaussian', guess=guess)


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
    filter_dict = filter_image(image, hp_median, hp_threshold, denoise_cycles, gauss_filter, com_threshold)

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
