""" @author: Guillaume Plateau, TAU Systems """

import os
import numpy as np
from typing import Optional, Any
import scipy.ndimage as simg
from geecs_api.tools.images.filtering import clip_outliers
from geecs_api.tools.line_analysis.fit_utility import fit_distribution


def fwhm(sd):
    return 2 * np.sqrt(2 * np.log(2)) * sd


def spot_analysis(image, mad_multiplier: float = 5.2, filter_size=5, threshold=0.5) -> Optional[dict[str, Any]]:
    """ @author: Guillaume Plateau, TAU Systems """
    analysis: Optional[dict[str, Any]]

    try:
        analysis = {}

        # find spot
        i_spot, j_spot = find_spot(image, mad_multiplier, filter_size, threshold)
        analysis['mad_multiplier'] = mad_multiplier
        analysis['gaussian_filter'] = filter_size
        analysis['threshold'] = threshold
        analysis['position'] = (i_spot, j_spot)

        # X-axis Gaussian fit
        data_x = image[i_spot, :]
        axis_x = np.arange(data_x.size)

        com_x = simg.center_of_mass(data_x)[0]
        std_x = abs(j_spot - com_x)
        guess_x = [0, image[i_spot, j_spot], com_x, std_x]

        # noinspection PyTypeChecker
        opt_x, fit_x = fit_distribution(axis_x, data_x, fit_type='gaussian', guess=guess_x)
        analysis['axis_x'] = axis_x
        analysis['data_x'] = data_x
        analysis['opt_x'] = opt_x[0]
        analysis['fit_x'] = fit_x

        # Y-axis Gaussian fit
        data_y = image[:, j_spot]
        axis_y = np.arange(data_y.size)

        com_y = simg.center_of_mass(data_y)[0]
        std_y = abs(i_spot - com_y)
        guess_y = [0, image[i_spot, j_spot], com_y, std_y]

        # noinspection PyTypeChecker
        opt_y, fit_y = fit_distribution(axis_y, data_y, fit_type='gaussian', guess=guess_y)
        analysis['axis_y'] = axis_y
        analysis['data_y'] = data_y
        analysis['opt_y'] = opt_y[0]
        analysis['fit_y'] = fit_y

    except Exception:
        analysis = None
        pass

    return analysis


# noinspection PyArgumentList
def find_spot(image, mad_multiplier: float = 5.2, filter_size: float = 5., threshold: float = 0.5):
    """ @author: Guillaume Plateau, TAU Systems """
    # MAD filter
    if mad_multiplier > 0:
        image = clip_outliers(image, mad_multiplier, nonzero_only=False)

    # gaussian filter
    image = simg.gaussian_filter(image, sigma=filter_size)

    # peak location
    i_max, j_max = np.where(image == image.max())
    i_max, j_max = i_max[0].item(), j_max[0].item()

    # center of mass of thresholded image
    val_max = image[i_max, j_max]
    binary = image > (threshold * val_max)
    binary = binary.astype(float)
    img_thr = image * binary

    i_spot, j_spot = simg.center_of_mass(img_thr)

    return int(round(i_spot)), int(round(j_spot))


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
