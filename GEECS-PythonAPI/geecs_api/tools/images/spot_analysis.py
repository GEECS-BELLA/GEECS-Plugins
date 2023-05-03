import matplotlib.pyplot as plt
import scipy.ndimage as simg
import scipy.optimize as sopt
import numpy as np
import os


def fit_distribution(x_data, y_data, fit_type='linear', guess=None):
    fit = y_data
    opt = None

    if fit_type == 'linear':
        opt, _ = sopt.curve_fit(linear_fit, x_data, y_data, p0=guess)
        fit = linear_fit(x_data, *opt)

    if fit_type == 'root':
        opt, _ = sopt.curve_fit(root_fit, x_data, y_data, p0=guess)
        fit = root_fit(x_data, *opt)

    if fit_type == 'fwhm_fit':
        opt, _ = sopt.curve_fit(fwhm_fit, x_data, y_data, p0=guess)
        fit = fwhm_fit(x_data, *opt)

    return tuple(opt), tuple([x_data, y_data, fit])


def linear_fit(x, m, b):
    return m * x + b


def root_fit(x, a, b, c, d):
    return a + b * ((x - c)**d)


def fwhm_fit(x, a, b, c):
    return a * np.sqrt(1 + ((x - b) / c)**2)


def gauss_fit(x, a, b, c, d):
    return a + b * np.exp(-(x - c)**2 / (2 * d**2))


def fwhm(sd):
    return 2 * np.sqrt(2 * np.log(2)) * sd


def spot_analysis(image, filter_size=5, threshold=0.5):
    i_spot, j_spot = find_spot(image, filter_size=filter_size, threshold=threshold)

    # X-axis Gaussian fit
    data_x = image[i_spot, :]
    axis_x = np.arange(data_x.size)

    com_x = simg.center_of_mass(data_x)[0]
    std_x = abs(j_spot - com_x)
    guess_x = [0, image[i_spot, j_spot], com_x, std_x]

    # noinspection PyTypeChecker
    opt_x, _ = sopt.curve_fit(gauss_fit, axis_x, data_x, guess_x)
    fit_x = gauss_fit(axis_x, *opt_x)

    # Y-axis Gaussian fit
    data_y = image[:, j_spot]
    axis_y = np.arange(data_y.size)

    com_y = simg.center_of_mass(data_y)[0]
    std_y = abs(i_spot - com_y)
    guess_y = [0, image[i_spot, j_spot], com_y, std_y]

    # noinspection PyTypeChecker
    opt_y, _ = sopt.curve_fit(gauss_fit, axis_y, data_y, guess_y)
    fit_y = gauss_fit(axis_y, *opt_y)

    return tuple(opt_x), tuple(opt_y), tuple([axis_x, data_x, fit_x]), tuple([axis_y, data_y, fit_y])


# noinspection PyArgumentList
def find_spot(image, filter_size=5, threshold=0.5):
    # Gaussian filter
    image = simg.gaussian_filter(image, sigma=filter_size)

    # Peak location
    i_max, j_max = np.where(image == image.max())
    i_max, j_max = i_max[0].item(), j_max[0].item()

    # Center of mass of thresholded image
    val_max = image[i_max, j_max]
    binary = image > (threshold * val_max)
    binary = binary.astype(float)
    img_thr = image * binary

    i_peak, j_peak = simg.center_of_mass(img_thr)

    return int(round(i_peak)), int(round(j_peak))


if __name__ == "__main__":
    f_path = os.path.normpath(r'C:\Users\gplateau\Box\SEN\SEN14.Prop_21_045_SF001770 Beta Unit Restart 2021'
                              + r'\Design-VerificationAndValidation\Data\210528 BS1 ENG07 Power Supply'
                              + r'\1327 70kV Max Emission\1404_70kV_e_0p65mA_f_1p570A_X_70mA_Y_-120mA')

    img, _ = avg_tiff(img_dir=f_path, min_imgs=1)
    x_opt, y_opt, x_fit, y_fit = spot_analysis(img)
    x_lim = [round((x_opt[2] - 2*fwhm(x_opt[3])) / 10) * 10, round((x_opt[2] + 2*fwhm(x_opt[3])) / 10) * 10]
    y_lim = [round((y_opt[2] - 2*fwhm(y_opt[3])) / 10) * 10, round((y_opt[2] + 2*fwhm(y_opt[3])) / 10) * 10]

    fig = plt.figure()
    fig.add_subplot(121)
    plt.plot(x_fit[0], x_fit[1], 'b-', label='data')
    plt.plot(x_fit[0], x_fit[2], 'r-', label='fit (FWHM = %.1f)' % fwhm(x_opt[3]))
    plt.gca().set_xlim(x_lim)
    plt.xlabel('X-axis')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left')

    fig.add_subplot(122)
    plt.plot(y_fit[0], y_fit[1], 'b-', label='data')
    plt.plot(y_fit[0], y_fit[2], 'r-', label='fit (FWHM = %.1f)' % fwhm(y_opt[3]))
    plt.gca().set_xlim(y_lim)
    plt.xlabel('Y-axis')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(r'C:\Users\gplateau\Documents\Data\Tmp', 'spot_fits.png'))
    # plt.show(block=True)
