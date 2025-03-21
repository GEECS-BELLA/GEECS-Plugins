""" @author: Guillaume Plateau, TAU Systems """

import numpy as np
from scipy.optimize import curve_fit


# noinspection PyTupleAssignmentBalance
def fit_distribution(x_data: np.ndarray, y_data: np.ndarray, fit_type='linear', guess=None, bounds=None):
    """
    fit_type: 'linear', 'root', 'gaussian'
    """
    fit: np.ndarray = y_data
    opt = err = None

    if fit_type == 'linear':
        if bounds:
            opt, err = curve_fit(linear_fit, x_data, y_data, p0=guess, bounds=bounds)
        else:
            opt, err = curve_fit(linear_fit, x_data, y_data, p0=guess)
        fit = linear_fit(x_data, *opt)

    if fit_type == 'root':
        if bounds:
            opt, err = curve_fit(root_fit, x_data, y_data, p0=guess, bounds=bounds)
        else:
            opt, err = curve_fit(root_fit, x_data, y_data, p0=guess)
        fit = root_fit(x_data, *opt)

    if fit_type == 'gaussian':
        if bounds:
            opt, err = curve_fit(gaussian_fit, x_data, y_data, p0=guess, bounds=bounds)
        else:
            opt, err = curve_fit(gaussian_fit, x_data, y_data, p0=guess)
        fit = gaussian_fit(x_data, *opt)

    err = np.sqrt(np.diag(err))
    return opt, err, fit


def linear_fit(x, m, b):
    return m * x + b


def root_fit(x, a, b, c, d):
    return a + b * ((x - c)**d)


def gaussian_fit(x, a, b, c, d):
    return a + b * np.exp(-(x - c)**2 / (2 * d**2))
