import numpy as np
from typing import Optional
from scipy import signal
from scipy.optimize import least_squares


def apply_butterworth_filter(data, order=int(1), crit_f=0.025, filt_type='low'):
    # generate butterworth filter
    b, a = signal.butter(order, crit_f, filt_type)

    # apply filter in forward and backward propogation
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def identify_primary_valley(data):
    # find min value (assuming first peak is largest and negative)
    min_ind = np.argmin(data)

    # find where signal goes to zero before spike
    count = 1
    test_val = data[min_ind]
    try:
        while test_val < 0:
            # define test ind and val
            test_ind = min_ind - count
            test_val = data[test_ind]
            count += 1
        # define range minimum
        valley_min = int(test_ind + 1)
    except:
        valley_min = 0

    # find where signal goes to zero after spike
    count = 1
    test_val = data[min_ind]
    try:
        while test_val < 0:
            # define test ind and val
            test_ind = min_ind + count
            test_val = data[test_ind]
            # increment count
            count += 1
        # define range minimum
        valley_max = int(test_ind)
    except:
        valley_max = len(data)

    # define array of indices corresponding to spike
    valley_ind = np.arange(valley_min, valley_max)

    return valley_ind


def get_sinusoidal_noise(data, signal_region: tuple[Optional[int], Optional[int]]):
    """
    Fits a sinusoidal function to the given region of "noise"

    Parameters
    ----------
    data - full numpy array or list of data
    signal_region - tuple representing the first and last index of the region with the beam signal, which we exclude

    Returns
    -------
    A numpy array representing the sinusoidal fit for the full data range
    """
    x_axis = np.arange(len(data))

    p1 = signal_region[0]
    p2 = signal_region[1]

    if p1 is None and p2 is not None:
        bg_data = data[p2:]
        bg_axis = x_axis[p2:]
    elif p2 is None and p1 is not None:
        bg_data = data[:p1]
        bg_axis = x_axis[:p1]
    else:
        bg_data = np.concatenate((data[:p1], data[p2:]))
        bg_axis = np.concatenate((x_axis[:p1], x_axis[p2:]))

    def sin_model(t, amplitude, frequency, phase, offset):
        return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    def residuals(p, x, y):
        return y - sin_model(x, *p)

    fft_data = np.fft.rfft(bg_data)
    fft_axis = np.fft.rfftfreq(len(bg_data), d=(bg_axis[1] - bg_axis[0]) if len(bg_axis) > 1 else 1)

    if len(fft_data) > 2:
        dominant_freq_idx = np.argmax(np.abs(fft_data)[1:]) + 1
    else:
        dominant_freq_idx = np.argmax(np.abs(fft_data))

    initial_freq = fft_axis[dominant_freq_idx]

    ave_val = np.mean(bg_data)
    std_val = np.std(bg_data)

    if bg_data[0] > ave_val + std_val:
        phi_est = np.pi/2
    elif bg_data[0] < ave_val - std_val:
        phi_est = -np.pi/2
    elif bg_data[100] > bg_data[0]:
        phi_est = 0
    else:
        phi_est = np.pi

    p0 = [std_val, initial_freq, phi_est, np.mean(bg_data)]

    try:
        result = least_squares(residuals, p0, args=(bg_axis[::4], bg_data[::4]), loss='soft_l1')
        params = result.x
    except RuntimeError:
        params = p0

    background_model = sin_model(x_axis, *params)

    return background_model


def test_func(data, dt, crit_f, calib):
    value = np.array(data)
    value = np.array(apply_butterworth_filter(value, order=int(1), crit_f=crit_f))

    signal_location = np.argmin(data)
    first_interval_end = signal_location - 100 if signal_location > 100 else None
    second_interval_start = signal_location + 600 if signal_location + 600 < len(value) else None

    # Perform the sinusoidal fit twice
    value -= get_sinusoidal_noise(data=value, signal_region=(first_interval_end, second_interval_start))
    value -= get_sinusoidal_noise(data=value, signal_region=(first_interval_end, second_interval_start))

    ind_roi = identify_primary_valley(value)
    value = np.array(value[ind_roi])
    integrated_signal = np.trapz(value, x=None, dx=dt)
    charge_pC = integrated_signal * -calib * 10 ** (12)

    return charge_pC


def B_Cave_ICT(data, dt, crit_f):
    calib = 0.2
    charge_pC = test_func(data, dt, crit_f, calib)
    return charge_pC


def Undulator_Exit_ICT(data, dt, crit_f):
    """ dt = 4e-9, crit_f = 0.125 """
    calib = 0.2 / 2.78
    charge_pC = test_func(data, dt, crit_f, calib)
    return charge_pC
