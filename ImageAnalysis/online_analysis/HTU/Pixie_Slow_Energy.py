import numpy as np
#calb = 1.9e+06 # 300p
#calb = 4.2e+06 # 1n
#calb = 1.2e+07 # 3n
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
def Integrate_Energy(y, dt, calb):
    # y is a numpy array
    # dt is extracted from the .DAT file of the pixie slow device
    #Calibration is a parameter to be passed into the function refer
    # to 09/26/2023 daily log for the calibration factor to pass

    # Potential pre-processing that might need to be done if the butter worth filter is not corresponding to the correct
    # bounds.
    #####
    # v = np.copy(y)
    # floor = np.mean(v[0:50])
    # vs = v - floor
    # vs = moving_average(data=vs, window_size=5)
    # y = np.copy(vs)
    ##### End of pre processing
    dx = 2
    ygrad = np.gradient(y)
    i = np.argmax(ygrad)
    y_ = y[i - int(4 * dx):i + int(40 * dx)]
    return np.trapz(y_, x=None, dx=dt)*calb # [mJ] if you used the number that's liste in the daily log