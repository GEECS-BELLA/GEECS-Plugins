"""
Fri 8-4-2023

Remaking function to take a Gaussian fit of each transverse slice and returning the array of fit parameters.  Using
ChatGPT suggestions and some human debugging.

@Chris
"""
import sys
import numpy as np

# sys.path.insert(0, "../")
import modules.MagSpecAnalysis as MagSpec
from scipy import optimize

const_HiResMagSpec_Resolution = 43

def Gaussian(x, amp, sigma, x0):
    return amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def FitDataSomething(data, axis, function, guess=[0., 0., 0.]):
    errfunc = lambda p, x, y: function(x, *p) - y
    p0 = guess
    p1, success = optimize.leastsq(errfunc, p0[:], args=(axis, data))
    return p1

def FitTransverseGaussianSlices(image, calibrationFactor=1, threshold=0.01, binsize=1):
    ny, nx = np.shape(image)
    xloc, yloc, maxval = MagSpec.FindMax(image)

    sigma_arr = np.zeros(nx)
    err_arr = np.zeros(nx)
    x0_arr = np.zeros(nx)
    amp_arr = np.zeros(nx)

    for i in range(int(nx/binsize)):
        if binsize == 1:
            slice_arr = image[:, i]
        else:
            slice_arr = np.average(image[:, binsize*i:(binsize*i)+binsize-1], axis=1)
        if np.max(slice_arr) > threshold * maxval:
            axis_arr = np.linspace(0, len(slice_arr), len(slice_arr)) * calibrationFactor
            axis_arr = np.flip(axis_arr)

            fit = FitDataSomething(slice_arr, axis_arr, Gaussian,
                                   guess=[max(slice_arr), 5 * const_HiResMagSpec_Resolution,
                                          axis_arr[np.argmax(slice_arr)]])
            amp_fit, sigma_fit, x0_fit = fit
            amp_arr[binsize*i:binsize*(i+1)] = amp_fit
            sigma_arr[binsize*i:binsize*(i+1)] = sigma_fit
            x0_arr[binsize*i:binsize*(i+1)] = x0_fit

            # Check to see that our x0 is within bounds (only an issue for vertically-clipped beams)
            if x0_arr[binsize*i] < axis_arr[-1] or x0_arr[binsize*i] > axis_arr[0]:
                sigma_arr[binsize*i:binsize*(i+1)] = 0
                x0_arr[binsize*i:binsize*(i+1)] = 0
                amp_arr[binsize*i:binsize*(i+1)] = 0
                err_arr[binsize*i:binsize*(i+1)] = 0
            else:
                func = Gaussian(axis_arr, *fit)
                error = np.sum(np.square(slice_arr - func))
                err_arr[binsize*i:binsize*(i+1)] = np.sqrt(error) * 1e3
        else:
            sigma_arr[binsize*i:binsize*(i+1)] = 0
            x0_arr[binsize*i:binsize*(i+1)] = 0
            amp_arr[binsize*i:binsize*(i+1)] = 0
            err_arr[binsize*i:binsize*(i+1)] = 0
    return sigma_arr, x0_arr, amp_arr, err_arr

"""
const_HiResMagSpec_Resolution = 43

def FitTransverseGaussianSlices(image, calibrationFactor=1, threshold=0.01):
    ny, nx = np.shape(image)
    xloc, yloc, maxval = MagSpec.FindMax(image)
    skipPlots = True  # False for debugging and/or make an animation book

    sigma_arr = np.zeros(nx)
    err_arr = np.zeros(nx)
    x0_arr = np.zeros(nx)
    amp_arr = np.zeros(nx)
    for i in range(nx):
        slice_arr = image[:, i]
        if np.max(slice_arr) > threshold * maxval:
            axis_arr = np.linspace(0, len(slice_arr), len(slice_arr))
            axis_arr = np.flip(axis_arr) * calibrationFactor
            # if i%100 == 0:
            #    skipPlots = False
            # else:
            #    skipPlots = True
            fit = FitDataSomething(slice_arr, axis_arr,
                                             Gaussian,
                                             guess=[max(slice_arr), 5 * const_HiResMagSpec_Resolution,
                                                    axis_arr[np.argmax(slice_arr)]])
            amp_arr[i], sigma_arr[i], x0_arr[i] = fit

            # Check to see that our x0 is within bounds (only an issue for vertically-clipped beams
            if x0_arr[i] < axis_arr[-1] or x0_arr[i] > axis_arr[0]:
                sigma_arr[i] = 0
                x0_arr[i] = 0
                amp_arr[i] = 0
                err_arr[i] = 0
            else:
                # TODO:  This error calculation below is "correct," but produces no meaningful error.  Please find a way
                func = Gaussian(fit, axis_arr)
                error = 0
                for j in range(len(slice_arr)):
                    error = error + np.square(slice_arr[j] - func[j])
                err_arr[i] = np.sqrt(error) * 1e3

        else:
            sigma_arr[i] = 0
            x0_arr[i] = 0
            amp_arr[i] = 0
            err_arr[i] = 0

    return sigma_arr, x0_arr, amp_arr, err_arr

def FitDataSomething(data, axis, function, guess = [0.,0.,0.]):
    errfunc = lambda p, x, y: function(p, x) - y
    p0 = guess
    p1, success = optimize.leastsq(errfunc, p0[:], args=(axis, data))
    return p1

def Gaussian(p, x):
    return  p[0]*np.exp(-.5*np.square(x-p[2])/np.square(p[1]))
"""

