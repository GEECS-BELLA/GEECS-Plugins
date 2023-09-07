# -*- coding: utf-8 -*-
"""
Created on Mon Aug 8 2023

Module for performing various analysis on MagSpec Images.

This updated version is cleaner and streamlined to work on LabView.  Most notably by not opening TDMS file or
the interpSpec to parse any additional information.  Just make sure some of the constants are correct.

@author: chris
"""

import numpy as np
import time
from scipy import optimize

from . import CedossMathTools as MathTools
from . import EnergyAxisLookup_HiRes as EnergyAxisLookup

def PrintTime(label, time_in, doPrint=False):
    if doPrint:
        print(label, time.perf_counter() - time_in)
    return time.perf_counter()


def AnalyzeImage(inputImage, inputParams):
    currentTime = time.perf_counter()
    doPrint = False

    numPixelCrop = inputParams["Pixel-Crop"]
    roiImage = inputImage[numPixelCrop: -numPixelCrop, numPixelCrop: -numPixelCrop]

    saturationValue = inputParams["Saturation-Value"]
    saturationCheck = SaturationCheck(roiImage, saturationValue)
    currentTime = PrintTime(" Saturation Check:", currentTime, doPrint=doPrint)

    threshold = inputParams["Threshold-Value"]
    image = ThresholdReduction(roiImage, threshold)
    currentTime = PrintTime(" Threshold Subtraction", currentTime, doPrint=doPrint)

    normalizationFactor = inputParams["Normalization-Factor"]
    image = NormalizeImage(image, normalizationFactor)
    currentTime = PrintTime(" Normalize Image:", currentTime, doPrint=doPrint)

    image = np.copy(image[::-1, ::-1])
    currentTime = PrintTime(" Rotate Image", currentTime, doPrint=doPrint)

    image_width = np.shape(image)[1]
    pixel_arr = np.linspace(0, image_width, image_width)
    energy_arr = EnergyAxisLookup.return_default_energy_axis(pixel_arr)
    currentTime = PrintTime(" Calculated Energy Axis:", currentTime, doPrint=doPrint)

    chargeOnCamera = np.sum(image)
    currentTime = PrintTime(" Charge on Camera", currentTime, doPrint=doPrint)

    clippedPercentage = CalculateClippedPercentage(image)
    currentTime = PrintTime(" Calculate Clipped Percentage", currentTime, doPrint=doPrint)

    calibrationFactor = inputParams["Transverse-Calibration"]
    charge_arr = CalculateChargeDensityDistribution(image, calibrationFactor)
    currentTime = PrintTime(" Charge Projection:", currentTime, doPrint=doPrint)

    if np.sum(charge_arr) == 0:
        peakCharge = float(0)
        averageEnergy = float(-1)
        energySpread = float(0)
        peakChargeEnergy = float(0)
        averageBeamSize = float(0)
        beamAngle = float(0)
        beamIntercept = float(0)
        projectedBeamSize = float(0)
        optimization_factor = float(0)
    else:
        peakCharge = CalculateMaximumCharge(charge_arr)
        currentTime = PrintTime(" Peak Charge:", currentTime, doPrint=doPrint)

        averageEnergy = CalculateAverageEnergy(charge_arr, energy_arr)
        currentTime = PrintTime(" Average Energy:", currentTime, doPrint=doPrint)

        energySpread = CalculateStandardDeviationEnergy(charge_arr, energy_arr, averageEnergy)
        currentTime = PrintTime(" Energy Spread:", currentTime, doPrint=doPrint)

        peakChargeEnergy = CalculatePeakEnergy(charge_arr, energy_arr)
        currentTime = PrintTime(" Energy at Peak Charge:", currentTime, doPrint=doPrint)

        central_energy = inputParams["Optimization-Central-Energy"]
        bandwidth_energy = inputParams["Optimization-Bandwidth-Energy"]
        optimization_factor = CalculateOptimizationFactor(charge_arr, energy_arr, central_energy, bandwidth_energy)
        currentTime = PrintTime(" Optimization Factor:", currentTime, doPrint=doPrint)

        doTransverse = inputParams["Do-Transverse-Calculation"]
        if doTransverse:
            binsize = inputParams["Transverse-Slice-Binsize"]
            sliceThreshold = inputParams["Transverse-Slice-Threshold"]
            sigma_arr, x0_arr, amp_arr, err_arr = TransverseSliceLoop(image, calibrationFactor=calibrationFactor,
                                                                      threshold=sliceThreshold, binsize=binsize)
            currentTime = PrintTime(" Gaussian Fits for each Slice:", currentTime, doPrint=doPrint)

            averageBeamSize = CalculateAverageSize(sigma_arr, amp_arr)
            currentTime = PrintTime(" Average Beam Size:", currentTime, doPrint=doPrint)

            linear_fit = FitBeamAngle(x0_arr, amp_arr, energy_arr)
            currentTime = PrintTime(" Beam Angle Fit:", currentTime, doPrint=doPrint)

            beamAngle = linear_fit[0]
            beamIntercept = linear_fit[1]
            projected_axis, projected_arr, projectedBeamSize = CalculateProjectedBeamSize(image, calibrationFactor)
            projectedBeamSize = projectedBeamSize * calibrationFactor
            PrintTime(" Projected Size:", currentTime, doPrint=doPrint)
        else:
            averageBeamSize = float(0.0)
            projectedBeamSize = float(0.0)
            beamAngle = float(0.0)
            beamIntercept = float(0.0)

    magSpecDict = {
        "Clipped-Percentage": float(clippedPercentage),  # 1
        "Saturation-Counts": int(saturationCheck),  # 2
        "Charge-On-Camera": float(chargeOnCamera),  # 3
        "Peak-Charge": float(peakCharge),  # 4
        "Peak-Charge-Energy": float(peakChargeEnergy),  # 5
        "Average-Energy": float(averageEnergy),  # 6
        "Energy-Spread": float(energySpread),  # 7
        "Energy-Spread-Percent": float(energySpread / averageEnergy),  # 8
        "Average-Beam-Size": float(averageBeamSize),
        "Projected-Beam-Size": float(projectedBeamSize),
        "Beam-Tilt": float(beamAngle),
        "Beam-Intercept": float(beamIntercept),
        "Beam-Intercept-100MeV": float(100 * beamAngle + beamIntercept),
        "Optimization-Factor": float(optimization_factor)
    }
    return image, magSpecDict, np.vstack((energy_arr, charge_arr))


def NormalizeImage(image, normalizationFactor):
    returnimage = np.copy(image) * normalizationFactor
    return returnimage


def ThresholdReduction(image, threshold):
    returnimage = np.copy(image) - threshold
    returnimage[np.where(returnimage < 0)] = 0
    return returnimage


def CalculateClippedPercentage(image):
    # roi_image = image[1:-1,1:-1]
    clipcheck = np.append(np.append(np.append(image[0, :], image[:, 0]), image[-1, :]), image[:, -1])
    maxval = np.max(image)
    if maxval != 0:
        return np.max(clipcheck) / maxval
    else:
        return 1.1


def CalculateProjectedBeamSize(image, calibrationFactor):
    skipPlots = True
    proj_arr = np.sum(image, axis=1)

    axis_arr = np.linspace(0, len(proj_arr) * calibrationFactor, len(proj_arr))
    axis_arr = np.flip(axis_arr)
    fit = MathTools.FitDataSomething(proj_arr, axis_arr,
                                     MathTools.Gaussian,
                                     guess=[max(proj_arr), 20 * calibrationFactor, axis_arr[np.argmax(proj_arr)]],
                                     supress=skipPlots)
    beamSize = fit[1]
    return axis_arr, proj_arr, beamSize


def CalculateChargeDensityDistribution(image, calibrationFactor):
    charge_arr = np.sum(image, axis=0) * calibrationFactor
    return charge_arr


def CalculateMaximumCharge(charge_arr):
    return np.max(charge_arr)


def CalculateAverageEnergy(charge_arr, energy_arr):
    return np.average(energy_arr, weights=charge_arr)


def CalculateStandardDeviationEnergy(charge_arr, energy_arr, average_energy=None):
    if average_energy is None:
        average_energy = CalculateAverageEnergy(charge_arr, energy_arr)
    return np.sqrt(np.average((energy_arr - average_energy) ** 2, weights=charge_arr))


def CalculatePeakEnergy(charge_arr, energy_arr):
    return energy_arr[np.argmax(charge_arr)]


def CalculateAverageSize(sigma_arr, amp_arr):
    return np.average(sigma_arr, weights=amp_arr)


def FitBeamAngle(x0_arr, amp_arr, energy_arr):
    linear_fit = np.polyfit(energy_arr, x0_arr, deg=1, w=np.power(amp_arr, 2))
    return linear_fit


def FindMax(image):
    ymax, xmax = np.unravel_index(np.argmax(image), image.shape)
    maxval = image[ymax, xmax]
    return xmax, ymax, maxval


def SaturationCheck(image, saturationValue):
    return len(np.where(image > saturationValue)[0])


def Gaussian(x, amp, sigma, x0):
    return amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def FitDataSomething(data, axis, function, guess=[0., 0., 0.]):
    errfunc = lambda p, x, y: function(x, *p) - y
    p0 = guess
    p1, success = optimize.leastsq(errfunc, p0[:], args=(axis, data))
    return p1


def TransverseSliceLoop(image, calibrationFactor=1, threshold=0.01, binsize=1, option=1):
    # option 0 for Gaussian fits, option 1 for moment statistics.  1 is usually super good enough
    ny, nx = np.shape(image)
    xloc, yloc, maxval = FindMax(image)

    sigma_arr = np.zeros(nx)
    err_arr = np.zeros(nx)
    x0_arr = np.zeros(nx)
    amp_arr = np.zeros(nx)

    for i in range(int(nx / binsize)):
        if binsize == 1:
            slice_arr = image[:, i]
        else:
            slice_arr = np.average(image[:, binsize * i:(binsize * i) + binsize - 1], axis=1)
        if np.max(slice_arr) > threshold * maxval:
            axis_arr = np.linspace(0, len(slice_arr), len(slice_arr)) * calibrationFactor
            axis_arr = np.flip(axis_arr)

            if option == 0:
                slice_sigma, slice_x0, slice_amp, slice_err = FitTransverseGaussianSlices(axis_arr, slice_arr)
            if option == 1:
                slice_sigma, slice_x0, slice_amp, slice_err = GetTransverseStatSlices(axis_arr, slice_arr)

        else:
            slice_sigma = 0
            slice_x0 = 0
            slice_amp = 0
            slice_err = 0
        sigma_arr[binsize * i:binsize * (i + 1)] = slice_sigma
        x0_arr[binsize * i:binsize * (i + 1)] = slice_x0
        amp_arr[binsize * i:binsize * (i + 1)] = slice_amp
        err_arr[binsize * i:binsize * (i + 1)] = slice_err
    return sigma_arr, x0_arr, amp_arr, err_arr


def GetTransverseStatSlices(axis_arr, slice_arr):
    amp_slice = np.average(slice_arr)
    x0_slice = np.average(np.average(axis_arr, weights=slice_arr))
    sigma_slice = np.sqrt(np.average(np.power(axis_arr - x0_slice, 2), weights=slice_arr))
    err_slice = 0
    return sigma_slice, x0_slice, amp_slice, err_slice


def FitTransverseGaussianSlices(axis_arr, slice_arr):
    fit = FitDataSomething(slice_arr, axis_arr, Gaussian,
                           guess=[max(slice_arr), 5 * 43,  # calibrationFactor,
                                  axis_arr[np.argmax(slice_arr)]])
    amp_fit, sigma_fit, x0_fit = fit

    # Check to see that our x0 is within bounds (only an issue for vertically-clipped beams)
    if x0_fit < axis_arr[-1] or x0_fit > axis_arr[0]:
        sigma_fit = 0
        x0_fit = 0
        amp_fit = 0
        err_fit = 0
    else:
        func = Gaussian(axis_arr, *fit)
        error = np.sum(np.square(slice_arr - func))
        err_fit = np.sqrt(error) * 1e3
    return sigma_fit, x0_fit, amp_fit, err_fit


def CalculateOptimizationFactor(charge_arr, energy_arr, central_energy, bandwidth_energy):
    gaussian_weight_function = Gaussian(energy_arr, 1.0, bandwidth_energy, central_energy)
    optimization_factor = np.sum(charge_arr * gaussian_weight_function)
    return optimization_factor
