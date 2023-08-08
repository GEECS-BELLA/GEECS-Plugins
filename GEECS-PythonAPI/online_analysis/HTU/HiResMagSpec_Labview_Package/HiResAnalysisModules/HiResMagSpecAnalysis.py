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

import HiResAnalysisModules.CedossMathTools_HiRes as MathTools
import HiResAnalysisModules.EnergyAxisLookup_HiRes as EnergyAxisLookup


def PrintTime(label, time_in, doPrint = False):
    if doPrint:
        print(label, time.perf_counter()-time_in)
    return time.perf_counter()

def AnalyzeImage(image, inputParams):
    currentTime = time.perf_counter()
    doPrint = False

    saturationValue = inputParams["Saturation-Value"]
    saturationCheck = SaturationCheck(image, saturationValue)
    currentTime = PrintTime(" Saturation Check:", currentTime, doPrint=doPrint)

    threshold = inputParams["Threshold-Value"]
    image = ThresholdReduction(image, threshold)
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
    else:
        peakCharge = CalculateMaximumCharge(charge_arr)
        currentTime = PrintTime(" Peak Charge:", currentTime, doPrint=doPrint)

        averageEnergy = CalculateAverageEnergy(charge_arr, energy_arr)
        currentTime = PrintTime(" Average Energy:", currentTime, doPrint=doPrint)

        energySpread = CalculateStandardDeviationEnergy(charge_arr, energy_arr, averageEnergy)
        currentTime = PrintTime(" Energy Spread:", currentTime, doPrint=doPrint)

        peakChargeEnergy = CalculatePeakEnergy(charge_arr, energy_arr)
        currentTime = PrintTime(" Energy at Peak Charge:", currentTime, doPrint=doPrint)

        # binsize = inputParams["Transverse-Slice-Binsize"]
        # sliceThreshold = inputParams["Transverse-Slice-Threshold"]
        # sigma_arr, x0_arr, amp_arr, err_arr = FitTransverseGaussianSlices(image, calibrationFactor=calibrationFactor,
                                                                          # threshold=sliceThreshold, binsize=binsize)
        # currentTime = PrintTime(" Gaussian Fits for each Slice:", currentTime, doPrint=doPrint)

        # averageBeamSize = CalculateAverageSize(sigma_arr, amp_arr)
        # currentTime = PrintTime(" Average Beam Size:", currentTime, doPrint=doPrint)

        # linear_fit = FitBeamAngle(x0_arr, amp_arr, energy_arr)
        # currentTime = PrintTime(" Beam Angle Fit:", currentTime, doPrint=doPrint)

        # beamAngle = linear_fit[0]
        # beamIntercept = linear_fit[1]
        # projected_axis, projected_arr, projectedBeamSize = CalculateProjectedBeamSize(image, calibrationFactor)
        # projectedBeamSize = projectedBeamSize * calibrationFactor
        # PrintTime(" Projected Size:", currentTime, doPrint=doPrint)

    magSpecDict = {
        "Clipped-Percentage": clippedPercentage,
        "Saturation-Counts": saturationCheck,
        "Charge-On-Camera": chargeOnCamera,
        "Peak-Charge": peakCharge,
        "Peak-Charge-Energy": peakChargeEnergy,
        "Average-Energy": averageEnergy,
        "Energy-Spread": energySpread,
        "Energy-Spread-Percent": energySpread/averageEnergy,
        # "Average-Beam-Size": averageBeamSize,
        # "Projected-Beam-Size": projectedBeamSize,
        # "Beam-Tilt": beamAngle,
        # "Beam-Intercept": beamIntercept,
        # "Beam-Intercept-100MeV": 100*beamAngle + beamIntercept
    }
    return magSpecDict


def NormalizeImage(image, normalizationFactor):
    returnimage = np.copy(image) * normalizationFactor
    return returnimage

"""
def PrintNormalization(shotnumber, tdms_filepath):
    charge_pC_vals = GetBeamCharge(tdms_filepath)
    charge = charge_pC_vals[shotnumber - 1]

    trigger_list, exposure_list = GetCameraTriggerAndExposure(tdms_filepath)

    # Assuming the image is good, find the factor, camera delay, and shutter duration and
    # print out the information for copy-pasting into this module.

    print("The following are the normalization factors,")
    print(" paste them into MagSpecAnalysis.py:")
    print("const_normalization_triggerdelay = ", trigger_list[shotnumber - 1])
    print("const_normalization_exposure =", exposure_list[shotnumber - 1])
    return
"""

"""
def LoadImage(superpath, scannumber, shotnumber, folderpath):
    fullpath = DirectoryFunc.CompileFileLocation(superpath, scannumber, shotnumber, folderpath, suffix=".png")
    image = pngTools.nBitPNG(fullpath)
    return image
"""

def ThresholdReduction(image, threshold):
    returnimage = np.copy(image) - threshold
    returnimage[np.where(returnimage < 0)] = 0
    return returnimage


def CalculateClippedPercentage(image):
    clipcheck = np.append(np.append(np.append(image[0, :], image[:, 0]), image[-1, :]), image[:, -1])
    maxval = np.max(image)
    if maxval != 0:
        return np.max(clipcheck) / maxval
    else:
        return 1.0


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
    """
    Projects the MagSpec image onto the energy axis
    
    Parameters
    ----------
    image : 2D Float Numpy Array
        The MagSpecImage.

    Returns
    -------
    charge_arr : 1D Float Numpy Array
        Summation of the charge for each slice in energy
    
    """
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

"""
def GetBeamCharge(tdms_filepath):
    # For future reference, can list all of the groups using tdms_file.groups()
    # and can list all of the groups, channels using group.channels()
    #
    # Also note, this loads the entire TDMS file into memory, and so a more
    # elegant usage of TdmsFile could be only reading the necessary picoscope data
    tdms_file = TdmsFile.read(tdms_filepath)
    picoscope_group = tdms_file['U-picoscope5245D']
    charge_pC_channel = picoscope_group['U-picoscope5245D charge pC']
    scan_charge_vals = np.asarray(charge_pC_channel[:], dtype=float)
    tdms_file = None
    return scan_charge_vals
"""
"""
def GetShotCharge(superpath, scannumber, shotnumber):
    tdms_filepath = DirectoryFunc.CompileTDMSFilepath(superpath, scannumber)
    charge_pC_vals = GetBeamCharge(tdms_filepath)
    return charge_pC_vals[shotnumber - 1]
"""
"""
def GetCameraTriggerAndExposure(tdms_filepath):
    tdms_file = TdmsFile.read(tdms_filepath)
    hiresmagcam_group = tdms_file['U_HiResMagCam']
    exposure_list = hiresmagcam_group['U_HiResMagCam Exposure']
    trigger_list = hiresmagcam_group['U_HiResMagCam TriggerDelay']
    tdms_file = None
    return trigger_list, exposure_list
"""

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

def FitTransverseGaussianSlices(image, calibrationFactor=1, threshold=0.01, binsize=1):
    ny, nx = np.shape(image)
    xloc, yloc, maxval = FindMax(image)

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
                                   guess=[max(slice_arr), 5 * calibrationFactor,
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
