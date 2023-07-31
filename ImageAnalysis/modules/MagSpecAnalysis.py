# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:45:38 2023

Module for performing various analysis on MagSpec Images

@author: chris
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from nptdms import TdmsFile

# sys.path.insert(0, "../")
import modules.CedossMathTools as MathTools
import modules.pngTools as pngTools
import modules.DirectoryModules as DirectoryFunc
import modules.EnergyAxisEstimate as EnergyAxis

# Important: Using constants like this is generally bad,
# should consider using a specific normalization file with
# this information instead

const_HiResMagSpec_Resolution = 43  # ums/pixel

# When doing the slice algorithm, only count slices where the peak is a particular percentage relative to image maximum
const_default_sliceThreshold = 0.02

# Factor to go from camera counts to pC/MeV
# Record: June 29th, Scan 23, HiResMagSpec
const_normalization_factor = 8.419118646207455e-07#3.308440355409613e-06
const_normalization_triggerdelay = 15.497208
const_normalization_exposure = 0.010000

# Record: July 25th, Scan 25, HiResMagSpec
const_normalization_triggerdelay =  15.497208
const_normalization_exposure = 0.010000
const_normalization_factor = 1.0510602507596621e-06


# Record: July 25th, Scan 24, HiResMagSpec
const_normalization_triggerdelay =  15.497208
const_normalization_exposure = 0.010000
const_normalization_factor = 1.1301095153900242e-06


def AnalyzeImage(image, tdms_filepath, interpSpec_filepath, shotnumber, hardlimit=None, sliceThreshold=0.02,
                 calibrationFactor=const_HiResMagSpec_Resolution):
    saturationCheck = SaturationCheck(image)

    image = ThresholdReduction(image, hardlimit)
    image = NormalizeImage(image, shotnumber, tdms_filepath)

    interpSpec_energy_arr, spec_charge_arr = ParseInterpSpec(interpSpec_filepath)
    image = np.copy(image[::-1, ::-1])
    image_width = np.shape(image)[1]
    pixel_arr = np.linspace(0, image_width, image_width)
    energy_arr = EnergyAxis.GetEstimatedEnergyAxis(pixel_arr, interpSpec_energy_arr)

    chargeOnCamera = np.sum(image)

    charge_pC_vals = GetBeamCharge(tdms_filepath)
    picoscopeCharge = charge_pC_vals[shotnumber - 1]

    clippedPercentage = CalculateClippedPercentage(image)

    charge_arr = CalculateChargeDensityDistribution(image)
    if np.sum(charge_arr) == 0:
        print("No beam this shot")
        peakCharge = 0
        averageEnergy = 0
        energySpread = 0
        peakChargeEnergy = 0
        averageBeamSize = 0
        beamAngle = 0
        beamIntercept = 0
        projectedBeamSize = 0
    else:
        peakCharge = CalculateMaximumCharge(charge_arr)
        averageEnergy = CalculateAverageEnergy(charge_arr, energy_arr)
        energySpread = CalculateStandardDeviationEnergy(charge_arr, energy_arr)
        peakChargeEnergy = CalculatePeakEnergy(charge_arr, energy_arr)

        sigma_arr, x0_arr, amp_arr, err_arr = FitTransverseGaussianSlices(image, calibrationFactor=calibrationFactor,
                                                                          threshold=sliceThreshold)
        averageBeamSize = CalculateAverageSize(sigma_arr, amp_arr)  # * const_HiResMagSpec_Resolution
        linear_fit = FitBeamAngle(x0_arr, amp_arr, energy_arr)
        beamAngle = linear_fit[0]  # * const_HiResMagSpec_Resolution
        beamIntercept = linear_fit[1]  # * const_HiResMagSpec_Resolution
        projected_axis, projected_arr, projectedBeamSize = CalculateProjectedBeamSize(image)
        projectedBeamSize = projectedBeamSize * const_HiResMagSpec_Resolution

    magSpecDict = {
        "Shot-Number": shotnumber,
        "Clipped-Percentage": clippedPercentage,
        "Saturation-Counts": saturationCheck,
        "Charge-On-Camera": chargeOnCamera,
        "Picoscope-Charge": picoscopeCharge,
        "Peak-Charge": peakCharge,
        "Peak-Charge-Energy": peakChargeEnergy,
        "Average-Energy": averageEnergy,
        "Energy-Spread": energySpread,
        "Average-Beam-Size": averageBeamSize,
        "Projected-Beam-Size": projectedBeamSize,
        "Beam-Tilt": beamAngle,
        "Beam-Intercept": beamIntercept
    }
    return magSpecDict


def AnalyzeImage_Interp(image, tdms_filepath, interpSpec_filepath, shotnumber, hardlimit=None, sliceThreshold=0.02,
                        calibrationFactor=const_HiResMagSpec_Resolution):
    image = ThresholdReduction(image, hardlimit)
    image = NormalizeImage(image, shotnumber, tdms_filepath)
    energy_arr, spec_charge_arr = ParseInterpSpec(interpSpec_filepath)
    chargeOnCamera = np.sum(image)

    charge_pC_vals = GetBeamCharge(tdms_filepath)
    picoscopeCharge = charge_pC_vals[shotnumber - 1]

    clippedPercentage = CalculateClippedPercentage(image)
    saturationCheck = SaturationCheck(image)

    charge_arr = CalculateChargeDensityDistribution(image)

    peakCharge = CalculateMaximumCharge(charge_arr)
    averageEnergy = CalculateAverageEnergy(charge_arr, energy_arr)
    energySpread = CalculateStandardDeviationEnergy(charge_arr, energy_arr)
    peakChargeEnergy = CalculatePeakEnergy(charge_arr, energy_arr)

    sigma_arr, x0_arr, amp_arr, err_arr = FitTransverseGaussianSlices(image, calibrationFactor=calibrationFactor,
                                                                      threshold=sliceThreshold)
    averageBeamSize = CalculateAverageSize(sigma_arr, amp_arr)  # * const_HiResMagSpec_Resolution
    linear_fit = FitBeamAngle(x0_arr, amp_arr, energy_arr)
    beamAngle = linear_fit[0]  # * const_HiResMagSpec_Resolution
    beamIntercept = linear_fit[1]  # * const_HiResMagSpec_Resolution
    projected_axis, projected_arr, projectedBeamSize = CalculateProjectedBeamSize(image)
    projectedBeamSize = projectedBeamSize * const_HiResMagSpec_Resolution

    magSpecDict = {
        "Clipped-Percentage": clippedPercentage,
        "Saturation-Counts": saturationCheck,
        "Charge-On-Camera": chargeOnCamera,
        "Picoscope-Charge": picoscopeCharge,
        "Peak-Charge": peakCharge,
        "Peak-Charge-Energy": peakChargeEnergy,
        "Average-Energy": averageEnergy,
        "Energy-Spread": energySpread,
        "Average-Beam-Size": averageBeamSize,
        "Projected-Beam-Size": projectedBeamSize,
        "Beam-Tilt": beamAngle,
        "Beam-Intercept": beamIntercept
    }
    return magSpecDict


def NormalizeImage(image, shotnumber, tdms_filepath):
    """
    Normalizes the HiResMagSpec image according to the saved normalization value.
    This value represents an image which is neither clipped or saturated

    Parameters
    ----------
    image : 2D Numpy Float Array
        The Image we are normalizing
    shotnumber : Int
        The shot number.
    tdms_filepath : String
        Filepath to the tdms file.

    Returns
    -------
    returnimage : 2D Numpy Float Array
        The normalized image.

    """
    # First, open up the tdms file and check that the camera delay
    # and shutter duration are close to expected
    trigger_list, exposure_list = GetCameraTriggerAndExposure(tdms_filepath)
    if float(trigger_list[shotnumber - 1]) != float(const_normalization_triggerdelay):
        print("WARNING: the trigger delay is not the same as the current normalization!")
        print(" This Image: ", trigger_list[shotnumber - 1])
        print(" Reference:  ", const_normalization_triggerdelay)
    if float(exposure_list[shotnumber - 1]) != float(const_normalization_exposure):
        print("WARNING: the exposure is not the same as the current normalization!")
        print(" This Image: ", exposure_list[shotnumber - 1])
        print(" Reference:  ", const_normalization_exposure)

    # Then, do the old quick normalization
    returnimage = np.copy(image) * const_normalization_factor
    return returnimage


def PrintNormalization(image, shotnumber, tdms_filepath):
    """
    For a given image, print out the normalization factor.
    If this is a good image, then copy the normalization factors to the top of this module

    Parameters
    ----------
    image : 2D Numpy Float Array
        The Image we are normalizing
    shotnumber : Int
        The shot number.
    tdms_filepath : String
        Filepath to the tdms file.

    Returns
    -------
    None.

    """
    charge_pC_vals = GetBeamCharge(tdms_filepath)
    charge = charge_pC_vals[shotnumber - 1]

    trigger_list, exposure_list = GetCameraTriggerAndExposure(tdms_filepath)

    # Assuming the image is good, find the factor, camera delay, and shutter duration and
    # print out the information for copy-pasting into this module.

    print("The following are the normalization factors,")
    print(" paste them into MagSpecAnalysis.py:")
    #print("const_normalization_factor =", charge / sum(sum(image)))
    print("const_normalization_triggerdelay = ", trigger_list[shotnumber - 1])
    print("const_normalization_exposure =", exposure_list[shotnumber - 1])
    return


def PlotEnergyProjection_Interp(image, tdms_filepath, interpSpec_filepath, shotnumber, plotInfo=None, analyzeDict=None,
                                doThreshold=False, doNormalize=False, hardlimit=None,
                                sliceThreshold=const_default_sliceThreshold):
    if analyzeDict is None:
        analyzeDict = AnalyzeImage_Interp(image, tdms_filepath, interpSpec_filepath, shotnumber, hardlimit=None,
                                          sliceThreshold=sliceThreshold)
    if doThreshold:
        image = ThresholdReduction(image, hardlimit)
    if doNormalize:
        image = NormalizeImage(image, shotnumber, tdms_filepath)

    energy_arr, spec_charge_arr = ParseInterpSpec(interpSpec_filepath)
    charge_arr = CalculateChargeDensityDistribution(image)
    peak_charge = analyzeDict['Peak-Charge']
    average_energy = analyzeDict['Average-Energy']
    peak_energy = analyzeDict['Peak-Charge-Energy']
    sigma_energy = analyzeDict['Energy-Spread']
    charge = analyzeDict['Picoscope-Charge']

    plt.plot(energy_arr, charge_arr, c='k', ls='solid',
             label="Peak Charge:" + "{:10.2f}".format(peak_charge) + " pC/MeV")
    plt.plot([average_energy, average_energy], [0, max(charge_arr) * 1.1], c='r', ls='dashed',
             label="Average Energy:" + "{:8.2f}".format(average_energy) + " MeV")
    plt.plot([peak_energy, peak_energy], [0, max(charge_arr) * 1.1], c='g', ls='dotted',
             label="Peak Energy:" + "{:13.2f}".format(peak_energy) + " MeV")
    plt.plot([average_energy - sigma_energy, average_energy + sigma_energy], [0.5 * peak_charge, 0.5 * peak_charge],
             c='r',
             ls='dotted', label="STD Energy Spread:" + "{:6.2f}".format(sigma_energy) + " MeV")
    plt.xlabel("Energy " + r'$(\mathrm{\ MeV})$')
    plt.ylabel("Charge Density " + r'$(\mathrm{\ pC/MeV})$')
    if plotInfo is not None:
        plt.title(plotInfo)
    plt.legend(title="Picoscope Charge:" + "{:10.2f}".format(charge) + " pC")
    plt.show()


def PlotBeamDistribution_Interp(image, tdms_filepath, interpSpec_filepath, shotnumber, plotInfo=None, analyzeDict=None,
                                doThreshold=False, doNormalize=False, hardlimit=None,
                                sliceThreshold=const_default_sliceThreshold,
                                calibrationFactor=const_HiResMagSpec_Resolution):
    if analyzeDict is None:
        analyzeDict = AnalyzeImage_Interp(image, tdms_filepath, interpSpec_filepath, shotnumber, hardlimit=None,
                                          sliceThreshold=sliceThreshold)
    if doThreshold:
        image = ThresholdReduction(image, hardlimit)
    if doNormalize:
        image = NormalizeImage(image, shotnumber, tdms_filepath)

    energy_arr, spec_charge_arr = ParseInterpSpec(interpSpec_filepath)
    sigma_arr, x0_arr, amp_arr, err_arr = FitTransverseGaussianSlices(image, calibrationFactor=calibrationFactor,
                                                                      threshold=sliceThreshold)

    # linear_fit = FitBeamAngle(x0_arr, amp_arr, energy_arr)
    projected_axis, projected_arr, projected_size = CalculateProjectedBeamSize(image,
                                                                               calibrationFactor=calibrationFactor)

    beamAngle = analyzeDict["Beam-Tilt"]
    beamIntercept = analyzeDict["Beam-Intercept"]
    anglefunc = energy_arr * beamAngle + beamIntercept

    vertSize = (np.shape(image)[0]) * calibrationFactor
    aspect_ratio = (energy_arr[-1] - energy_arr[0]) / vertSize
    projected_factor = 0.3 * (energy_arr[-1] - energy_arr[0]) / max(projected_arr)

    clippedFactor = analyzeDict['Clipped-Percentage']

    plt.imshow(image, aspect=aspect_ratio, extent=(energy_arr[0], energy_arr[-1], 0, vertSize))
    plt.plot(energy_arr, anglefunc, c='white', ls='dashed',
             label="Slope:" + "{:10.2f}".format(beamAngle) + " " + r'$\mathrm{\mu}$' + "m/MeV")
    plt.plot(projected_arr * projected_factor + min(energy_arr), projected_axis, c='orange', ls='dotted',
             label="Projected RMS Size:" + "{:8.2f}".format(projected_size) + " " + r'$\mathrm{\mu}$' + "m")
    plt.xlabel("Energy " + r'$(\mathrm{\ MeV})$')
    plt.ylabel("Transverse Position " + r'$(\mathrm{\mu m})$')
    if plotInfo is not None:
        plt.title(plotInfo)
    plt.ylim([0, vertSize])
    plt.xlim([min(energy_arr), max(energy_arr)])
    plt.legend(title="Clipped Percentage: " + "{:.1f}".format(clippedFactor * 100) + " %")

    plt.show()


def PlotSliceStatistics_Interp(image, tdms_filepath, interpSpec_filepath, shotnumber, plotInfo=None, analyzeDict=None,
                               doThreshold=False, doNormalize=False, hardlimit=None,
                               sliceThreshold=const_default_sliceThreshold,
                               calibrationFactor=const_HiResMagSpec_Resolution):
    if analyzeDict is None:
        analyzeDict = AnalyzeImage_Interp(image, tdms_filepath, interpSpec_filepath, shotnumber, hardlimit=None,
                                          sliceThreshold=sliceThreshold, calibrationFactor=calibrationFactor)
    if doThreshold:
        image = ThresholdReduction(image, hardlimit)
    if doNormalize:
        image = NormalizeImage(image, shotnumber, tdms_filepath)

    energy_arr, spec_charge_arr = ParseInterpSpec(interpSpec_filepath)
    sigma_arr, x0_arr, amp_arr, err_arr = FitTransverseGaussianSlices(image, calibrationFactor=calibrationFactor,
                                                                      threshold=sliceThreshold)
    average_size = analyzeDict['Average-Beam-Size']
    camera_charge = analyzeDict['Charge-On-Camera']

    plt.errorbar(energy_arr, sigma_arr, yerr=err_arr * 10, c='b', ls='dotted', label="Relative Error in Fit")
    plt.plot(energy_arr, sigma_arr, c='r', ls='solid', label="Tranverse Size of Slice")
    plt.plot([energy_arr[0], energy_arr[-1]], [average_size, average_size], c='g', ls='dashed',
             label="Average Size:" + "{:10.2f}".format(average_size) + r'$\mathrm{\ \mu m}$')
    plt.xlabel("Energy " + r'$(\mathrm{\ MeV})$')
    plt.ylabel("Transverse Beam Size " + r'$(\mathrm{\mu m})$')
    if plotInfo is not None:
        plt.title(plotInfo)
    plt.legend(title="Charge on Camera:" + "{:10.2f}".format(camera_charge) + " pC")
    plt.show()


def PlotEnergyProjection(image, tdms_filepath, interpSpec_filepath, shotnumber, plotInfo=None, analyzeDict=None,
                         doThreshold=False, doNormalize=False, hardlimit=None,
                         sliceThreshold=const_default_sliceThreshold):
    if analyzeDict is None:
        analyzeDict = AnalyzeImage(image, tdms_filepath, interpSpec_filepath, shotnumber, hardlimit=None,
                                   sliceThreshold=sliceThreshold)
    if doThreshold:
        image = ThresholdReduction(image, hardlimit)
    if doNormalize:
        image = NormalizeImage(image, shotnumber, tdms_filepath)

    interpSpec_energy_arr, spec_charge_arr = ParseInterpSpec(interpSpec_filepath)
    image = np.copy(image[::-1, ::-1])
    image_width = np.shape(image)[1]
    pixel_arr = np.linspace(0, image_width, image_width)
    energy_arr = EnergyAxis.GetEstimatedEnergyAxis(pixel_arr, interpSpec_energy_arr)

    charge_arr = CalculateChargeDensityDistribution(image)
    peak_charge = analyzeDict['Peak-Charge']
    average_energy = analyzeDict['Average-Energy']
    peak_energy = analyzeDict['Peak-Charge-Energy']
    sigma_energy = analyzeDict['Energy-Spread']
    charge = analyzeDict['Picoscope-Charge']

    plt.plot(energy_arr, charge_arr, c='k', ls='solid',
             label="Peak Charge:" + "{:10.2f}".format(peak_charge) + " pC/MeV")
    plt.plot([average_energy, average_energy], [0, max(charge_arr) * 1.1], c='r', ls='dashed',
             label="Average Energy:" + "{:8.2f}".format(average_energy) + " MeV")
    plt.plot([peak_energy, peak_energy], [0, max(charge_arr) * 1.1], c='g', ls='dotted',
             label="Peak Energy:" + "{:13.2f}".format(peak_energy) + " MeV")
    plt.plot([average_energy - sigma_energy, average_energy + sigma_energy], [0.5 * peak_charge, 0.5 * peak_charge],
             c='r',
             ls='dotted', label="STD Energy Spread:" + "{:6.2f}".format(sigma_energy) + " MeV")
    plt.xlabel("Energy " + r'$(\mathrm{\ MeV})$')
    plt.ylabel("Charge Density " + r'$(\mathrm{\ pC/MeV})$')
    if plotInfo is not None:
        plt.title(plotInfo)
    plt.legend(title="Picoscope Charge:" + "{:10.2f}".format(charge) + " pC")
    plt.show()


def PlotBeamDistribution(image, tdms_filepath, interpSpec_filepath, shotnumber, plotInfo=None, analyzeDict=None,
                         doThreshold=False, doNormalize=False, hardlimit=None,
                         sliceThreshold=const_default_sliceThreshold, calibrationFactor=const_HiResMagSpec_Resolution):
    if analyzeDict is None:
        analyzeDict = AnalyzeImage(image, tdms_filepath, interpSpec_filepath, shotnumber, hardlimit=None,
                                   sliceThreshold=sliceThreshold)
    if doThreshold:
        image = ThresholdReduction(image, hardlimit)
    if doNormalize:
        image = NormalizeImage(image, shotnumber, tdms_filepath)

    interpSpec_energy_arr, spec_charge_arr = ParseInterpSpec(interpSpec_filepath)
    image = np.copy(image[::-1, ::-1])
    image_width = np.shape(image)[1]
    pixel_arr = np.linspace(0, image_width, image_width)
    energy_arr = EnergyAxis.GetEstimatedEnergyAxis(pixel_arr, interpSpec_energy_arr)
    # sigma_arr, x0_arr, amp_arr, err_arr = FitTransverseGaussianSlices(image, calibrationFactor = calibrationFactor, threshold = sliceThreshold)

    # linear_fit = FitBeamAngle(x0_arr, amp_arr, energy_arr)
    projected_axis, projected_arr, projected_size = CalculateProjectedBeamSize(image,
                                                                               calibrationFactor=calibrationFactor)

    beamAngle = analyzeDict["Beam-Tilt"]
    beamIntercept = analyzeDict["Beam-Intercept"]
    anglefunc = energy_arr * beamAngle + beamIntercept

    vertSize = (np.shape(image)[0]) * calibrationFactor
    aspect_ratio = (energy_arr[-1] - energy_arr[0]) / vertSize
    projected_factor = 0.3 * (energy_arr[-1] - energy_arr[0]) / max(projected_arr)

    clippedFactor = analyzeDict['Clipped-Percentage']

    fig, ax = plt.subplots(1, 1)

    maxx, maxy, maxval = FindMax(image)

    color_choice = 2  # 1 for saturation check, 2 for a tropical vacation, anything else for normal

    if color_choice == 1:
        colors_normal = plt.cm.Greens_r(np.linspace(0, 0.80, 256))
        colors_saturated = plt.cm.jet(np.linspace(0.99, 1, 5))
        all_colors = np.vstack((colors_normal, colors_saturated))
        colormap = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)
        divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=maxval * .99999, vmax=maxval)
        print("Saturation Counts:", analyzeDict["Saturation-Counts"])
        print("Saturation Percentage:", analyzeDict["Saturation-Counts"]/(np.shape(image)[0]*np.shape(image)[1])*100,"%")

    elif color_choice == 2 and maxval != 0:
        colors_water = plt.cm.terrain(np.linspace(0, 0.17, 256))
        colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
        all_colors = np.vstack((colors_water, colors_land))
        colormap = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)
        divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=.4 * maxval, vmax=maxval)

    else:
        colormap = None
        divnorm = None

    img = ax.imshow(image, aspect=aspect_ratio, extent=(energy_arr[0], energy_arr[-1], 0, vertSize), norm=divnorm,
                    cmap=colormap)
    cbar = fig.colorbar(img, label='Charge Density ' + r'$\mathrm{(pC/(\mu m\times MeV))}$')
    plt.plot(energy_arr, anglefunc, c='white', ls='dashed',
             label="Slope:" + "{:10.2f}".format(beamAngle) + " " + r'$\mathrm{\mu}$' + "m/MeV")
    plt.plot(projected_arr * projected_factor + min(energy_arr), projected_axis, c='orange', ls='dotted',
             label="Projected RMS Size:" + "{:8.2f}".format(projected_size) + " " + r'$\mathrm{\mu}$' + "m")
    ax.set_xlabel("'Approximate' Energy " + r'$(\mathrm{\ MeV})$')
    ax.set_ylabel("Transverse Position " + r'$(\mathrm{\mu m})$')

    # This is where I would have the non-linear ticks, but they don't work with the beam angle
    """
    energy_ticks = np.arange(80,130.1,5)
    energy_ticks = energy_ticks[np.where(energy_ticks < np.max(energy_arr))[0]]
    energy_ticks = energy_ticks[np.where(energy_ticks > np.min(energy_arr))[0]]

    print(energy_arr)

    pixel_ticks = np.zeros(len(energy_ticks))
    for i in range(len(energy_arr)):
        for j in range(len(pixel_ticks)):
            if energy_arr[i] < energy_ticks[j]:
                pixel_ticks[j] = energy_arr[i]
    print(pixel_ticks)

    label_ticks = np.zeros(len(energy_ticks))
    for i in range(len(energy_ticks)):
        label_ticks[i] = str(energy_ticks[i])

    print(label_ticks)

    ax.set_xticks(pixel_ticks)
    ax.set_xticklabels(label_ticks)
    """
    if plotInfo is not None:
        plt.title(plotInfo)
    plt.ylim([0, vertSize])
    plt.xlim([min(energy_arr), max(energy_arr)])
    plt.legend(title="Clipped Percentage: " + "{:.1f}".format(clippedFactor * 100) + " %")

    plt.show()


def PlotSliceStatistics(image, tdms_filepath, interpSpec_filepath, shotnumber, plotInfo=None, analyzeDict=None,
                        doThreshold=False, doNormalize=False, hardlimit=None,
                        sliceThreshold=const_default_sliceThreshold, calibrationFactor=const_HiResMagSpec_Resolution):
    if analyzeDict is None:
        analyzeDict = AnalyzeImage(image, tdms_filepath, interpSpec_filepath, shotnumber, hardlimit=None,
                                   sliceThreshold=sliceThreshold, calibrationFactor=calibrationFactor)
    if doThreshold:
        image = ThresholdReduction(image, hardlimit)
    if doNormalize:
        image = NormalizeImage(image, shotnumber, tdms_filepath)

    interpSpec_energy_arr, spec_charge_arr = ParseInterpSpec(interpSpec_filepath)
    image = np.copy(image[::-1, ::-1])
    image_width = np.shape(image)[1]
    pixel_arr = np.linspace(0, image_width, image_width)
    energy_arr = EnergyAxis.GetEstimatedEnergyAxis(pixel_arr, interpSpec_energy_arr)

    sigma_arr, x0_arr, amp_arr, err_arr = FitTransverseGaussianSlices(image, calibrationFactor=calibrationFactor,
                                                                      threshold=sliceThreshold)
    average_size = analyzeDict['Average-Beam-Size']
    camera_charge = analyzeDict['Charge-On-Camera']

    plt.errorbar(energy_arr, sigma_arr, yerr=err_arr * 10, c='b', ls='dotted', label="Relative Error in Fit")
    plt.plot(energy_arr, sigma_arr, c='r', ls='solid', label="Tranverse Size of Slice")
    plt.plot([energy_arr[0], energy_arr[-1]], [average_size, average_size], c='g', ls='dashed',
             label="Average Size:" + "{:10.2f}".format(average_size) + r'$\mathrm{\ \mu m}$')
    plt.xlabel("Energy " + r'$(\mathrm{\ MeV})$')
    plt.ylabel("Transverse Beam Size " + r'$(\mathrm{\mu m})$')
    if plotInfo is not None:
        plt.title(plotInfo)
    plt.legend(title="Charge on Camera:" + "{:10.2f}".format(camera_charge) + " pC")
    plt.show()


def ParseInterpSpec(filename):
    """
    Parses the data saved in the Spec folder to get the momentum array and
    the charge density array.  Although ideally we will use our own analysis
    
    Parameters
    ----------
    filename : String
        Full path to the txt file.

    Returns
    -------
    momentum_arr : Float Numpy Array
        Arrays of the momentum.
    charge_arr : Float Numpy Array
        Arrays of the charge density per momentum.
    
    """
    with open(filename) as f:
        lines = f.readlines()
        nSlices = len(lines) - 1
        momentum_arr = np.zeros(nSlices)
        charge_arr = np.zeros(nSlices)

        i = 0
        for line in lines:
            line = line.strip()
            columns = line.split('\t')
            if columns[0] != 'Momentum_GeV/c':
                momentum_arr[i] = float(columns[0])
                charge_arr[i] = float(columns[1])
                i = i + 1
        f.close()
    return momentum_arr, charge_arr


def LoadImage(superpath, scannumber, shotnumber, folderpath, tdms_filepath=None, doThreshold=True, hardlimit=None,
              doNormalize=True):
    fullpath = DirectoryFunc.CompileFileLocation(superpath, scannumber, shotnumber, folderpath, suffix=".png")
    image = pngTools.nBitPNG(fullpath)
    if doThreshold:
        image = ThresholdReduction(image, hardlimit)
    if doNormalize:
        if tdms_filepath is None:
            tdms_filepath = DirectoryFunc.CompileTDMSFilepath(superpath, scannumber)
        image = NormalizeImage(image, shotnumber, tdms_filepath)
    return image


def ThresholdReduction(image, hardlimit=None, skipPlots=True):
    """
    Attempts to do a quick reduction of the background by subtracting a fixed threshold off of the data
    Can either manually supply a "hardlimit" to do this, or if none is provided then we take Gaussian
    fits of the data off of the maximum to find what it asymptotes to

    Parameters
    ----------
    image : 2D Numpy Float Array
        Image array.
    hardlimit : Float, optional
        If a specific threshold is given rather than using Gaussian fits. The default is None.
    skipPlots : Boolean, optional
        If True, skips plotting Gaussian fits in MathTools. The default is True.

    Returns
    -------
    returnimage : 2D Numpy Float Array
        A separate instance of the initial image array that is threshold subtracted.

    """
    method = 1

    if hardlimit is None:
        if method == 0:
            xmax, ymax, maxval = FindMax(image)
            xslice = image[ymax, :]
            yslice = image[:, xmax]
            xfit = MathTools.FitDataSomething(xslice, np.linspace(0, len(xslice), len(xslice)),
                                              MathTools.GaussianOffset,
                                              guess=[max(xslice), 5, np.argmax(xslice), min(xslice)], supress=skipPlots)
            yfit = MathTools.FitDataSomething(yslice, np.linspace(0, len(yslice), len(yslice)),
                                              MathTools.GaussianOffset,
                                              guess=[max(yslice), 5, np.argmax(yslice), min(yslice)], supress=skipPlots)
            threshold = np.max([xfit[3], yfit[3]])
        else:
            width = 2
            topslab = SlabAverage(image, -1, -1-width, 'x')
            botslab = SlabAverage(image, 0, width, 'x')
            leftslab = SlabAverage(image, -1, -1-width, 'y')
            rightslab = SlabAverage(image, 0, width, 'y')
            threshold = np.min([topslab, botslab, leftslab, rightslab])*4.0
    else:
        threshold = hardlimit
    returnimage = np.copy(image)
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            returnimage[i][j] = image[i][j] - threshold
            if returnimage[i][j] < 0:
                returnimage[i][j] = 0
    return returnimage


def SlabAverage(image, ind0, indf, axis):
    slice_array = np.linspace(ind0, indf, np.abs(indf-ind0)+1)
    average = 0
    if axis == 'x':
        for i in slice_array:
            average = average + np.average(image[int(i),:])
        average = average / len(slice_array)
    elif axis == 'y':
        for i in slice_array:
            average = average + np.average(image[:,int(i)])
        average = average / len(slice_array)
    else:
        print("SlabAverage needs 'x' or 'y'!")
    return average


# NOT YET IMPLEMENTED
def BackgroundSubtraction(image, background):
    print("JUST A PLACEHOLDER")
    return image


def CalculateClippedPercentage(image):
    clipcheck = np.append(np.append(np.append(image[0, :], image[:, 0]), image[-1, :]), image[:, -1])
    xmax, ymax, maxval = FindMax(image)
    if maxval != 0:
        return np.max(clipcheck) / maxval
    else:
        return 1.0


def CalculateProjectedBeamSize(image, calibrationFactor=const_HiResMagSpec_Resolution):
    skipPlots = True
    proj_arr = np.zeros(np.shape(image)[0])
    for i in range(len(proj_arr)):
        image_slice = image[i, :]
        proj_arr[i] = np.sum(image_slice)

    axis_arr = np.linspace(0, len(proj_arr) * calibrationFactor, len(proj_arr))
    axis_arr = np.flip(axis_arr)
    fit = MathTools.FitDataSomething(proj_arr, axis_arr,
                                     MathTools.Gaussian,
                                     guess=[max(proj_arr), 20 * calibrationFactor, axis_arr[np.argmax(proj_arr)]],
                                     supress=skipPlots)
    beamSize = fit[1]
    return axis_arr, proj_arr, beamSize


def CalculateChargeDensityDistribution(image):
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
    charge_arr = np.zeros(np.shape(image)[1])
    for i in range(len(charge_arr)):
        image_slice = image[:, i]
        charge_arr[i] = np.sum(image_slice)
    return charge_arr


def CalculateMaximumCharge(charge_arr):
    return np.max(charge_arr)


def CalculateAverageEnergy(charge_arr, energy_arr):
    if np.sum(charge_arr) != 0:
        return np.average(energy_arr, weights=charge_arr)
    else:
        return 0


def CalculateStandardDeviationEnergy(charge_arr, energy_arr):
    average_energy = CalculateAverageEnergy(charge_arr, energy_arr)
    if np.sum(charge_arr) != 0:
        return np.sqrt(np.average((energy_arr - average_energy) ** 2, weights=charge_arr))
    else:
        return 0


def CalculatePeakEnergy(charge_arr, energy_arr):
    return energy_arr[np.argmax(charge_arr)]


def FitTransverseGaussianSlices(image, calibrationFactor=1, threshold=0.01):
    ny, nx = np.shape(image)
    xloc, yloc, maxval = FindMax(image)
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
            fit = MathTools.FitDataSomething(slice_arr, axis_arr,
                                             MathTools.Gaussian,
                                             guess=[max(slice_arr), 5 * const_HiResMagSpec_Resolution,
                                                    axis_arr[np.argmax(slice_arr)]], supress=skipPlots)
            amp_arr[i], sigma_arr[i], x0_arr[i] = fit

            # Check to see that our x0 is within bounds (only an issue for vertically-clipped beams
            if x0_arr[i] < axis_arr[-1] or x0_arr[i] > axis_arr[0]:
                sigma_arr[i] = 0;
                x0_arr[i] = 0;
                amp_arr[i] = 0;
                err_arr[i] = 0
            else:
                # TODO:  This error calculation below is "correct," but produces no meaningful error.  Please find a way
                func = MathTools.Gaussian(fit, axis_arr)
                error = 0
                for j in range(len(slice_arr)):
                    error = error + np.square(slice_arr[j] - func[j])
                err_arr[i] = np.sqrt(error) * 1e3

        else:
            sigma_arr[i] = 0;
            x0_arr[i] = 0;
            amp_arr[i] = 0;
            err_arr[i] = 0

    return sigma_arr, x0_arr, amp_arr, err_arr


def CalculateAverageSize(sigma_arr, amp_arr):
    if np.sum(amp_arr) != 0:
        return np.average(sigma_arr, weights=amp_arr)
    else:
        return 0


def FitBeamAngle(x0_arr, amp_arr, energy_arr):
    linear_fit = np.polyfit(energy_arr, x0_arr, deg=1, w=np.power(amp_arr, 2))
    secretPlot = False
    if secretPlot:
        # Secret Plot of the Angle Fit
        plt.plot(energy_arr, x0_arr)
        energy_axis = np.linspace(energy_arr[0], energy_arr[-1], 50)
        plt.plot(energy_axis, energy_axis * linear_fit[0] + linear_fit[1])
        plt.plot(energy_arr, amp_arr * 0.3 * max(x0_arr) / max(amp_arr))
        plt.show()

    return linear_fit


def GetSizeStatistics_Full(image, energy_arr, sliceThreshold=const_default_sliceThreshold):
    sigma_arr, x0_arr, amp_arr, err_arr = FitTransverseGaussianSlices(image,
                                                                      calibrationFactor=const_HiResMagSpec_Resolution,
                                                                      threshold=sliceThreshold)
    average_size = CalculateAverageSize(sigma_arr, amp_arr)
    axis, vals, projected_size = CalculateProjectedBeamSize(image)
    linear_fit = FitBeamAngle(x0_arr, amp_arr, energy_arr)
    return average_size, projected_size, linear_fit[0]


def GetBeamCharge(tdms_filepath):
    """
    From the TDMS file, grabs the measured beam charge for each shot

    Parameters
    ----------
    tdms_filepath : String
        Filepath to the TDMS file.

    Returns
    -------
    scan_charge_vals : 1D Numpy Float Array
        Array of beam charge measurements.
        
    """
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


def GetShotCharge(superpath, scannumber, shotnumber):
    tdms_filepath = DirectoryFunc.CompileTDMSFilepath(superpath, scannumber)
    charge_pC_vals = GetBeamCharge(tdms_filepath)
    return charge_pC_vals[shotnumber - 1]


def GetCameraTriggerAndExposure(tdms_filepath):
    """
    From the TDMS file, grabs the list of camera exposures and trigger delays

    Parameters
    ----------
    tdms_filepath : String
        Filepath to the TDMS file.

    Returns
    -------
    trigger_list : 1D Numpy Float Array
        Array of recorded trigger delays
    exposure_list : 1D Numpy Float Array
        Array of recorded exposure times

    """
    tdms_file = TdmsFile.read(tdms_filepath)
    hiresmagcam_group = tdms_file['U_HiResMagCam']
    exposure_list = hiresmagcam_group['U_HiResMagCam Exposure']
    trigger_list = hiresmagcam_group['U_HiResMagCam TriggerDelay']
    tdms_file = None
    return trigger_list, exposure_list


def FindMax(image):
    """
    Grabs the coordinates and value of the maximum of the image

    Parameters
    ----------
    image : 2D Numpy Float Array
        The image array.

    Returns
    -------
    xmax : Int
        Horizontal position of maximum.
    ymax : Int
        Vertical position of maximum.
    maxcal : Float
        The value of the pixel at the maximum.

    """
    ny, nx = np.shape(image)
    max_ind = np.argmax(image)
    xmax = int(max_ind % nx)
    ymax = int(np.floor(max_ind / nx))
    maxval = image[ymax, xmax]
    return xmax, ymax, maxval


def SaturationCheck(image):
    nx, ny = np.shape(image)
    saturation_counts = 0
    maximum_value = 0
    for i in range(nx):
        for j in range(ny):
            if image[i][j] > maximum_value:
                maximum_value = image[i][j]
                saturation_counts = 1
            if image[i][j] == maximum_value:
                saturation_counts = saturation_counts + 1
    return saturation_counts - 1
