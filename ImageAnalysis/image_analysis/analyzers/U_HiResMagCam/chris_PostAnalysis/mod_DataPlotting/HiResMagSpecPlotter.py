"""
Thursday 8-10-2023

Migration of my plotting scripts in the original MagSpecAnalysis module to be used with the streamlined LabView experience.

@Chris
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#import HiResAnalysisModules.HiResMagSpecAnalysis as MagSpec
#import HiResAnalysisModules.EnergyAxisLookup_HiRes as EnergyAxisLookup
import image_analysis.analyzers.U_HiResMagCam.OnlineAnalysisModules.HiResMagSpecAnalysis as MagSpec
import image_analysis.analyzers.U_HiResMagCam.OnlineAnalysisModules.EnergyAxisLookup_HiRes as EnergyAxisLookup


def PlotEnergyProjection(image, analyzeDict, inputParams, plotInfo=None,
                         doThreshold=False, doNormalize=False):

    normalization_factor = inputParams["Normalization-Factor"]
    if doThreshold:
        threshold = inputParams["Threshold-Value"]
        image = MagSpec.ThresholdReduction(image, threshold)
    if doNormalize:
        image = MagSpec.NormalizeImage(image, normalization_factor)

    image = np.copy(image[::-1, ::-1])
    image_width = np.shape(image)[1]
    pixel_arr = np.linspace(0, image_width, image_width)
    energy_arr = EnergyAxisLookup.return_default_energy_axis(pixel_arr)

    transverse_calibration = inputParams["Transverse-Calibration"]
    charge_arr = MagSpec.CalculateChargeDensityDistribution(image, transverse_calibration)
    peak_charge = analyzeDict['Peak-Charge']
    average_energy = analyzeDict['Average-Energy']
    peak_energy = analyzeDict['Peak-Charge-Energy']
    sigma_energy = analyzeDict['Energy-Spread']
    charge = analyzeDict['Charge-On-Camera']

    plt.plot(energy_arr, charge_arr, c='k', ls='solid',
             label="Peak Charge:" + "{:10.2f}".format(peak_charge) + " pC/MeV")
    plt.plot([average_energy, average_energy], [0, max(charge_arr) * 1.1], c='r', ls='dashed',
             label="Average Energy:" + "{:8.2f}".format(average_energy) + " MeV")
    plt.plot([peak_energy, peak_energy], [0, max(charge_arr) * 1.1], c='g', ls='dotted',
             label="Peak Energy:" + "{:13.2f}".format(peak_energy) + " MeV")
    plt.plot([average_energy - sigma_energy, average_energy + sigma_energy], [0.5 * peak_charge, 0.5 * peak_charge],
             c='r', ls='dotted', label="STD Energy Spread:" + "{:6.2f}".format(sigma_energy) + " MeV")
    plt.xlabel("Energy " + r'$(\mathrm{\ MeV})$')
    plt.ylabel("Charge Density " + r'$(\mathrm{\ pC/MeV})$')
    if plotInfo is not None:
        plt.title(plotInfo)
    plt.legend(title="Charge on Camera:" + "{:10.2f}".format(charge) + " pC")
    plt.show()


def PlotBeamDistribution(image, analyzeDict, inputParams, plotInfo=None,
                         doThreshold=False, doNormalize=False, style=0):
    # 1 for saturation check, 2 for a tropical vacation, anything else for normal
    normalization_factor = inputParams["Normalization-Factor"]
    if doThreshold:
        threshold = inputParams["Threshold-Value"]
        image = MagSpec.ThresholdReduction(image, threshold)
    if doNormalize:
        image = MagSpec.NormalizeImage(image, normalization_factor)

    image = np.copy(image[::-1, ::-1])
    image_width = np.shape(image)[1]
    pixel_arr = np.linspace(0, image_width, image_width)
    energy_arr = EnergyAxisLookup.return_default_energy_axis(pixel_arr)

    transverse_calibration = inputParams["Transverse-Calibration"]
    projected_axis, projected_arr, projected_size = MagSpec.CalculateProjectedBeamSize(image,
                                                                               calibrationFactor=transverse_calibration)

    beamAngle = analyzeDict["Beam-Tilt"]
    beamIntercept = analyzeDict["Beam-Intercept"]
    anglefunc = energy_arr * beamAngle + beamIntercept

    vertSize = (np.shape(image)[0]) * transverse_calibration
    aspect_ratio = (energy_arr[-1] - energy_arr[0]) / vertSize
    projected_factor = 0.3 * (energy_arr[-1] - energy_arr[0]) / max(projected_arr)

    clippedFactor = analyzeDict['Clipped-Percentage']

    fig, ax = plt.subplots(1, 1)

    maxx, maxy, maxval = MagSpec.FindMax(image)

    if style == 1:
        colors_normal = plt.cm.Greens_r(np.linspace(0, 0.80, 256))
        colors_saturated = plt.cm.jet(np.linspace(0.99, 1, 5))
        all_colors = np.vstack((colors_normal, colors_saturated))
        colormap = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)
        divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=maxval * .99999, vmax=maxval)
        print("Saturation Counts:", analyzeDict["Saturation-Counts"])
        print("Saturation Percentage:", analyzeDict["Saturation-Counts"]/(np.shape(image)[0]*np.shape(image)[1])*100,"%")

    elif style == 2 and maxval != 0:
        colors_water = plt.cm.terrain(np.linspace(0, 0.17, 256))
        colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
        all_colors = np.vstack((colors_water, colors_land))
        colors_saturated = plt.cm.rainbow(np.linspace(0.99, 1, 1))
        all_colors = np.vstack((all_colors, colors_saturated))
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

    # This is where I would have the non-linear ticks, but they don't easily work with the beam angle
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


def PlotSliceStatistics(image, analyzeDict, inputParams, plotInfo=None,
                        doThreshold=False, doNormalize=False):
    normalization_factor = inputParams["Normalization-Factor"]
    if doThreshold:
        threshold = inputParams["Threshold-Value"]
        image = MagSpec.ThresholdReduction(image, threshold)
    if doNormalize:
        image = MagSpec.NormalizeImage(image, normalization_factor)

    image = np.copy(image[::-1, ::-1])
    image_width = np.shape(image)[1]
    pixel_arr = np.linspace(0, image_width, image_width)
    energy_arr = EnergyAxisLookup.return_default_energy_axis(pixel_arr)

    transverse_calibration = inputParams["Transverse-Calibration"]
    binsize = inputParams["Transverse-Slice-Binsize"]
    sliceThreshold = inputParams["Transverse-Slice-Threshold"]

    sigma_arr, x0_arr, amp_arr, err_arr = MagSpec.TransverseSliceLoop(image, calibrationFactor=transverse_calibration,
                                                                      threshold=sliceThreshold, binsize=binsize, option=1)

    average_size = analyzeDict['Average-Beam-Size']
    camera_charge = analyzeDict['Charge-On-Camera']

    #plt.errorbar(energy_arr, sigma_arr, yerr=err_arr * 10, c='b', ls='dotted', label="Relative Error in Fit")
    plt.plot(energy_arr, sigma_arr, c='r', ls='solid', label="Tranverse Size of Slice")
    plt.plot([energy_arr[0], energy_arr[-1]], [average_size, average_size], c='g', ls='dashed',
             label="Average Size:" + "{:10.2f}".format(average_size) + r'$\mathrm{\ \mu m}$')
    plt.plot(energy_arr, amp_arr * 0.5*np.max(sigma_arr)/np.max(amp_arr), c='k', ls='dashed', label="Relative Weight")

    plt.xlabel("Energy " + r'$(\mathrm{\ MeV})$')
    plt.ylabel("Transverse Beam Size " + r'$(\mathrm{\mu m})$')
    if plotInfo is not None:
        plt.title(plotInfo)
    plt.legend(title="Charge on Camera:" + "{:10.2f}".format(camera_charge) + " pC")
    plt.show()
