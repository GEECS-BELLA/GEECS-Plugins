"""
Wed 7/26/2023

Loop over a full scan directory of nice HiResMagSpec images.  Mostly interested in the picoscope charge, sum of camera
counts, clipped percentage, and saturation counts.  For good shots, plot the relation between picoscope charge and
camera counts.  Find a fit, and keep that as the camera calibration constant.  Make not of trigger delay

Could just use the "AnalyzeImage" function to get everything, but honestly I would rather just quickly calculate
the info I need.  Mostly because I don't know how to store variables in consoles for pycharm yet.  :(

@Chris
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

rootpath = os.path.abspath("../../../../../")
sys.path.insert(0, rootpath)

import ImageAnalysis.image_analysis.analyzers.calibration_scripts.modules_image_processing.pngTools as pngTools
import ImageAnalysis.image_analysis.analyzers.calibration_scripts.modules_image_processing.shot_charge_reader as charge_tdms
import ImageAnalysis.image_analysis.analyzers.UC_HiResMagCam as mag_spec_caller
import ImageAnalysis.image_analysis.analyzers.online_analysis_modules.directory_functions as directory_functions
import ImageAnalysis.image_analysis.analyzers.online_analysis_modules.mag_spec_analysis as mag_spec_analysis
import ImageAnalysis.image_analysis.analyzers.online_analysis_modules.math_tools as math_tools


def linear(x, a):
    return a * x


# Locate the scan directory


doPrint = False
normalizationCheck = False

sampleCase = 2
if sampleCase == 1:
    data_day = 29
    data_month = 6
    data_year = 2023
    scan_number = 23

elif sampleCase == 2:
    data_day = 25
    data_month = 7
    data_year = 2023
    scan_number = 24

else:
    print("Pick a valid sample case!")
    data_day = 0
    data_month = 0
    data_year = 0
    scan_number = 0

super_path = directory_functions.compile_daily_path(data_day, data_month, data_year)
image_name = "U_HiResMagCam"

inputParams = mag_spec_caller.UC_HiResMagCamImageAnalyzer(
    noise_threshold=100,
    edge_pixel_crop=1,
    saturation_value=4095,
    normalization_factor=1,  # 7.643283839778091e-07,
    transverse_calibration=43,
    do_transverse_calculation=False,  # True,
    transverse_slice_threshold=0.02,
    transverse_slice_binsize=5,
    optimization_central_energy=100.0,
    optimization_bandwidth_energy=2.0).build_input_parameter_dictionary()

num_shots = directory_functions.get_number_of_shots(super_path, scan_number, image_name)
shot_arr = np.array(range(num_shots)) + 1

clipping_arr = np.zeros(num_shots)
saturation_arr = np.zeros(num_shots)
picoscope_charge_arr = np.zeros(num_shots)
camera_counts_arr = np.zeros(num_shots)

for i in range(len(shot_arr)):
    if i % 10 == 0:
        print(i / len(shot_arr) * 100, "%")
    shot_number = shot_arr[i]
    if doPrint:
        print("Shot Number:", shot_number)
    fullpath = directory_functions.compile_file_location(super_path, scan_number, shot_number, image_name,
                                                         suffix=".png")
    image = pngTools.nBitPNG(fullpath)
    if doPrint:
        print("Loaded")

    processed_image = image.astype(np.float32)
    returned_image, MagSpecDict, lineouts = mag_spec_analysis.analyze_image(processed_image, inputParams)
    clipping_arr[i] = MagSpecDict['camera_clipping_factor']
    saturation_arr[i] = MagSpecDict['camera_saturation_counts']
    camera_counts_arr[i] = MagSpecDict['total_charge_pC']
    if doPrint:
        print("Clipped Percentage:", clipping_arr[i])
        print("Saturation Counts:", saturation_arr[i])
        print("Camera Counts:", camera_counts_arr[i])

    """
    clipping_arr[i] = MagSpecAnalysis.CalculateClippedPercentage(image)
    print("Clipped Percentage:", clipping_arr[i])
    max_pixel_value = 4095
    saturation_arr[i] = MagSpecAnalysis.SaturationCheck(image, max_pixel_value)
    print("Saturation Counts:", saturation_arr[i])
   """

    picoscope_charge_arr[i] = charge_tdms.get_shot_charge(super_path, scan_number, shot_number)
    if doPrint:
        print("Picoscope Charge:", picoscope_charge_arr[i])
print()

clip_tolerance = 0.001
clip_pass = np.where(clipping_arr < clip_tolerance)[0]
clip_picoscope_charge_arr = picoscope_charge_arr[clip_pass]
clip_camera_counts_arr = camera_counts_arr[clip_pass]
print("Non-Clipped:", len(clip_pass))

sat_tolerance = 50
sat_pass = np.where(saturation_arr < sat_tolerance)[0]
sat_picoscope_charge_arr = picoscope_charge_arr[sat_pass]
sat_camera_counts_arr = camera_counts_arr[sat_pass]
print("Non-Saturated:", len(sat_pass))

print("Minimum Picoscope Reading:  ", np.min(picoscope_charge_arr))
print("Minimum Total Camera Counts:", np.min(camera_counts_arr))
print()

if normalizationCheck:
    min_camera = 2
else:
    min_camera = 2e6
all_conditions = [
    [clipping_arr, '<', clip_tolerance],
    [saturation_arr, '<', sat_tolerance],
    [camera_counts_arr, '>', min_camera]
]
both_pass = math_tools.get_inequality_indices(all_conditions)
both_picoscope_charge_arr = picoscope_charge_arr[both_pass]
both_camera_counts_arr = camera_counts_arr[both_pass]

# A polyfit is alright, but we would want the y-intercept to be zero
lfit = np.polyfit(both_camera_counts_arr, both_picoscope_charge_arr, 1)
# So instead we just fit to a linear function and change the lfit values to this fit
popt, pcov = curve_fit(linear, both_camera_counts_arr, both_picoscope_charge_arr)
lfit[0] = popt[0]
lfit[1] = 0
print("Fit Values:", lfit)

print("Calibration Constants:")
tdms_filepath = charge_tdms.compile_tdms_filepath(super_path, scan_number)
charge_tdms.print_normalization(num_shots, tdms_filepath)
print("const_normalization_factor =", lfit[0])

axis = np.linspace(min(camera_counts_arr), max(camera_counts_arr), 50)
slope = axis * lfit[0] + lfit[1]
# slope2 = axis * MagSpecAnalysis.const_normalization_factor + lfit[1]

plt.scatter(camera_counts_arr, picoscope_charge_arr, color="r", marker="o", label="All Shots")
plt.scatter(both_camera_counts_arr, both_picoscope_charge_arr, color="b", marker="o", label="Good Shots")
plt.scatter(clip_camera_counts_arr, clip_picoscope_charge_arr, color="k", marker="1", label="Not Clipped")
plt.scatter(sat_camera_counts_arr, sat_picoscope_charge_arr, color="k", marker="2", label="Unsaturated")
plt.plot(axis, slope, c='k', ls='dashed', label="Fit: " + '{:.3e}'.format(lfit[0]))
# plt.plot(axis, slope2, c = 'g', ls = 'dashed', label="Old Calibration")
if normalizationCheck:
    plt.xlabel("Calibrated Camera Charge (pC)")
else:
    plt.xlabel("Camera Counts")
plt.ylabel("Picoscope Charge (pC)")
plotInfo = directory_functions.compile_plot_info(data_day, data_month, data_year, scan_number, shot=None,
                                                 camera_name="Camera Charge Calibration")
plt.title(plotInfo)
plt.legend()
plt.show()
