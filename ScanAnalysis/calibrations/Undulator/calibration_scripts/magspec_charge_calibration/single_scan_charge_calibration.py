"""
Wed 7/26/2023

Loop over a full scan directory of nice HiResMagSpec images.  Mostly interested in the picoscope charge, sum of camera
counts, clipped percentage, and saturation counts.  For good shots, plot the relation between picoscope charge and
camera counts.  Find a fit, and keep that as the camera calibration constant.  Make not of trigger delay

Could just use the "AnalyzeImage" function to get everything, but honestly I would rather just quickly calculate
the info I need.  Mostly because I don't know how to store variables in consoles for pycharm yet.  :(

Thu 12/12/2024 - After ScanAnalyzer was created, TODO would be nice to remake this calibration script...

@Chris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from image_analysis.utils import read_imaq_png_image
import calibrations.Undulator.shot_charge_reader as charge_reader
from image_analysis.analyzers.UC_GenericMagSpecCam import UC_GenericMagSpecCamAnalyzer
from geecs_data_utils import ScanData

def linear(x, a):
    return a * x


doPrint = False
normalizationCheck = False
key = 'U_BCaveICT Charge pC'
sampleCase = 2
if sampleCase == 1:
    data_day = 9
    data_month = 5
    data_year = 2024
    scan_number = 51  # 30, 50, 51
    image_name = "UC_HiResMagCam"

elif sampleCase == 101:
    data_day = 29
    data_month = 6
    data_year = 2023
    scan_number = 23
    image_name = "U_HiResMagCam"

elif sampleCase == 102:
    data_day = 25
    data_month = 7
    data_year = 2023
    scan_number = 24
    image_name = "U_HiResMagCam"

elif sampleCase == 2:
    data_day = 6
    data_month = 3
    data_year = 2025
    scan_number = 22
    image_name = "UC_HiResMagCam"
    #image_name = "UC_BCaveMagSpecCam1"
    key = 'U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC'

else:
    print("Pick a valid sample case!")
    data_day = 0
    data_month = 0
    data_year = 0
    scan_number = 0
    image_name = None

scan_tag = ScanData.get_scan_tag(data_year, data_month, data_day, scan_number, experiment_name='Undulator')
scan_data = ScanData(tag=scan_tag, load_scalars=True)

calibration_analyzer = UC_GenericMagSpecCamAnalyzer(
    mag_spec_name='hires',
    noise_threshold=100,
    roi=[1, -1, 1, -1],
    saturation_value=4095,
    normalization_factor=1,  # 6.218486e-6  OLD: 7.643283839778091e-07,
    transverse_calibration=43,
    do_transverse_calculation=False,  # True,
    transverse_slice_threshold=0.02,
    transverse_slice_binsize=5,
    optimization_central_energy=100.0,
    optimization_bandwidth_energy=2.0)

input_params = calibration_analyzer.build_input_parameter_dictionary()

picoscope_charge_arr = scan_data.data_frame[key]
num_shots = len(picoscope_charge_arr)

shot_arr = np.array(range(num_shots))+1

clipping_arr = np.zeros(num_shots)
saturation_arr = np.zeros(num_shots)
camera_counts_arr = np.zeros(num_shots)
camera_counts_arr_alt = np.zeros(num_shots)


for i in range(len(shot_arr)):
    if i % 10 == 0:
        print(i / len(shot_arr) * 100, "%")
    shot_number = shot_arr[i]
    if doPrint:
        print("Shot Number:", shot_number)

    fullpath = ScanData.get_device_shot_path(scan_tag, image_name, int(shot_number))
    try:
        image = read_imaq_png_image(Path(fullpath)) * 1.0
        if doPrint:
            print("Loaded")

        analyzer_returns = calibration_analyzer.analyze_image(image)
        mag_spec_dict = analyzer_returns["analyzer_return_dictionary"]
        clipping_arr[i] = mag_spec_dict['camera_clipping_factor']
        saturation_arr[i] = mag_spec_dict['camera_saturation_counts']
        camera_counts_arr[i] = mag_spec_dict['total_charge_pC']
        if doPrint:
            print("Clipped Percentage:", clipping_arr[i])
            print("Saturation Counts:", saturation_arr[i])
            print("Camera Counts:", camera_counts_arr[i])
            print("Picoscope Charge:", picoscope_charge_arr[i])
    except FileNotFoundError:
        print(f"No file for shot {shot_number}")
        clipping_arr[i] = 100
        saturation_arr[i] = 100
        camera_counts_arr[i] = 0

"""
    image_name_alt = 'UC_BCaveMagSpecCam1'
    fullpath_alt = ScanData.get_device_shot_path(scan_tag, image_name_alt, int(shot_number))
    try:
        image = read_imaq_png_image(Path(fullpath_alt)) * 1.0
        camera_counts_arr_alt[i] = np.sum(image)
    except FileNotFoundError:
        print(f"No file for shot {shot_number}")
        camera_counts_arr_alt[i] = 0
plt.scatter(camera_counts_arr, camera_counts_arr_alt)
plt.show()
"""

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

both_pass = np.where((clipping_arr < clip_tolerance) &
                     (saturation_arr < sat_tolerance) &
                     (camera_counts_arr > min_camera))[0]

both_picoscope_charge_arr = picoscope_charge_arr[both_pass]
both_camera_counts_arr = camera_counts_arr[both_pass]

linear_fit = np.zeros(2)
p_opt, p_cov = curve_fit(linear, both_camera_counts_arr, both_picoscope_charge_arr)
linear_fit[0] = p_opt[0]
linear_fit[1] = 0
print("Fit Values:", linear_fit)

# TODO Replace with something better
# print("Calibration Constants:")
# tdms_filepath = charge_reader.compile_tdms_filepath(super_path, scan_number)
# charge_reader.print_normalization(num_shots, tdms_filepath)
# print("const_normalization_factor =", linear_fit[0])

axis = np.linspace(min(camera_counts_arr), max(camera_counts_arr), 50)
slope = axis * linear_fit[0] + linear_fit[1]
# slope2 = axis * MagSpecAnalysis.const_normalization_factor + linear_fit[1]

plt.scatter(camera_counts_arr, picoscope_charge_arr, color="r", marker="o", label="All Shots")
plt.scatter(both_camera_counts_arr, both_picoscope_charge_arr, color="b", marker="o", label="Good Shots")
plt.scatter(clip_camera_counts_arr, clip_picoscope_charge_arr, color="k", marker="1", label="Not Clipped")
plt.scatter(sat_camera_counts_arr, sat_picoscope_charge_arr, color="k", marker="2", label="Unsaturated")
plt.plot(axis, slope, c='k', ls='dashed', label="Fit: " + '{:.3e}'.format(linear_fit[0]))
# plt.plot(axis, slope2, c = 'g', ls = 'dashed', label="Old Calibration")
if normalizationCheck:
    plt.xlabel("Calibrated Camera Charge (pC)")
else:
    plt.xlabel("Camera Counts")
plt.ylabel("Picoscope Charge (pC)")

plt.title(f"{data_day}/{data_month}/{data_year}:Scan{scan_number}: Camera Charge Calibration")
plt.legend()
plt.show()
