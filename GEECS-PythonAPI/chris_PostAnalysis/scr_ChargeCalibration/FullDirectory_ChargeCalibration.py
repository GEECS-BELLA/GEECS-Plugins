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
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, "../../")
import chris_PostAnalysis.mod_ImageProcessing.ShotChargeReader as ChargeTDMS
import chris_PostAnalysis.mod_ImageProcessing.pngTools as pngTools
import online_analysis.HTU.OnlineAnalysisModules.DirectoryModules as DirectoryFunc
import online_analysis.HTU.OnlineAnalysisModules.CedossMathTools as MathTools
import online_analysis.HTU.OnlineAnalysisModules.HiResMagSpecAnalysis as MagSpecAnalysis


# print("Not yet fixed with new API!!")
# sys.exit()

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
    data_day = 0;
    data_month = 0;
    data_year = 0;
    scan_number = 0

superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
image_name = "U_HiResMagCam"

inputParams = {
    "Threshold-Value": 100,  # 230,                    # Large enough to remove noise level
    "Pixel-Crop": 1,  # Number of edge pixels to crop
    "Saturation-Value": 4095,  # Just below the maximum for 2^12
    "Normalization-Factor": 1,  # NO NORMALIZATION HERE
    "Transverse-Calibration": 43,  # ums / pixel
    "Do-Transverse-Calculation": False,  # False if you want to skip
    "Transverse-Slice-Threshold": 0.02,
    "Transverse-Slice-Binsize": 5
}

num_shots = DirectoryFunc.GetNumberOfShots(superpath, scan_number, image_name)
shot_arr = np.array(range(num_shots)) + 1

clipcheck_arr = np.zeros(num_shots)
saturation_arr = np.zeros(num_shots)
picoscopecharge_arr = np.zeros(num_shots)
cameracounts_arr = np.zeros(num_shots)

for i in range(len(shot_arr)):
    if i % 10 == 0:
        print(i / len(shot_arr) * 100, "%")
    shot_number = shot_arr[i]
    if doPrint:
        print("Shot Number:", shot_number)
    fullpath = DirectoryFunc.CompileFileLocation(superpath, scan_number, shot_number, image_name, suffix=".png")
    image = pngTools.nBitPNG(fullpath)
    if doPrint:
        print("Loaded")

    processed_image = image.astype(np.float32)
    returned_image, MagSpecDict = MagSpecAnalysis.AnalyzeImage(processed_image, inputParams)
    clipcheck_arr[i] = MagSpecDict['Clipped-Percentage']
    saturation_arr[i] = MagSpecDict['Saturation-Counts']
    cameracounts_arr[i] = MagSpecDict['Charge-On-Camera']
    if doPrint:
        print("Clipped Percentage:", clipcheck_arr[i])
        print("Saturation Counts:", saturation_arr[i])
        print("Camera Counts:", cameracounts_arr[i])

    """
    clipcheck_arr[i] = MagSpecAnalysis.CalculateClippedPercentage(image)
    print("Clipped Percentage:", clipcheck_arr[i])
    max_pixel_value = 4095
    saturation_arr[i] = MagSpecAnalysis.SaturationCheck(image, max_pixel_value)
    print("Saturation Counts:", saturation_arr[i])
   """

    picoscopecharge_arr[i] = ChargeTDMS.GetShotCharge(superpath, scan_number, shot_number)
    if doPrint:
        print("Picoscope Charge:", picoscopecharge_arr[i])

clip_tolerance = 0.001
clip_pass = np.where(clipcheck_arr < clip_tolerance)[0]
clip_picoscopecharge_arr = picoscopecharge_arr[clip_pass]
clip_cameracounts_arr = cameracounts_arr[clip_pass]
# clip_saturation_arr = saturation_arr[clip_pass]
print("Non-Clipped:", len(clip_pass))

sat_tolerance = 50
sat_pass = np.where(saturation_arr < sat_tolerance)[0]
sat_picoscopecharge_arr = picoscopecharge_arr[sat_pass]
sat_cameracounts_arr = cameracounts_arr[sat_pass]
print("Non-Saturated:", len(sat_pass))

"""# Old way before i made my nice function
both_pass = np.array([])
for j in range(len(clip_pass)):
    for k in range(len(sat_pass)):
        if clip_pass[j] == sat_pass[k]:
            both_pass = np.append(both_pass, int(clip_pass[j]))
both_pass = both_pass.astype(int)"""

print("Minimum Picoscope Reading:  ", np.min(picoscopecharge_arr))
print("Minimum Total Camera Counts:", np.min(cameracounts_arr))

if normalizationCheck:
    min_camera = 2
else:
    min_camera = 2e6
all_conditions = [
    [clipcheck_arr, '<', clip_tolerance],
    [saturation_arr, '<', sat_tolerance],
    [cameracounts_arr, '>', min_camera]
]
both_pass = MathTools.GetInequalityIndices(all_conditions)
both_picoscopecharge_arr = picoscopecharge_arr[both_pass]
both_cameracounts_arr = cameracounts_arr[both_pass]

lfit = np.polyfit(both_cameracounts_arr, both_picoscopecharge_arr, 1)
popt, pcov = curve_fit(linear, both_cameracounts_arr, both_picoscopecharge_arr)
lfit[0] = popt[0]
lfit[1] = 0
print("Fit Values:", lfit)

print("Calibration Constants:")
tdms_filepath = ChargeTDMS.CompileTDMSFilepath(superpath, scan_number)
ChargeTDMS.PrintNormalization(shot_number, tdms_filepath)
print("const_normalization_factor =", lfit[0])

axis = np.linspace(min(cameracounts_arr), max(cameracounts_arr), 50)
slope = axis * lfit[0] + lfit[1]
# slope2 = axis * MagSpecAnalysis.const_normalization_factor + lfit[1]

plt.scatter(cameracounts_arr, picoscopecharge_arr, color="r", marker="o", label="All Shots")
plt.scatter(both_cameracounts_arr, both_picoscopecharge_arr, color="b", marker="o", label="Good Shots")
plt.scatter(clip_cameracounts_arr, clip_picoscopecharge_arr, color="k", marker="1", label="Not Clipped")
plt.scatter(sat_cameracounts_arr, sat_picoscopecharge_arr, color="k", marker="2", label="Unsaturated")
plt.plot(axis, slope, c='k', ls='dashed', label="Fit: " + '{:.3e}'.format(lfit[0]))
# plt.plot(axis, slope2, c = 'g', ls = 'dashed', label="Old Calibration")
if normalizationCheck:
    plt.xlabel("Calibrated Camera Charge (pC)")
else:
    plt.xlabel("Camera Counts")
plt.ylabel("Picoscope Charge (pC)")
plotInfo = DirectoryFunc.CompilePlotInfo(data_day, data_month, data_year, scan_number, shot=None,
                                         cameraStr="Camera Charge Calibration")
plt.title(plotInfo)
plt.legend()
plt.show()
