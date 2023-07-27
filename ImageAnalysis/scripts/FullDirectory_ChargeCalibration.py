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

sys.path.insert(0, "../")
import modules.MagSpecAnalysis as MagSpecAnalysis
import modules.DirectoryModules as DirectoryFunc

def linear(x, a):
    return a * x

# Locate the scan directory

sampleCase = 1
if sampleCase == 1:
    data_day = 29
    data_month = 6
    data_year = 2023
    scan_number = 23

else:
    print("Pick a valid sample case!")
    data_day = 0; data_month = 0; data_year = 0; scan_number = 0

superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
image_name = "U_HiResMagCam"

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
    print("Shot Number:", shot_number)
    image = MagSpecAnalysis.LoadImage(superpath, scan_number, shot_number, image_name, doNormalize=False)
    print("Loaded")
    clipcheck_arr[i] = MagSpecAnalysis.CalculateClippedPercentage(image)
    print("Clipped Percentage:", clipcheck_arr[i])
    saturation_arr[i] = MagSpecAnalysis.SaturationCheck(image)
    print("Saturation Counts:", saturation_arr[i])
    picoscopecharge_arr[i] = MagSpecAnalysis.GetShotCharge(superpath, scan_number, shot_number)
    print("Picoscope Charge:", picoscopecharge_arr[i])
    cameracounts_arr[i] = np.sum(image)
    print("Camera Counts:", cameracounts_arr[i])

clip_tolerance = 0.001
clip_pass = np.where(clipcheck_arr < clip_tolerance)[0]
clip_picoscopecharge_arr = picoscopecharge_arr[clip_pass]
clip_cameracounts_arr = cameracounts_arr[clip_pass]
#clip_saturation_arr = saturation_arr[clip_pass]
print("Non-Clipped:", len(clip_pass))

sat_tolerance = 50
sat_pass = np.where(saturation_arr < sat_tolerance)[0]
sat_picoscopecharge_arr = picoscopecharge_arr[sat_pass]
sat_cameracounts_arr = cameracounts_arr[sat_pass]
print("Non-Saturated:", len(sat_pass))

both_pass = np.array([])
for j in range(len(clip_pass)):
    for k in range(len(sat_pass)):
        if clip_pass[j] == sat_pass[k]:
            both_pass = np.append(both_pass, int(clip_pass[j]))
both_pass = both_pass.astype(int)
both_picoscopecharge_arr = picoscopecharge_arr[both_pass]
both_cameracounts_arr = cameracounts_arr[both_pass]

lfit = np.polyfit(cameracounts_arr, picoscopecharge_arr, 1)
popt, pcov = curve_fit(f, cameracounts_arr, picoscopecharge_arr)
lfit[0] = popt[0]
lfit[1] = 0
print("Fit Values:",lfit)

axis = np.linspace(min(cameracounts_arr), max(cameracounts_arr),50)
slope = axis*lfit[0] + lfit[1]
slope2 = axis * MagSpecAnalysis.const_normalization_factor + lfit[1]

plt.scatter(cameracounts_arr, picoscopecharge_arr, color = "r", marker="o", label="All Shots")
plt.scatter(both_cameracounts_arr, both_picoscopecharge_arr, color = "b", marker="o", label="Good Shots")
plt.scatter(clip_cameracounts_arr, clip_picoscopecharge_arr, color = "k", marker="1", label="Not Clipped")
plt.scatter(sat_cameracounts_arr, sat_picoscopecharge_arr, color = "k", marker="2", label="Unsaturated")
plt.plot(axis, slope, c = 'k', ls = 'dashed', label="Fit")
#plt.plot(axis, slope2, c = 'g', ls = 'dashed', label="Old Calibration")
plt.xlabel("Camera Counts")
plt.ylabel("Picoscope Charge (MeV)")
plt.legend()
