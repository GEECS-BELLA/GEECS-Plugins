"""
Tues 8-15-2023

Analyzes a single image of HiResMagCam and displays plots

@Chris
"""

import sys
import os
import time
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

rootpath = os.path.abspath("../../../../../../")
sys.path.insert(0, rootpath)

import ImageAnalysis.image_analysis.analyzers.U_HiResMagCam.chris_PostAnalysis.mod_ImageProcessing.pngTools as pngTools
#import ImageAnalysis.image_analysis.analyzers.U_HiResMagCam.chris_PostAnalysis.mod_DataPlotting.HiResMagSpecPlotter as MagPlotter
import ImageAnalysis.image_analysis.analyzers.U_HiResMagCam.OnlineAnalysisModules.DirectoryModules as DirectoryFunc
import ImageAnalysis.image_analysis.analyzers.U_HiResMagCam.U_HiResMagSpec as MagSpecCaller


def generate_elliptical_gaussian(amplitude, height, width, center_x, center_y, sigma_x, sigma_y, angle_deg):
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    x, y = np.meshgrid(x, y)

    # Apply rotation to the coordinates
    angle_rad = np.radians(angle_deg)
    x_rot = (x - center_x) * np.cos(angle_rad) - (y - center_y) * np.sin(angle_rad) + center_x
    y_rot = (x - center_x) * np.sin(angle_rad) + (y - center_y) * np.cos(angle_rad) + center_y

    # Calculate the Gaussian function
    exponent = -((x_rot - center_x) ** 2 / (2 * sigma_x ** 2) + (y_rot - center_y) ** 2 / (2 * sigma_y ** 2))
    gaussian = amplitude*np.exp(exponent)

    return gaussian


# Define parameters for the elliptical Gaussian
amplitude = 6000
height = 50
width = 50
center_x = 40
center_y = 25
sigma_x = 7
sigma_y = 3
angle_deg = 30

start = time.perf_counter()
# Generate the elliptical Gaussian array
elliptical_gaussian_array = generate_elliptical_gaussian(amplitude, height, width, center_x, center_y, sigma_x, sigma_y, angle_deg)
print("Elapsed Time: ", time.perf_counter() - start, "s")

"""
data_day = 29  # 29#9
data_month = 6  # 6#8
data_year = 2023
scan_number = 23  # 23#9
shot_number = 54

superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
# image_name = "UC_TestCam"
image_name = "U_HiResMagCam"

fullpath = DirectoryFunc.CompileFileLocation(superpath, scan_number, shot_number, image_name, suffix=".png")
raw_image = pngTools.nBitPNG(fullpath)
"""

start = time.perf_counter()
#returned_image, analyzeDict, inputParams = MagSpecCaller.HiResMagSpec_Dictionary(raw_image)
returned_image, analyzeDict, inputParams = MagSpecCaller.U_HiResMagSpecImageAnalyzer().analyze_image(elliptical_gaussian_array)
print("Elapsed Time: ", time.perf_counter() - start, "s")
print(analyzeDict)

plt.imshow(elliptical_gaussian_array)
plt.show()

"""
num_pixel = inputParams["Pixel-Crop"]
raw_image = raw_image[num_pixel:-num_pixel, num_pixel:-num_pixel]
doThreshold = True
doNormalization = True

plotInfo = DirectoryFunc.CompilePlotInfo(data_day, data_month, data_year, scan_number, shot_number, "U_HiResMagSpec")
MagPlotter.PlotEnergyProjection(raw_image, analyzeDict, inputParams, plotInfo=plotInfo, doThreshold=doThreshold,
                                doNormalize=doNormalization)
MagPlotter.PlotSliceStatistics(raw_image, analyzeDict, inputParams, plotInfo=plotInfo, doThreshold=doThreshold,
                               doNormalize=doNormalization)
MagPlotter.PlotBeamDistribution(raw_image, analyzeDict, inputParams, plotInfo=plotInfo, doThreshold=doThreshold,
                                doNormalize=doNormalization, style=0)
"""
