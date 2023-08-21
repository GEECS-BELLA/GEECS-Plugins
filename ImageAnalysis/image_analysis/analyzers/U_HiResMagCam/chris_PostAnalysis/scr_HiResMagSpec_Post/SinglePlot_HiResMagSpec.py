"""
Tues 8-15-2023

Analyzes a single image of HiResMagCam and displays plots

@Chris
"""

import sys
import os
import time

rootpath = os.path.abspath("../../../../../")
sys.path.insert(0, rootpath)

import image_analysis.analyzers.U_HiResMagCam.chris_PostAnalysis.mod_ImageProcessing.pngTools as pngTools
import image_analysis.analyzers.U_HiResMagCam.chris_PostAnalysis.mod_DataPlotting.HiResMagSpecPlotter as MagPlotter
import image_analysis.analyzers.U_HiResMagCam.OnlineAnalysisModules.DirectoryModules as DirectoryFunc
import image_analysis.analyzers.U_HiResMagCam.U_HiResMagSpec as MagSpecCaller


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

start = time.perf_counter()
returned_image, analyzeDict, inputParams = MagSpecCaller.HiResMagSpec_Dictionary(raw_image)
print("Elapsed Time: ", time.perf_counter() - start, "s")
print(analyzeDict)

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
