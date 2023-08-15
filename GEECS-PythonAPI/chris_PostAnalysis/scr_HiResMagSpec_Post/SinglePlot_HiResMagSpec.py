"""
Tues 8-15-2023

Analyzes a single image of HiResMagCam and displays plots

@Chris
"""

import sys
import time

sys.path.insert(0, "../../")
import chris_PostAnalysis.mod_ImageProcessing.pngTools as pngTools
import chris_PostAnalysis.mod_DataPlotting.HiResMagSpecPlotter as MagPlotter
import online_analysis.HTU.OnlineAnalysisModules.DirectoryModules as DirectoryFunc
import online_analysis.HTU.HiResMagSpec_LabView as MagSpecCaller


data_day = 9  # 29#9
data_month = 8  # 6#8
data_year = 2023
scan_number = 9  # 23#9
shot_number = 28

superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
image_name = "UC_TestCam"
# image_name = "U_HiResMagCam"

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
                                doNormalize=doNormalization, style=2)