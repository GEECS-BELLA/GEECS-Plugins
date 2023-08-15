"""
Mon 8-7-2023

A wrapper function to call my Hi Res Mag Spec Analysis in the framework of the Labview code.

Inputs an image and outputs a list of doubles.

All constants are defined in the function.

@ Chris
"""

from array import array

import HiResAnalysisModules.HiResMagSpecAnalysis as MagSpecAnalysis
import HiResAnalysisModules.DirectoryModules_HiRes as DirectoryFunc
#import HiResAnalysisModules.pngTools_HiRes as pngTools

def HiResMagSpec_LabView(image):

    # Factor to go from camera counts to pC/MeV
    # Record: July 25th, Scan 24, HiResMagSpec
    normalization_triggerdelay = 15.497208
    normalization_exposure = 0.010000
    normalization_factor = 1.1301095153900242e-06

    inputParams = {
        "Threshold-Value": 230,                         # Large enough to remove noise level
        "Saturation-Value": 4095,                       # Just below the maximum for 2^12
        "Normalization-Factor": normalization_factor,   # Only valid for the above camera settings
        "Transverse-Calibration": 43,                   # ums / pixel
        "Transverse-Slice-Threshold": 0.02,
        "Transverse-Slice-Binsize": 5
    }

    MagSpecDict = MagSpecAnalysis.AnalyzeImage(image, inputParams)
    values = array('d', MagSpecDict.values())
    return (image,list(values))


"""
if __name__ == '__main__':

    data_day = 29
    data_month = 6
    data_year = 2023
    scan_number = 23
    shot_number = 20

    superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
    image_name = "U_HiResMagCam"

    fullpath = DirectoryFunc.CompileFileLocation(superpath, scan_number, shot_number, image_name, suffix=".png")
    raw_image = pngTools.nBitPNG(fullpath)

    results = HiResMagSpec_LabView(raw_image)
    print(results)
"""
