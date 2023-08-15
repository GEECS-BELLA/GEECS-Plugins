"""
Mon 8-7-2023

A wrapper function to call my Hi Res Mag Spec Analysis in the framework of the Labview code.

Inputs an image and outputs a list of doubles. For the "_LabView" function.

The "_Dictionary" function is the same, just the list of doubles is instead a dictionary

All constants are defined in the function.

The imports a bit ugly, but haven't had a chance to debug how paths work in the LabView implementation.  Works though.

@ Chris
"""

from array import array
import numpy as np

# Either importing with the path set to GEECS-PythonAPI (as is the case for post-analysis scripts elsewhere)
#  or importing with the path set to this location (which is the case for when run on LabView)

try:
    import online_analysis.HTU.OnlineAnalysisModules.HiResMagSpecAnalysis as MagSpecAnalysis
    print("Imported Via GEECS-PythonAPI")
except ImportError:
    try:
        import OnlineAnalysisModules.HiResMagSpecAnalysis as MagSpecAnalysis
        print("Imported Via GEECS-PythonAPI.online_analysis.HTU")
    except ImportError:
        print("Modules not found!  Check your paths!")

"""
import OnlineAnalysisModules.HiResMagSpecPlotter as MagPlotter
import OnlineAnalysisModules.DirectoryModules as DirectoryFunc
import OnlineAnalysisModules.pngTools_HiRes as pngTools
"""

def HiResMagSpec_LabView(image):
    returned_image, MagSpecDict, inputParams = HiResMagSpec_Dictionary(image)
    values = array('d', MagSpecDict.values())
    return (returned_image, list(values))


def HiResMagSpec_Dictionary(image):

    # Factor to go from camera counts to pC/MeV
    # Record: July 25th, Scan 24, HiResMagSpec
    normalization_triggerdelay = 15.497208
    normalization_exposure = 0.010000
    normalization_factor = 1.1301095153900242e-06

    inputParams = {
        "Threshold-Value": 100,#230,                    # Large enough to remove noise level
        "Pixel-Crop": 1,                                # Number of edge pixels to crop
        "Saturation-Value": 4095,                       # Just below the maximum for 2^12
        "Normalization-Factor": normalization_factor,   # Only valid for the above camera settings
        "Transverse-Calibration": 43,                   # ums / pixel
        "Do-Transverse-Calculation": True,              # False if you want to skip
        "Transverse-Slice-Threshold": 0.02,
        "Transverse-Slice-Binsize": 5
    }
    processed_image = image.astype(np.float32)
    returned_image, MagSpecDict = MagSpecAnalysis.AnalyzeImage(processed_image, inputParams)
    unnormalized_image = returned_image / normalization_factor
    uint_image = unnormalized_image.astype(np.uint16)
    return uint_image, MagSpecDict, inputParams

# I don't run through this anymore, all plotting is handled in the chris_PostAnalysis folder of GEECS-PythonAPI
"""
if __name__ == '__main__':

    data_day = 9#29#9
    data_month = 8#6#8
    data_year = 2023
    scan_number = 9#23#9
    shot_number = 28

    superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
    image_name = "UC_TestCam"
    #image_name = "U_HiResMagCam"

    fullpath = DirectoryFunc.CompileFileLocation(superpath, scan_number, shot_number, image_name, suffix=".png")
    raw_image = pngTools.nBitPNG(fullpath)

    start = time.perf_counter()
    returned_image, analyzeDict, inputParams = HiResMagSpec_Dictionary(raw_image)
    print("Elapsed Time: ", time.perf_counter()-start, "s")
    print(analyzeDict)

    num_pixel = inputParams["Pixel-Crop"]
    raw_image = raw_image[num_pixel:-num_pixel, num_pixel:-num_pixel]
    doThreshold = True
    doNormalization = True

    plotInfo = DirectoryFunc.CompilePlotInfo(data_day, data_month, data_year, scan_number, shot_number, "U_HiResMagSpec")
    MagPlotter.PlotEnergyProjection(raw_image, analyzeDict, inputParams, plotInfo=plotInfo, doThreshold=doThreshold, doNormalize=doNormalization)
    MagPlotter.PlotSliceStatistics(raw_image, analyzeDict, inputParams, plotInfo=plotInfo, doThreshold=doThreshold, doNormalize=doNormalization)
    MagPlotter.PlotBeamDistribution(raw_image, analyzeDict, inputParams, plotInfo=plotInfo, doThreshold=doThreshold, doNormalize=doNormalization, style=2)
"""
