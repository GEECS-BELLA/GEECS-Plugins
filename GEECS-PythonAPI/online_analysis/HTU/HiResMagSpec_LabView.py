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


def HiResMagSpec_LabView(image):
    returned_image, MagSpecDict, inputParams = HiResMagSpec_Dictionary(image)
    values = array('d', MagSpecDict.values())
    return (returned_image, list(values))


def HiResMagSpec_Dictionary(image):

    # Factor to go from camera counts to pC/MeV
    # Depends on trigger delay, exposure, and the threshold value for magspec analysis
    # Record: July 25th, Scan 24, HiResMagSpec
    normalization_triggerdelay = 15.497208
    normalization_exposure = 0.010000
    normalization_thresholdvalue = 100 # 230
    normalization_factor = 7.643283839778091e-07 # 1.1301095153900242e-06

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
