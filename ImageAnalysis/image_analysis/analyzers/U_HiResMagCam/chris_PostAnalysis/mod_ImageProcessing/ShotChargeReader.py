"""
Wed 8-16-2023

Module hosting functions to read the saved shot beam charges from the tdms file.  Used in some older analysis scripts
and the camera-charge calibration script

@Chris
"""

import numpy as np
from nptdms import TdmsFile


def CompileTDMSFilepath(superpath, scannumber):
    scanpath = "Scan" + "{:03d}".format(scannumber)
    tdms_filepath = superpath + "/" + scanpath + "/" + scanpath + ".tdms"
    return tdms_filepath


def PrintNormalization(shotnumber, tdms_filepath):
    charge_pC_vals = GetBeamCharge(tdms_filepath)
    charge = charge_pC_vals[shotnumber - 1]

    trigger_list, exposure_list = GetCameraTriggerAndExposure(tdms_filepath)

    # Assuming the image is good, find the factor, camera delay, and shutter duration and
    # print out the information for copy-pasting into this module.

    print("The following are the normalization factors,")
    print(" paste them into MagSpecAnalysis.py:")
    print("const_normalization_triggerdelay = ", trigger_list[shotnumber - 1])
    print("const_normalization_exposure =", exposure_list[shotnumber - 1])
    return


def GetBeamCharge(tdms_filepath):
    # For future reference, can list all of the groups using tdms_file.groups()
    # and can list all of the groups, channels using group.channels()
    #
    # Also note, this loads the entire TDMS file into memory, and so a more
    # elegant usage of TdmsFile could be only reading the necessary picoscope data
    tdms_file = TdmsFile.read(tdms_filepath)
    picoscope_group = tdms_file['U-picoscope5245D']
    charge_pC_channel = picoscope_group['U-picoscope5245D charge pC']
    scan_charge_vals = np.asarray(charge_pC_channel[:], dtype=float)
    tdms_file = None
    return scan_charge_vals


def GetShotCharge(superpath, scannumber, shotnumber):
    tdms_filepath = CompileTDMSFilepath(superpath, scannumber)
    charge_pC_vals = GetBeamCharge(tdms_filepath)
    return charge_pC_vals[shotnumber - 1]


def GetCameraTriggerAndExposure(tdms_filepath):
    tdms_file = TdmsFile.read(tdms_filepath)
    hiresmagcam_group = tdms_file['U_HiResMagCam']
    exposure_list = hiresmagcam_group['U_HiResMagCam Exposure']
    trigger_list = hiresmagcam_group['U_HiResMagCam TriggerDelay']
    tdms_file = None
    return trigger_list, exposure_list
