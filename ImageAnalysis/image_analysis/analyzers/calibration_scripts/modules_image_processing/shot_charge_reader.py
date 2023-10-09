"""
Wed 8-16-2023

Module hosting functions to read the saved shot beam charges from the tdms file.  Used in some older analysis scripts
and the camera-charge calibration script

@Chris
"""

import numpy as np
from nptdms import TdmsFile


def compile_tdms_filepath(super_path, scan_number):
    scan_path = "Scan" + "{:03d}".format(scan_number)
    tdms_filepath = super_path + "/" + scan_path + "/" + scan_path + ".tdms"
    return tdms_filepath


def print_normalization(shot_number, tdms_filepath):
    trigger_list, exposure_list = get_camera_trigger_and_exposure(tdms_filepath)
    print("const_normalization_triggerdelay = ", trigger_list[shot_number - 1])
    print("const_normalization_exposure =", exposure_list[shot_number - 1])
    return


def get_beam_charge(tdms_filepath):
    # Note, this loads the entire TDMS file into memory, and so a more
    # elegant usage of TdmsFile could be only reading the necessary picoscope data
    tdms_file = TdmsFile.read(tdms_filepath)
    picoscope_group = tdms_file['U-picoscope5245D']
    charge_pc_channel = picoscope_group['U-picoscope5245D charge pC']
    scan_charge_vals = np.asarray(charge_pc_channel[:], dtype=float)
    return scan_charge_vals


def get_shot_charge(super_path, scan_number, shot_number):
    tdms_filepath = compile_tdms_filepath(super_path, scan_number)
    charge_pc_vals = get_beam_charge(tdms_filepath)
    return charge_pc_vals[shot_number - 1]


def get_camera_trigger_and_exposure(tdms_filepath):
    tdms_file = TdmsFile.read(tdms_filepath)
    hiresmagcam_group = tdms_file['U_HiResMagCam']
    exposure_list = hiresmagcam_group['U_HiResMagCam Exposure']
    trigger_list = hiresmagcam_group['U_HiResMagCam TriggerDelay']
    return trigger_list, exposure_list
