# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:45:38 2023

Module for performing various analysis on MagSpec Images

@author: chris
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile

#sys.path.insert(0, "../")
import modules.CedossMathTools as MathTools
import modules.pngTools as pngTools

#Important: Using constants like this is generally bad,
# should consider using a specific normalization file with
# this information isntead

#Factor to go from camera counts to pC/MeV
#Record: June 29th, Scan 23, Shot 1 (HiRes ONLY)
const_normalization_factor = 3.308440355409613e-06
const_normalization_triggerdelay = 15.497208
const_normalization_exposure = 0.010000

def NormalizeImage(image, shotnumber, tdms_filepath):
    """
    Normalizes the HiResMagSpec image according to the saved normalization value.
    This value represents an image which is neither clipped or saturated

    Parameters
    ----------
    image : 2D Numpy Float Array
        The Image we are normalizing
    shotnumber : Int
        The shot number.
    tdms_filepath : String
        Filepath to the tdms file.

    Returns
    -------
    returnimage : 2D Numpy Float Array
        The normalized image.

    """
    #First, open up the tdms file and check that the camera delay
    # and shutter duration are close to expected
    trigger_list, exposure_list = GetCameraTriggerAndExposure(tdms_filepath)
    if float(trigger_list[shotnumber-1]) != float(const_normalization_triggerdelay):
        print("WARNING: the trigger delay is not the same as the current normalization!")
        print(" This Image: ",trigger_list[shotnumber-1])
        print(" Reference:  ",const_normalization_triggerdelay)
    if float(exposure_list[shotnumber-1]) != float(const_normalization_exposure):
        print("WARNING: the exposure is not the same as the current normalization!")
        print(" This Image: ",exposure_list[shotnumber-1])
        print(" Reference:  ",const_normalization_exposure)
    
    #Then, do the old quick normalization
    returnimage = np.copy(image) * const_normalization_factor
    return returnimage

def PrintNormalization(image, shotnumber ,tdms_filepath):
    """
    For a given image, print out the normalization factor.
    If this is a good image, then copy the normalization factors to the top of this module

    Parameters
    ----------
    image : 2D Numpy Float Array
        The Image we are normalizing
    shotnumber : Int
        The shot number.
    tdms_filepath : String
        Filepath to the tdms file.

    Returns
    -------
    None.

    """
    charge_pC_vals = GetBeamCharge(tdms_filepath)
    charge=charge_pC_vals[shotnumber-1]
    
    trigger_list, exposure_list = GetCameraTriggerAndExposure(tdms_filepath)

    #Assuming the image is good, find the factor, camera delay, and shutter duration and
    # print out the information for copy-pasting into this module.
    
    print("The following are the normalization factors,")
    print(" paste them into MagSpecAnalysis.py:")
    print("const_normalization_factor =",charge/sum(sum(image)))
    print("const_normalization_triggerdelay = ",trigger_list[shotnumber-1])
    print("const_normalization_exposure =",exposure_list[shotnumber-1])
    return

def CompileImageDirectory(superpath, scannumber, shotnumber, imagename, suffix=".png"):
    """
    Given the scan and shot number, compile the full path to the image
    
    Parameters
    ----------
    superpath : String
        Path to the folder where the datasets are stored.
        (NOTE:  Will update more once I understand the file structure of where things are saved)
    scannumber : Int
        The set number.
    shotnumber : Int
        The shot number.
    imagename : String
        Name of the cameara's image.
    suffix : String
        What filetype it is.  Defaults to .png

    Returns
    -------
    The filepath for the image.
    
    """
    scanpath = "Scan"+"{:03d}".format(scannumber)
    shotpath = scanpath + "_" + imagename + "_" + "{:03d}".format(shotnumber) + suffix
    fullpath = superpath + "/" + scanpath + "/" + imagename + "/" + shotpath
    
    return fullpath

def CompileTDMSFilepath(superpath,scannumber):
    """
    Given the scan number, compile the path to the tdms file

    Parameters
    ----------
    superpath : String
        Path to the folder where the datasets are stored.
        (NOTE:  Will update more once I understand the file structure of where things are saved).
    scannumber : Int
        The set number.

    Returns
    -------
    tdms_filepath : String
        The filepath for the tdms file.
    
    """
    scanpath = "Scan"+"{:03d}".format(scannumber)
    tdms_filepath = superpath + "/" + scanpath + "/" + scanpath + ".tdms"
    return tdms_filepath

def ParseInterpSpec(filename):
    """
    Parses the data saved in the Spec folder to get the momentum array and
    the charge density array.  Although ideally we will use our own analysis
    
    Parameters
    ----------
    filename : String
        Full path to the txt file.

    Returns
    -------
    momentum_arr : Float Numpy Array
        Arrays of the momentum.
    charge_arr : Float Numpy Array
        Arrays of the charge density per momentum.
    
    """
    with open(filename) as f:
        lines = f.readlines()
        nSlices = len(lines)-1
        momentum_arr = np.zeros(nSlices)
        charge_arr = np.zeros(nSlices)
        
        i = 0
        for line in lines:
            line = line.strip()
            columns = line.split('\t')
            if columns[0] != 'Momentum_GeV/c':
                momentum_arr[i] = float(columns[0])
                charge_arr[i] = float(columns[1])
                i = i + 1
        f.close()
    return momentum_arr, charge_arr

def LoadImage(superpath, scannumber, shotnumber, folderpath, tdms_filepath = None, doThreshold = True, hardlimit = None, doNormalize=True):
    fullpath = CompileImageDirectory(superpath, scannumber, shotnumber, folderpath)
    if tdms_filepath is None:
        tdms_filepath = CompileTDMSFilepath(superpath, scannumber)
    image = pngTools.nBitPNG(fullpath)
    if doThreshold:
        image = ThresholdReduction(image, hardlimit)
    if doNormalize:
        image = NormalizeImage(image, shotnumber, tdms_filepath)
    return image

def ThresholdReduction(image, hardlimit = None, skipPlots = True):
    """
    Attempts to do a quick reduction of the background by subtracting a fixed threshold off of the data
    Can either manually supply a "hardlimit" to do this, or if none is provided then we take Gaussian
    fits of the data off of the maximum to find what it asymptotes to

    Parameters
    ----------
    image : 2D Numpy Float Array
        Image array.
    hardlimit : Float, optional
        If a specific threshold is given rather than using Gaussian fits. The default is None.
    skipPlots : Boolean, optional
        If True, skips plotting Gaussian fits in MathTools. The default is True.

    Returns
    -------
    returnimage : 2D Numpy Float Array
        A separate instance of the initial image array that is threshold subtracted.

    """
    if hardlimit is None:
        xmax, ymax, maxval = FindMax(image)
        xslice = image[ymax,:]
        yslice = image[:,xmax]
        xfit = MathTools.FitDataSomething(xslice, np.linspace(0,len(xslice),len(xslice)), 
                                   MathTools.GaussianOffset, guess=[max(xslice),5,np.argmax(xslice),min(xslice)], supress = skipPlots)
        yfit = MathTools.FitDataSomething(yslice, np.linspace(0,len(yslice),len(yslice)), 
                                   MathTools.GaussianOffset, guess=[max(yslice),5,np.argmax(yslice),min(yslice)], supress = skipPlots)
        threshold = np.max([xfit[3],yfit[3]])
    else:
        threshold = hardlimit
        
    returnimage = np.copy(image)
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            returnimage[i][j] = image[i][j] - threshold
            if returnimage[i][j] < 0:
                returnimage[i][j] = 0
    return returnimage

#NOT YET IMPLEMENTED
def BackgroundSubtraction(image, background):
    print("JUST A PLACEHOLDER")
    return image

def CalculateProjectedBeamSize(image):
    skipPlots = True
    proj_arr = np.zeros(np.shape(image)[0])
    for i in range(len(proj_arr)):
        image_slice = image[i,:]
        proj_arr[i]=np.sum(image_slice)
    
    axis_arr = np.linspace(0,len(proj_arr),len(proj_arr))
    axis_arr = np.flip(axis_arr)
    fit = MathTools.FitDataSomething(proj_arr, axis_arr, 
                  MathTools.Gaussian, guess=[max(proj_arr),5,axis_arr[np.argmax(proj_arr)]], supress = skipPlots)
    
    return axis_arr, proj_arr, fit[1]

def CalculateChargeDensityDistribution(image):
    """
    Projects the MagSpec image onto the energy axis
    
    Parameters
    ----------
    image : 2D Float Numpy Array
        The MagSpecImage.

    Returns
    -------
    charge_arr : 1D Float Numpy Array
        Summation of the charge for each slice in energy
    
    """
    charge_arr = np.zeros(np.shape(image)[1])
    for i in range(len(charge_arr)):
        image_slice = image[:,i]
        charge_arr[i]=np.sum(image_slice)
    return charge_arr

def CalculateMaximumCharge(charge_arr):
    return np.max(charge_arr)

def CalculateAverageEnergy(charge_arr, energy_arr):
    return np.average(energy_arr,weights=charge_arr)

def CalculateStandardDeviationEnergy(charge_arr, energy_arr):
    average_energy = CalculateAverageEnergy(charge_arr, energy_arr)
    return np.sqrt(np.average((energy_arr-average_energy)**2, weights=charge_arr))

def CalculatePeakEnergy(charge_arr, energy_arr):
    return energy_arr[np.argmax(charge_arr)]

def FitTransverseGaussianSlices(image, threshold=0.01):
    ny, nx = np.shape(image)
    xloc, yloc, maxval = FindMax(image)
    skipPlots = True #False for debugging and/or make an animation book
    
    sigma_arr = np.zeros(nx)
    err_arr = np.zeros(nx)
    x0_arr = np.zeros(nx)
    amp_arr = np.zeros(nx)
    for i in range(nx):
        slice_arr = image[:,i]
        if np.max(slice_arr) > threshold*maxval:
            axis_arr = np.linspace(0,len(slice_arr),len(slice_arr))
            axis_arr = np.flip(axis_arr)
            fit = MathTools.FitDataSomething(slice_arr, axis_arr, 
                          MathTools.Gaussian, guess=[max(slice_arr),5,axis_arr[np.argmax(slice_arr)]], supress = skipPlots)
            amp_arr[i], sigma_arr[i], x0_arr[i] = fit
            
            func = MathTools.Gaussian(fit, axis_arr)
            error = 0
            for j in range(len(slice_arr)):
                error = error + np.square(slice_arr[j]-func[j])
            err_arr[i] = np.sqrt(error)/max(slice_arr)
        else:
            sigma_arr[i] = 0; x0_arr[i] = 0; amp_arr[i] = 0; err_arr[i] = 0

    return sigma_arr, x0_arr, amp_arr, err_arr

def CalculateAverageSize(sigma_arr, amp_arr):
    return np.average(sigma_arr, weights=amp_arr)

def FitBeamAngle(x0_arr, amp_arr, energy_arr):
    linear_fit = np.polyfit(energy_arr, x0_arr, deg=1, w=amp_arr)
    return linear_fit

def GetSizeStatistics_Full(image, energy_arr, threshold=0.01):
    sigma_arr, x0_arr, amp_arr, err_arr = FitTransverseGaussianSlices(image, threshold)
    average_size = CalculateAverageSize(sigma_arr, amp_arr)
    axis, vals, projected_size = CalculateProjectedBeamSize(image)
    linear_fit = FitBeamAngle(x0_arr, amp_arr, energy_arr)
    return average_size, projected_size, linear_fit[0]

def GetBeamCharge(tdms_filepath):
    """
    From the TDMS file, grabs the measured beam charge for each shot

    Parameters
    ----------
    tdms_filepath : String
        Filepath to the TDMS file.

    Returns
    -------
    scan_charge_vals : 1D Numpy Float Array
        Array of beam charge measurements.
        
    """
    #For future reference, can list all of the groups using tdms_file.groups()
    # and can list all of the groups, channels using group.channels()
    #
    #Also note, this loads the entire TDMS file into memory, and so a more
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
    return charge_pC_vals[shotnumber-1]

def GetCameraTriggerAndExposure(tdms_filepath):
    """
    From the TDMS file, grabs the list of camera exposures and trigger delays

    Parameters
    ----------
    tdms_filepath : String
        Filepath to the TDMS file.

    Returns
    -------
    trigger_list : 1D Numpy Float Array
        Array of recorded trigger delays
    exposure_list : 1D Numpy Float Array
        Array of recorded exposure times

    """
    tdms_file = TdmsFile.read(tdms_filepath)
    hiresmagcam_group = tdms_file['U_HiResMagCam']
    exposure_list = hiresmagcam_group['U_HiResMagCam Exposure']
    trigger_list = hiresmagcam_group['U_HiResMagCam TriggerDelay']
    tdms_file = None
    return trigger_list, exposure_list

def FindMax(image):
    """
    Grabs the coordinates and value of the maximum of the image

    Parameters
    ----------
    image : 2D Numpy Float Array
        The image array.

    Returns
    -------
    xmax : Int
        Horizontal position of maximum.
    ymax : Int
        Vertical position of maximum.
    maxcal : Float
        The value of the pixel at the maximum.

    """
    ny, nx = np.shape(image)
    max_ind = np.argmax(image)
    xmax = int(max_ind%nx)
    ymax = int(np.floor(max_ind/nx))
    maxval = image[ymax,xmax]
    return xmax, ymax, maxval

