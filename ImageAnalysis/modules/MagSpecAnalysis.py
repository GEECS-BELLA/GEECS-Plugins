# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:45:38 2023

Module for performing various analysis on MagSpec Images

@author: chris
"""
import sys
import numpy as np
from nptdms import TdmsFile

#sys.path.insert(0, "../")
from modules import CedossMathTools as MathTools

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
    tdms_file = TdmsFile.read("C:/Users/chris/Desktop/cedoss_htu_data/Scan023/Scan023.tdms")
    picoscope_group = tdms_file['U-picoscope5245D']
    charge_pC_channel = picoscope_group['U-picoscope5245D charge pC']
    scan_charge_vals = np.asarray(charge_pC_channel[:], dtype=float)
    tdms_file = None
    return scan_charge_vals

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