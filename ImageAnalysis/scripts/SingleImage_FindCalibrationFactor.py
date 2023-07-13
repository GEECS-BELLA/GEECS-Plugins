# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:48:42 2023

A quick script that plots the 2D image and prints out the
calibration to go from camera counts to pC charge density

@author: chris
"""

import sys
#import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
import modules.MagSpecAnalysis as MagSpecAnalysis
import modules.pngTools as pngTools

#I use another version of pngTools because the current version
# is incompatible with my version of python?  Eventually merge these...
#sys.path.insert(0, "../../dataanalysis-notebook/functions/")
#from pngTools import nBitPNG

superpath = "C:/Users/chris/Desktop/cedoss_htu_data/"

scannumber = 23
shotnumber = 1

#Loads in a 2d png image

folderpath = "U_HiResMagCam-interp"
fullpath = MagSpecAnalysis.CompileImageDirectory(superpath, scannumber, shotnumber, folderpath)
image = pngTools.nBitPNG(fullpath)
image = MagSpecAnalysis.ThresholdReduction(image)

tdms_filepath = MagSpecAnalysis.CompileTDMSFilepath(superpath, scannumber)

print("Check that this is neither clipped nor saturated:")
plt.imshow(image, aspect=.1)
plt.show()

MagSpecAnalysis.PrintNormalization(image, shotnumber, tdms_filepath)

