# -*- coding: utf-8 -*-
"""
This script uses the modules for loading MagSpec images to display quick
information about the MagSpec images.

This initial test script uses data saved on my personal latop

Doss - 7/10/2023
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
from modules import MagSpecAnalysis
from modules import pngTools

#I use another version of pngTools because the current version
# is incompatible with my version of python?  Eventually merge these...
#sys.path.insert(0, "../../dataanalysis-notebook/functions/")
#from pngTools import nBitPNG

superpath = "C:/Users/chris/Desktop/cedoss_htu_data/"

scannumber = 23
shotnumber = 1

#Loads in the 2 arrays from the interpSpec

folderpath = "U_HiResMagCam-interpSpec"
suffix = ".txt"
specpath = MagSpecAnalysis.CompileImageDirectory(superpath, scannumber, shotnumber, folderpath, suffix)

mom_arr, charge_arr = MagSpecAnalysis.ParseInterpSpec(specpath)


#Loads in a 2d png image

folderpath = "U_HiResMagCam-interp"
fullpath = MagSpecAnalysis.CompileImageDirectory(superpath, scannumber, shotnumber, folderpath)
image = pngTools.nBitPNG(fullpath)
image = MagSpecAnalysis.ThresholdReduction(image)

tdms_filepath = MagSpecAnalysis.CompileTDMSFilepath(superpath, scannumber)
charge_pC_vals = MagSpecAnalysis.GetBeamCharge(tdms_filepath)
charge=charge_pC_vals[shotnumber-1]

image = image * charge/sum(sum(image))

calc_charge_arr = MagSpecAnalysis.CalculateChargeDensityDistribution(image)

peak_charge = np.max(calc_charge_arr)
average_energy = np.average(mom_arr,weights=calc_charge_arr)
peak_energy = mom_arr[np.argmax(calc_charge_arr)]

#plt.plot(mom_arr,charge_arr)
plt.plot(mom_arr,calc_charge_arr,c='k',ls='solid',label="Peak Charge:"+"{:10.2f}".format(peak_charge)+" pC/MeV")
plt.plot([average_energy,average_energy],[0,max(calc_charge_arr)*1.1],c='r',ls='dashed', label="Average Energy:"+"{:8.2f}".format(average_energy)+" MeV")
plt.plot([peak_energy,peak_energy],[0,max(calc_charge_arr)*1.1],c='g',ls='dotted', label="Peak Energy:"+"{:13.2f}".format(peak_energy)+" MeV")
plt.xlabel("Energy "+r'$(\mathrm{\ MeV})$')
plt.ylabel("Charge Density "+r'$(\mathrm{\ pC/MeV})$')
plt.title("Sample Data:  Scan "+str(scannumber)+", Shot "+str(shotnumber))
plt.legend(title="Total Charge:"+"{:10.2f}".format(charge)+" pC")
plt.show()

plt.imshow(image, aspect=1, extent=(mom_arr[0],mom_arr[-1],-10,10))
plt.show()
