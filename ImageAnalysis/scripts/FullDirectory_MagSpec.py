# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:31:45 2023

Loop over the full directory, and get a bunch of arrays for each value.

Ideally, this is where I would export to the s file, but here I just plot

@author: chris
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
import modules.MagSpecAnalysis as MagSpecAnalysis

superpath = "C:/Users/chris/Desktop/cedoss_htu_data/"
folderpath = "U_HiResMagCam-interp"
scannumber = 23

skipLoop = True

if not skipLoop:
    spec_folderpath = "U_HiResMagCam-interpSpec"
    suffix = ".txt"
    testshot = 1
    specpath = MagSpecAnalysis.CompileImageDirectory(superpath, scannumber, testshot, spec_folderpath, suffix)
    energy_arr, spec_charge_arr = MagSpecAnalysis.ParseInterpSpec(specpath)
    
    scanpath = "Scan"+"{:03d}".format(scannumber)
    num_shots = len(os.listdir(superpath + "/" + scanpath + "/" + spec_folderpath + "/"))
    
    shot_arr = np.array(range(num_shots))+1
    
    clipcheck_arr = np.zeros(num_shots)
    picoscopecharge_arr = np.zeros(num_shots)
    peakcharge_arr = np.zeros(num_shots)
    averageenergy_arr = np.zeros(num_shots)
    sigmaenergy_arr = np.zeros(num_shots)
    peakenergy_arr = np.zeros(num_shots)
    averagesize_arr = np.zeros(num_shots)
    projectedsize_arr = np.zeros(num_shots)
    beamtilt_arr = np.zeros(num_shots)
    
    for i in range(len(shot_arr)):
        if i%10==0:
            print(i/len(shot_arr)*100,"%")
        shotnumber = shot_arr[i]
        image = MagSpecAnalysis.LoadImage(superpath, scannumber, shotnumber, folderpath)
        charge_arr = MagSpecAnalysis.CalculateChargeDensityDistribution(image)
                    
        clipcheck = np.append(np.append(np.append(image[0,:],image[:,0]),image[-1,:]),image[:,-1])
        xmax, ymax, maxval = MagSpecAnalysis.FindMax(image)
        clipcheck_arr[i] = np.max(clipcheck)/maxval
            
        picoscopecharge_arr[i] =  MagSpecAnalysis.GetShotCharge(superpath, scannumber, shotnumber)
        peakcharge_arr[i] =    MagSpecAnalysis.CalculateMaximumCharge(charge_arr)
        averageenergy_arr[i] = MagSpecAnalysis.CalculateAverageEnergy(charge_arr, energy_arr)
        sigmaenergy_arr[i] =   MagSpecAnalysis.CalculateStandardDeviationEnergy(charge_arr, energy_arr)
        peakenergy_arr[i] =    MagSpecAnalysis.CalculatePeakEnergy(charge_arr, energy_arr)
    
        threshold = 0.01
        ave, proj, fit = MagSpecAnalysis.GetSizeStatistics_Full(image, energy_arr, threshold=threshold)
        averagesize_arr[i] = ave
        projectedsize_arr[i] = proj
        beamtilt_arr[i] = fit

tolerance_arr = np.array([0.0001, 0.01, 0.05])
for j in range(len(tolerance_arr)):
    tolerance = tolerance_arr[j]
    tol_arr = np.where(clipcheck_arr < tolerance)[0]
    print(len(tol_arr),"shots with clipped intensity <",tolerance*100,"%")

tolval = 0.05
tol_arr = np.where(clipcheck_arr < tolval)[0]

plot_peakcharge = peakcharge_arr[tol_arr]
plot_averageenergy = averageenergy_arr[tol_arr]
plot_sigmaenergy = sigmaenergy_arr[tol_arr]

plt.scatter(plot_peakcharge,plot_averageenergy,c=plot_sigmaenergy/plot_averageenergy)
plt.colorbar(label="RMS SEnergy Spread (%)")
plt.ylabel("Average Energy (MeV)")
plt.xlabel("Peak Charge Density "+r'$(\mathrm{pC/MeV})$')
plt.title("Sample Data: "+"Scan "+str(scannumber)+", Clipping Tolerance "+str(tolval*100)+"% ("+str(len(tol_arr))+" Shots)")
plt.show()
