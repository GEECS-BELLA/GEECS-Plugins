import numpy as np
from scipy import signal

def integrate(data,dt,calib):
    value=np.array(data)
    bkg=np.mean(value[0:1000])
    value=value-bkg
    integrated_signal = np.trapz(value, x=None, dx=dt) 
    charge_pC = integrated_signal * calib
    
    return charge_pC
    
def calculate_centroid(data, dt):
    value = np.array(data)
    bkg = np.mean(value[0:500])  # Calculate DC offset using the first 500 samples
    value = value - bkg  # Subtract DC offset
    thresholded_values = np.where(value < 1, 0, value)
    times = np.arange(len(data)) * dt  # Create a time array that matches data array
    centroid = np.sum(times * thresholded_values) / np.sum(thresholded_values) if np.sum(thresholded_values) != 0 else 0
    return centroid
    
def Rm148_MLink(data,dt,crit_f):
    calib = 1
    energy = integrate(data,dt,calib)
    return energy
    
def BCave_MLink(data,dt,crit_f):
    return 0
    
def null_function(data,dt,crit_f):
    return 0
    
def centroid_function(data,dt,crit_f):
    cent = calculate_centroid(data,dt)
    return cent
    