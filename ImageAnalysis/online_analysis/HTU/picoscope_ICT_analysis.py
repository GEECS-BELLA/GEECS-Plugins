import numpy as np
from scipy import signal

def apply_butterworth_filter(data, order=int(1), crit_f=0.025, filt_type='low'):
    
    #generate butterworth filter
    b, a = signal.butter(order, crit_f, filt_type)
    
    #apply filter in forward and backward propogation
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data
    
def identify_primary_valley(data):

    # find min value (assuming first peak is largest and negative)
    min_ind = np.argmin(data)
    
    # find where signal goes to zero before spike
    count = 1
    test_val = data[min_ind]
    try:
        while test_val < 0:
            # define test ind and val
            test_ind = min_ind - count
            test_val = data[test_ind]
            count += 1
        # define range minimum
        valley_min = int(test_ind + 1)
    except:
        valley_min = 0
    
    #find where signal goes to zero after spike
    count = 1
    test_val = data[min_ind]
    try:
        while test_val < 0:
            # define test ind and val
            test_ind = min_ind + count
            test_val = data[test_ind]
            #increment count
            count += 1
        # define range minimum
        valley_max = int(test_ind)
    except:
        valley_max = len(data)
    
    # define array of indices corresponding to spike
    valley_ind = np.arange(valley_min, valley_max)
    
    return valley_ind

def test_func(data,dt,crit_f,calib):
    value=np.array(data)
    bkg=np.mean(value[0:100])
    value=value-bkg
    value=np.array(apply_butterworth_filter(value,order=int(1),crit_f=crit_f))
    ind_roi = identify_primary_valley(value)
    value=np.array(value[ind_roi])
    integrated_signal = np.trapz(value, x=None, dx=dt) 
    charge_pC = integrated_signal* -calib * 10**(12)
    
    return charge_pC
    
def B_Cave_ICT(data,dt,crit_f):
    calib=0.2
    charge_pC=test_func(data,dt,crit_f,calib)
    return charge_pC
    
def Undulator_Exit_ICT(data,dt,crit_f):
    calib=0.2/2.78
    charge_pC=test_func(data,dt,crit_f,calib)
    return charge_pC
    