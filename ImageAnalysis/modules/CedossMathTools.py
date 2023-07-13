# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:03:37 2023

Functions for generic math and statistics

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def FitDataSomething(data, axis, function, guess = [0.,0.,0.], datlabel = 'Simulation',log=False,supress=False):
    errfunc = lambda p, x, y: function(p, x) - y
    p0 = guess
    p1, success = optimize.leastsq(errfunc,p0[:], args=(axis, data))
    if supress==False:
        if log==True:
            plt.semilogy(axis, data, label=datlabel)
        else:
            plt.plot(axis, data, label=datlabel)
        plt.plot(axis, function(p1,axis), label="Fitted "+ function.__name__ +" profile")
        plt.legend()
        plt.grid()
        plt.show()
    return p1

def Linear(p, x):
    return p[0]*x + p[1]

def Gaussian(p, x):
    return  p[0]*np.exp(-.5*np.square(x-p[2])/np.square(p[1]))

def GaussianOffset(p, x):
    return  p[0]*(np.exp(-.5*np.square(x-p[2])/np.square(p[1]))+p[3]/p[0])
