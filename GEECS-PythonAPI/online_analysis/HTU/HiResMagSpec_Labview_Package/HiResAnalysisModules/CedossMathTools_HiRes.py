# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:03:37 2023

Functions for generic math and statistics

@author: chris
"""

import numpy as np
from scipy import optimize

def GetInequalityIndices(inputList):
    #inputList=[['inputArray1', 'inputSign1', 'inputCompare1']]

    returnIndices = None
    for i in range(len(inputList)):
        inputArray = inputList[i][0]
        inputSign = inputList[i][1]
        inputCompare = inputList[i][2]
        if inputSign == '=':
            inequalityIndices = np.where(np.array(inputArray) == np.array(inputCompare))[0]
        elif inputSign == '!=':
            inequalityIndices = np.where(np.array(inputArray) != np.array(inputCompare))[0]
        elif inputSign == '<':
            inequalityIndices = np.where(np.array(inputArray) < np.array(inputCompare))[0]
        elif inputSign == '<=':
            inequalityIndices = np.where(np.array(inputArray) <= np.array(inputCompare))[0]
        elif inputSign == '>':
            inequalityIndices = np.where(np.array(inputArray) > np.array(inputCompare))[0]
        elif inputSign == '>=':
            inequalityIndices = np.where(np.array(inputArray) >= np.array(inputCompare))[0]
        else:
            print("ERROR!!  Please use appropriate inequality sign:")
            print("         =      !=      <      <=      >      >=")
            inequalityIndices = np.array([])

        if returnIndices is None:
            returnIndices = inequalityIndices
        elif len(inequalityIndices) > 0:
            mergedIndices = np.array([])
            for j in range(len(returnIndices)):
                for k in range(len(inequalityIndices)):
                    if returnIndices[j] == inequalityIndices[k]:
                        mergedIndices = np.append(mergedIndices, returnIndices[j])
            mergedIndices = mergedIndices.astype(int)
            returnIndices = np.copy(mergedIndices)

    return returnIndices


def FitDataSomething(data, axis, function, guess = [0.,0.,0.], datlabel = 'Simulation',log=False,supress=False):
    errfunc = lambda p, x, y: function(p, x) - y
    p0 = guess
    p1, success = optimize.leastsq(errfunc, p0[:], args=(axis, data))
    return p1


def Linear(p, x):
    return p[0]*x + p[1]

def Gaussian(x, amp, sigma, x0):
    return amp * np.exp(-0.5 * np.square((x - x0) / sigma))

def Gaussian(p, x):
    return  p[0]*np.exp(-.5*np.square(x-p[2])/np.square(p[1]))

def GaussianOffset(p, x):
    return  p[0]*(np.exp(-.5*np.square(x-p[2])/np.square(p[1]))+p[3]/p[0])
