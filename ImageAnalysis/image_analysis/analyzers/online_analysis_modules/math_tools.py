# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:03:37 2023

Functions for generic math and statistics

@author: chris
"""

import numpy as np
from scipy import optimize

def get_inequality_indices(input_list):
    #input_list=[['input_array1', 'input_sign1', 'input_compare1']]

    return_indices = None
    for i in range(len(input_list)):
        input_array = input_list[i][0]
        input_sign = input_list[i][1]
        input_compare = input_list[i][2]
        if input_sign == '=':
            inequality_indices = np.where(np.array(input_array) == np.array(input_compare))[0]
        elif input_sign == '!=':
            inequality_indices = np.where(np.array(input_array) != np.array(input_compare))[0]
        elif input_sign == '<':
            inequality_indices = np.where(np.array(input_array) < np.array(input_compare))[0]
        elif input_sign == '<=':
            inequality_indices = np.where(np.array(input_array) <= np.array(input_compare))[0]
        elif input_sign == '>':
            inequality_indices = np.where(np.array(input_array) > np.array(input_compare))[0]
        elif input_sign == '>=':
            inequality_indices = np.where(np.array(input_array) >= np.array(input_compare))[0]
        else:
            print("ERROR!!  Please use appropriate inequality sign:")
            print("         =      !=      <      <=      >      >=")
            inequality_indices = np.array([])

        if return_indices is None:
            return_indices = inequality_indices
        elif len(inequality_indices) > 0:
            merged_indices = np.array([])
            for j in range(len(return_indices)):
                for k in range(len(inequality_indices)):
                    if return_indices[j] == inequality_indices[k]:
                        merged_indices = np.append(merged_indices, return_indices[j])
            merged_indices = merged_indices.astype(int)
            return_indices = np.copy(merged_indices)

    return return_indices


def fit_data_something(data, axis, function, guess = [0., 0., 0.]):
    errfunc = lambda p, x, y: function(p, x) - y
    p0 = guess
    p1, success = optimize.leastsq(errfunc, p0[:], args=(axis, data))
    return p1

def func_linear(p, x):
    return p[0]*x + p[1]

def func_gaussian(p, x):
    return p[0]*np.exp(-.5*np.square(x-p[2])/np.square(p[1]))

def func_gaussian_offset(p, x):
    return p[0]*(np.exp(-.5*np.square(x-p[2])/np.square(p[1]))+p[3]/p[0])
