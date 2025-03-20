#! /usr/bin/python

#----

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

hasodata_file_path = './../DATAS/data_phase_computation.has'

hasoslopes = wkpy.HasoSlopes(has_file_path = hasodata_file_path)

"""Create the Zernike modal coeffs object and compute coefficients values from slopes
"""
modal_coef = wkpy.ModalCoef(
    modal_normalisation = wkpy.E_ZERNIKE_NORM.STD,
    nb_coeffs_total = 32,
    hasoslopes = hasoslopes)

"""Retrieve the Zernike coefficients given this pupil
"""
size = modal_coef.get_dim()
data_coeffs, data_indexes, pupil = modal_coef.get_data()
for i in range(size):
    print('zernike coeff at index' + str(data_indexes[i]) + ' : ' + str(data_coeffs[i]))
