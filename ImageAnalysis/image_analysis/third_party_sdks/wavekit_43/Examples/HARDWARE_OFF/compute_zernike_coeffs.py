#! /usr/bin/python

#----

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

hasodata_file_path = './../DATAS/data_phase_computation.has'

"""Create ComputePhaseSet object to configure the phase computation
"""
compute_phase_set = wkpy.ComputePhaseSet(type_phase = wkpy.E_COMPUTEPHASESET.MODAL_ZERNIKE)
"""Load the hasoslopes contained into HAS file
"""
hasoslopes = wkpy.HasoSlopes(has_file_path = hasodata_file_path)

"""Create the pupil object from the hasoslopes
"""
pupil = wkpy.Pupil(hasoslopes = hasoslopes)
"""Compute the geometric parameters of the Zernike pupil from a cPupil input
"""
center, radius = wkpy.ComputePupil.fit_zernike_pupil(
    pupil,
    wkpy.E_PUPIL_DETECTION.AUTOMATIC,
    wkpy.E_PUPIL_COVERING.INSCRIBED,
    False
    )

"""Create the Zernike modal coeffs object and set preferences for modal phase reconstruction
"""
modal_coef = wkpy.ModalCoef(modal_type = wkpy.E_MODAL.ZERNIKE)
modal_coef.set_zernike_prefs(
    wkpy.E_ZERNIKE_NORM.STD,
    32,
    None,
    wkpy.ZernikePupil_t(
        center,
        radius
        )
    )

hasodata = wkpy.HasoData(hasoslopes = hasoslopes)
wkpy.Compute.coef_from_hasodata(compute_phase_set, hasodata, modal_coef)
"""Retrieve the Zernike coefficients given this pupil
"""
size = modal_coef.get_dim()
data_coeffs, data_indexes, pupil = modal_coef.get_data()
for i in range(size):
    print('zernike coeff at index' + str(data_indexes[i]) + ' : ' + str(data_coeffs[i]))
