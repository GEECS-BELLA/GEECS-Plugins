#! /usr/bin/python

#----

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

hasodata_file_path = './../DATAS/data_phase_computation.has'

"""Load an HAS file
"""
hasoslopes = wkpy.HasoSlopes(has_file_path = hasodata_file_path)

"""Create Phase object from HasoSlopes
"""
phase = wkpy.Phase(
    hasoslopes = hasoslopes,
    type_ = wkpy.E_COMPUTEPHASESET.ZONAL,
    filter_ = [True, True, True, True, True]
    )

"""Retrieve statistics datas
"""
stats = phase.get_statistics()
print('RMS value : '+str(stats.rms))
print('PV  value : '+str(stats.pv))
