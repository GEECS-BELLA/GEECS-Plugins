#! /usr/bin/python

#----

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

hasodata_file_path = './../DATAS/data_phase_computation.has'

"""Load an HAS file
Create ComputePhaseSet setted on zonal reconstruction
"""
compute_phase_set = wkpy.ComputePhaseSet(type_phase = wkpy.E_COMPUTEPHASESET.ZONAL)
"""Create HasoData object
"""
hasodata = wkpy.HasoData(has_file_path = hasodata_file_path)

"""Create Phase object from computed HasoData
"""
dimensions = hasodata.get_dimensions()
phase = wkpy.Compute.phase_zonal(
    compute_phase_set,
    hasodata
    )

"""Retrieve statistics datas
"""
stats = phase.get_statistics()
print('RMS value : '+str(stats.rms))
print('PV  value : '+str(stats.pv))
