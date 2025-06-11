#! /usr/bin/python

#----

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

hasodata_file_path = './../DATAS/data_phase_computation.has'
config_file_path = './../DATAS/config_file_haso.dat'

"""Get HasoSlopes from HAS file
"""
hasoslopes = wkpy.HasoSlopes(has_file_path = hasodata_file_path)

"""Get Pupil object from HasoSlopes object
"""
pupil = wkpy.Pupil(hasoslopes = hasoslopes)
"""Get HasoData from HasoSlopes object
"""
hasodata = wkpy.HasoData(hasoslopes = hasoslopes)

"""Create cComputePhaseSet object to configure the phase computation.
Set TypePhase = E_COMPUTEPHASESET_ZONAL to use zonal reconstruction
"""
compute_phase_set = wkpy.ComputePhaseSet(type_phase = wkpy.E_COMPUTEPHASESET.ZONAL)
"""Compute phase
"""
phase = wkpy.Compute.phase_zonal(compute_phase_set, hasodata)
properties = hasoslopes.get_geometric_properties()

"""HasoField constructor from HasoSlopes and Phase
"""
hasofield = wkpy.HasoField(
    config_file_path = config_file_path,
    hasoslopes = hasoslopes,
    phase = phase,
    curv_radius = properties[2], #Radius curvature
    wavelength = 675,
    oversampling = 0
    )

wavelength = hasofield.get_wavelength()

"""Surface constructor from dimensions and steps
Dimensions may be set to dummy values (> 0) since the surface will be correctly resized by Imop_HasoField_PSF
"""
psf_surface = hasofield.psf(
    False,
    True,
    0.0,
    config_file_path
    )
statistics = psf_surface.get_statistics()
print('PSF min : '+str(statistics.min))
print('PSF max : '+str(statistics.max))
