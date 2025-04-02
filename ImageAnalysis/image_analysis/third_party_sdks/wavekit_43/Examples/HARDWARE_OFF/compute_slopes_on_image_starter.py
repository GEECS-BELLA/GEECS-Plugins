#! /usr/bin/python

#----

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

config_file_path = './../DATAS/config_file_haso.dat'
image_file_path = './../DATAS/data_image.himg'

"""Create the needed objects : image, HasoEngine to compute the slopes and hasoslopes
"""
image = wkpy.Image(image_file_path = image_file_path)

"""By default the start subpupil is set by the one in the configuration file. 
it is close to the center of the sensor.
In this example it is arbitrary set to (20,20)
"""
start_subpupil = wkpy.uint2D(16,16)

hasoslopes = wkpy.HasoSlopes(
    image = image,
    config_file_path = config_file_path,
    start_subpupil = start_subpupil
    )

"""Save slopes
"""
save_file_path = './../OUT_FILES/data_phase_computation_example.has'
hasoslopes.save_to_file(
    save_file_path,
    '',
    ''
    )
print('HasoSlopes saved as '+'data_phase_computation_example.has'+' in Examples/OUT_FILES')