#! /usr/bin/python

#----
""" WARNING ! This example is Hardware dependant.
Please ensure that hardware is connected,
And use your own Haso configuration file and acquisition system.
"""
#----

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

"""Get configuration file path
"""
config_file_path = input('Haso Configuration file path : \n')
print('Configuration file path set to :' + config_file_path)

"""Camera constructor from Haso configuration file
"""
camera = wkpy.Camera(config_file_path = config_file_path)
camera.connect()

"""Set camera exposure duration, and check the value
"""
exposure_duration = 1000
camera.set_parameter_value('exposure_duration_us',exposure_duration)
print('Exposure duration requested :' + str(exposure_duration))
print('Exposure duration applied :'   + str(camera.get_parameter_value('exposure_duration_us')))

"""Set the number of images to sum
"""
image_to_sum = 5
camera.set_nb_images_to_sum(image_to_sum)
print('Nb images summed : ' + str(camera.get_nb_images_to_sum()))

"""Start acquisition
"""
camera.start(
    wkpy.E_CAMERA_ACQUISITION_MODE.LAST,
    wkpy.E_CAMERA_SYNCHRONIZATION_MODE.ASYNCHRONOUS
    )

"""Acquire Image
"""
is_image_ready = False
while( not(is_image_ready)):
    is_image_ready = camera.async_image_ready()
image = camera.get_raw_image()

"""Save raw image
"""
image.save('../OUT_FILES/asynchronous_acquisition_image.himg', 'asynchronous acquisition example')
print('Image captured saved as : asynchronous_acquisition_image.himg in Examples/OUT_FILES')

"""Wait for the camera to be receptive again, then stop camera and delete the camera object
"""
is_image_ready = False
while( not(is_image_ready)):
    is_image_ready = camera.async_image_ready()
camera.stop()
camera.disconnect()
