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
print('Exposure duration requested : ' + str(exposure_duration) + 'us')
print('Exposure duration applied : '   + str(camera.get_parameter_value('exposure_duration_us')) + 'us')

"""Set the number of images to sum
"""
image_to_sum = 5
camera.set_nb_images_to_sum(image_to_sum)
print('Nb images summed : ' + str(camera.get_nb_images_to_sum()) + ' images')

"""Set camera timeout
"""
timeout = 4000
camera.set_timeout(timeout)
print('Timeout requested : ' + str(timeout)                  + 'ms')
print('Timeout applied : '   + str(camera.get_timeout()) + 'ms')

"""Start acquisition
"""
camera.start(
    wkpy.E_CAMERA_ACQUISITION_MODE.NEW,
    wkpy.E_CAMERA_SYNCHRONIZATION_MODE.SYNCHRONOUS
    )

"""Acquire Image
"""
image = camera.get_raw_image()

"""Save raw image
"""
image.save('../OUT_FILES/synchronous_acquisition_image_1000.himg', 'synchronous acquisition example')
print('Image captured saved as : synchronous_acquisition_image_1000.himg in Examples/OUT_FILES')

"""Set camera exposure duration, and check the value
"""
exposure_duration = 2000
camera.set_parameter_value('exposure_duration_us',exposure_duration)
print('Exposure duration requested : ' + str(exposure_duration) + 'us')
print('Exposure duration applied : '   + str(camera.get_parameter_value('exposure_duration_us')) + 'us')

"""Acquire Image
"""
image = camera.get_raw_image()

"""Save raw image
"""
image.save('../OUT_FILES/synchronous_acquisition_image_2000.himg', 'synchronous acquisition example')
print('Image captured saved as : synchronous_acquisition_image_2000.himg in Examples/OUT_FILES')

"""Stop camera and delete the camera object
"""
camera.stop()
camera.disconnect()
