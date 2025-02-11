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
wfs_config_file_path = input('Haso configuration file path : \n')
print('Haso configuration file path set to :' + wfs_config_file_path)
wfc_config_file_path = input('Corrector configuration file path : \n')
print('Corrector configuration file path set to :' + wfc_config_file_path)

"""Instantiate objects
"""
camera = wkpy.Camera(config_file_path = wfs_config_file_path)
wavefrontcorrector = wkpy.WavefrontCorrector(config_file_path = wfc_config_file_path)
corr_data_manager = wkpy.CorrDataManager(
    haso_config_file_path = wfs_config_file_path,
    wfc_config_file_path = wfc_config_file_path
    )

"""Connect to hardware
"""
camera.connect()
wavefrontcorrector.connect(True)    

push_pull_value = 0.2
corr_data_manager.set_calibration_prefs(push_pull_value)
size = corr_data_manager.get_calibration_matrix_size()
print('Calibration matrix size : ('+str(size.X)+' x '+str(size.Y)+')')

"""HasoSlopes array to store computed slopes
"""
specs = corr_data_manager.get_specifications()
hasoslopes_array = []
for i in range(2 * specs.nb_actuators):
    hasoslopes_array.append(
        wkpy.HasoSlopes(
            dimensions = specs.dimensions,
            serial_number = specs.haso_serial_number
            )
        )

"""Calibration process
"""
start_subpupil = wkpy.uint2D(16,16)
for i in range(specs.nb_actuators):
    prefs = corr_data_manager.get_actuator_prefs(i)
    if(prefs.validity==wkpy.E_ACTUATOR_CONDITIONS.VALID):
        push, pull = corr_data_manager.get_calibration_commands(i)
        """Get push interaction slopes
        """
        wavefrontcorrector.move_to_absolute_positions(push)
        image = camera.snap_raw_image()
        hasoslopes_array[(2 * i)] = wkpy.HasoSlopes(
            image = image,
            config_file_path = wfs_config_file_path,
            start_subpupil = start_subpupil
            )
        """Get pull interaction slopes
        """
        wavefrontcorrector.move_to_absolute_positions(pull)
        image = camera.snap_raw_image()
        hasoslopes_array[(2 * i) + 1] = wkpy.HasoSlopes(
            image = image,
            config_file_path = wfs_config_file_path,
            start_subpupil = start_subpupil
            )
        print('Calibration process experiment for actuator ' + str(i+1) + '/' + str(specs.nb_actuators) + ' succeed.')

"""Compute interaction matrix
"""
corr_data_manager.compute_interaction_matrix(hasoslopes_array)
corr_data_manager.save_backup_file('./../OUT_FILES/correction_data_backup_starter.aoc', 'Exemple Test')
print('Correction data saved to file correction_data_backup_starter.aoc in OUT_FILES directory.')

"""Set computation prefs
"""
nb_kept_modes = 32
corr_data_manager.set_command_matrix_prefs(
    nb_kept_modes,
    False
    )
corr_data_manager.compute_command_matrix()
"""Get vector of singular values
"""
influence_array = corr_data_manager.get_diagnostic_singular_vector()
for i in range(nb_kept_modes):
    print('Singular value at index '+str(i)+': '+str(influence_array[i]))

camera.stop()
camera.disconnect()

del camera
del wavefrontcorrector
del corr_data_manager
del hasoslopes_array
del image
