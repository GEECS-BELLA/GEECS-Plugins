import msvcrt

#! /usr/bin/python

#----
""" WARNING ! This example is Hardware dependant.
Please ensure that hardware is connected,
And use your own Haso configuration file and acquisition system.
"""
""" WARNING ! If you have no Correction data backup file,
Please exit this example and run the correction_data example
"""
#----

print('WARNING ! If you have no Correction data backup file, \nPlease exit this example and run the correction_data example')

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

"""Get configuration file path
"""
wfc_correction_file_path = input('Correction data backup file path (.aoc) : \n')
print('Correction data backup file path set to :' + wfc_correction_file_path)
has_config_file_path = input('Haso configuration file path : \n')
print('Haso configuration file path set to :' + has_config_file_path)
wfc_config_file_path = input('Corrector configuration file path : \n')
print('Corrector configuration file path set to :' + wfc_config_file_path)

"""Create camera, wavefrontcorrector and corrdatamanager objects
"""
camera = wkpy.Camera(config_file_path = has_config_file_path)
wavefront_corrector = wkpy.WavefrontCorrector(config_file_path = wfc_config_file_path)
corr_data_manager = wkpy.CorrDataManager(haso_config_file_path = has_config_file_path, interaction_matrix_file_path = wfc_correction_file_path)
"""Connect hardware
"""
camera.connect()
camera.set_parameter_value('exposure_duration_us', 1000)
wavefront_corrector.connect(True)

"""Acquire an image
"""
image = camera.snap_raw_image()
start_subpupil = wkpy.uint2D(16,16)
ref_hasoslopes = wkpy.HasoSlopes(
    image = image,
    config_file_path = has_config_file_path,
    start_subpupil = start_subpupil
    )
wkpy.SlopesPostProcessor.apply_filter(
    ref_hasoslopes,
    False,
    False,
    False,
    True,
    True,
    True
    )
wavefront_corrector.set_temporization(20)

corr_data_manager.set_command_matrix_prefs(
    32,
    False
    )
corr_data_manager.compute_command_matrix()

loop_smoothing = wkpy.LoopSmoothing(level = "MEDIUM")
gain = 0.8

"""Loop
"""
try :
    while True:
        image = camera.snap_raw_image()
        hasoslopes = wkpy.HasoSlopes(
            image = image,
            config_file_path = has_config_file_path,
            start_subpupil = start_subpupil
            )
        delta_hasoslopes = wkpy.SlopesPostProcessor.apply_substractor(hasoslopes, ref_hasoslopes)
        phase = wkpy.Phase(
            hasoslopes = delta_hasoslopes,
            type_ = wkpy.E_COMPUTEPHASESET.ZONAL,
            filter_ = [True, True, True, True, True]
            )
        print ('RMS value : ' + str(phase.get_statistics().rms) + '\t(Press any key or Ctrl+C to stop the closed loop)')
    
        delta_commands, applied_gain = corr_data_manager.compute_closed_loop_iteration(
            delta_hasoslopes,
            False,
            loop_smoothing,
            gain
            )
        wavefront_corrector.move_to_relative_positions(
            delta_commands
            )
    
        if msvcrt.kbhit():
            break
except KeyboardInterrupt:
    pass
"""End of loop
"""
camera.stop()
camera.disconnect()
del loop_smoothing
del phase
del delta_hasoslopes
del ref_hasoslopes
del hasoslopes
del image
del corr_data_manager
del wavefront_corrector
del camera
