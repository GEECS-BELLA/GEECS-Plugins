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
wavefront_corrector_set = wkpy.WavefrontCorrectorSet(config_file_path = wfc_config_file_path)
corr_data_manager = wkpy.CorrDataManager(haso_config_file_path = has_config_file_path, interaction_matrix_file_path = wfc_correction_file_path)
"""Connect hardware
"""
camera.connect()
wavefront_corrector.connect(True)

"""Acquire an image
"""
camera.start(
    wkpy.E_CAMERA_ACQUISITION_MODE.NEW,
    wkpy.E_CAMERA_SYNCHRONIZATION_MODE.SYNCHRONOUS
    )
camera.set_parameter_value('exposure_duration_us', 1000)
image = camera.get_raw_image()

"""Compute reference slopes target tilt and focus
"""
hasoengine = wkpy.HasoEngine(config_file_path = has_config_file_path)
"""Get computed slopes
"""
hasoslopes = hasoengine.compute_slopes(
    image,
    False
    )[1] #Get only computed slopes
"""Create ref_hasoslopeserence slopes
"""
ref_hasoslopes = wkpy.HasoSlopes(hasoslopes = hasoslopes)
processor_list = wkpy.SlopesPostProcessorList()
processor_list.insert_filter(0, False, False, False, True, True, True)

hasodata = wkpy.HasoData(hasoslopes = ref_hasoslopes)
hasodata.apply_slopes_post_processor_list(processor_list)
ref_hasoslopes = hasodata.get_hasoslopes()[0] #Get only computed slopes

processor_list.delete_processor(0)
processor_list.insert_substractor(0, ref_hasoslopes, "")

hasodata.set_hasoslopes(hasoslopes)
hasodata.apply_slopes_post_processor_list(processor_list)    
delta_hasoslopes = hasodata.get_hasoslopes()[0]

"""Set wavefrontcorrector pref_hasoslopeserences
"""
wavefront_corrector.set_temporization(20)

"""Compute command matrix
"""
corr_data_manager.set_command_matrix_prefs(32, False)
corr_data_manager.compute_command_matrix()

"""Loop with RMS printing
"""
compute_phase_set = wkpy.ComputePhaseSet(type_phase = wkpy.E_COMPUTEPHASESET.ZONAL)
loop_smoothing = wkpy.LoopSmoothing(level = "MEDIUM")
gain = 0.8

"""Loop
"""
try :
    while True:
        image = camera.get_raw_image()
        hasoslopes = hasoengine.compute_slopes(
            image, 
            False
            )[1]
        hasodata.set_hasoslopes(hasoslopes)
        hasodata.apply_slopes_post_processor_list(processor_list)
        delta_hasoslopes = hasodata.get_hasoslopes()[0]
        phase = wkpy.Compute.phase_zonal(compute_phase_set, hasodata)
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
del compute_phase_set
del phase
del delta_hasoslopes
del hasodata
del processor_list
del ref_hasoslopes
del hasoengine
del hasoslopes
del image
del corr_data_manager
del wavefront_corrector_set
del wavefront_corrector
del camera
