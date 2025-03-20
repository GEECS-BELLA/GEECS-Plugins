#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_phase as imop_phase
import io_wavekit_hasoslopes as imop_hslp

class CorrDataManager(object):
    """Class CorrDataManager
    
    - Constructor from Config Files :
        - **haso_config_file_path** - string : Absolute path to Haso configuration file (\*.dat)
        - **wfc_config_file_path** - string : Absolute path to WavefrontCorrector configuration file (\*.dat)
    
    - Constructor from Backup File :
        - **haso_config_file_path** - string : Absolute path to Haso configuration file (\*.dat)
        - **interaction_matrix_file_path** - string : Absolute path to CorrDataManager dump file (\*.aoc)
    
    - Constructor from Copy :
        - **corrdatamanager** - CorrDataManager : CorrDataManager object to copy
    """
        
    def __init_from_configs_files(
        self,
        haso_config_file_path,
        wfc_config_file_path
        ):
        """CorrDataManager constructor from a Haso configuration file and a wavefront configuration file.
        
        :param haso_config_file_path: Absolute path to Haso configuration file
        :type haso_config_file_path: string
        :param wfc_config_file_path: Absolute path to WavefrontCorrector configuration file
        :type wfc_config_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_NewFromConfigsFiles(
                message,
                ctypes.pointer(self.corrdatamanager),
                ctypes.c_char_p(haso_config_file_path.encode('utf-8')),
                ctypes.c_char_p(wfc_config_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_configs_files',exception)

    def __init_from_backup_file(
        self,
        haso_config_file_path,
        interaction_matrix_file_path
        ):
        """CorrDataManager constructor from a CorrDataManager dump file.
        
        :param haso_config_file_path: Absolute path to Haso configuration file
        :type haso_config_file_path: string
        :param interaction_matrix_file_path:  Absolute path to CorrDataManager dump file
        :type interaction_matrix_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_NewFromBackupFile(
                message,
                ctypes.pointer(self.corrdatamanager),
                ctypes.c_char_p(haso_config_file_path.encode('utf-8')),
                ctypes.c_char_p(interaction_matrix_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_backup_file',exception)

    def __init_from_copy(
        self,
        corrdatamanager
        ):
        """CorrDataManager copy constructor.
        
        :param corrdatamanager: CorrDataManager object to be copied
        :type corrdatamanager: CorrDataManager
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_NewFromCopy(
                message,
                corrdatamanager.corrdatamanager,
                ctypes.pointer(self.corrdatamanager)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_copy',exception)

    def __init__(self, **kwargs):
        """CorrDataManager constructor
        """
        self.corrdatamanager = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('corrdatamanager' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['corrdatamanager'])
            if(arg_size == 2):
                if('haso_config_file_path' in kwargs
                   and 'wfc_config_file_path' in kwargs
                   ):
                    entered = 1
                    self.__init_from_configs_files(kwargs['haso_config_file_path'], kwargs['wfc_config_file_path'])
                if('haso_config_file_path' in kwargs
                   and 'interaction_matrix_file_path' in kwargs
                   ):
                    entered = 1
                    self.__init_from_backup_file(kwargs['haso_config_file_path'], kwargs['interaction_matrix_file_path'])
        except Exception as exception :
            raise Exception(__name__+' : init',exception)
        if(entered == 0):            
            raise Exception(__name__+' : init','---CAN NOT CREATE CORRDATAMANAGER OBJECT---')

    def __del_obj__(self):
        """CorrDataManager copy destructor.
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_CorrDataManager_Delete(message, self.corrdatamanager)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_specifications(self):
        """Haso and Wavefront corrector specifications getter.
        
        :return: Haso specifications : serial number, dimensions; Wavefront corrector specifications : serial number, name, actuators count
        :rtype: CorrDataManagerSpecs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb_actuators = ctypes.c_int()
            wavefrontcorrector_name = ctypes.create_string_buffer(256)
            wavefrontcorrector_serial_number = ctypes.create_string_buffer(256)
            haso_slopes_size = imop_struct.uint2D(0, 0)
            haso_slopes_step = imop_struct.float2D(0.0, 0.0)
            haso_serial_number = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_GetSpecifications(
                message,
                self.corrdatamanager,
                ctypes.byref(nb_actuators),
                wavefrontcorrector_name,
                wavefrontcorrector_serial_number,
                ctypes.byref(haso_slopes_size),
                ctypes.byref(haso_slopes_step),
                haso_serial_number
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.CorrDataManagerSpecs_t(
                haso_serial_number.value.decode('utf-8'),
                imop_struct.dimensions(haso_slopes_size, haso_slopes_step),
                wavefrontcorrector_serial_number.value.decode('utf-8'),
                wavefrontcorrector_name.value.decode('utf-8'),
                nb_actuators.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_specifications',exception)

    def load_backup_file(
        self,
        backup_file_path
        ):
        """Load data extracted from a CorrDataManager backup file into current CorrDataManager object.
        
        :param backup_file_path: Absolute path to CorrDataManager backup file (\*.aoc)
        :type backup_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_LoadBackupFile(
                message,
                self.corrdatamanager,
                ctypes.c_char_p(backup_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : load_backup_file',exception)

    def save_backup_file(
        self,
        backup_file_path,
        comments
        ):
        """Dump CorrDataManager data to file.
        
        :param backup_file_path: Absolute path to CorrDataManager backup file (\*.aoc)
        :type backup_file_path: string
        :param comments: User-defined comment to write to backup file
        :type comments: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_SaveBackupFile(
                message,
                self.corrdatamanager,
                ctypes.c_char_p(backup_file_path.encode('utf-8')),
                ctypes.c_char_p(comments.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : save_backup_file',exception)

    def set_microlenses_focal_length(
        self,
        microlenses_focal_length_um
        ):
        """Set Micro lenses focal length.
        Provided for backward compatibility with old formatted interaction matrix backup files.
        This value is used for correction loop computations.
        If a focal length value has already been set in the correction data manager, it won't be replaced.

        :param backup_file_path: Micro lenses focal length (um)
        :type backup_file_path: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_SetMicrolensesFocalLength(
                message,
                self.corrdatamanager,
                ctypes.c_float(microlenses_focal_length_um)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_microlenses_focal_length',exception)

    def reset_interaction_matrix(
        self
        ):
        """Reset Interaction matrix, greatest common pupil and Wavefront corrector preferences to the last stable state.
        Restore the CorrDataManager to last interaction matrix computation state or last loading from a backup file.

        .. warning:: This function clears command matrix and diagnostics.
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_ResetInteractionMatrix(
                message,
                self.corrdatamanager
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : reset_interaction_matrix',exception)

    def get_actuators_count(
        self
        ):
        """Get Wavefront corrector actuators number.

        :return: WavefrontCorrector actuators count
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            actuator_count = ctypes.c_int()
            self.dll.Imop_CorrDataManager_GetActuatorsCount(
                message,
                self.corrdatamanager,
                ctypes.byref(actuator_count)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return actuator_count.value
        except Exception as exception:
            raise Exception(__name__+' : get_actuators_count',exception)

    def get_actuator_prefs(
        self,
        actuator_index
        ):
        """Get Preferences for one wavefront corrector actuator.
        
        .. seealso:: WavefrontCorrector.get_preferences for details on wavefront corrector preferences

        :param actuator_index: Index of the actuator
        :type actuator_index: int
        :return: lowest, highest command value, status, fixed position
        :rtype: ActuatorPrefs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            min_ = ctypes.c_float()
            max_ = ctypes.c_float()
            validity = ctypes.c_int()
            fixed_value = ctypes.c_float()
            self.dll.Imop_CorrDataManager_GetActuatorPrefs(
                message,
                self.corrdatamanager,
                ctypes.c_int(actuator_index),
                ctypes.byref(min_),
                ctypes.byref(max_),
                ctypes.byref(validity),
                ctypes.byref(fixed_value)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.ActuatorPrefs_t(
                min_.value,
                max_.value,
                validity.value,
                fixed_value.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_actuator_prefs',exception)

    def get_corrector_init_prefs(self):
        """Get Wavefront corrector actuators preferences.
        
        .. seealso:: WavefrontCorrector.get_preferences for details on wavefront corrector preferences

        :return: Corrector preferences : lowest command, highest command, validity, fixed positions for each actuator
        :rtype: CorrectorPrefs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb_actuators = self.get_actuators_count()
            min_ = numpy.zeros(nb_actuators, dtype = numpy.single)
            max_ = numpy.zeros(nb_actuators, dtype = numpy.single)
            validity = numpy.zeros(nb_actuators, dtype = numpy.intc)
            fixed_value = numpy.zeros(nb_actuators, dtype = numpy.single)
            self.dll.Imop_CorrDataManager_GetCorrectorInitPrefs.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
            ]
            self.dll.Imop_CorrDataManager_GetCorrectorInitPrefs(
                message,
                self.corrdatamanager,
                min_,
                max_,
                validity,
                fixed_value
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.CorrectorPrefs_t(
                    min_.tolist(),
                    max_.tolist(),
                    validity.tolist(),
                    fixed_value.tolist()
                )
        except Exception as exception:
            raise Exception(__name__+' : get_corrector_init_prefs',exception)
            
    def get_corrector_prefs(self):
        """Get Wavefront corrector actuators preferences.
        
        .. seealso:: WavefrontCorrector.get_preferences for details on wavefront corrector preferences

        :return: Corrector preferences : lowest command, highest command, validity, fixed positions for each actuator
        :rtype: CorrectorPrefs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb_actuators = self.get_actuators_count()
            min_ = numpy.zeros(nb_actuators, dtype = numpy.single)
            max_ = numpy.zeros(nb_actuators, dtype = numpy.single)
            validity = numpy.zeros(nb_actuators, dtype = numpy.intc)
            fixed_value = numpy.zeros(nb_actuators, dtype = numpy.single)
            self.dll.Imop_CorrDataManager_GetCorrectorPrefs.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
            ]
            self.dll.Imop_CorrDataManager_GetCorrectorPrefs(
                message,
                self.corrdatamanager,
                min_,
                max_,
                validity,
                fixed_value
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.CorrectorPrefs_t(
                    min_.tolist(),
                    max_.tolist(),
                    validity.tolist(),
                    fixed_value.tolist()
                )
        except Exception as exception:
            raise Exception(__name__+' : get_corrector_prefs',exception)

    def set_actuator_prefs(
        self,
        actuator_index,
        actuator_prefs
        ):
        """Set Preferences for one wavefront corrector actuator.
        The new preferences have to be compliant with the Wavefront corrector specifications.
        If the new preferences are compliant with the one used for the last calibration process or the last loaded ones, \
        the interaction matrix is recomputed,
        else the interaction matrix is erased and a new calibration process (or loading from file or backup restoration) is needed.
        
        .. seealso:: WavefrontCorrector.get_preferences for details on wavefront corrector preferences

        :param actuator_index: Index of the actuator
        :type actuator_index: int
        :param actuator_prefs: Structure containing lowest, highest command, validity value, fixed position
        :type actuator_prefs: ActuatorPrefs_t
        :return: if true, interaction matrix has been erased
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            new_data = ctypes.c_bool()
            self.dll.Imop_CorrDataManager_SetActuatorPrefs(
                message,
                self.corrdatamanager,
                ctypes.c_int(actuator_index),
                ctypes.c_float(actuator_prefs.min_value),
                ctypes.c_float(actuator_prefs.max_value),
                ctypes.c_int(actuator_prefs.validity),
                ctypes.c_float(actuator_prefs.fixed_value),
                ctypes.byref(new_data)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return new_data.value
        except Exception as exception:
            raise Exception(__name__+' : set_actuator_prefs',exception)
            
    def set_corrector_prefs(
        self,
        prefs
        ):
        """Set Wavefront corrector actuators preferences.
        The new preferences have to be compliant with the Wavefront corrector specifications \
        read from configuration file).
        If the new preferences are compliant with the one used for the last calibration process \
        or the last loaded ones, the interaction matrix is recomputed,
        else the interaction matrix is erased and a new calibration process \
        (or loading from file or backup restoration) is needed.
        
        .. seealso:: WavefrontCorrector.get_preferences for details on wavefront corrector preferences

        :param prefs_lists: Structure containing lowest, highest command values, validity values, fixed positions lists
        :type prefs_lists: CorrectorPrefs_t
        :return: if true, indicates that interaction matrix has been erased
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            new_data_needed = ctypes.c_bool()
            self.dll.Imop_CorrDataManager_SetCorrectorPrefs.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ]
            self.dll.Imop_CorrDataManager_SetCorrectorPrefs(
                message,
                self.corrdatamanager,
                numpy.array(prefs.min_array, dtype = numpy.single),
                numpy.array(prefs.max_array, dtype = numpy.single),
                numpy.array(prefs.validity_array, dtype = numpy.intc),
                numpy.array(prefs.fixed_value_array, dtype = numpy.single),
                ctypes.byref(new_data_needed)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return new_data_needed.value
        except Exception as exception:
            raise Exception(__name__+' : set_corrector_prefs',exception)
            
    def check_relative_position_saturation(
        self,
        current_positions_array,
        delta_commands_array
        ):
        """Count the number of actuators which will be saturated \
        (reaching min or max of their positions) if relative command is applied

        :param current_positions_array: List of current wavefront corrector positions
        :type current_positions_array: float
        :param delta_commands_array: List of delta commands
        :type delta_commands_array: float
        :return: Count of actuators which will be at min or max of their positions if command is applied
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            saturated_act_count = ctypes.c_int()
            self.dll.Imop_CorrDataManager_CheckRelativePositionsSaturation.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ]
            self.dll.Imop_CorrDataManager_CheckRelativePositionsSaturation(
                message,
                self.corrdatamanager,
                numpy.array(current_positions_array, dtype = numpy.single),
                numpy.array(delta_commands_array, dtype = numpy.single),
                ctypes.byref(saturated_act_count)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return saturated_act_count.value
        except Exception as exception:
            raise Exception(__name__+' : check_relative_position_saturation',exception)
            
    def set_calibration_prefs(
        self,
        push_pull_value
        ):
        """Calibration process supposes back and forth moves on all actuators.
        Use this function to calibrate the amplitude of these moves.

        :param push_pull_value: In case experiment type is "push-pull" (type 0), indicates the actuators displacement amplitude
        :type push_pull_value: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_SetCalibrationPrefs(
                message,
                self.corrdatamanager,
                ctypes.c_float(push_pull_value)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_calibration_prefs',exception)
            
    def get_calibration_commands(
        self,
        actuator_index
        ):
        """Get the two wavefront corrector commands corresponding to the actuator actuator_index calibration.
        Command application must be performed using the functions of WFC.
        Corresponding HasoSlopes must be computed on appropriate Image captured by the Haso sensor,
        Function returns an error if actuator is invalid. If so, please do not move the actuator and continue with the next one.

        :param actuator_index: Index of the actuator
        :type actuator_index: int
        :return: Wavefront corrector commands associated to the first move (push), Wavefront corrector commands associated to the second move (pull)
        :rtype: tuple(float list[], float list[])
        """
        try:
            nb_actuators = self.get_specifications().nb_actuators
            message = ctypes.create_string_buffer(256)
            first_command = numpy.zeros(nb_actuators, dtype = numpy.single)
            second_command = numpy.zeros(nb_actuators, dtype = numpy.single)
            self.dll.Imop_CorrDataManager_GetCalibrationCommands.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_int,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_CorrDataManager_GetCalibrationCommands(
                message,
                self.corrdatamanager,
                ctypes.c_int(actuator_index),
                first_command,
                second_command               
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                first_command.tolist(),
                second_command.tolist()
                )
        except Exception as exception:
            raise Exception(__name__+' : get_calibration_commands',exception)
            
    def get_calibration_matrix_size(self):
        """Get Calibration matrix size.
        
        :return: Size of calibration matrix
        :rtype: uint2D
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = imop_struct.uint2D(0, 0)
            self.dll.Imop_CorrDataManager_GetCalibrationMatrixSize(
                message,
                self.corrdatamanager,
                ctypes.byref(size)                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size
        except Exception as exception:
            raise Exception(__name__+' : get_calibration_matrix_size',exception)
            
    def compute_interaction_matrix(
        self,
        hasoslopes_list
        ):
        """Compute interaction matrix from collected calibration data (two HasoSlopes measured per actuator).
        
        .. seealso:: Calibration Matrix computation details
        
        :param hasoslopes_list: HasoSlopes measured for each calibration step.
        :type hasoslopes_list: HasoSlopes list[] 
        """
        try:
            message = ctypes.create_string_buffer(256)
            array_size = len(hasoslopes_list)
            hasoslopes_array = (ctypes.c_void_p*array_size)()
            for i in range(array_size):
                hasoslopes_array[i] = hasoslopes_list[i].hasoslopes
            self.dll.Imop_CorrDataManager_ComputeInteractionMatrix(
                message,
                self.corrdatamanager,
                ctypes.c_int(array_size),
                ctypes.pointer(hasoslopes_array)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : compute_interaction_matrix',exception)
            
    def update_interaction_matrix(
        self,
        index,
        hasoslopes_push,
        hasoslopes_pull
        ):
        """Compute interaction matrix from collected calibration data (two HasoSlopes measured per actuator).
        
        :param index: Index of the actuator
        :type index: int
        :param hasoslopes_push: HasoSlopes measured for the push movement of the actuator
        :type hasoslopes_push: HasoSlopes
        :param hasoslopes_pull: HasoSlopes measured for the pull movement of the actuator
        :type hasoslopes_pull: HasoSlopes
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_UpdateInteractionMatrix(
                message,
                self.corrdatamanager,
                ctypes.c_int(index),
                hasoslopes_push.hasoslopes,
                hasoslopes_pull.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : update_interaction_matrix',exception)
            
    def get_interaction_matrix_size(self):
        """Get Interaction matrix size.
        
        :return: Size of interaction matrix
        :rtype: uint2D
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = imop_struct.uint2D(0, 0)
            self.dll.Imop_CorrDataManager_GetInteractionMatrixSize(
                message,
                self.corrdatamanager,
                ctypes.byref(size)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size
        except Exception as exception:
            raise Exception(__name__+' : get_interaction_matrix_size',exception)
            
    def get_interaction_matrix_buffer(self):
        """Get Interaction matrix content.
        
        :return: 2D Array of interaction matrix content
        :rtype: 2D float numpy array
        """
        try:
            message = ctypes.create_string_buffer(256)
            data_size = self.get_interaction_matrix_size()
            data = numpy.zeros((data_size.Y,data_size.X), dtype = numpy.single)
            self.dll.Imop_CorrDataManager_GetInteractionMatrixBuffer.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_CorrDataManager_GetInteractionMatrixBuffer(
                message,
                self.corrdatamanager,
                data
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return data
        except Exception as exception:
            raise Exception(__name__+' : get_interaction_matrix_buffer',exception)
            
    def set_command_matrix_prefs(
        self,
        nb_kept_modes,
        tilt_filtering = False
        ):
        """Set Command matrix computation preferences.
        
        :param nb_kept_modes: Number of influences (starting from the highest) to keep
        :type nb_kept_modes: int
        :param tilt_filtering: If True, computed command matrix does not take tilt into account
        :type tilt_filtering: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_SetCommandMatrixPrefs(
                message,
                self.corrdatamanager,
                ctypes.c_int(nb_kept_modes),
                ctypes.c_bool(tilt_filtering)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_command_matrix_prefs',exception)
            
    def compute_command_matrix(self):
        """Computes command matrix.
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_ComputeCommandMatrix(
                message,
                self.corrdatamanager
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : compute_command_matrix',exception)
            
    def get_command_matrix_size(self):
        """Get Command matrix size.
        
        :return: Size of command matrix
        :rtype: uint2D
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = imop_struct.uint2D(0, 0)
            self.dll.Imop_CorrDataManager_GetCommandMatrixSize(
                message,
                self.corrdatamanager,
                ctypes.byref(size)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size
        except Exception as exception:
            raise Exception(__name__+' : get_command_matrix_size',exception)
            
    def get_command_matrix_buffer(self):
        """Get Command matrix content.
        
        :return: 2D Array of command matrix content
        :rtype: 2D float numpy array
        """
        try:
            message = ctypes.create_string_buffer(256)
            data_size = self.get_command_matrix_size()
            data = numpy.zeros((data_size.Y, data_size.X), dtype = numpy.single)
            self.dll.Imop_CorrDataManager_GetCommandMatrixBuffer.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_CorrDataManager_GetCommandMatrixBuffer(
                message,
                self.corrdatamanager,
                data
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return data
        except Exception as exception:
            raise Exception(__name__+' : get_command_matrix_buffer',exception)
            
    def get_greatest_common_pupil(self):
        """Compute the greatest common pupil.
        
        :return: Greatest common pupil as a 2D Array
        :rtype: 2D bool numpy array
        """
        try:
            message = ctypes.create_string_buffer(256)
            data_size = self.get_specifications().dimensions.size
            data = numpy.zeros((data_size.Y, data_size.X), dtype = numpy.bool_)
            self.dll.Imop_CorrDataManager_GetGreatestCommonPupil.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.bool_, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_CorrDataManager_GetGreatestCommonPupil(
                message,
                self.corrdatamanager,
                data
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return data
        except Exception as exception:
            raise Exception(__name__+' : get_greatest_common_pupil',exception)
            
    def apply_pupil_mask(
        self,
        pupil_mask
        ):
        """Set the greatest common pupil to the intersection between itself and the pupil_mask.
        If the pupil_mask has more "1" values than the current greatest common pupil, it won't have any effect on it (Reduce pupil size only).
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_ApplyPupilMask(
                message,
                self.corrdatamanager,
                pupil_mask.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : apply_pupil_mask',exception)
            
    def remove_pupil_mask(self):
        """Restore the greatest common pupil.
        Cancel the effects of pupil mask(s) : \
        Restore the greatest common pupil to last interaction matrix computation value or last loading from a backup value.
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CorrDataManager_RemovePupilMask(
                message,
                self.corrdatamanager
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : remove_pupil_mask',exception)
            
    def get_actuator_influence(
        self,
        actuator_index,
        use_actuator_pupil = False
        ):
        """Get actuator influence as HasoSlopes.
        The influence pupil may be the pupil where actuator influence has been computed or the greatest common pupil of all the actuators 
        (CorrDataManager greatest common pupil).
        Returns 0 HasoSlopes if actuator in not valid.
        
        :param actuator_index: Actuator index
        :type actuator_index: int
        :param use_actuator_pupil: If true, influence pupil will be the actuator's one, else it will be the corrdata_manager one
        :type use_actuator_pupil: bool
        :return: HasoSlopes object
        :rtype: HasoSlopes
        """
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = imop_hslp.HasoSlopes(
                dimensions = self.get_specifications().dimensions,
                serial_number = self.get_specifications().haso_serial_number
                )
            self.dll.Imop_CorrDataManager_GetActuatorInfluence(
                message,
                self.corrdatamanager,
                ctypes.c_int(actuator_index),
                ctypes.c_bool(use_actuator_pupil),
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out
        except Exception as exception:
            raise Exception(__name__+' : get_actuator_influence',exception)
            
    def get_diagnostic_singular_vector(self):
        """Get vector of singular values.
        
        :return: Array of influences, sorted in descending order.
        :rtype: float list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            data = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            self.dll.Imop_CorrDataManager_GetDiagnosticSingularVector.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_CorrDataManager_GetDiagnosticSingularVector(
                message,
                self.corrdatamanager,
                data
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return data.tolist()
        except Exception as exception:
            raise Exception(__name__+' : get_diagnostic_singular_vector',exception)
            
    def get_diagnostic_eigen_vector(
        self,
        mode_index
        ):
        """Get HasoSlopes object and commands array associated to an influence.
        
        :param mode_index: Influence index
        :type mode_index: int
        :return: HasoSlopes object, Command array
        :rtype: tuple(HasoSlopes, float list[])
        """
        try:
            message = ctypes.create_string_buffer(256)
            data = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            hasoslopes_out = imop_hslp.HasoSlopes(
                dimensions = self.get_specifications().dimensions,
                serial_number = self.get_specifications().haso_serial_number
                )
            self.dll.Imop_CorrDataManager_GetDiagnosticEigenVectors.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p, 
                ctypes.c_int,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_CorrDataManager_GetDiagnosticEigenVectors(
                message,
                self.corrdatamanager,
                ctypes.c_int(mode_index),
                hasoslopes_out.hasoslopes,
                data
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                hasoslopes_out,
                data.tolist()
                )
        except Exception as exception:
            raise Exception(__name__+' : get_diagnostic_eigen_vector',exception)
            
    def compute_delta_command_from_delta_slopes(
        self,
        delta_slopes
        ):
        """Compute the relative wavefront corrector commands corresponding to a delta_slopes variation.
        This function uses the CorrDataManager command matrix and wavefront corrector preferences to compute the output command.
        Typical application is to compute the delta slopes as the difference between the current measured slopes
        and references slopes (for example, references slopes can be created from a particular set of Zernike coefficients,
        see Imop_HasoSlopes_NewFromModalCoef).
        
        :param delta_slopes: Slopes delta (stored as HasoSlopes)
        :type delta_slopes: HasoSlopes
        :return: Array of size actuators count, containing the commands deltas (relatives variations to apply from actuators initial positions)
        :rtype: float list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            delta_commands = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            self.dll.Imop_CorrDataManager_ComputeDeltaCommandFromDeltaSlopes.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_CorrDataManager_ComputeDeltaCommandFromDeltaSlopes(
                message,
                self.corrdatamanager,
                delta_slopes.hasoslopes,
                delta_commands
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return delta_commands.tolist()
        except Exception as exception:
            raise Exception(__name__+' : compute_delta_command_from_delta_slopes',exception)
            
    def compute_command_from_modalcoef(
        self,
        modalcoef
        ):
        """Compute the relative wavefront corrector commands corresponding to a modal coefficients data.
        This function uses the CorrDataManager command matrix and wavefront corrector preferences to compute the output command.
        Typical application is to compute commands from slopes computed from modal coeffecients data.
        
        :param modalcoef: ModalCoef 
        :type modalcoef: ModalCoef
        :return: Array of size actuators count, containing the commands deltas (relatives variations to apply from actuators initial positions)
        :rtype: float list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            delta_commands = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            self.dll.Imop_CorrDataManager_ComputeCommandFromModalCoefs.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_CorrDataManager_ComputeCommandFromModalCoefs(
                message,
                self.corrdatamanager,
                modalcoef.modalcoef,
                delta_commands
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return delta_commands.tolist()
        except Exception as exception:
            raise Exception(__name__+' : compute_command_from_modalcoef',exception)
            
    def compute_closed_loop_iteration(
        self,
        delta_slopes,
        do_tilt_check,
        smoothing_params,
        specified_loop_gain
        ):
        """Compute the relative wavefront corrector commands corresponding to a delta_phase variation.
        This function uses the CorrDataManager command matrix and wavefront corrector preferences to compute the output command.
        Typical application is to compute the delta phase as the difference between the current measured phase
        and reference phase (for example, reference phase can be created from a particular set of Zernike coefficients).
        
        :param delta_slopes: Array of target slopes deltas (stored as HasoSlopes)
        :type delta_slopes: HasoSlopes
        :param do_tilt_check: Indicates if the computed commands deltas must be adapted to avoid huge tilts
        :type do_tilt_check: bool
        :param smoothing_params: Closed Loop smoothing params object
        :type smoothing_params: LoopSmoothing
        :param specified_loop_gain: User specified loop smoothing gain
        :type specified_loop_gain: float
        :return: Array of size actuators count, containing the commands deltas (relatives variations to apply from actuators initial positions), Loop smoothing gain really applied
        :rtype: tuple(float list[], float)
        """
        try:
            message = ctypes.create_string_buffer(256)
            delta_command = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            applied_loop_gain = ctypes.c_float()
            self.dll.Imop_CorrDataManager_ComputeClosedLoopIteration.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_void_p
                ]
            self.dll.Imop_CorrDataManager_ComputeClosedLoopIteration(
                message,
                self.corrdatamanager,
                delta_slopes.hasoslopes,
                delta_command,
                ctypes.c_bool(do_tilt_check),
                smoothing_params.loopsmoothing,
                ctypes.c_float(specified_loop_gain),
                ctypes.byref(applied_loop_gain)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                delta_command.tolist(),
                applied_loop_gain.value
                )
        except Exception as exception:
            raise Exception(__name__+' : compute_closed_loop_iteration',exception)
            
    def compute_slopes_from_command(
        self,
        command_array
        ):
        """Simulation of the expected slopes given a wavefront corrector command.
        Function returns an error if command is not compliant with the current corrdata_manager wavefront corrector preferences.
        
        :param command_array: Array of size actuators count, containing for each actuator its current position
        :type command_array: float list[]
        :return: HasoSlopes object, where the simulated slopes are stored
        :rtype: HasoSlopes
        """
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = imop_hslp.HasoSlopes(
                dimensions = self.get_specifications().dimensions,
                serial_number = self.get_specifications().haso_serial_number
                )
            self.dll.Imop_CorrDataManager_ComputeSlopesFromCommand.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ]
            self.dll.Imop_CorrDataManager_ComputeSlopesFromCommand(
                message,
                self.corrdatamanager,
                numpy.array(command_array, dtype = numpy.single),
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out
        except Exception as exception:
            raise Exception(__name__+' : compute_slopes_from_command',exception)