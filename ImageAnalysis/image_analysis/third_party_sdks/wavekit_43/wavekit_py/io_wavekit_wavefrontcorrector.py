#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_enum as imop_enum
import io_wavekit_wavefrontcorrectorset as imop_wfcset

class WavefrontCorrector(object):
    """Class WavefrontCorrector
    
    - Constructor from Config File :
        - **config_file_path** - string : Absolute path to Wavefront corrector configuration file (\*.dat)
    """
    def __init_(
        self,
        config_file_path
        ):
        """WavefrontCorrector constructor
        
        :param config_file_path: Absolute path to Wavefront corrector configuration file
        :type config_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_NewInstanceFromConfigFile(
                message,
                ctypes.pointer(self.wavefrontcorrector),
                ctypes.c_char_p(config_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_',exception)

    def __init__(self,  **kwargs):
        """WavefrontCorrector constructor
        """
        self.wavefrontcorrector = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        self.nb_actuators = 0
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('config_file_path' in kwargs):
                    entered = 1
                    self.__init_(kwargs['config_file_path'])
            wavefrontcorrectorset = imop_wfcset.WavefrontCorrectorSet(config_file_path = kwargs['config_file_path'])
            self.nb_actuators = wavefrontcorrectorset.get_actuators_count()
            del wavefrontcorrectorset
        except Exception as exception:
            raise Exception(__name__+' : init',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE WAVEFRONTCORRECTOR OBJECT---')

    def __del_obj__(self):
        """WavefrontCorrector Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_WavefrontCorrector_Delete(message, self.wavefrontcorrector)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def connect(
        self,
        set_init_state_from_config_file = False
        ):
        """Establish connection to the device and sets the wavefront corrector to its initial positions
        if set_init_state_from_config_file is true, mirror will be moved to reach to initial positions specified in configuration file.
        else initial positions will be read from device internal memory if possible or throw an error if not.
        
        .. seealso:: WavefrontCorrectorSet.get_specifications to check if your wavefront corrector has an internal memory.
        
        :param set_init_state_from_config_file: if true, using initial positions from configuration file. Otherwise using internal memory
        :type set_init_state_from_config_file: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_Connect(
                message,
                self.wavefrontcorrector,
                ctypes.c_bool(set_init_state_from_config_file)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : connect',exception)

    def disconnect(self):
        """Close connection to the device
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_Disconnect(
                message,
                self.wavefrontcorrector
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : disconnect',exception)

    def call_specific_feature(
        self,
        feature_name
        ):
        """Perform an action corresponding to feature_name.
        
        :param feature_name: Name of the action
        :type feature_name: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_CallSpecificFeature(
                message,
                self.wavefrontcorrector,
                ctypes.c_char_p(feature_name.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : call_specific_feature',exception)

    def get_preferences(
        self
        ):
        """Get current preferences
        
        :return: Time to stabilization (ms), Corrector preferences (lowest command value, highest command value, status and fixed positions for each actuator)
        :rtype: tuple(int, CorrectorPrefs_t)
        """
        try:
            message = ctypes.create_string_buffer(256)
            sleep_after_movement_out = ctypes.c_int()
            min_out = numpy.zeros(self.nb_actuators, dtype = numpy.float32)
            max_out = numpy.zeros(self.nb_actuators, dtype = numpy.float32)
            validity_out = numpy.zeros(self.nb_actuators, dtype = numpy.intc)
            fixed_values_out = numpy.zeros(self.nb_actuators, dtype = numpy.float32)
            self.dll.Imop_WavefrontCorrector_GetPreferences.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                ctypes.POINTER(ctypes.c_int),
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS") 
                ]
            self.dll.Imop_WavefrontCorrector_GetPreferences(
                message,
                self.wavefrontcorrector,
                ctypes.byref(sleep_after_movement_out),
                min_out,
                max_out,
                validity_out,
                fixed_values_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                sleep_after_movement_out.value,
                imop_struct.CorrectorPrefs_t(
                    min_out.tolist(),
                    max_out.tolist(),
                    validity_out.tolist(),
                    fixed_values_out.tolist()
                    )
                )
        except Exception as exception:
            raise Exception(__name__+' : get_preferences',exception)

    def assert_equal_preferences(
        self,
        sleep_after_movement,
        prefs
        ):
        """Assert requested preferences are equal to current preferences
        
        :param sleep_after_movement: Time to stabilization (ms)
        :type sleep_after_movement: int
        
        :param prefs : Corrector preferences
        :type prefs : CorrectorPrefs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_AssertEqualPreferences.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                ctypes.c_int,
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_WavefrontCorrector_AssertEqualPreferences(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(sleep_after_movement),
                numpy.array(prefs.min_array, numpy.float32),
                numpy.array(prefs.max_array, numpy.float32),
                numpy.array(prefs.validity_array, numpy.intc),
                numpy.array(prefs.fixed_value_array, numpy.float32)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : assert_equal_preferences',exception)

    def check_user_preferences(
        self,
        sleep_after_movement,
        prefs
        ):
        """Assert requested preferences fulfill specifications constraints
        
        :param sleep_after_movement: Time to stabilization (ms)
        :type sleep_after_movement: int
        
        :param prefs : Corrector preferences
        :type prefs : CorrectorPrefs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_CheckUserPreferences.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p,
                ctypes.c_int,
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_WavefrontCorrector_CheckUserPreferences(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(sleep_after_movement),
                numpy.array(prefs.min_array, numpy.float32),
                numpy.array(prefs.max_array, numpy.float32),
                numpy.array(prefs.validity_array, numpy.intc),
                numpy.array(prefs.fixed_value_array, numpy.float32)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : check_user_preferences',exception)

    def set_preferences(
        self,
        sleep_after_movement,
        prefs
        ):
        """Apply requested preferences, clip or ignore if specifications constraints are not fulfilled
        
        :param sleep_after_movement: Time to stabilization (ms)
        :type sleep_after_movement: int
        
        :param prefs : Corrector preferences
        :type prefs : CorrectorPrefs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_SetPreferences.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p, 
                ctypes.c_int,
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_WavefrontCorrector_SetPreferences(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(sleep_after_movement),
                numpy.array(prefs.min_array, numpy.float32),
                numpy.array(prefs.max_array, numpy.float32),
                numpy.array(prefs.validity_array, numpy.intc),
                numpy.array(prefs.fixed_value_array, numpy.float32)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_preferences',exception)

    def get_temporization(
        self
        ):
        """Get current WavefrontCorrector preference 
        
        :return: Time to stabilization (ms)
        :rtype: int 
        """
        try:
            message = ctypes.create_string_buffer(256)
            temporization_out = ctypes.c_int()
            self.dll.Imop_WavefrontCorrector_GetTemporization(
                message,
                self.wavefrontcorrector,
                ctypes.byref(temporization_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return temporization_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_temporization',exception)

    def set_temporization(
        self,
        temporization
        ):
        """Set current WavefrontCorrector preference, 
        clip or ignore if specifications constraints are not fulfilled
        
        :param temporization: Time to stabilization (ms)
        :type temporization: int 
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_SetTemporization(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(temporization)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_temporization',exception)

    def get_current_positions(
        self
        ):
        """Get current actuators positions
        
        :return: Array containing the actuators positions
        :rtype: float list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            positions_out = numpy.zeros(self.nb_actuators, dtype = numpy.single)
            self.dll.Imop_WavefrontCorrector_GetCurrentPositions.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_WavefrontCorrector_GetCurrentPositions(
                message,
                self.wavefrontcorrector,
                positions_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return positions_out.tolist()
        except Exception as exception:
            raise Exception(__name__+' : get_current_positions',exception)

    def check_relative_positions(
        self,
        positions_array
        ):
        """Assert if requested relative positions satisfy current preferences.
        If actuator condition is E_ACTUATOR_CONDITIONS.INVALID, no error is thrown, since
        position will be ignored by the Wavefront corrector.
        
        :param positions_array: Requested relatives positions
        :type positions_array: float list (size = number of actuators)
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_CheckRelativePositions.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_WavefrontCorrector_CheckRelativePositions(
                message,
                self.wavefrontcorrector,
                numpy.array(positions_array, dtype = numpy.float32)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : check_relative_positions',exception)

    def move_to_relative_positions(
        self,
        positions_array
        ):
        """Move to requested relative positions, clip according to current preferences
        
        :param positions_array: Requested relatives positions
        :type positions_array: float list[] (size = number of actuators)
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_MoveToRelativePositions.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_WavefrontCorrector_MoveToRelativePositions(
                message,
                self.wavefrontcorrector,
                numpy.array(positions_array, dtype = numpy.float32)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : move_to_relative_positions',exception)

    def check_absolute_positions(
        self,
        positions_array
        ):
        """Assert if requested absolute positions satisfy current preferences.
        If actuator condition is E_ACTUATOR_CONDITIONS.INVALID, no error is thrown, since
        position will be ignored by the Wavefront corrector
        
        :param positions_array: Requested absolutes positions
        :type positions_array: float[] (list)
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_CheckAbsolutePositions.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_WavefrontCorrector_CheckAbsolutePositions(
                message,
                self.wavefrontcorrector,
                numpy.array(positions_array, dtype = numpy.float32)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : check_absolute_positions',exception)

    def move_to_absolute_positions(
        self,
        positions_array
        ):
        """Move to requested absolute positions, clip according to current preferences
        
        :param positions_array: Requested absolutes positions
        :type positions_array: float list[] (size = number of actuators)
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_MoveToAbsolutePositions.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_WavefrontCorrector_MoveToAbsolutePositions(
                message,
                self.wavefrontcorrector,
                numpy.array(positions_array, dtype = numpy.float32)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : move_to_absolute_positions',exception)

    def compute_delta_command_amplitude(
        self,
        delta_commands_array
        ):
        """Compute a relative move amplitude as the max displacement percentage among all the valid actuators.
        For one actuator : Displacement percentage = displacement / (max - min) * 100
        
        :param delta_commands_array: Delta command to analyze
        :type delta_commands_array: float list[]
        :return: Computed delta command amplitude
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            amplitude_out = ctypes.c_int()
            self.dll.Imop_WavefrontCorrector_ComputeDeltaCommandAmplitude.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ]
            self.dll.Imop_WavefrontCorrector_ComputeDeltaCommandAmplitude(
                message,
                self.wavefrontcorrector,
                numpy.array(delta_commands_array, dtype = numpy.float32),
                ctypes.byref(amplitude_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return amplitude_out.value
        except Exception as exception:
            raise Exception(__name__+' : compute_delta_command_amplitude',exception)

    def get_actuator_preferences(
        self,
        index
        ):
        """Single actuator management
        Set and get position and preferences of one actuator given its index
        
        :param index: Index of the actuator
        :type index: int
        :return: lowest command, highest command, status, fixed position
        :rtype: ActuatorPrefs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            min_out = ctypes.c_float()
            max_out = ctypes.c_float()
            validity_out = ctypes.c_int()
            fixed_values_out = ctypes.c_float()
            self.dll.Imop_WavefrontCorrector_GetActuatorPreferences(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(index),
                ctypes.byref(min_out),
                ctypes.byref(max_out),
                ctypes.byref(validity_out),
                ctypes.byref(fixed_values_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.ActuatorPrefs_t(
                min_out.value,
                max_out.value,
                validity_out.value,
                fixed_values_out.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_actuator_preferences',exception)

    def check_actuator_preferences(
        self,
        index,
        min_value,
        max_value,
        validity,
        fixed_value
        ):
        """Assert requested preferences fulfill specifications constraints
        
        :param index: Index of the actuator
        :type index: int
        :param min_value: lowest command
        :type min_value: float
        :param max_value: highest command
        :type max_value: float
        :param validity: status
        :type validity: int
        :param fixed_value: fixed position
        :type fixed_value: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_CheckActuatorUserPreferences(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(index),
                ctypes.c_float(min_value),
                ctypes.c_float(max_value),
                ctypes.c_int(validity),
                ctypes.c_float(fixed_value)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : check_actuator_preferences',exception)

    def set_actuator_preferences(
        self,
        index,
        min_value,
        max_value,
        validity,
        fixed_value
        ):
        """Set preferences of actuator
        
        :param index: Index of the actuator
        :type index: int
        :param min_value: lowest command
        :type min_value: float
        :param max_value: highest command
        :type max_value: float
        :param validity: status
        :type validity: int
        :param fixed_value: fixed position
        :type fixed_value: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_SetActuatorPreferences(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(index),
                ctypes.c_float(min_value),
                ctypes.c_float(max_value),
                ctypes.c_int(validity),
                ctypes.c_float(fixed_value)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_actuator_preferences',exception)

    def get_actuator_current_position(
        self,
        index
        ):
        """Get current position of actuator
        
        :param index: Index of the actuator
        :type index: int
        :return: Position
        :rtype: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            position_out = ctypes.c_float()
            self.dll.Imop_WavefrontCorrector_GetActuatorCurrentPosition(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(index),
                ctypes.byref(position_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return position_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_actuator_current_position',exception)

    def check_actuator_relative_position(
        self,
        index,
        position
        ):
        """Assert if requested relative position satisfy current preferences.
        If actuator is invalid, no error is thrown, since
        position will be ignored by the Wavefront corrector
        
        :param index: Index of the actuator
        :type index: int 
        :param position: Requested relatives position
        :type position: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_CheckActuatorRelativePosition(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(index),
                ctypes.c_float(position)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : check_actuator_relative_position',exception)

    def move_actuator_to_relative_position(
        self,
        index,
        position
        ):
        """Move to requested relative position, clip according to current preferences
        
        :param index: Index of the actuator
        :type index: int 
        :param position: Requested relatives position
        :type position: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_MoveActuatorToRelativePosition(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(index),
                ctypes.c_float(position)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : move_actuator_to_relative_position',exception)

    def check_actuator_absolute_position(
        self,
        index,
        position
        ):
        """Assert if requested absolute position satisfy current preferences.
        If actuator is invalid, no error is thrown, since
        position will be ignored by the Wavefront corrector
        
        :param index: Index of the actuator
        :type index: int 
        :param position: Requested absolutes position
        :type position: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_CheckActuatorAbsolutePosition(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(index),
                ctypes.c_float(position)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : check_actuator_absolute_position',exception)

    def move_actuator_to_absolute_position(
        self,
        index,
        position
        ):
        """Move to requested absolute position, clip according to current preferences
        
        :param index: Index of the actuator
        :type index: int 
        :param position: Requested absolutes position
        :type position: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_MoveActuatorToAbsolutePosition(
                message,
                self.wavefrontcorrector,
                ctypes.c_int(index),
                ctypes.c_float(position)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : move_actuator_to_absolute_position',exception)

    def get_parameter_value(
        self,
        parameter_name
        ):
        """Wavefront corrector parameter getter for parameter
        
        :param parameter_name: Name of the parameter
        :type parameter_name: string 
        :return: Value
        :rtype: int or bool or double or string
        """
        try:
            message = ctypes.create_string_buffer(256)
            wavefrontcorrset = imop_wfcset.WavefrontCorrectorSet(wavefrontcorrector = self)
            type_ = wavefrontcorrset.get_parameter_type(parameter_name)
            if(type_ == imop_enum.E_TYPES.INT) :
                value_out = ctypes.c_int()
                self.dll.Imop_WavefrontCorrector_GetParameterInt(
                    message,
                    self.wavefrontcorrector,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(value_out)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return value_out.value
            if(type_ == imop_enum.E_TYPES.REAL) :
                value_out = ctypes.c_double()
                self.dll.Imop_WavefrontCorrector_GetParameterReal(
                    message,
                    self.wavefrontcorrector,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(value_out)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return value_out.value
            if(type_ == imop_enum.E_TYPES.BOOL) :
                value_out = ctypes.c_bool()
                self.dll.Imop_WavefrontCorrector_GetParameterBoolean(
                    message,
                    self.wavefrontcorrector,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(value_out)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return value_out.value
            if(type_ == imop_enum.E_TYPES.STRING) :
                value_out = ctypes.create_string_buffer(256)
                self.dll.Imop_WavefrontCorrector_GetParameterString(
                    message,
                    self.wavefrontcorrector,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    value_out
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return value_out.value.decode('utf-8')        
            raise Exception('IO_Error','Unknown parameter type')
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_value',exception)

    def diff_wavefrontcorrectorset(
        self,
        wavefrontcorrectorset
        ):
        """Load and apply WavefrontCorrector parameters from a WavefrontCorrectorSet object
        
        :param wavefrontcorrectorset: *WavefrontCorrectorSet* object
        :type wavefrontcorrectorset: WavefrontCorrectorSet 
        :return: number of settable parameters whose values differ, number of onfly settable parameters whose values differ, number of settable connection parameters whose values differ
        :rtype: tuple(int, int, int)
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb_nonequal_settable_params = ctypes.c_int()
            nb_nonequals_onfly_settable_params = ctypes.c_int()
            nb_nonequals_settable_connection_params = ctypes.c_int()
            self.dll.Imop_WavefrontCorrector_DiffWavefrontCorrectorSet(
                message,
                self.wavefrontcorrector,
                wavefrontcorrectorset.wavefrontcorrectorset,
                ctypes.byref(nb_nonequal_settable_params),
                ctypes.byref(nb_nonequals_onfly_settable_params),
                ctypes.byref(nb_nonequals_settable_connection_params)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                nb_nonequal_settable_params.value,
                nb_nonequals_onfly_settable_params.value,
                nb_nonequals_settable_connection_params.value
                )
        except Exception as exception:
            raise Exception(__name__+' : diff_wavefrontcorrectorset',exception)

    def reset_parameters(
        self,
        reset_connection_params,
        reset_params
        ):
        """Reset WavefrontCorrector parameters to factory values
        
        :param reset_connection_params: try to load and apply factory connection parameters if true
        :type reset_connection_params: bool
        :param reset_params: try to load and apply factory parameters if true
        :type reset_params: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_ResetParameters(
                message,
                self.wavefrontcorrector,
                ctypes.c_bool(reset_connection_params),
                ctypes.c_bool(reset_params)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : reset_parameters',exception)

    def set_wavefrontcorrectorset(
        self,
        wavefrontcorrectorset,
        load_and_apply_connection_params,
        load_and_apply_params
        ):
        """Load and apply WavefrontCorrector parameters from a WavefrontCorrectorSet object
        If WavefrontCorrector is moving, only on fly settable parameters will be applied
        
        :param wavefrontcorrectorset: WavefrontCorrectorSet object
        :type wavefrontcorrectorset: WavefrontCorrectorSet
        :param load_and_apply_connection_params: try to load and apply connection parameters if true
        :type load_and_apply_connection_params: bool
        :param load_and_apply_params: try to load and apply parameters if true
        :type load_and_apply_params: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_SetWavefrontCorrectorSet(
                message,
                self.wavefrontcorrector,
                wavefrontcorrectorset.wavefrontcorrectorset,
                ctypes.c_bool(load_and_apply_connection_params),
                ctypes.c_bool(load_and_apply_params)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_wavefrontcorrectorset',exception)

    def set_parameter_value(
        self,
        parameter_name,
        parameter_value
        ):
        """Set WavefrontCorrector parameter
        
        :param parameter_name: Name of the parameter
        :type parameter_name: string
        :param parameter_value: Value of the parameter
        :type parameter_value: int or double or bool or string
        """
        try:
            message = ctypes.create_string_buffer(256)
            wavefrontcorrset = imop_wfcset.WavefrontCorrectorSet(wavefrontcorrector = self)
            type_ = wavefrontcorrset.get_parameter_type(parameter_name)
            if(type_ == imop_enum.E_TYPES.INT) :
                self.dll.Imop_WavefrontCorrector_SetParameterInt(
                    message,
                    self.wavefrontcorrector,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_int(parameter_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return
            if(type_ == imop_enum.E_TYPES.REAL) :
                self.dll.Imop_WavefrontCorrector_SetParameterReal(
                    message,
                    self.wavefrontcorrector,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_float(parameter_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return
            if(type_ == imop_enum.E_TYPES.BOOL) :
                self.dll.Imop_WavefrontCorrector_SetParameterBoolean(
                    message,
                    self.wavefrontcorrector,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_bool(parameter_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return
            if(type_ == imop_enum.E_TYPES.STRING) :
                self.dll.Imop_WavefrontCorrector_SetParameterString(
                    message,
                    self.wavefrontcorrector,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_char_p(parameter_value.encode('utf-8'))
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return
            raise Exception('IO_Error','Unknown parameter type')
        except Exception as exception:
            raise Exception(__name__+' : set_parameter_value',exception)

    @staticmethod
    def get_positions_from_file(
        position_file_path
        ):
        """Get Wavefront corrector positions from positions file
        
        :param position_file_path: Positions file path
        :type position_file_path: string
        :return: Positions values
        :rtype: float list[]
        """
        dll   = imop_library.load_dll()
        
        try:
            nb_actuator_out = WavefrontCorrector.get_nb_actuator_from_position_file(position_file_path)
            message = ctypes.create_string_buffer(256)
            positions_out = numpy.zeros((nb_actuator_out), dtype = numpy.float32)
            dll.Imop_WavefrontCorrector_GetPositionsFromFile.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p, 
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            dll.Imop_WavefrontCorrector_GetPositionsFromFile(
                message,
                ctypes.c_char_p(position_file_path.encode('utf-8')),
                positions_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return positions_out.tolist()
        except Exception as exception:
            raise Exception(__name__+' : get_positions_from_file',exception)

    @staticmethod
    def get_nb_actuator_from_position_file(
        wfc_position_file_path
        ):
        """Get Wavefront corrector number actuator
        
        :param wfc_position_file_path: Positions file path
        :type wfc_position_file_path: string
        :return: Wavefront corrector actuators count
        :rtype: int
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            nb_actuator_out = ctypes.c_int()
            dll.Imop_WavefrontCorrector_GetInfoFromWavefrontCorrectorPositionFile(
                message,
                ctypes.c_char_p(wfc_position_file_path.encode('utf-8')),
                ctypes.byref(nb_actuator_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return nb_actuator_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_nb_actuator_from_position_file',exception)

    def save_current_positions_to_file(
        self,
        pmc_file_path
        ):
        """Save current Wavefront corrector positions to positions file
        
        :param pmc_file_path: Positions file path
        :type pmc_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrector_SaveCurrentPositionsToFile(
                message,
                self.wavefrontcorrector,
                ctypes.c_char_p(pmc_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : save_current_positions_to_file',exception)
