#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_enum as imop_enum
import io_wavekit_structure as imop_struct

class WavefrontCorrectorSet(object):
    """Class WavefrontCorrectorSet
    
    - Constructor from Config File :
        - **config_file_path** - string : Absolute path to WavefrontCorrector configuration file (\*.dat)
    
    - Constructor from WavefrontCorrector :
        - **wavefrontcorrector** - WavefrontCorrector : WavefrontCorrector object
    
    - Constructor from Copy :
        - **wavefrontcorrectorset** - WavefrontCorrectorSet : WavefrontCorrectorSet object to copy
    """
    def __init_from_config_file(
        self,
        config_file_path
        ):
        """WavefrontCorrectorSet
        
        :param config_file_path: Absolute path to WavefrontCorrector configuration file
        :type config_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrectorSet_NewFromConfigFile(
                message,
                ctypes.pointer(self.wavefrontcorrectorset),
                ctypes.c_char_p(config_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_config_file',exception)

    def __init_from_wavefrontcorrector(
        self,
        wavefrontcorrector
        ):
        """WavefrontCorrectorSet constructor from corrector
        
        :param wavefrontcorrector: WavefrontCorrector object
        :type wavefrontcorrector: WavefrontCorrector
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrectorSet_NewFromCorrector(
                message,
                wavefrontcorrector.wavefrontcorrector,
                ctypes.pointer(self.wavefrontcorrectorset)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_wavefrontcorrector',exception)

    def __init_from_copy(
        self,
        wavefrontcorrectorset
        ):
        """WavefrontCorrectorSet constructor from copy
        
        :param wavefrontcorrectorset: WavefrontCorrectorSet object
        :type wavefrontcorrectorset: WavefrontCorrectorSet
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrectorSet_NewFromCopy(
                message,
                wavefrontcorrectorset.wavefrontcorrectorset,
                ctypes.pointer(self.wavefrontcorrectorset)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_copy',exception)

    def __init__(self, **kwargs):
        """WavefrontCorrectorSet Constructor
        """
        self.wavefrontcorrectorset = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('config_file_path' in kwargs):
                    entered = 1
                    self.__init_from_config_file(kwargs['config_file_path'])
                if('wavefrontcorrector' in kwargs):
                    entered = 1
                    self.__init_from_wavefrontcorrector(kwargs['wavefrontcorrector'])
                if('wavefrontcorrectorset' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['wavefrontcorrectorset'])
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE WAVEFRONTCORRECTORSET OBJECT---')
                    
    def __del_obj__(self):
        """WavefrontCorrectorSet Destructor"""
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_WavefrontCorrectorSet_Delete(message, self.wavefrontcorrectorset)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_specifications(self):
        """Get WavefrontCorrector specifications 
        
        :return: WavefrontCorrector specifications : WFC name, WFC Serial number, revision, Driver name, actuators count, stabilization delay, indicator of whether Wavefront corrector electronic can store its position
        :rtype: Config_t, CorrectorSpecs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            driver_name = ctypes.create_string_buffer(256)
            wavefrontcorrector_name = ctypes.create_string_buffer(256)
            wavefrontcorrector_serial_number = ctypes.create_string_buffer(256)
            revision = ctypes.c_int()
            actuators_count = ctypes.c_int()
            min_sleep_after_movement = ctypes.c_int()
            has_internal_memory = ctypes.c_bool()
            self.dll.Imop_WavefrontCorrectorSet_GetSpecifications(
                message,
                self.wavefrontcorrectorset,
                driver_name,
                wavefrontcorrector_name,
                wavefrontcorrector_serial_number,
                ctypes.byref(revision),
                ctypes.byref(actuators_count),
                ctypes.byref(min_sleep_after_movement),
                ctypes.byref(has_internal_memory)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                imop_struct.Config_t (
                    wavefrontcorrector_name.value.decode('utf-8'),
                    wavefrontcorrector_serial_number.value.decode('utf-8'),
                    revision.value,
                    driver_name.value.decode('utf-8')
                    ),
                imop_struct.CorrectorSpecs_t (
                    actuators_count.value,
                    min_sleep_after_movement.value,
                    has_internal_memory.value
                    )
                )
        except Exception as exception:
            raise Exception(__name__+' : get_specifications',exception)

    def get_actuators_count(self):
        """Get WavefrontCorrector actuators count
        
        :return: Actuators count
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            actuators_count = ctypes.c_int()
            self.dll.Imop_WavefrontCorrectorSet_GetActuatorsCount(
                message,
                self.wavefrontcorrectorset,
                ctypes.byref(actuators_count)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return actuators_count.value
        except Exception as exception:
            raise Exception(__name__+' : get_actuators_count',exception)

    def get_initial_preferences(self):
        """Get WavefrontCorrector constraints
        
        :return: Corrector preferences (lowest command value, highest command value, status and fixed positions for each actuator), initial positions
        :rtype: tuple(CorrectorPrefs_t, float list[])
        """
        try:
            message = ctypes.create_string_buffer(256)
            min_out = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            max_out = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            validity_out = numpy.zeros(self.get_actuators_count(), dtype = numpy.intc)
            fixed_values_out = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            init_positions_out = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            self.dll.Imop_WavefrontCorrectorSet_GetInitialPreferences.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")   
                ]
            self.dll.Imop_WavefrontCorrectorSet_GetInitialPreferences(
                message,
                self.wavefrontcorrectorset,
                max_out,
                min_out,
                validity_out,
                fixed_values_out,
                init_positions_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                imop_struct.CorrectorPrefs_t(
                    min_out.tolist(),
                    max_out.tolist(),
                    validity_out.tolist(),
                    fixed_values_out.tolist()
                    ),
                init_positions_out.tolist()
                )
        except Exception as exception:
            raise Exception(__name__+' : get_initial_preferences',exception)

    def get_flat_mirror_positions(self):
        """Get WavefrontCorrector commands to flat mirror shape
        
        :return: Command value
        :rtype: float list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            flat_command_out = numpy.zeros(self.get_actuators_count(), dtype = numpy.single)
            self.dll.Imop_WavefrontCorrectorSet_GetFlatMirrorPositions.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")  
                ]
            self.dll.Imop_WavefrontCorrectorSet_GetFlatMirrorPositions(
                message,
                self.wavefrontcorrectorset,
                flat_command_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return flat_command_out.tolist()
        except Exception as exception:
            raise Exception(__name__+' : get_flat_mirror_positions',exception)

    def get_specific_feature_list_size(self):
        """Get WavefrontCorrector features list size
        
        :return: Features list size
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = ctypes.c_int()
            self.dll.Imop_WavefrontCorrectorSet_GetSpecificFeaturesListSize(
                message,
                self.wavefrontcorrectorset,
                ctypes.byref(size_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_specific_feature_list_size',exception)

    def get_specific_feature_name(
        self,
        index
        ):
        """Get WavefrontCorrector features name
        
        :param index: Feature index
        :type index: int
        :return: Features name
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            feature_name_out = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrectorSet_GetSpecificFeatureName(
                message,
                self.wavefrontcorrectorset,
                ctypes.c_int(index),
                feature_name_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return feature_name_out.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_specific_feature_name',exception)

    def get_parameter_list_size(self):
        """Get WavefrontCorrector parameters list size
        
        :return: Parameters list size (total count of wavefront corrector parameters)
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = ctypes.c_int()
            self.dll.Imop_WavefrontCorrectorSet_GetParameterListSize(
                message,
                self.wavefrontcorrectorset,
                ctypes.byref(size_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_list_size',exception)

    def get_parameter_name(
        self,
        index
        ):
        """Get WavefrontCorrector parameters list size
        
        :param index: Parameter index
        :type index: int
        :return: Parameter name
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            name_out = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrectorSet_GetParameterName(
                message,
                self.wavefrontcorrectorset,
                ctypes.c_int(index),
                name_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error', message.value)
            return name_out.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_name',exception)

    def get_parameter_type(
        self,
        parameter_name
        ):
        """Get WavefrontCorrector parameter type
        
        :param parameter_name: Parameter name
        :type parameter_name: string
        :return: E_TYPES_T Parameter type value
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            type_out = ctypes.c_int()
            self.dll.Imop_WavefrontCorrectorSet_GetParameterType(
                message,
                self.wavefrontcorrectorset,
                ctypes.c_char_p(parameter_name.encode('utf-8')),
                ctypes.byref(type_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return type_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_type',exception)

    def get_parameter_option(
        self,
        parameter_name
        ):
        """Get WavefrontCorrector parameter options
        
        :param parameter_name: Parameter name
        :type parameter_name: string
        :return: Parameter options (has_default_value, has_limits, is_connection_parameter, is_gettable, is_gettable_onfly, is_settable, is_settable_onfly, is_string_enum)
        :rtype: ParameterOption_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            is_settable = ctypes.c_bool()
            is_gettable = ctypes.c_bool()
            has_default_value = ctypes.c_bool()
            has_limits = ctypes.c_bool()
            is_string_enum = ctypes.c_bool()
            is_settable_on_fly = ctypes.c_bool()
            is_gettable_on_fly = ctypes.c_bool()
            is_connection_parameter = ctypes.c_bool()
            self.dll.Imop_WavefrontCorrectorSet_GetParameterOption(
                message,
                self.wavefrontcorrectorset,
                ctypes.c_char_p(parameter_name.encode('utf-8')),
                ctypes.byref(is_settable),
                ctypes.byref(is_gettable),
                ctypes.byref(has_default_value),
                ctypes.byref(has_limits),
                ctypes.byref(is_string_enum),
                ctypes.byref(is_settable_on_fly),
                ctypes.byref(is_gettable_on_fly),
                ctypes.byref(is_connection_parameter)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.ParameterOption_t(
                has_default_value.value,
                has_limits.value,
                is_connection_parameter.value,
                is_gettable.value,
                is_gettable_on_fly.value,
                is_settable.value,
                is_settable_on_fly.value,
                is_string_enum.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_option',exception)

    def get_parameter_limits(
        self,
        parameter_name
        ):
        """Get WavefrontCorrector parameters specifications.
        Each parameter comes with specifications given its type 
        and has therefore a dedicated specification value(s) getter.
        Please ensure your retrieved the parameters type given its name to call the approriate function.
        
        :param parameter_name: Parameter name
        :type parameter_name: string
        :return: min, max
        :rtype: tuple(int or double, int or double)
        """
        try:
            message = ctypes.create_string_buffer(256)
            type_ = self.get_parameter_type(parameter_name)
            if(type_ == imop_enum.E_TYPES.INT) :
                min_ = ctypes.c_int()
                max_ = ctypes.c_int()
                self.dll.Imop_WavefrontCorrectorSet_GetParameterIntLimits(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(min_),
                    ctypes.byref(max_)                    
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return (
                    min_.value,
                    max_.value
                    )
            elif(type_ == imop_enum.E_TYPES.REAL) :
                min_ = ctypes.c_double()
                max_ = ctypes.c_double()
                self.dll.Imop_WavefrontCorrectorSet_GetParameterRealLimits(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(min_),
                    ctypes.byref(max_)                    
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return (
                    min_.value,
                    max_.value
                    )
            else:
                raise Exception('IO_Error','Parameter type must be int or double)')
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_limits',exception)

    def get_available_values_list_size(
        self,
        parameter_name
        ):
        """If parameter is a string to be selected among a finite set of strings, return the size of this set.
        
        .. seealso:: WavefrontCorrectorSet.get_parameter_option
        
        :param parameter_name: Parameter name
        :type parameter_name: string
        :return: Size of the list (count of strings expected in that parameter)
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = ctypes.c_int()
            self.dll.Imop_WavefrontCorrectorSet_GetAvailableValuesListSize(
                message,
                self.wavefrontcorrectorset,
                ctypes.c_char_p(parameter_name.encode('utf-8')),
                ctypes.byref(size_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_available_values_list_size',exception)

    def get_available_value(
        self,
        parameter_name,
        index
        ):
        """If parameter is a string to be selected among a finite set of strings,
        extract the possible value located at the index position from this set.
        
        .. seealso:: WavefrontCorrectorSet.get_parameter_option
        
        :param parameter_name: Parameter name
        :type parameter_name: string
        :param index: index
        :type index: int
        :return: Value at index
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            val_out = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrectorSet_GetAvailableValue(
                message,
                self.wavefrontcorrectorset,
                ctypes.c_char_p(parameter_name.encode('utf-8')),
                ctypes.c_uint(index),
                ctypes.byref(val_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return val_out.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_available_value',exception)

    def get_parameter_value(
        self,
        parameter_name,
        get_factory_value = False
        ):
        """Get WavefrontCorrector parameter
        
        :param parameter_name: Parameter name
        :type parameter_name: string
        :param get_factory_value: True to get the factory value, False to get current value (default : False)
        :type get_factory_value: bool
        :return: Parameter value
        :rtype: int or double or bool or string
        """
        try:
            message = ctypes.create_string_buffer(256)
            type_ = self.get_parameter_type(parameter_name)
            if(type_ == imop_enum.E_TYPES.INT) :
                val_out = ctypes.c_int()
                self.dll.Imop_WavefrontCorrectorSet_GetParameterIntValue(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(val_out),
                    ctypes.c_bool(get_factory_value)          
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return val_out.value
            if(type_ == imop_enum.E_TYPES.REAL) :
                val_out = ctypes.c_double()
                self.dll.Imop_WavefrontCorrectorSet_GetParameterRealValue(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(val_out),
                    ctypes.c_bool(get_factory_value)           
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return val_out.value
            if(type_ == imop_enum.E_TYPES.BOOL) :
                val_out = ctypes.c_bool()
                self.dll.Imop_WavefrontCorrectorSet_GetParameterBooleanValue(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(val_out),
                    ctypes.c_bool(get_factory_value)          
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return val_out.value
            if(type_ == imop_enum.E_TYPES.STRING) :
                val_out = ctypes.create_string_buffer(256)
                self.dll.Imop_WavefrontCorrectorSet_GetParameterStringValue(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    val_out,
                    ctypes.c_bool(get_factory_value)        
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return val_out.value.decode('utf-8')
            raise Exception('IO_Error','Unknown parameter type')
        except Exception as exception:
            raise Exception(__name__+' : get_parameter',exception)

    def reset(self):
        """Reset parameters to factory values
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrectorSet_ResetParameters(
                message,
                self.wavefrontcorrectorset
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : reset',exception)

    def set_parameter_value(
        self,
        parameter_name,
        parameter_value
        ):
        """Set WavefrontCorrector parameter
        
        :param parameter_name: Parameter name
        :type parameter_name: string
        :param parameter_value: Parameter value
        :type parameter_value: int or double or bool or string
        """
        try:
            message = ctypes.create_string_buffer(256)
            type_ = self.get_parameter_type(parameter_name)
            if(type_ == imop_enum.E_TYPES.INT) :
                self.dll.Imop_WavefrontCorrectorSet_SetParameterIntValue(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_int(parameter_value)           
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            elif(type_ == imop_enum.E_TYPES.REAL) :
                self.dll.Imop_WavefrontCorrectorSet_SetParameterRealValue(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_double(parameter_value)           
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            elif(type_ == imop_enum.E_TYPES.BOOL) :
                self.dll.Imop_WavefrontCorrectorSet_SetParameterBooleanValue(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_bool(parameter_value)           
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            elif(type_ == imop_enum.E_TYPES.STRING) :
                self.dll.Imop_WavefrontCorrectorSet_SetParameterStringValue(
                    message,
                    self.wavefrontcorrectorset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_char_p(parameter_value.encode('utf-8'))        
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            else:
                raise Exception('IO_Error','Unknown parameter type')
        except Exception as exception:
            raise Exception(__name__+' : set_parameter',exception)

    def save(
        self,
        file_path
        ):
        """Save WavefrontCorrectorSet to a file
        
        :param file_path: Absolute path to the file
        :type file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrectorSet_Save(
                message,
                self.wavefrontcorrectorset,
                ctypes.c_char_p(file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : save',exception)

    def load(
        self,
        file_path
        ):
        """Load file content to WavefrontCorrectorSet object
        
        :param file_path: Absolute path to the file
        :type file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_WavefrontCorrectorSet_Load(
                message,
                self.wavefrontcorrectorset,
                ctypes.c_char_p(file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : load',exception)
