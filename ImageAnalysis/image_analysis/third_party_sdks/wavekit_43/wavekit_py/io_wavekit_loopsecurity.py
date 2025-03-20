#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library
    
class LoopSecurity(object):
    """Class LoopSecurity
    
    - Constructor from Parameters :
        - **max_saturated_act_count** - int : Maximum tolerated count of saturated actuators
        - **max_camera_fill_pct** - int : Maximum tolerated saturation percentage on camera signal
        - **max_pupil_size_variation** - int : Maximum tolerated variation of the pupil size, relatively to the command matrix pupil size
        - **max_dwf_pv** - float : Maximum tolerated delta wavefront peak-to-valley
    
    - Constructor from Level :
        - **level** - string : Security level. Values are 'LOW', 'MEDIUM' or 'HIGH'
    """

    def __init_from_params(
        self,
        max_saturated_act_count,
        max_camera_fill_pct,
        max_pupil_size_variation,
        max_dwf_pv
        ):
        """Manual construction
        
        :param max_saturated_act_count: Maximum tolerated count of saturated actuators
        :type max_saturated_act_count: int
        :param max_camera_fill_pct: Maximum tolerated saturation percentage on camera signal
        :type max_camera_fill_pct: int
        :param max_pupil_size_variation: Maximum tolerated variation of the pupil size, relatively to the command matrix pupil size
        :type max_pupil_size_variation: int
        :param max_dwf_pv: Maximum tolerated delta wavefront peak-to-valley
        :type max_dwf_pv: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_LoopSecurityParams_New(
                message,
                ctypes.c_int(max_saturated_act_count),
                ctypes.c_int(max_camera_fill_pct),
                ctypes.c_int(max_pupil_size_variation),
                ctypes.c_float(max_dwf_pv),
                ctypes.pointer(self.loopsecurity)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_params',exception)

    def __init_from_levels(
        self,
        level
        ):
        """Automatic construction
        
        :param level: low security parameters('LOW')
        medium security parameters('MEDIUM')
        high security parameters('HIGH')
        :type level: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(level == "LOW") : 
                self.dll.Imop_LoopSecurityParams_NewLow(
                    message,
                    ctypes.pointer(self.loopsecurity)
                    )
            elif(level == "MEDIUM") : 
                self.dll.Imop_LoopSecurityParams_NewMedium(
                    message,
                    ctypes.pointer(self.loopsecurity)
                    )
            elif(level == "HIGH") : 
                self.dll.Imop_LoopSecurityParams_NewHigh(
                    message,
                    ctypes.pointer(self.loopsecurity)
                    )
            else :
                raise Exception('IO_Error','level value must be ''LOW'', ''MEDIUM'' or ''HIGH''')
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_levels',exception)

    def __init__(
        self,
        **kwargs
        ):
        """LoopSecurity Constructor
        """
        self.loopsecurity = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 4):
                if('max_saturated_act_count' in kwargs
                   and 'max_camera_fill_pct' in kwargs
                   and 'max_pupil_size_variation' in kwargs
                   and 'max_dwf_pv' in kwargs):
                    entered = 1
                    self.__init_from_params(kwargs['max_saturated_act_count'], kwargs['max_camera_fill_pct'], kwargs['max_pupil_size_variation'], kwargs['max_dwf_pv'])
            if(arg_size == 1):
                if('level' in kwargs):
                    entered = 1
                    self.__init_from_levels(kwargs['level'])        
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):            
            raise Exception('IO_Error','---CAN NOT CREATE LOOPSECURITY OBJECT---')
    
    def __del_obj__(self):
        """LoopSecurity Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_LoopSecurityParams_Delete(message, self.loopsecurity)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def set_parameters(
        self,
        max_saturated_act_count,
        max_camera_fill_pct,
        max_pupil_size_variation,
        max_dwf_pv
        ):
        """Set tolerances
        
        :param max_saturated_act_count: Maximum tolerated count of saturated actuators
        :type max_saturated_act_count: int
        :param max_camera_fill_pct: Maximum tolerated saturation percentage on camera signal
        :type max_camera_fill_pct: int
        :param max_pupil_size_variation: Maximum tolerated variation of the pupil size, relatively to the command matrix pupil size
        :type max_pupil_size_variation: int
        :param max_dwf_pv: Maximum tolerated delta wavefront peak-to-valley
        :type max_dwf_pv: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_LoopSecurityParams_Set(
                message,
                self.loopsecurity,
                ctypes.c_int(max_saturated_act_count),
                ctypes.c_int(max_camera_fill_pct),
                ctypes.c_int(max_pupil_size_variation),
                ctypes.c_float(max_dwf_pv)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_parameters',exception)

    def get_parameters(self):
        """Get tolerances
        
        :return: Maximum tolerated count of saturated actuators, saturation percentage on camera signal, variation of the pupil size, loop smoothing gain
        :rtype: tuple(int, int, int, float)
        """
        try:
            message = ctypes.create_string_buffer(256)
            max_saturated_act_count = ctypes.c_int()
            max_camera_fill_pct = ctypes.c_int()
            max_pupil_size_variation = ctypes.c_int()
            max_dwf_pv = ctypes.c_float()
            self.dll.Imop_LoopSecurityParams_Get(
                message,
                self.loopsecurity,
                ctypes.byref(max_saturated_act_count),
                ctypes.byref(max_camera_fill_pct),
                ctypes.byref(max_pupil_size_variation),
                ctypes.byref(max_dwf_pv)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                max_saturated_act_count.value,
                max_camera_fill_pct.value,
                max_pupil_size_variation.value,
                max_dwf_pv.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_parameters',exception)

    def check(
        self,
        activation_params,
        camera_fill_percentage,
        corrdata_manager,
        delta_slopes,
        current_corrector_positions,
        delta_commands
        ):
        """Perform security checks required by security_activation using security_params values

        :param activation_params: Closed Loop activation params object
        :type activation_params: LoopSecurityActivation
        :param camera_fill_percentage: Saturation percentage on camera signal
        :type camera_fill_percentage: float
        :param corrdata_manager: CorrDataManager object
        :type corrdata_manager: CorrDataManager
        :param delta_slopes: HasoSlopes object containing the current AO residual slopes
        :type delta_slopes: HasoSlopes
        :param current_corrector_positions: Array containing the actuators positions
        :type current_corrector_positions: float list[]
        :param delta_commands: Array containing the actuators positions
        :type delta_commands: float list[]
        :return: actuators saturation check, saturation percentage on camera signal check, pupil size variation check, residual wavefront peak-to-valley check
        :rtype: tuple(bool, bool, bool, bool)
        """
        try:
            message = ctypes.create_string_buffer(256)
            is_ok_saturated_act_count = ctypes.c_bool()
            is_ok_camera_fill_pct = ctypes.c_bool()
            is_ok_pupil_size_variation = ctypes.c_bool()
            is_ok_dwf_pv = ctypes.c_bool()       
            self.dll.Imop_LoopSecurity_Check.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ]  
            self.dll.Imop_LoopSecurity_Check(
                message,
                activation_params.loopsecurityactivation,
                self.loopsecurity,
                ctypes.c_float(camera_fill_percentage),
                corrdata_manager.corrdatamanager,
                delta_slopes.hasoslopes,
                numpy.array(current_corrector_positions, dtype = numpy.single),
                numpy.array(delta_commands, dtype = numpy.single),
                ctypes.byref(is_ok_saturated_act_count),
                ctypes.byref(is_ok_camera_fill_pct),
                ctypes.byref(is_ok_pupil_size_variation),
                ctypes.byref(is_ok_dwf_pv)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                is_ok_saturated_act_count.value,
                is_ok_camera_fill_pct.value,
                is_ok_pupil_size_variation.value,
                is_ok_dwf_pv.value
                )
        except Exception as exception:
            raise Exception(__name__+' : check',exception)
