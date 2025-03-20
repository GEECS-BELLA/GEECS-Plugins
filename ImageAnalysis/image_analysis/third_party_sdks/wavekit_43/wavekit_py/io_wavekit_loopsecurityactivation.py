#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library
    
class LoopSecurityActivation(object):
    """Class LoopSecurityActivation
    
    - Constructor from Parameters :
        - **do_check_saturated_act_count** - bool : Activates the requested actuators positions check, test will return false if too many acturators reach their boundaries
        - **do_check_camera_fill_pct** - bool : Activates the camera saturation level check
        - **do_check_pupil_size_variation** - bool : Activates the pupil size variation check, test will return false if pupil is too different from the interaction matrix pupil
        - **do_check_dwf_pv** - bool : Activates the delta wavefront peak-to-valley check
    
    - Constructor from Level :
        - **level** - string : Security level. Values are 'LOW', 'MEDIUM' or 'HIGH'
    """

    def __init_from_params(
        self,
        do_check_saturated_act_count,
        do_check_camera_fill_pct,
        do_check_pupil_size_variation,
        do_check_dwf_pv
        ):
        """Manual construction
        
        :param do_check_saturated_act_count: Activates the requested actuators positions check, test will return false if too many acturators reach their boundaries
        :type do_check_saturated_act_count: bool
        :param do_check_camera_fill_pct: Activates the camera saturation level check
        :type do_check_camera_fill_pct: bool
        :param do_check_pupil_size_variation: Activates the pupil size variation check, test will return false if pupil is too different from the interaction matrix pupil
        :type do_check_pupil_size_variation: bool
        :param do_check_dwf_pv: Activates the delta wavefront peak-to-valley check
        :type do_check_dwf_pv: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_LoopSecurityActivation_New(
                message,
                ctypes.c_bool(do_check_saturated_act_count),
                ctypes.c_bool(do_check_camera_fill_pct),
                ctypes.c_bool(do_check_pupil_size_variation),
                ctypes.c_bool(do_check_dwf_pv),
                ctypes.pointer(self.loopsecurityactivation)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_params',exception)

    def __init_from_levels(
        self,
        level
        ):
        """Automatic construction
        
        :param level: low security activations('LOW')
        medium security activations('MEDIUM')
        HIGH security activations('HIGH')
        :type level: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(level == "LOW") : 
                self.dll.Imop_LoopSecurityActivation_NewLow(
                    message,
                    ctypes.pointer(self.loopsecurityactivation)
                    )
            elif(level == "MEDIUM") : 
                self.dll.Imop_LoopSecurityActivation_NewMedium(
                    message,
                    ctypes.pointer(self.loopsecurityactivation)
                    )
            elif(level == "HIGH") : 
                self.dll.Imop_LoopSecurityActivation_NewHigh(
                    message,
                    ctypes.pointer(self.loopsecurityactivation)
                    )
            else :
                raise Exception('IO_Error','level value must be ''LOW'', ''MEDIUM'' or ''HIGH''')
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_levels',exception)

    def __init__(self, **kwargs):
        """LoopSecurityActivation constructor
        """
        self.loopsecurityactivation = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 4):
                if('do_check_saturated_act_count' in kwargs
                   and 'do_check_camera_fill_pct' in kwargs
                   and 'do_check_pupil_size_variation' in kwargs
                   and 'do_check_dwf_pv' in kwargs):
                    entered = 1
                    self.__init_from_params(kwargs['do_check_saturated_act_count'], kwargs['do_check_camera_fill_pct'], kwargs['do_check_pupil_size_variation'], kwargs['do_check_dwf_pv'])
            if(arg_size == 1):
                if('level' in kwargs):
                    entered = 1
                    self.__init_from_levels(kwargs['level'])        
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):            
            raise Exception('IO_Error','---CAN NOT CREATE LOOPSECURITYACTIVATION OBJECT---')
    
    def __del_obj__(self):
        """LoopSecurityActivation destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_LoopSecurityActivation_Delete(message, self.loopsecurityactivation)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def set_activation_parameters(
        self,
        do_check_saturated_act_count,
        do_check_camera_fill_pct,
        do_check_pupil_size_variation,
        do_check_dwf_pv
        ):
        """Set switches states
        
        :param do_check_saturated_act_count: Activates the requested actuators positions check, test will return false if too many acturators reach their boundaries
        :type do_check_saturated_act_count: bool
        :param do_check_camera_fill_pct: Activates the camera saturation level check
        :type do_check_camera_fill_pct: bool
        :param do_check_pupil_size_variation: Activates the pupil size variation check, test will return false if pupil is too different from the interaction matrix pupil
        :type do_check_pupil_size_variation: bool
        :param do_check_dwf_pv: Activates the delta wavefront peak-to-valley check
        :type do_check_dwf_pv: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_LoopSecurityActivation_Set(
                message,
                self.loopsecurityactivation,
                ctypes.c_bool(do_check_saturated_act_count),
                ctypes.c_bool(do_check_camera_fill_pct),
                ctypes.c_bool(do_check_pupil_size_variation),
                ctypes.c_bool(do_check_dwf_pv)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_activation_parameters',exception)

    def get_activation_parameters(
        self
        ):
        """Get switches states
        
        :return: requested actuators positions check,  camera saturation level check,  pupil size variation check, delta wavefront peak-to-valley check
        :rtype: tuple(bool, bool, bool, bool)
        """
        try:
            message = ctypes.create_string_buffer(256)
            do_check_saturated_act_count = ctypes.c_bool()
            do_check_camera_fill_pct = ctypes.c_bool()
            do_check_pupil_size_variation = ctypes.c_bool()
            do_check_dwf_pv = ctypes.c_bool()
            self.dll.Imop_LoopSecurityActivation_Get(
                message,
                self.loopsecurityactivation,
                ctypes.byref(do_check_saturated_act_count),
                ctypes.byref(do_check_camera_fill_pct),
                ctypes.byref(do_check_pupil_size_variation),
                ctypes.byref(do_check_dwf_pv)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                do_check_saturated_act_count.value,
                do_check_camera_fill_pct.value,
                do_check_pupil_size_variation.value,
                do_check_dwf_pv.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_activation_parameters',exception)
