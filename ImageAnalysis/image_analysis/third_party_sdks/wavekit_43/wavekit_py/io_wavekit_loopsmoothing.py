#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_wavekit_structure as imop_struct

import io_thirdparty_load_library as imop_library

class LoopSmoothing(object):
    """Class LoopSmoothing 
    
    - Constructor from Parameters :
        - **low_gain_factor** - float : lowest value for gain factor
        - **high_gain_factor** - float : highest value for gain factor
        - **low_amplitude** - int : lowest amplitude value before switching to linear behavior
        - **high_amplitude** - int : highest amplitude value where linear behavior stops
    
    - Constructor from Level :
        - **level** - string : Security level. Values are 'LOW', 'MEDIUM' or 'HIGH'
    """

    def __init_from_params(
        self,
        low_gain_factor,
        high_gain_factor,
        low_amplitude,
        high_amplitude
        ):
        """LoopSmoothing parameters constructor.
        
        :param low_gain_factor: lowest value for gain factor
        :type low_gain_factor: float
        :param high_gain_factor: highest value for gain factor
        :type high_gain_factor: float
        :param low_amplitude: lowest amplitude value before switching to linear behavior
        :type low_amplitude: int
        :param high_amplitude: highest amplitude value where linear behavior stops
        :type high_amplitude: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_LoopSmoothingParams_New(
                message,
                ctypes.c_float(low_gain_factor),
                ctypes.c_float(high_gain_factor),
                ctypes.c_int(low_amplitude),
                ctypes.c_int(high_amplitude),
                ctypes.pointer(self.loopsmoothing)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:            
            raise Exception('IO_Error','---CAN NOT CREATE LOOPSMOOTHING OBJECT---')

    def __init_from_levels(
        self,
        level
        ):
        """LoopSmoothing parameters constructor.
        
        :param level: loop smoothing parameters constructor
        Don't smooth ("LOW")
        Smooth a little ("MEDIUM")
        Smooth a lot ("HIGH")
        :type level: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(level == "LOW") : 
                self.dll.Imop_LoopSmoothingParams_NewLow(
                    message,
                    ctypes.pointer(self.loopsmoothing)
                    )
            elif(level == "MEDIUM") : 
                self.dll.Imop_LoopSmoothingParams_NewMedium(
                    message,
                    ctypes.pointer(self.loopsmoothing)
                    )
            elif(level == "HIGH") : 
                self.dll.Imop_LoopSmoothingParams_NewHigh(
                    message,
                    ctypes.pointer(self.loopsmoothing)
                    )
            else :
                raise Exception('IO_Error','level value must be ''LOW'', ''MEDIUM'' or ''HIGH''')
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_levels',exception)

    def __init__(self, **kwargs):
        """LoopSmoothing constructor
        """
        self.loopsmoothing = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 4):
                if('low_gain_factor' in kwargs
                   and 'high_gain_factor' in kwargs
                   and 'low_amplitude' in kwargs
                   and 'high_amplitude' in kwargs):
                    entered = 1
                    self.__init_from_params(kwargs['low_gain_factor'], kwargs['high_gain_factor'], kwargs['low_amplitude'], kwargs['high_amplitude'])
            if(arg_size == 1):
                if('level' in kwargs):
                    entered = 1
                    self.__init_from_levels(kwargs['level'])
        except Exception as exception:            
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE PUPIL OBJECT---')
    
    def __del_obj__(self):
        """LoopSmoothing destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_LoopSmoothingParams_Delete(message, self.loopsmoothing)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)
            
    def set_params(
        self,
        low_gain_factor,
        high_gain_factor,
        low_amplitude,
        high_amplitude
        ):
        """Set closed loop smoothing parameters.
        
        :param low_gain_factor: lowest value for gain factor
        :type low_gain_factor: float
        :param high_gain_factor: highest value for gain factor
        :type high_gain_factor: float
        :param low_amplitude: lowest amplitude value before switching to linear behavior
        :type low_amplitude: int
        :param high_amplitude: highest amplitude value where linear behavior stops
        :type high_amplitude: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_LoopSmoothingParams_Set(
                message,
                self.loopsmoothing,
                ctypes.c_float(low_gain_factor),
                ctypes.c_float(high_gain_factor),
                ctypes.c_int(low_amplitude),
                ctypes.c_int(high_amplitude)                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set',exception)
            
    def get_params(
        self
        ):
        """Get closed loop smoothing parameters.
        
        :return: lowest value , highest value for gain factor, lowest amplitude , highest amplitude value where linear behavior stops
        :rtype: tuple(float, float, int, int)
        """
        try:
            message = ctypes.create_string_buffer(256)
            low_gain_factor = ctypes.c_float()
            high_gain_factor = ctypes.c_float()
            low_amplitude = ctypes.c_int()
            high_amplitude = ctypes.c_int()
            self.dll.Imop_LoopSmoothingParams_Get(
                message,
                self.loopsmoothing,
                ctypes.byref(low_gain_factor),
                ctypes.byref(high_gain_factor),
                ctypes.byref(low_amplitude),
                ctypes.byref(high_amplitude)                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (low_gain_factor.value,
                    high_gain_factor.value,
                    low_amplitude.value,
                    high_amplitude.value
                    )
        except Exception as exception:
            raise Exception(__name__+' : get',exception)
            
    def compute_gain_factor(
        self,
        amplitude
        ):
        """Compute smoothing gain as a decreasing function of delta command amplitude.
        This function is called in the CorrDataManager.compute_closed_loop_iteration function,
        where the applied_loop_gain is computed as the product of the computed factor and the specified_loop_gain.
        
        :return: multiplicative factor to apply for final smoothing gain computation
        :rtype: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            factor = ctypes.c_float()
            self.dll.Imop_LoopSmoothingParams_ComputeGainFactor(
                message,
                self.loopsmoothing,
                ctypes.c_int(amplitude),
                ctypes.byref(factor)                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return factor.value
        except Exception as exception:
            raise Exception(__name__+' : compute_gain_factor',exception)