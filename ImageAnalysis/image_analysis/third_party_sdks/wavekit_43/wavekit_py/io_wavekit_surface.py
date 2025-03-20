#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct

class Surface(object):
    """Class Surface
    
    - Constructor from Dimensions :
        - **dimensions** - Dimensions : Dimensions of the Surface
    
    - Constructor from Copy :
        - **surface** - Surface : Surface to copy
    """
    def __init_(
        self, 
        dimensions
        ):
        """Surface constructor from size and steps
        
        :param dimensions: dimensions of the surface
        :type dimensions: dimensions
        """
        try:
            if type(dimensions) is not imop_struct.dimensions:
                raise Exception('IO_Error', 'dimensions must be an io_wavekit_structure.dimensions class')
            message = ctypes.create_string_buffer(256)
            size = dimensions.size
            steps = dimensions.steps
            self.dll.Imop_Surface_New(
                message,
                ctypes.byref(size),
                ctypes.byref(steps),
                ctypes.pointer(self.surface)
                )               
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_',exception)

    def __init_from_copy(
        self, 
        surface
        ):
        """Surface constructor from copy
        
        :param surface: Surface to copy
        :type surface: Surface
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Surface_NewFromCopy(
                message,
                ctypes.pointer(self.surface),
                surface.surface
                )               
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_copy',exception)
            
    def __init__(self, **kwargs):
        """Surcace Constructor
        """
        self.surface = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('dimensions' in kwargs):
                    entered = 1
                    self.__init_(kwargs['dimensions'])
                if('surface' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['surface'])
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE SURFACE OBJECT---')
            
    def __del_obj__(self):
        """Surcace Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_Surface_Delete(message, self.surface)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_data(self):
        """Get read access to Surface buffer
        
        :return: Surface buffer
        :rtype: 2D float numpy array
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = self.get_dimensions().size
            float_arr = numpy.zeros(
                (size.Y, size.X), 
                dtype = numpy.single
                )
            self.dll.Imop_Surface_GetData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_Surface_GetData(
                message,
                self.surface,
                float_arr
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return float_arr
        except Exception as exception:
            raise Exception(__name__+' : get_data',exception)

    def set_data(
        self, 
        float_arr
        ):
        """Get write access to Surface buffer
        
        :param float_arr: Surface buffer to set
        :type float_arr: float numpy array
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Surface_SetData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_Surface_SetData(
                message,
                self.surface,
                float_arr
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_data',exception)

    def get_dimensions(self):
        """Get Surface properties
        
        :return: Dimensions of the Surface object
        :rtype: dimensions
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = imop_struct.uint2D(0,0)
            steps = imop_struct.float2D(0.0,0.0)
            self.dll.Imop_Surface_GetDimensions(
                message,
                self.surface,
                ctypes.byref(size),
                ctypes.byref(steps)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.dimensions(size, steps)
        except Exception as exception:
            raise Exception(__name__+' : get_dimensions',exception)

    def get_statistics(self):
        """Get Surface statistics
        
        :return: Surface statistics as tuple : (root mean square deviation, peak to valley, maximum, minimum)
        :rtype: tuple(float, float, float, float)
        """
        try:
            message = ctypes.create_string_buffer(256)
            rms = ctypes.c_double()
            pv = ctypes.c_double()
            max_ = ctypes.c_double()
            min_ = ctypes.c_double()
            self.dll.Imop_Surface_GetStatistics(
                message,
                self.surface,
                ctypes.byref(rms),
                ctypes.byref(pv),
                ctypes.byref(max_),
                ctypes.byref(min_)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.Statistics_t(
                    rms.value,
                    pv.value,
                    max_.value,
                    min_.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_statistics',exception)
