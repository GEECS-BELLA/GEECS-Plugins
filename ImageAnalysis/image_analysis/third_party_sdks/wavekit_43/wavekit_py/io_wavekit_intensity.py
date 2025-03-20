#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_pupil as imop_pupil

class Intensity(object):
    """Class Intensity
    
    - Constructor from Dimensions :
        - **dimensions** - dimensions : Dimensions of Intensity map
    
    - Constructor from HasoSlopes :
        - **hasoslopes** - HasoSlopes : HasoSlopes object
    """

    def __init_from_dimensions(
        self,
        dimensions
        ):
        """*Intensity* constructor from dimensions and steps
        All the elements of the Phase values buffer are set to zero and the elements
        of the pupil are set to true.
        
        :param dimensions: Intensity dimensions of the pupil
        :type dimensions: dimensions
        """
        try:
            message = ctypes.create_string_buffer(256)
            if type(dimensions) is not imop_struct.dimensions:
                raise Exception('IO_Error', 'dimensions must be an io_wavekit_structure.dimensions class')
            size = dimensions.size
            steps = dimensions.steps
            self.dll.Imop_Intensity_NewFromDimAndSteps(
                message,
                ctypes.byref(size),
                ctypes.byref(steps),
                ctypes.pointer(self.intensity)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_dimensions',exception)

    def __init_from_hasoslopes(
        self,
        hasoslopes
        ):
        """*Intensity* constructor from HasoSlopes
        Extract intensity values from HasoSlopes object and builds Intensity object
        
        :param hasoslopes: HasoSlopes object
        :type hasoslopes: HasoSlopes
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Intensity_NewFromSlopes(
                message,
                ctypes.pointer(self.intensity),
                hasoslopes.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_hasoslopes',exception)

    def __init__(self, **kwargs):
        """Intensity Constructor
        """
        self.intensity = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('dimensions' in kwargs):
                    entered = 1
                    self.__init_from_dimensions(kwargs['dimensions'])  
                if('hasoslopes' in kwargs):
                    entered = 1
                    self.__init_from_hasoslopes(kwargs['hasoslopes'])
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):            
            raise Exception('IO_Error','---CAN NOT CREATE INTENSITY OBJECT---')

    def __del_obj__(self):
        """Intensity Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_Intensity_Delete(message, self.intensity)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_dimensions(self):
        """ Get Intensity dimensions
        
        :return: Intensity dimensions of the pupil
        :rtype: dimensions
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = imop_struct.uint2D(0, 0)
            steps_out = imop_struct.float2D(0.0, 0.0)
            self.dll.Imop_Intensity_GetDimensionsAndSteps(
                message,
                self.intensity,
                ctypes.byref(size_out),
                ctypes.byref(steps_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.dimensions(size_out, steps_out)
        except Exception as exception:
            raise Exception(__name__+' : get_dimensions',exception)

    def get_data(self):
        """Get Intensity buffer and pupil
        
        :return: Intensity values buffer, pupil object
        :rtype: tuple(float 2D numpy.array, Pupil)
        """
        try:
            message = ctypes.create_string_buffer(256)
            dim = self.get_dimensions()
            buffer_out = numpy.zeros(
                (dim.size.Y, dim.size.X),
                dtype = numpy.single
                )
            pupil_out = imop_pupil.Pupil(
                dimensions = dim,
                value = 1
                )                             
            self.dll.Imop_Intensity_GetData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ]            
            self.dll.Imop_Intensity_GetData(
                message,
                self.intensity,
                buffer_out,
                pupil_out.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                buffer_out, 
                pupil_out
                )
        except Exception as exception:
            raise Exception(__name__+' : get_data',exception)

    def get_statistics(self):
        """Get computed statistics of Intensity 
        
        :return: Intensity root mean square deviation, peak to valley, minimum, maximum
        :rtype: tuple(double, double, double, double)
        """
        try:
            message = ctypes.create_string_buffer(256)     
            rms = ctypes.c_double() 
            pv = ctypes.c_double() 
            max_ = ctypes.c_double() 
            min_ = ctypes.c_double()
            self.dll.Imop_Intensity_GetStatistics(
                message,
                self.intensity,
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

    def resize(
        self,
        resize_factor,
        do_erode
        ):
        """Resize and interpolate Intensity
        
        :param resize_factor: resize factor : output intensity width (or height) = factor * input intensity width (or height)
        :type resize_factor: uchar
        :param do_erode: if equal to 1, intensity borders are eroded to avoid weird reconstructed values
        :type do_erode: uchar
        :return: Resized Intensity object
        :rtype: Intensity
        """
        try:
            message = ctypes.create_string_buffer(256)   
            intensity_out = Intensity(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    )
                )
            self.dll.Imop_Intensity_Resize(
                message,
                self.intensity,
                ctypes.c_ubyte(resize_factor),
                ctypes.c_ubyte(do_erode),
                intensity_out.intensity
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return intensity_out
        except Exception as exception:
            raise Exception(__name__+' : resize',exception)

    def set_data(
        self,
        datas,
        pupil):
        """Set Intensity buffer and pupil
        
        :param datas: values buffer
        :type datas: float 2D numpy.array
        :param pupil: pupil object
        :type pupil: Pupil
        """
        try:
            message = ctypes.create_string_buffer(256)           
            self.dll.Imop_Intensity_SetData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ]    
            self.dll.Imop_Intensity_SetData(
                message,
                self.intensity,
                datas,
                pupil.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_data',exception)
