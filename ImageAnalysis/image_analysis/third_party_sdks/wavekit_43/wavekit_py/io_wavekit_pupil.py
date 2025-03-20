#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct

class Pupil(object):
    """Class Pupil
    
    - Constructor from Dimensions :
        - **dimensions** - dimensions : Pupil dimensions 
        - **value** - bool : Inital value
    
    - Constructor from HasoSlopes :
        - **hasoslopes** - HasoSlopes : HasoSlopes dimensions 
    
    - Constructor from Legendre Pupil :
        - **dimensions** - dimensions : Pupil dimensions 
        - **center** - float2D : Center of the rectangle shape (nb subpupils)
        - **rectangle_size** - float2D : Shape (half-width, half-height) of the rectangle (nb subpupils)
    
    - Constructor from Zernike Pupil :
        - **dimensions** - dimensions : Pupil dimensions
        - **center** - float2D : Center of the circle shape
        - **radius** - float : Radius of the circle shape
    
    - Constructor from Copy :
        - **pupil** - Pupil : Pupil object to copy
    """

    def __init_from_dimensions(
        self, 
        dimensions,
        value
        ):
        """Pupil constructor from size, steps and inital value
        
        :param dimensions: Pupil dimensions (array dimensions, Step x and y between the subpupils)
        :type dimensions: dimensions
        :param value: Inital value
        :type value: bool
        """
        try:
            if type(dimensions) is not imop_struct.dimensions:
                raise Exception('IO_Error', 'dimensions must be an io_wavekit_structure.dimensions class')
            message = ctypes.create_string_buffer(256)
            size = dimensions.size
            steps = dimensions.steps
            self.dll.Imop_Pupil_NewFromDimensions(
                message,
                ctypes.pointer(self.pupil),
                ctypes.byref(size),
                ctypes.byref(steps),
                ctypes.c_ubyte(value)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_dimensions',exception)

    def __init_from_hasoslopes(
        self, 
        hasoslopes
        ):
        """Pupil constructor from HasoSlopes
        
        :param hasoslopes: HasoSlopes input
        :type hasoslopes: HasoSlopes
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Pupil_NewFromHasoSlopes(
                message,
                ctypes.pointer(self.pupil),
                hasoslopes.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_hasoslopes',exception)

    def __init_from_legendrepupil(
        self, 
        dimensions, 
        center, 
        rectangle_size
        ):
        """Rectangular Pupil constructor from dimensions, steps and rectangle shape
        
        :param dimensions: Pupil dimensions (array dimensions, Step x and y between the subpupils)
        :type dimensions: dimensions
        :param center: Center of the rectangle shape (nb subpupils)
        :type center: float2D
        :param rectangle_size: Shape (half-width, half-height) of the rectangle (nb subpupils)
        :type rectangle_size: float2D
        """
        try:
            if type(dimensions) is not imop_struct.dimensions:
                raise Exception('IO_Error', 'dimensions must be an io_wavekit_structure.dimensions class')
            if type(center) is not imop_struct.float2D:
                raise Exception('IO_Error', 'center must be an io_wavekit_structure.float2D class')
            if type(rectangle_size) is not imop_struct.float2D:
                raise Exception('IO_Error', 'rectangle_size must be an io_wavekit_structure.float2D class')
            message = ctypes.create_string_buffer(256)
            size = dimensions.size
            steps = dimensions.steps
            self.dll.Imop_Pupil_NewFromLegendrePupil(
                message,
                ctypes.pointer(self.pupil),
                ctypes.byref(steps),
                ctypes.byref(size),
                ctypes.byref(center),
                ctypes.byref(rectangle_size)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_legendrepupil',exception)

    def __init_from_zernikepupil(
        self,
        dimensions,
        center,
        radius
        ):
        """Rectangular Pupil constructor from dimensions, steps and rectangle shape
        
        :param dimensions: Pupil dimensions (array dimensions, Step x and y between the subpupils)
        :type dimensions: dimensions
        :param center: Center of the rectangle shape (nb subpupils)
        :type center: float2D
        :param radius: Radius of the circle shape (nb subpupils)
        :type radius: float
        """
        try:
            if type(dimensions) is not imop_struct.dimensions:
                raise Exception('IO_Error', 'dimensions must be an io_wavekit_structure.dimensions class')
            if type(center) is not imop_struct.float2D:
                raise Exception('IO_Error', 'center must be an io_wavekit_structure.float2D class')
            message = ctypes.create_string_buffer(256)
            size = dimensions.size
            steps = dimensions.steps
            self.dll.Imop_Pupil_NewFromZernikePupil(
                message,
                ctypes.pointer(self.pupil),
                ctypes.byref(steps),
                ctypes.byref(size),
                ctypes.byref(center),
                ctypes.byref(ctypes.c_float(radius))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_zernikepupil',exception) 

    def __init_from_copy(
        self,
        pupil
        ):
        """Pupil constructor from copy
        
        :param pupil: Pupil object
        :type pupil: Pupil
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Pupil_NewFromCopy(
                message,
                ctypes.pointer(self.pupil),
                pupil.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_copy',exception) 

    def __init__(self, **kwargs):
        """Pupil Constructor
        """
        self.pupil = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 2):
                if('dimensions' in kwargs
                   and 'value' in kwargs):
                    entered = 1
                    self.__init_from_dimensions(kwargs['dimensions'], kwargs['value'])
            if(arg_size == 1):
                if('hasoslopes' in kwargs):
                    entered = 1
                    self.__init_from_hasoslopes(kwargs['hasoslopes'])  
                if('pupil' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['pupil'])
            if(arg_size == 3):
                if('dimensions' in kwargs
                   and 'center' in kwargs
                   and 'rectangle_size' in kwargs):
                    entered = 1
                    self.__init_from_legendrepupil(kwargs['dimensions'], kwargs['center'], kwargs['rectangle_size'])
                if('dimensions' in kwargs
                   and 'center' in kwargs
                   and 'radius' in kwargs):
                    entered = 1
                    self.__init_from_zernikepupil(kwargs['dimensions'], kwargs['center'], kwargs['radius'])
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE PUPIL OBJECT---')
    
    def __del_obj__(self):
        """Pupil Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_Pupil_Delete(message, self.pupil)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def __or__(
        self,
        pupil
        ):
        """OR operator
        
        :param pupil: Pupil object
        :type pupil: Pupil
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Pupil_OR(
                message,
                self.pupil,
                pupil.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __or__',exception)

    def __and__(
        self,
        pupil
        ):
        """AND operator
        
        :param pupil: Pupil object
        :type pupil: Pupil
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Pupil_AND(
                message,
                self.pupil,
                pupil.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __and__',exception)

    def __invert__(
        self
        ):
        """INVERT operator
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Pupil_INVERT(
                message,
                self.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __invert__',exception)

    def get_barycenter(self):
        """Get Pupil barycenter
        
        :return: Pupil barycenter
        :rtype: float2D
        """
        try:
            message = ctypes.create_string_buffer(256)
            ret = imop_struct.float2D(0.0,0.0)
            self.dll.Imop_Pupil_GetBarycenter(
                message,
                self.pupil,
                ctypes.byref(ret)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return ret
        except Exception as exception:
            raise Exception(__name__+' : get_barycenter',exception)

    def get_nb_enlighted_subapertures(self):
        """Get Pupil enlighted subapertures number
        
        :return: Number of enlighted subapertures
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            ret = ctypes.c_int()
            self.dll.Imop_Pupil_GetNbEnlightedSubapertures(
                message,
                self.pupil,
                ctypes.byref(ret)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return ret.value
        except Exception as exception:
            raise Exception(__name__+' : get_nb_enlighted_subapertures',exception)

    def resize(
        self, 
        resize_factor
        ):
        """Resize pupil
        
        :param resize_factor: resize factor : output pupil width (or height) = factor * input pupil width (or height)
        :type resize_factor: uchar
        :return: Resized Pupil
        :rtype: Pupil
        """
        try:
            message = ctypes.create_string_buffer(256)
            pupil_out = Pupil(
                dimensions = self.get_dimensions(),
                value = True
                )
            self.dll.Imop_Pupil_Resize(
                message,
                self.pupil,
                ctypes.c_ubyte(resize_factor),
                pupil_out.pupil                                                           
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return pupil_out
        except Exception as exception:
            raise Exception(__name__+' : resize',exception)

    def has_central_occultation(self):
        """Find if pupil has one or more central occultation(s)
        
        :return: True if Pupil has one or more central occultation(s)
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            ret = ctypes.c_bool()
            self.dll.Imop_Pupil_HasCentralOccultation(
                message,
                self.pupil,
                ctypes.byref(ret)                                                          
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return ret.value
        except Exception as exception:
            raise Exception(__name__+' : has_central_occultation',exception)

    def get_regions_stats(self):
        """Get statistics on pupil regions
        
        :return: Number of regions where subpupils are off, enlighted, Total number of regions
        :rtype: tuple(int, int, int)
        """
        try:
            message = ctypes.create_string_buffer(256)
            black = ctypes.c_int()
            white = ctypes.c_int()
            number = ctypes.c_int()
            self.dll.Imop_Pupil_RegionsStats(
                message,
                self.pupil,
                ctypes.byref(black),  
                ctypes.byref(white), 
                ctypes.byref(number)                                                          
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                black.value,
                white.value, 
                number.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_regions_stats',exception)

    def get_regions_positions(self):
        """Get statistics on pupil regions
        
        :return: Array of bottom edge, Array of left edge, Array of right edge, Array of top edge position for each region's bounding rectangle, Array region's type (0 if black region)
        :rtype: tuple(int list[], int list[], int list[], int list[], int list[])
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb = self.get_regions_stats()[2]
            black  = ctypes.c_int()
            white  = ctypes.c_int()
            number = ctypes.c_int()
            bottom = numpy.zeros(nb, dtype = numpy.intc)
            left   = numpy.zeros(nb, dtype = numpy.intc)
            right  = numpy.zeros(nb, dtype = numpy.intc)
            top    = numpy.zeros(nb, dtype = numpy.intc)
            values = numpy.zeros(nb, dtype = numpy.intc)            
            self.dll.Imop_Pupil_RegionsPositions.argtypes = [
                ctypes.c_char_p, #Message
                ctypes.c_void_p, #Pupil
                ctypes.c_void_p, #NB Black
                ctypes.c_void_p, #NB White
                ctypes.c_void_p, #NB Regions
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"), #Bottoms
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"), #Lefts
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"), #Rights
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS"), #Tops
                numpy.ctypeslib.ndpointer(numpy.intc, flags="C_CONTIGUOUS")  #Values
                ]
            self.dll.Imop_Pupil_RegionsPositions(
                message,
                self.pupil,
                ctypes.byref(black),  
                ctypes.byref(white), 
                ctypes.byref(number),
                bottom,
                left,
                right,
                top,
                values
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                bottom.tolist(), 
                left.tolist(), 
                right.tolist(), 
                top.tolist(), 
                values.tolist()
                )
        except Exception as exception:
            raise Exception(__name__+' : get_regions_positions',exception)

    def get_dimensions(self):
        """Get Pupil size and steps
        
        :return: Dimensions of the pupil
        :rtype: dimensions
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = imop_struct.uint2D(0,0)
            steps = imop_struct.float2D(0.0,0.0)
            self.dll.Imop_Pupil_GetDimensions(
                message,
                self.pupil,
                ctypes.byref(size),  
                ctypes.byref(steps)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.dimensions(size, steps)
        except Exception as exception:
            raise Exception(__name__+' : get_dimensions',exception)

    def get_data(self):
        """Get Pupil buffer
        
        :return: Pupil buffer
        :rtype: bool 2D numpy.array
        """
        try:
            message = ctypes.create_string_buffer(256)            
            dimensions = self.get_dimensions()
            values = numpy.zeros(
                (dimensions.size.Y, dimensions.size.X),
                dtype = numpy.bool_
                )            
            self.dll.Imop_Pupil_GetData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p, #Pupil
                numpy.ctypeslib.ndpointer(numpy.bool_, flags="C_CONTIGUOUS")  #Values
                ]
            self.dll.Imop_Pupil_GetData(
                message,
                self.pupil,
                values
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return values
        except Exception as exception:
            raise Exception(__name__+' : get_data',exception)

    def set_data(
        self,
        datas
        ):
        """Set Pupil buffer
        
        :param datas: Pupil buffer
        :type datas: bool 2D numpy.array
        """
        try:
            message = ctypes.create_string_buffer(256)            
            self.dll.Imop_Pupil_SetData.argtypes = [
                ctypes.c_char_p, #Message
                ctypes.c_void_p, #Pupil
                numpy.ctypeslib.ndpointer(numpy.bool_, flags="C_CONTIGUOUS")  #Values
                ]
            self.dll.Imop_Pupil_SetData(
                message,
                self.pupil,
                datas
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_data',exception)

    def set_data_from_text(
        self,
        text_file_path
        ):
        """Set Pupil buffer from text data.
        
        :param text_file_path: text file to read data from
        :type text_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Pupil_SetDataFromText(
                message,
                self.pupil,
                ctypes.c_char_p(text_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_data_from_text',exception)

    def save_data_to_text(
        self,
        text_file_path
        ):
        """Save Pupil buffer to text file.
        
        :param text_file_path: text file in which to save data
        :type text_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Pupil_SaveDataToText(
                message,
                self.pupil,
                ctypes.c_char_p(text_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : save_data_to_text',exception)
