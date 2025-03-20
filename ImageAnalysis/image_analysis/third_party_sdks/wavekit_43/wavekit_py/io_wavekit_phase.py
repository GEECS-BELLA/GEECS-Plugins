#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_pupil as imop_pupil
import io_wavekit_enum as imop_enum
class Phase(object):
    """Class Phase
    
    - Constructor from Copy :
        - **phase** - Phase : Phase object to copy
    
    - Constructor from Dimensions :
        - **dimensions** - dimensions : Phase dimensions 
    
    - Constructor from Pupil : 
        - **pupil** - Pupil : Pupil object
        - **default_value** - float : Phase default value
    
    - Constructor from HasoSlopes : 
        - **hasoslopes** - HasoSlopes : HasoSlopes object
        - **type_** - E_COMPUTEPHASESET : Phase reconstruction mode
        
        Modal reconstruction, Legendre basis
        Modal reconstruction, Zernike basis
        Zonal reconstruction      
        
        - **filter_** - bool list[] : Array of aberrations to filter 
        
        filter[0] : tiltx
        filter[1] : tilty
        filter[2] : curvature
        filter[3] : astigmatism 0 degree
        filter[4] : astigmatism 45 degree
        
        - **nb_coeffs** - uchar : Optional number of coefficients to use for reconstruction
    
    - Constructor from ModalCoef : 
        - **modalcoef** - ModalCoef : ModalCoef object
        - **filter_** - bool list[] : Array of aberrations to filter      
        
        filter[0] : tiltx
        filter[1] : tilty
        filter[2] : curvature
        filter[3] : astigmatism 0 degree
        filter[4] : astigmatism 45 degree
    """
    
    def __init_from_copy(
        self, 
        phase
        ):
        """*Phase* constructor from copy
        
        :param phase: Phase object
        :type phase: Phase
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Phase_NewFromCopy(
                message,
                phase.phase,
                ctypes.pointer(self.phase)
            )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_copy',exception)
    
    def __init_from_dimensions(
        self, 
        dimensions
        ):
        """*Phase* constructor from dimensions
        All the elements of the Phase values buffer are set to zero and the elements
        of the pupil are set to true.
        
        :param dimensions: Phase dimensions containing size and steps
        :type dimensions: dimensions
        """
        try:
            message = ctypes.create_string_buffer(256)
            if type(dimensions) is not imop_struct.dimensions:
                raise Exception('IO_Error', 'dimensions must be an io_wavekit_structure.dimensions class')
            size = dimensions.size
            steps = dimensions.steps
            self.dll.Imop_Phase_NewFromDimAndSteps(
                message,
                ctypes.byref(size),
                ctypes.byref(steps),
                ctypes.pointer(self.phase)
            )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_dimensions',exception)

    def __init_from_pupil(
        self, 
        pupil,
        default_value
        ):
        """*Phase* constructor from pupil
        
        :param pupil: Phase dimensions containing size and steps
        :type pupil: dimensions
        :param default_value: Phase default value
        :type default_value: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Phase_NewFromPupil(
                message,
                pupil.pupil,
                ctypes.c_float(default_value),
                ctypes.pointer(self.phase)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_pupil',exception)

    def __init_from_hasoslopes(
        self, 
        hasoslopes, 
        type_, 
        filter_, 
        nb_coeffs = None
        ):
        """*Phase* constructor from HasoSlopes.
        This constructor computes the Phase using the selected method :
        1 = Modal reconstruction, Legendre basis
        2 = Modal reconstruction, Zernike basis
        3 = Zonal reconstruction
        and creates the corresponding *Phase* object.
        The Phase reconstruction parameters (ComputePhaseSet) are all set to their default values
        and cannot be modified, except the aberration filter (False = aberration is removed) and
        the number of coefficients to be used for modal reconstructions (before filtering).
        The correspondance between aberrations and polynomial modes (Modal reconstructions only) is
        automatically computed.
        - filter[0] : tiltx
        - filter[1] : tilty
        - filter[2] : curvature
        - filter[3] : astigmatism 0 degree
        - filter[4] : astigmatism 45 degree
        The projection pupil (Modal reconstructions only) is automatically computed from the slopes natural pupil.
        
        :param hasoslopes: HasoSlopes object
        :type hasoslopes: HasoSlopes
        :param type_: Phase reconstruction mode (Modal Legendre, Modal Zernike or Zonal only)
        :type type_: E_COMPUTEPHASESET
        :param filter_: Array of aberrations to filter
        Filter must be a 5 elements array
        :type filter_: bool list[]
        :param nb_coeffs: Optional number of coefficients to use for reconstruction.
        This parameters is mandatory if case Phase reconstruction mode is Modal Legendre or Modal Zernike,
        This parameters is ignored in reconstruction mode Zonal
        :type nb_coeffs: dimensions
        """
        try:
            message = ctypes.create_string_buffer(256)
            entered = False
            if (len(filter_) !=5):
                raise Exception('IO_Error', 'Filter size must be 5')
            if(nb_coeffs == None
               and type_ == imop_enum.E_COMPUTEPHASESET.ZONAL
               ) :     
                self.dll.Imop_Phase_NewFromSlopes.argtypes = [
                    ctypes.c_char_p, 
                    ctypes.c_void_p, 
                    ctypes.c_void_p,
                    ctypes.c_ubyte,  
                    numpy.ctypeslib.ndpointer(numpy.ubyte, flags="C_CONTIGUOUS"), 
                    ctypes.c_void_p     
                    ]  
                entered = True          
                self.dll.Imop_Phase_NewFromSlopes(
                    message,
                    ctypes.pointer(self.phase),
                    hasoslopes.hasoslopes,
                    type_,
                    numpy.array(filter_, dtype = numpy.ubyte),
                    ctypes.byref(ctypes.c_ubyte(0))
                    )
            elif(nb_coeffs != None):
                self.dll.Imop_Phase_NewFromSlopes.argtypes = [
                    ctypes.c_char_p, 
                    ctypes.c_void_p, 
                    ctypes.c_void_p,
                    ctypes.c_ubyte,  
                    numpy.ctypeslib.ndpointer(numpy.ubyte, flags="C_CONTIGUOUS"), 
                    ctypes.c_void_p               
                    ]
                entered = True  
                self.dll.Imop_Phase_NewFromSlopes(
                    message,
                    ctypes.pointer(self.phase),
                    hasoslopes.hasoslopes,
                    type_,
                    numpy.array(filter_, dtype = numpy.ubyte),
                    ctypes.byref(ctypes.c_ubyte(nb_coeffs))
                    )
            else:
                raise Exception(__name__+' : __init_from_hasoslopes','nb_coeffs can\'t be null with MODAL option')
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_hasoslopes',exception)

    def __init_from_modal_coef(
        self, 
        modalcoef, 
        filter_
        ):
        """*Phase* constructor from *ModalCoef*.
        This constructor computes the slopes on the input *ModalCoef* object
        depending on the projection type (Legendre or Zernike)
        and number of coefficients,
        then it computes the Phase and creates the corresponding Phase object.
        The slopes computation and Phase reconstruction parameters (ComputePhaseSet)
        are all set to their default values and cannot be modified.
        Modal coeffs must have a projection pupil set in their preferences.
        
        .. seealso:: ComputePupil.fit_zernike_pupil to fit a circular pupil to a natural slopes pupil for Zernike projection
        
        .. seealso:: ComputePupil.fit_legendre_pupil to fit rectangular pupil to a natural slopes pupil for Legendre projection
        
        .. seealso:: ModalCoef.set_zernike_prefs for setting preferences for a *ModalCoef* of type Zernike
        
        .. seealso:: ModalCoef.set_legendre.prefs for setting preferences for a *ModalCoef* of type Legendre
        
        :param modalcoef: ModalCoef object
        :type modalcoef: ModalCoef
        :param filter_: Array of aberrations to filter
        :type filter_: uchar list[]
        """
        try:
            message = ctypes.create_string_buffer(256)             
            self.dll.Imop_Phase_NewFromModalCoef.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.ubyte, flags="C_CONTIGUOUS")                     
                ]       
            self.dll.Imop_Phase_NewFromModalCoef(
                message,
                ctypes.pointer(self.phase),
                modalcoef.modalcoef,
                numpy.array(filter_, numpy.ubyte)
                )               
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_modal_coef',exception)
            
    def __init__(self, **kwargs):
        """Phase Constructor
        """
        self.phase = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('dimensions' in kwargs):
                    entered = 1
                    self.__init_from_dimensions(kwargs['dimensions'])
                if('phase' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['phase'])
            if(arg_size == 2):
                if('pupil' in kwargs
                   and 'default_value' in kwargs):
                    entered = 1
                    self.__init_from_pupil(kwargs['pupil'], kwargs['default_value'])
                if('modalcoef' in kwargs
                   and 'filter_' in kwargs):
                    entered = 1
                    self.__init_from_modal_coef(kwargs['modalcoef'], kwargs['filter_'])
            if(arg_size == 3):
                if('hasoslopes' in kwargs
                   and 'type_' in kwargs
                   and 'filter_' in kwargs):
                    entered = 1
                    self.__init_from_hasoslopes(kwargs['hasoslopes'], kwargs['type_'], kwargs['filter_'])
            if(arg_size == 4):
                if('hasoslopes' in kwargs
                   and 'type_' in kwargs
                   and 'filter_' in kwargs
                   and 'nb_coeffs' in kwargs):
                    entered = 1
                    self.__init_from_hasoslopes(kwargs['hasoslopes'], kwargs['type_'], kwargs['filter_'], kwargs['nb_coeffs'])
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE PHASE OBJECT---')
            
    def __del_obj__(self):
        """Phase Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_Phase_Delete(
            message, 
            self.phase
            )
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_data(self):
        """Get Phase buffer and pupil 
        
        :return: Phase values buffer, pupil object
        :rtype: tuple(2D float numpy array, Pupil)
        """
        try:
            message = ctypes.create_string_buffer(256)
            dim = self.get_dimensions()
            size = dim.size
            float_arr = numpy.zeros(
                (size.Y, size.X),
                dtype = numpy.single
                )
            pup_ret = imop_pupil.Pupil(
                dimensions = dim,
                value = True
                )
            self.dll.Imop_Phase_GetData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ]
            self.dll.Imop_Phase_GetData(
                message,
                self.phase,
                float_arr,
                pup_ret.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                float_arr, 
                pup_ret
                )
        except Exception as exception:
            raise Exception(__name__+' : get_data',exception)

    def get_dimensions(self):
        """Get Phase dimensions
        
        :return: Dimensions
        :rtype: dimensions
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = imop_struct.uint2D(0,0)
            steps = imop_struct.float2D(0.0,0.0)
            self.dll.Imop_Phase_GetDimensionsAndSteps(
                message,
                self.phase,
                ctypes.byref(size),
                ctypes.byref(steps)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.dimensions(
                size, 
                steps
                )
        except Exception as exception:
            raise Exception(__name__+' : get_dimensions',exception)

    def get_pupil(self):
        """Get Phase pupil
        
        :return: Pupil object
        :rtype: Pupil
        """
        try:
            message = ctypes.create_string_buffer(256)
            dim = self.get_dimensions()
            pup_ret = imop_pupil.Pupil(
                dimensions = dim,
                value = True
                )
            self.dll.Imop_Phase_GetPupil(
                message,
                self.phase,
                pup_ret.pupil
                )            
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return pup_ret
        except Exception as exception:
            raise Exception(__name__+' : get_pupil',exception)

    def get_statistics(self):
        """Get computed statistics
        
        :return: Phase statistics : (root mean square deviation, peak to valley, maximum, minimum)
        :rtype: tuple (float, float, float, float)
        """
        try:
            message = ctypes.create_string_buffer(256)
            rms_out = ctypes.c_double()
            pv_out = ctypes.c_double()
            max_out = ctypes.c_double()
            min_out = ctypes.c_double()
            self.dll.Imop_Phase_GetStatistics(
                message,
                self.phase,
                ctypes.byref(rms_out),
                ctypes.byref(pv_out),
                ctypes.byref(max_out),
                ctypes.byref(min_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.Statistics_t(
                rms_out.value, 
                pv_out.value, 
                max_out.value, 
                min_out.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_statistics',exception)

    def resize(
        self, 
        resize_factor, 
        do_erode
        ):
        """Resize and interpolate Phase
        
        :param resize_factor: resize factor : output phase width (or height) = factor * input phase width (or height)
        :type resize_factor: uchar
        :param do_erode: resize factor : if equal to 1, intensity borders are eroded to avoid weird reconstructed values.
        :type do_erode: uchar
        :return: Resized Phase
        :rtype: Phase
        """
        try:
            message = ctypes.create_string_buffer(256)
            dim = self.get_dimensions()
            phase_ret = Phase(dimensions = dim)
            self.dll.Imop_Phase_Resize(
                message,
                self.phase,
                ctypes.c_ubyte(resize_factor),
                ctypes.c_ubyte(do_erode),
                phase_ret.phase
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return phase_ret
        except Exception as exception:
            raise Exception(__name__+' : resize',exception)

    def set_data(
        self, 
        buffer_array, 
        pupil
        ):
        """Set Phase buffer and pupil
        
        :param buffer_array: Phase values buffer
        :type buffer_array: 2D float numpy array
        :param pupil: pupil object
        :type pupil: Pupil
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Phase_SetData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ]
            self.dll.Imop_Phase_SetData(
                message,
                self.phase,
                buffer_array,
                pupil.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_data',exception)
        
        
