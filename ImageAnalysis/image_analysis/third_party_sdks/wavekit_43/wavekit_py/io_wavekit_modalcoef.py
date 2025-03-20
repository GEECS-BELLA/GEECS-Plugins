#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_pupil as imop_pupil
import io_wavekit_structure as imop_struct

class ModalCoef(object):
    """Class ModalCoef
    
    - Constructor from Modal Type :
        - **modal_type** - E_MODAL : Polynomial basis type
    
    - Constructor from Data :
        - **modal_type** - E_MODAL : Polynomial basis type
        - **polyindex** - uint list[] : Array of modal coefficient indices
        - **coef** - float list[] : Array of modal coefficient values
        - **mask** - Pupil : Mask object stored as a pupil object
    
    - Constructor from File :
        - **modal_type** - E_MODAL : Polynomial basis type
        - **coef_file_path** - string : Modal coefficients file path
    
    - Constructor Zernike from Slopes :
        - **modal_normalization** - E_ZERNIKE_NORM : Normalization Phase coefficients enumeration
        - **nb_coeffs_total** - uint : Number of Zernike coefficients used in the polynomial base
        - **hasoslopes** - HasoSlopes : HasoSlopes object
    
    - Constructor Legendre from Slopes :
        - **nb_coeffs_total** - uint : Number of Legendre coefficients used in the polynomial base
        - **hasoslopes** - HasoSlopes : HasoSlopes object
    
    - Constructor from Copy :
        - **modalcoef** - ModalCoef : ModalCoef object to copy
    """

    def __init_(
        self,
        modal_type
        ):
        """ModalCoef constructor
        
        :param modal_type: *E_MODAL* Polynomial basis type
        :type modal_type: uchar
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ModalCoef_New(
                message,
                ctypes.pointer(self.modalcoef),
                modal_type
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_',exception)

    def __init_from_data(
        self,
        modal_type,
        polyindex,
        coef,
        mask
        ):
        """ModalCoef constructor from reconstruction informations
        
        .. warning:: This method does not set the ModalCoef preferences.
        
        :param modal_type: *E_MODAL* Polynomial basis type
        :type modal_type: uchar
        :param polyindex: Array of modal coefficient indices
        :type polyindex: uint list[]
        :param coef: Array of modal coefficient values
        :type coef: float list[]
        :param mask: Mask object stored as a pupil object
        :type mask: Pupil
        """
        try:
            if(len(coef) != len(polyindex)):
                raise Exception('IO_Error','polyindex and coef size cannot be different.')
            message = ctypes.create_string_buffer(256)            
            self.dll.Imop_ModalCoef_NewFromData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_ubyte,
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p,
                ctypes.c_uint
                ] 
            self.dll.Imop_ModalCoef_NewFromData(
                message,
                ctypes.pointer(self.modalcoef),
                modal_type,
                numpy.array(polyindex, dtype = numpy.uintc),
                numpy.array(coef, dtype = numpy.single),
                mask.pupil,
                ctypes.c_uint(len(coef))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_data',exception)

    def __init_from_file(
        self,
        modal_type,
        coef_file_path
        ):
        """ModalCoef constructor from modal coefficients file
        
        :param modal_type: *E_MODAL_T* Polynomial basis type
        :type modal_type: uchar
        :param polyindex: Modal coefficients file path
        :type polyindex: string
        :return: Comments
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            comments_out = ctypes.create_string_buffer(256)
            self.dll.Imop_ModalCoef_NewFromFile(
                message,
                ctypes.pointer(self.modalcoef),
                modal_type,
                ctypes.c_char_p(coef_file_path.encode('utf-8')),
                comments_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return comments_out.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : __init_from_file',exception)

    def __init_zernike_from_slopes(
        self,
        modal_normalization,
        nb_coeffs_total,
        hasoslopes
        ):
        """ModalCoef constructor from slopes object
        Uses the auto-detected circular inscribed pupil.
        
        .. warning:: This method does not handle central occulation and returns the nb_coeffs_total first Zernike coefficients unfiltered
        
        :param modal_normalization: *E_ZERNIKE_NORM* Normalization Phase coefficients enumeration
        :type modal_normalization: uchar
        :param nb_coeffs_total: Number of Zernike coefficients used in the polynomial base
        :type nb_coeffs_total: uint
        :param hasoslopes: HasoSlopes object
        :type hasoslopes: HasoSlopes
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ModalCoef_NewZernikeFromSlopes(
                message,
                ctypes.pointer(self.modalcoef),
                ctypes.c_ubyte(modal_normalization),
                ctypes.c_uint(nb_coeffs_total),
                hasoslopes.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_zernike_from_slopes',exception)

    def __init_legendre_from_slopes(
        self,
        nb_coeffs_total,
        hasoslopes, 
        normalization = 0
        ):
        """ModalCoef constructor from slopes object
        Uses the auto-detected square inscribed pupil.
        
        .. warning:: This method does not handle central occulation and returns the nb_coeffs_total first Zernike coefficients unfiltered
        
        :param nb_coeffs_total: Number of Legendre coefficients used in the polynomial base
        :type nb_coeffs_total: uint
        :param hasoslopes: HasoSlopes object
        :type hasoslopes: HasoSlopes
        :param normalization: E_LEGENDRE_NORM_T Normalization Phase coefficients enumeration.
        :type normalization: ubyte
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ModalCoef_NewLegendreFromSlopes(
                message,
                ctypes.pointer(self.modalcoef),
                ctypes.c_uint(nb_coeffs_total),
                hasoslopes.hasoslopes,
                ctypes.c_ubyte(normalization)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_legendre_from_slopes',exception)

    def __init_from_copy(
        self,
        modalcoef
        ):
        """ModalCoef constructor from copy
        
        :param modalcoef: Object to copy
        :type modalcoef: ModalCoef
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ModalCoef_NewFromCopy(
                message,
                ctypes.pointer(self.modalcoef),
                modalcoef.modalcoef
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_copy',exception)

    def __init__(self, **kwargs):
        """ModalCoef Constructor
        """
        self.modalcoef = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('modal_type' in kwargs):
                    entered = 1
                    self.__init_(kwargs['modal_type'])
                if('modalcoef' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['modalcoef'])
            if(arg_size == 4):
                if('modal_type' in kwargs
                   and 'polyindex' in kwargs
                   and 'coef' in kwargs
                   and 'mask' in kwargs):
                    entered = 1
                    self.__init_from_data(kwargs['modal_type'], kwargs['polyindex'],kwargs['coef'], kwargs['mask'])
            if(arg_size == 2):
                if('modal_type' in kwargs
                   and 'coef_file_path' in kwargs):
                    entered = 1
                    self.__init_from_file(kwargs['modal_type'], kwargs['coef_file_path'])
                if('nb_coeffs_total' in kwargs
                   and 'hasoslopes' in kwargs):
                    entered = 1
                    self.__init_legendre_from_slopes(kwargs['nb_coeffs_total'],kwargs['hasoslopes'])
            if(arg_size == 3):
                if('modal_normalization' in kwargs
                   and 'nb_coeffs_total' in kwargs
                   and 'hasoslopes' in kwargs):
                    entered = 1
                    self.__init_zernike_from_slopes(kwargs['modal_normalization'], kwargs['nb_coeffs_total'], kwargs['hasoslopes'])
        except Exception as exception :
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):            
            raise Exception('IO_Error','---CAN NOT CREATE MODALCOEF OBJECT---')

    def __del_obj__(self):
        """ModalCoef Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_ModalCoef_Delete(message, self.modalcoef)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def load(
        self, 
        coef_file_path
        ):
        """Load modal coefficients values and preferences from file
        
        :param coef_file_path: Modal coefficients file path
        :type coef_file_path: string
        :return: Comments
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            comments_out = ctypes.create_string_buffer(256)
            self.dll.Imop_ModalCoef_Load(
                message,
                self.modalcoef,
                ctypes.c_char_p(coef_file_path.encode('utf-8')),
                comments_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return comments_out.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : load',exception)

    def save(
        self, 
        coef_file_path, 
        comments
        ):
        """Save modal coefficients values and preferences to file
        
        :param coef_file_path: Modal coefficients file path
        :type coef_file_path: string
        :param comments: Comments
        :type comments: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ModalCoef_Save(
                message,
                self.modalcoef,
                ctypes.c_char_p(coef_file_path.encode('utf-8')),
                ctypes.c_char_p(comments.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : save',exception)

    @staticmethod
    def get_type_from_file(coef_file_path):
        """Get polynomial coefficients type from file
        
        :return: type of modal coefficients
        :rtype: E_MODAL
        """
        dll = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            type_out = ctypes.c_ubyte()
            dll.Imop_ModalCoef_GetTypeFromFile(
                message,
                ctypes.c_char_p(coef_file_path.encode('utf-8')),
                ctypes.byref(type_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return type_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_type_from_file',exception)

    def get_data(self):
        """Get polynomial coefficients informations and values
        
        :return: Array of modal coefficient values, indices, Mask object stored as a pupil object
        :rtype: tuple(float list[], uint list[], Pupil)
        """
        try:
            message = ctypes.create_string_buffer(256)
            dim = self.get_dim()
            coef_out = numpy.zeros(dim, dtype = numpy.single)
            index_out = numpy.zeros(dim, dtype = numpy.uintc)
            pup_out = imop_pupil.Pupil(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    ),
                value = True
                )
            self.dll.Imop_ModalCoef_GetData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ] 
            self.dll.Imop_ModalCoef_GetData(
                message,
                self.modalcoef,
                coef_out,
                index_out,
                pup_out.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                coef_out.tolist(), 
                list(map(int, index_out.tolist())), 
                pup_out
                )
        except Exception as exception:
            raise Exception(__name__+' : get_data',exception)

    def set_data(
        self,
        coef_array,
        index_array,
        pupil
        ):
        """Set polynomial coefficients informations and values
        
        :param coef_array: Array of modal coefficient values
        :type coef_array: float list[]
        :param index_array: Array of modal coefficient indices
        :type index_array: uint list[]
        :param pupil: Mask object stored as a pupil object
        :type pupil: Pupil
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ModalCoef_SetData.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_void_p
                ] 
            self.dll.Imop_ModalCoef_SetData(
                message,
                self.modalcoef,
                numpy.array(coef_array, numpy.single),
                numpy.array(index_array, numpy.uintc),
                ctypes.c_int(len(coef_array)),
                pupil.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_data',exception)

    def get_coefs_values(self):
        """Get polynomial coefficients values
        
        :return: Array of modal coefficient values, Array of modal coefficient indices
        :rtype: tuple(float list[], uint list[])
        """
        try:
            message = ctypes.create_string_buffer(256)
            dim = self.get_dim()
            coef_out = numpy.zeros(dim, dtype = numpy.single)
            index_out = numpy.zeros(dim, dtype = numpy.uintc)
            self.dll.Imop_ModalCoef_GetCoefsValues.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS")
                ] 
            self.dll.Imop_ModalCoef_GetCoefsValues(
                message,
                self.modalcoef,
                coef_out,
                index_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                coef_out.tolist(), 
                list(map(int, index_out.tolist()))
                )
        except Exception as exception:
            raise Exception(__name__+' : get_coefs_values',exception)

    def set_coefs_values(
        self,
        coef_array,
        index_array
        ):
        """Set polynomial coefficients values
        
        :param coef_array: Array of modal coefficient values
        :type coef_array: float list[]
        :param coef_array: Array of modal coefficient indices
        :type coef_array: int list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            dim = self.get_dim()
            self.dll.Imop_ModalCoef_SetCoefsValues.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),
                ctypes.c_int
                ] 
            self.dll.Imop_ModalCoef_SetCoefsValues(
                message,
                self.modalcoef,
                numpy.array(coef_array, dtype = numpy.single),
                numpy.array(index_array, dtype = numpy.uintc),
                ctypes.c_int(len(coef_array))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_coefs_values',exception)

    def get_dim(self):
        """Get polynomial coefficients count
        
        :return: Number of modal coefficients
        :rtype: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = ctypes.c_uint()
            self.dll.Imop_ModalCoef_GetDim(
                message,
                self.modalcoef,
                ctypes.byref(size_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_dim',exception)

    def get_type(self):
        """Get polynomial coefficients type
        
        :return: type of modal coefficients
        :rtype: uchar
        """
        try:
            message = ctypes.create_string_buffer(256)
            type_out = ctypes.c_ubyte()
            self.dll.Imop_ModalCoef_GetType(
                message,
                self.modalcoef,
                ctypes.byref(type_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return type_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_type',exception)

    def set_zernike_prefs(
        self,
        zernike_normalization,
        nb_zernike_coefs_total,
        coefs_to_filter,
        projection_pupil
        ):
        """Set the Zernike preferences for modal phase projection
        
        .. warning:: ModalCoef object must be of type E_MODAL.ZERNIKE
        
        :param zernike_normalization: Normalization Phase coefficients enumeration
        :type zernike_normalization: E_ZERNIKE_NORM
        :param nb_zernike_coefs_total: Number of Zernike coefficients used in the polynomial base
        :type nb_zernike_coefs_total: uint
        :param coefs_to_filter: List of indices of Zernike coefficients to be ignored in the polynomial base while projecting
        :type coefs_to_filter: uint list[]
        :param projection_pupil: Zernike projection pupil (center, radius)
        :type projection_pupil: ZernikePupil_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(coefs_to_filter is None):
                self.dll.Imop_ModalCoef_SetZernikePrefs(
                    message,
                    self.modalcoef,
                    ctypes.c_ubyte(zernike_normalization),
                    ctypes.c_uint(nb_zernike_coefs_total),
                    ctypes.c_uint(0),
                    None,
                    ctypes.byref(projection_pupil.center),
                    ctypes.c_float(projection_pupil.radius)
                    )
            else:
                self.dll.Imop_ModalCoef_SetZernikePrefs.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.c_ubyte,
                    ctypes.c_uint,
                    ctypes.c_uint,
                    numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),
                    ctypes.c_void_p,
                    ctypes.c_float
                    ] 
                self.dll.Imop_ModalCoef_SetZernikePrefs(
                    message,
                    self.modalcoef,
                    ctypes.c_ubyte(zernike_normalization),
                    ctypes.c_uint(nb_zernike_coefs_total),
                    ctypes.c_uint(len(coefs_to_filter)),
                    numpy.array(coefs_to_filter, dtype = numpy.uintc),
                    ctypes.byref(projection_pupil.center),
                    ctypes.c_float(projection_pupil.radius)
                    )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_zernike_prefs',exception)

    def get_zernike_prefs(self):
        """Get preferences for phase projection from Zernike decomposition
        
        .. warning:: ModalCoef object must be of type E_MODAL.ZERNIKE
        
        :return: Normalization Phase coefficients enumeration, Number of Zernike coefficients, List of indices of Zernike coefficients ignored, Zernike projection pupil (center, radius)
        :rtype: tuple(E_ZERNIKE_NORM, uint, uint list[], ZernikePupil_t)
        """
        try:
            message = ctypes.create_string_buffer(256)
            zernike_normalization = ctypes.c_ubyte()
            nb_zernike_coefs_total = ctypes.c_uint()
            nb_zernike_coefs_to_filter = ctypes.c_uint()
            dim = self.get_dim()
            coefs_to_filter = numpy.zeros(dim, dtype = numpy.uintc)
            projection_pupil_center = imop_struct.float2D(0.0, 0.0)
            projection_pupil_radius = ctypes.c_float()
            self.dll.Imop_ModalCoef_GetZernikePrefs.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),
                ctypes.c_void_p,
                ctypes.c_void_p
                ] 
            self.dll.Imop_ModalCoef_GetZernikePrefs(
                message,
                self.modalcoef,
                ctypes.byref(zernike_normalization),
                ctypes.byref(nb_zernike_coefs_total),
                ctypes.byref(nb_zernike_coefs_to_filter),
                coefs_to_filter,
                ctypes.byref(projection_pupil_center),
                ctypes.byref(projection_pupil_radius)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            if nb_zernike_coefs_to_filter.value == 0 :
                coefs_to_filter_returned = None
            else:
                coefs_to_filter_returned = coefs_to_filter.tolist()[:nb_zernike_coefs_to_filter.value] #get the first nb_zernike_coefs_to_filter elements of coefs_to_filter
            return (
                zernike_normalization.value,
                nb_zernike_coefs_total.value,
                coefs_to_filter_returned,
                imop_struct.ZernikePupil_t(
                    projection_pupil_center,
                    projection_pupil_radius.value
                    )
                )        
        except Exception as exception:
            raise Exception(__name__+' : get_zernike_prefs',exception)

    def set_legendre_prefs(
        self,
        nb_coeffs_total,
        projection_pupil,
        normalization = 0
        ):
        """Set the Legendre preferences for modal phase projection
        
        .. warning:: ModalCoef object must be of type E_MODAL.LEGENDRE
        
        :param nb_coeffs_total: Number of Legendre coefficients used in the polynomial projection basis
        :type nb_coeffs_total: uint
        :param projection_pupil: Legendre projection pupil (center, halfsize)
        :type projection_pupil: LegendrePupil_t
        :param normalization: E_LEGENDRE_NORM_T Normalization Phase coefficients enumeration.
        :type normalization: ubyte
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ModalCoef_SetLegendrePrefs(
                message,
                self.modalcoef,
                nb_coeffs_total,
                ctypes.byref(projection_pupil.center),
                ctypes.byref(projection_pupil.halfsize),
                ctypes.c_ubyte(normalization)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_legendre_prefs',exception)

    def get_legendre_prefs(self):
        """Get preferences for phase projection from Legendre decomposition
        
        .. warning:: ModalCoef object must be of type E_MODAL.LEGENDRE
        
        :return: Number of Legendre coefficients used, Legendre projection pupil (center, half size), E_LEGENDRE_NORM_T Normalization Phase coefficients enumeration.
        :rtype: tuple(uint, LegendrePupil_t, ubyte)
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb_coeffs_total = ctypes.c_uint()
            projection_pupil_center = imop_struct.float2D(0.0, 0.0)
            projection_pupil_halfsize = imop_struct.float2D(0.0, 0.0)
            normalization = ctypes.c_ubyte()
            self.dll.Imop_ModalCoef_GetLegendrePrefs(
                message,
                self.modalcoef,
                ctypes.byref(nb_coeffs_total),
                ctypes.byref(projection_pupil_center),
                ctypes.byref(projection_pupil_halfsize),
                ctypes.byref(normalization)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                nb_coeffs_total.value,
                imop_struct.LegendrePupil_t(
                    projection_pupil_center,
                    projection_pupil_halfsize
                    ),
                normalization.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_legendre_prefs',exception)

    @staticmethod
    def addition(
        modalCoefA,
        modalCoefB
        ):
        try:
            dll   = imop_library.load_dll()
            message = ctypes.create_string_buffer(256)
            res = ModalCoef(modalcoef = modalCoefA)
            dll.Imop_ModalCoef_Addition(
                message,
                modalCoefA.modalcoef,
                modalCoefB.modalcoef,
                ctypes.pointer(res.modalcoef)
            )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return res
        except Exception as exception:
            raise Exception(__name__+' : addition',exception)
            
    def __add__(self, modalCoef):
        return ModalCoef.addition(self, modalCoef)

    @staticmethod
    def soustraction(
        modalCoefA,
        modalCoefB
        ):
        try:
            dll   = imop_library.load_dll()
            message = ctypes.create_string_buffer(256)
            res = ModalCoef(modalcoef = modalCoefA)
            dll.Imop_ModalCoef_Subtraction(
                message,
                modalCoefA.modalcoef,
                modalCoefB.modalcoef,
                ctypes.pointer(res.modalcoef)
            )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return res
        except Exception as exception:
            raise Exception(__name__+' : soustraction',exception)
            
    def __sub__(self, modalCoef):
        return ModalCoef.soustraction(self, modalCoef)
