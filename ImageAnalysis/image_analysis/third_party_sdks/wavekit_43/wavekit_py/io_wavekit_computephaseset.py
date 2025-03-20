#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_enum as imop_enum

class ComputePhaseSet(object):
    """Class ComputePhaseSet
    
    - Constructor from Phase Type :
        - **type_phase** - E_COMPUTEPHASESET : Phase reconstruction mode
    
    - Constructor from HasoData :
        - **hasodata** - HasoData : Processed slopes extacted from HasoData Object
    
    - Constructor from Copy :
        - **computephaseset** - ComputePhaseSet : Object to copy
    """
    
    def __init_from_type_phase(
        self,
        type_phase
        ):
        """ComputePhaseSet constructor
        from E_COMPUTEPHASESET value
        
        .. warning:: E_COMPUTEPHASESET for authorized values
        
        :param type_phase: Phase reconstruction mode
        :type type_phase: E_COMPUTEPHASESET
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ComputePhaseSet_NewFromTypePhase(
                message,
                ctypes.pointer(self.computephaseset),
                ctypes.c_ubyte(type_phase)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_type_phase',exception)

    def __init_from_hasodata(
        self,
        hasodata
        ):
        """ComputePhaseSet constructor
        from the processed slopes of a HasoData
        
        :param hasodata: HasoData from where the processed slopes are extracted
        :type hasodata: HasoData object
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ComputePhaseSet_NewFromHasoData(
                message,
                ctypes.pointer(self.computephaseset),
                hasodata.hasodata
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_hasodata',exception)

    def __init_from_copy(
        self,
        computephaseset
        ):
        """ComputePhaseSet constructor
        from the processed slopes of a HasoData
        
        :param computephaseset: Object to copy
        :type computephaseset: ComputePhaseSet object
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ComputePhaseSet_NewFromCopy(
                message,
                ctypes.pointer(self.computephaseset),
                computephaseset.computephaseset
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_copy',exception)

    def __init__(self,**kwargs):
        """ComputePhaseSet constructor
        """
        self.computephaseset = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('type_phase' in kwargs):
                    entered = 1
                    self.__init_from_type_phase(kwargs['type_phase'])
                if('hasodata' in kwargs):
                    entered = 1
                    self.__init_from_hasodata(kwargs['hasodata'])
                if('computephaseset' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['computephaseset'])
        except Exception as exception:
            raise Exception(__name__+' : init',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE COMPUTEPHASESET OBJECT---')

    def __del_obj__(self):
        """ComputePhaseSet destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_ComputePhaseSet_Delete(message, self.computephaseset)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_type_phase(self):
        """Get phase reconstruction type.
        
        :return: Reconstruction type of the ComputePhaseSet object.
        :rtype: E_COMPUTEPHASESET
        """
        try:
            message = ctypes.create_string_buffer(256)
            phase_type = ctypes.c_ubyte()
            self.dll.Imop_ComputePhaseSet_GetTypePhase(
                message,
                self.computephaseset,
                ctypes.byref(phase_type)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return phase_type.value
        except Exception as exception:
            raise Exception(__name__+' : get_type_phase',exception)

    def set_zonal_prefs(
        self,
        nb_weak_iterations,
        nb_max_iterations,
        residual_limit
        ):
        """Set parameters for zonal phase reconstruction.

        .. warning:: ComputePhaseSet object must be of type E_COMPUTEPHASESET.ZONAL
        
        :param nb_weak_iterations: Maximum number of weak iterations
        :type nb_weak_iterations: uint
        :param nb_max_iterations: Maximum number of iterations
        :type nb_max_iterations: uint
        :param residual_limit: Residual variation limit
        :type residual_limit: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ComputePhaseSet_SetZonalPrefs(
                message,
                self.computephaseset,
                ctypes.c_uint(nb_weak_iterations),
                ctypes.c_uint(nb_max_iterations),
                ctypes.c_float(residual_limit)                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_zonal_prefs',exception)

    def get_zonal_prefs(self):
        """Get parameters for zonal phase reconstruction.

        .. warning:: ComputePhaseSet object must be of type E_COMPUTEPHASESET.ZONAL
        
        :return: Maximum number of weak iterations, Maximum number of iterations, Residual variation limit
        :rtype: tuple(uint, uint, float)
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb_weak_iterations = ctypes.c_uint()
            nb_max_iterations = ctypes.c_uint()
            residual_limit = ctypes.c_float()
            self.dll.Imop_ComputePhaseSet_GetZonalPrefs(
                message,
                self.computephaseset,
                ctypes.byref(nb_weak_iterations),
                ctypes.byref(nb_max_iterations),
                ctypes.byref(residual_limit)                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                nb_weak_iterations.value,
                nb_max_iterations.value,
                residual_limit.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_zonal_prefs',exception)

    def set_zonal_spyder_prefs(
        self,
        spyder_arm_size
        ):
        """Set parameter for zonal phase reconstruction.

        .. warning:: ComputePhaseSet object must be of type E_COMPUTEPHASESET.ZONAL
        
        :param spyder_arm_size: Spyder arm size
        :type spyder_arm_size: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_ComputePhaseSet_SetZonalSpyderOption(
                message,
                self.computephaseset,
                ctypes.c_uint(spyder_arm_size)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_zonal_spyder_prefs',exception)

    def get_zonal_spyder_prefs(self):
        """Get parameters for zonal phase reconstruction.

        .. warning:: ComputePhaseSet object must be of type E_COMPUTEPHASESET.ZONAL
        
        :return: Spyder arm size
        :rtype: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            spyder_arm_size = ctypes.c_uint()
            self.dll.Imop_ComputePhaseSet_GetZonalSpyderOption(
                message,
                self.computephaseset,
                ctypes.byref(spyder_arm_size)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return spyder_arm_size.value
        except Exception as exception:
            raise Exception(__name__+' : get_zonal_spyder_prefs',exception)

    def set_modal_filter(
        self,
        coeffs_to_filter
        ):
        """Set filter for phase reconstruction from modal decomposition.
        The filter is defined by the indices of polynomials that must be filtered.

        .. warning:: ComputePhaseSet object type must be E_COMPUTEPHASESET.MODAL or E_COMPUTEPHASESET.MODAL_ZONAL
        
        :param nb_coeffs_to_filter: List of coefficients to filter in the polynomial base
        :type nb_coeffs_to_filter: uint list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb_coeffs_to_filter = ctypes.c_uint(len(coeffs_to_filter))
            list_sent = numpy.array(coeffs_to_filter, dtype = numpy.uintc)
            self.dll.Imop_ComputePhaseSet_SetModalFilter.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_uint,
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS")
                ]             
            self.dll.Imop_ComputePhaseSet_SetModalFilter(
                message,
                self.computephaseset,
                nb_coeffs_to_filter,
                list_sent
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_modal_filter',exception)

    def get_modal_filter(self):
        """Get filter for phase reconstruction from Modal decomposition.
        The filter is defined by the indices of polynomials that must be filtered.

        .. warning:: ComputePhaseSet object type must be E_COMPUTEPHASESET.MODAL or E_COMPUTEPHASESET.MODAL_ZONAL
        
        :return: Number of the coefficients to filter in the polynomial base, List of coefficients to filter in the polynomial base
        :rtype: tuple(uint, uint list[])
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb_coeffs_to_filter = ctypes.c_uint()
            coeffs_to_filter = numpy.zeros(128, dtype = numpy.uintc)        
            self.dll.Imop_ComputePhaseSet_GetModalFilter.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS")
                ]             
            #we need to make a first call to get the array size before recalling it
            self.dll.Imop_ComputePhaseSet_GetModalFilter(
                message,
                self.computephaseset,
                ctypes.byref(nb_coeffs_to_filter),
                coeffs_to_filter
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            #then we make another call to get the values
            coeffs_to_filter = numpy.zeros(nb_coeffs_to_filter.value, dtype = numpy.uintc)
            self.dll.Imop_ComputePhaseSet_GetModalFilter(
                message,
                self.computephaseset,
                ctypes.byref(nb_coeffs_to_filter),
                coeffs_to_filter
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            
            return (
                nb_coeffs_to_filter.value, 
                coeffs_to_filter.tolist()
                )
        except Exception as exception:
            raise Exception(__name__+' : get_modal_filter',exception)

    def set_zonal_filter(
        self,
        aberrations_filters
        ):
        """Set filter for zonal phase reconstruction.
        The aberrations to filter are selected using the boolean values in the aberrations_filters array
        (0 = aberration is removed).
        - params[0] : tiltx
        - params[1] : tilty
        - params[2] : curvature
        - params[3] : astigmatism 0 degree
        - params[4] : astigmatism 45 degree

        .. warning:: ComputePhaseSet object must be of type E_COMPUTEPHASESET.ZONAL
        
        :param aberrations_filters: Number of the coefficients to filter in the polynomial base, List of coefficients to filter in the polynomial base
        :type aberrations_filters: bool list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            list_sent = numpy.array(aberrations_filters, dtype = numpy.ubyte)
            
            self.dll.Imop_ComputePhaseSet_SetZonalFilter.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.ubyte, flags="C_CONTIGUOUS")
                ]             
            self.dll.Imop_ComputePhaseSet_SetZonalFilter(
                message,
                self.computephaseset,
                list_sent
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_zonal_filter',exception)

    def get_zonal_filter(self):
        """Get filter for zonal phase reconstruction.
        The filter is defined by the indices of array \p params that must be filtered.
        see ComputePhaseSet.get_legendre_filter for details.

        .. warning:: ComputePhaseSet object must be of type E_COMPUTEPHASESET.ZONAL
        
        :return: Array of aberrations to filter
        :rtype: bool list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            aberrations_filters = numpy.zeros(5, dtype = numpy.ubyte)
            self.dll.Imop_ComputePhaseSet_GetZonalFilter.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.ubyte, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_ComputePhaseSet_GetZonalFilter(
                message,
                self.computephaseset,
                aberrations_filters
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return [bool(ele) for ele in aberrations_filters]
        except Exception as exception:
            raise Exception(__name__+' : get_zonal_filter',exception)