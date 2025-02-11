#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_pupil as imop_pupil
import io_wavekit_hasoslopes as imop_hslp

class SlopesPostProcessorList(object):
    """Class SlopesPostProcessorList
    
    - Default Constructor 
    
    - Constructor from Copy :
        - **slopespostprocessorlist** - SlopesPostProcessorList : SlopesPostProcessorList object to copy
    
    - Constructor from Has File :
        - **has_file_path** - string : Path to (\*.has) file
    
    - Constructor from applied Has File :
        - **has_file_path_applied** - string : Path to (\*.has) file
    """
    def __init_(self):
        """SlopesPostProcessorList default constructor
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_New(
                message,
                ctypes.pointer(self.slopespostprocessorlist)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_',exception)

    def __init_from_copy(
        self,
        slopespostprocessorlist
        ):
        """SlopesPostProcessorList constructor from copy 
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_NewFromCopy(
                message,
                ctypes.pointer(self.slopespostprocessorlist),
                slopespostprocessorlist.slopespostprocessorlist
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_copy',exception)

    def __init_from_has_file(
        self,
        has_file_path
        ):
        """SlopesPostProcessorList constructor from a .has file
        
        :param has_file_path: Path to .has file
        :type has_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_NewFromHasFile(
                message,
                ctypes.pointer(self.slopespostprocessorlist),
                ctypes.c_char_p(has_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_has_file',exception)

    def __init_from_applied_has_file(
        self,
        has_file_path_applied
        ):
        """SlopesPostProcessorList constructor from a .has file
        
        :param has_file_path_applied: Path to .has file
        :type has_file_path_applied: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_NewFromAppliedInHasFile(
                message,
                ctypes.pointer(self.slopespostprocessorlist),
                ctypes.c_char_p(has_file_path_applied.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_applied_has_file',exception)
            
    def __init__(self, **kwargs):
        """SlopesPostProcessorList Constructor
        """
        self.slopespostprocessorlist = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 0):
                entered = 1
                self.__init_()
            if(arg_size == 1):
                if('slopespostprocessorlist' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['slopespostprocessorlist'])
            if(arg_size == 1):
                if('has_file_path' in kwargs):
                    entered = 1
                    self.__init_from_has_file(kwargs['has_file_path'])
                if('has_file_path_applied' in kwargs):
                    entered = 1
                    self.__init_from_applied_has_file(kwargs['has_file_path_applied'])
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE SLOPESPOSTPROCESSORLIST OBJECT---')
            
    def __del_obj__(self):
        """SlopesPostProcessorList Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_SlopesPostProcessorList_Delete(message, self.slopespostprocessorlist)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_processor_list_size(self):
        """Get the size of the SlopesPostProcessorList
        
        :return: Processor list size
        :rsize: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = ctypes.c_uint()
            self.dll.Imop_SlopesPostProcessorList_GetProcessorListSize(
                message,
                self.slopespostprocessorlist,
                ctypes.byref(size)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size.value
        except Exception as exception:
            raise Exception(__name__+' : get_processor_list_size',exception)

    def get_processor_list_names(self):
        """Get names of the the SlopesPostProcessors contained into the SlopesPostProcessorList
        
        :return: List of processors names
        :rsize: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            names = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_GetProcessorListNames(
                message,
                self.slopespostprocessorlist,
                names
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return names.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_processor_list_names',exception)

    def delete_processor(
        self, 
        position
        ):
        """Delete a single processor of the SlopesPostProcessorList
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_DeleteProcessor(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : delete_processor',exception)

    def insert_slopespostprocessorlist(
        self,
        position,
        slopespostprocessorlist_to_add
        ):
        """Insert a SlopesPostProcessorList in the SlopesPostProcessorList
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param slopespostprocessorlist_to_add: SlopesPostProcessorList object
        :type slopespostprocessorlist_to_add: SlopesPostProcessorList
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertSlopesPostProcessorList(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                slopespostprocessorlist_to_add.slopespostprocessorlist
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_slopespostprocessorlist',exception)

    def insert_filter(
        self,
        position,
        apply_tiltx_filter,
        apply_tilty_filter,
        apply_curv_filter,
        apply_astig0_filter,
        apply_astig45_filter,
        apply_others_filter
        ):
        """Insert a "Filter" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param apply_tiltx_filter: Activate x tilt filter
        :type apply_tiltx_filter: uchar
        :param apply_tilty_filter: Activate y tilt filter
        :type apply_tilty_filter: uchar
        :param apply_curv_filter: Activate curvature filter
        :type apply_curv_filter: uchar
        :param apply_astig0_filter: Activate 0 degree astigmatism filter
        :type apply_astig0_filter: uchar
        :param apply_astig45_filter: Activate 45 degree astigmatism filter
        :type apply_astig45_filter: uchar
        :param apply_others_filter: Activate all other aberrations filter
        :type apply_others_filter: uchar
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertFilter(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.c_ubyte(apply_tiltx_filter),
                ctypes.c_ubyte(apply_tilty_filter),
                ctypes.c_ubyte(apply_curv_filter),
                ctypes.c_ubyte(apply_astig0_filter),
                ctypes.c_ubyte(apply_astig45_filter),
                ctypes.c_ubyte(apply_others_filter)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_filter',exception)

    def get_filter(
        self,
        position
        ):
        """Get parameters of the "Filter" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return:  X tilt filter state, Y tilt filter state, Curvature filter state, 0 degree astigmatism filter state, 45 degree astigmatism filter state, All other aberrations filter state
        :rtype: tuple(bool, bool, bool, bool, bool, bool)
        """
        try:
            message = ctypes.create_string_buffer(256)
            apply_tiltx_filter = ctypes.c_ubyte()
            apply_tilty_filter = ctypes.c_ubyte()
            apply_curv_filter = ctypes.c_ubyte()
            apply_astig0_filter = ctypes.c_ubyte()
            apply_astig45_filter = ctypes.c_ubyte()
            apply_others_filter = ctypes.c_ubyte()
            self.dll.Imop_SlopesPostProcessorList_GetFilterParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.byref(apply_tiltx_filter),
                ctypes.byref(apply_tilty_filter),
                ctypes.byref(apply_curv_filter),
                ctypes.byref(apply_astig0_filter),
                ctypes.byref(apply_astig45_filter),
                ctypes.byref(apply_others_filter)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.Filter_t(
                bool(apply_tiltx_filter), 
                bool(apply_tilty_filter), 
                bool(apply_curv_filter), 
                bool(apply_astig0_filter), 
                bool(apply_astig45_filter), 
                bool(apply_others_filter)
                )
        except Exception as exception:
            raise Exception(__name__+' : get_filter',exception)

    def insert_pupil(
        self,
        position,
        pupil_file_path,
        pupil
        ):
        """Insert a "Pupil" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param pupil_file_path: Pupil file path. No check is performed to assert that the Pupil parameter matches the pupil stored in pupil_file_path
        :type pupil_file_path: string
        :param pupil: pupil object
        :type pupil: Pupil
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertPupil(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.c_char_p(pupil_file_path.encode('utf-8')),
                pupil.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_pupil',exception)

    def get_pupil(
        self,
        position
        ):
        """Insert a "Pupil" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return: pupil object, Pupil file path
        :rtype: tuple(Pupil, string)
        """
        try:
            message = ctypes.create_string_buffer(256)
            file_out = ctypes.create_string_buffer(256)
            pup_out = imop_pupil.Pupil(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                ),
                value = True
                )
            self.dll.Imop_SlopesPostProcessorList_GetPupilParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                file_out,
                pup_out.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                pup_out, 
                file_out.value.decode('utf-8')
                )
        except Exception as exception:
            raise Exception(__name__+' : get_pupil',exception)

    def insert_substractor(
        self,
        position,
        hasoslopes,
        comment
        ):
        """Insert a "Substractor" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param hasoslopes: HasoSlopes to set in the processor
        :type hasoslopes: HasoSlopes
        :param comment: Comments
        :type comment: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertSubstractor(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.c_char_p(comment.encode('utf-8')),
                hasoslopes.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_substractor',exception)

    def get_substractor(
        self,
        position
        ):
        """Get parameters of the "Substractor" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return: HasoSlopes, Comments
        :rtype: tuple(HasoSlopes, string)
        """
        try:
            message = ctypes.create_string_buffer(256)
            comment_out = ctypes.create_string_buffer(256)
            hasoslopes_out = imop_hslp.HasoSlopes(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1, 1), 
                    imop_struct.float2D(1.0, 1.0)
                    ),
                serial_number = ''
                )
            self.dll.Imop_SlopesPostProcessorList_GetSubstractorParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                comment_out,
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                hasoslopes_out, 
                comment_out.value.decode('utf-8')
                )
        except Exception as exception:
            raise Exception(__name__+' : get_substractor',exception)

    def insert_adder(
        self,
        position,
        hasoslopes,
        comment
        ):
        """Insert a "Adder" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param hasoslopes: HasoSlopes to set in the processor
        :type hasoslopes: HasoSlopes
        :param comment: Comments
        :type comment: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertAdder(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.c_char_p(comment.encode('utf-8')),
                hasoslopes.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_adder',exception)

    def get_adder(
        self,
        position
        ):
        """Get parameters of the "Adder" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return: HasoSlopes, Comments
        :rtype: tuple(HasoSlopes, string)
        """
        try:
            message = ctypes.create_string_buffer(256)
            comment_out = ctypes.create_string_buffer(256)
            hasoslopes_out = imop_hslp.HasoSlopes(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1, 1), 
                    imop_struct.float2D(1.0, 1.0)
                    ),
                serial_number = ''
                )
            self.dll.Imop_SlopesPostProcessorList_GetAdderParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                comment_out,
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                hasoslopes_out,
                comment_out.value.decode('utf-8')
                )
        except Exception as exception:
            raise Exception(__name__+' : get_adder',exception)

    def insert_scaler(
        self,
        position,
        scale_factor,
        comment
        ):
        """Insert a "Scaler" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param scale_factor: Scale factor
        :type scale_factor: float
        :param comment: Comments
        :type comment: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertScaler(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.c_char_p(comment.encode('utf-8')),
                ctypes.c_float(scale_factor)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_scaler',exception)

    def get_scaler(
        self,
        position
        ):
        """Get parameters of the "Scaler" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return: Scale factor, Comments
        :rtype: tuple(float, string)
        """
        try:
            message = ctypes.create_string_buffer(256)
            comment_out = ctypes.create_string_buffer(256)
            scale_factor_out = ctypes.c_float()
            self.dll.Imop_SlopesPostProcessorList_GetScalerParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                comment_out,
                ctypes.byref(scale_factor_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                scale_factor_out.value, 
                comment_out.value.decode('utf-8')
                )
        except Exception as exception:
            raise Exception(__name__+' : get_scaler',exception)

    def insert_perfect_lens(
        self,
        position,
        focal_lens,
        keep_residual_curvature
        ):
        """Insert a "PerfectLens" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param focal_lens: Perfect lens focal length (m)
        :type focal_lens: float
        :param keep_residual_curvature: Keep the residual curvature
        :type keep_residual_curvature: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertPerfectLens(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.c_float(focal_lens),
                ctypes.c_bool(keep_residual_curvature)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_perfect_lens',exception)

    def get_perfect_lens(
        self,
        position
        ):
        """Get parameters of the "PerfectLens" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return: Perfect lens focal length (m), Keep the residual curvature
        :rtype: tuple(float, bool)
        """
        try:
            message = ctypes.create_string_buffer(256)
            focal_lens_out = ctypes.c_float()
            keep_residual_curvature_out = ctypes.c_bool()
            self.dll.Imop_SlopesPostProcessorList_GetPerfectLensParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.byref(focal_lens_out),
                ctypes.byref(keep_residual_curvature_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                focal_lens_out.value,
                keep_residual_curvature_out.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_perfect_lens',exception)

    def insert_double_path(
        self,
        position,
        active_on_tilt,
        active_on_curv
        ):
        """Insert a "DoublePath" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param active_on_tilt: Is Double path active on tilt
        :type active_on_tilt: bool
        :param active_on_curv: Is Double path is active on curvature
        :type active_on_curv: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertDoublePath(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.c_bool(active_on_tilt),
                ctypes.c_bool(active_on_curv)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_double_path',exception)

    def get_double_path(
        self,
        position
        ):
        """Get parameters of the "DoublePath" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return: Is Double path active on tilt, Is Double path is active on curvature
        :rtype: tuple(bool, bool)
        """
        try:
            message = ctypes.create_string_buffer(256)
            activated_on_tilt = ctypes.c_bool()
            activated_on_curvature = ctypes.c_bool()
            self.dll.Imop_SlopesPostProcessorList_GetDoublePathParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.byref(activated_on_tilt),
                ctypes.byref(activated_on_curvature)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                activated_on_tilt.value, 
                activated_on_curvature.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_double_path',exception)

    def insert_neighbor_extension(
        self,
        position
        ):
        """Insert a "NeighborExtension" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertNeighborExtension(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_neighbor_extension',exception)

    def insert_shut_of_boundaries(
        self,
        position,
        shutoff_radius
        ):
        """Insert a "ShutOfBoundaries" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param shutoff_radius: Radius of the boundary neighbourood where sub-pupils must be shut off
        :type shutoff_radius: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertShutOfBoundaries(
                message,
                self.slopespostprocessorlist,
                ctypes.c_uint(position),
                ctypes.c_uint(shutoff_radius)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_shut_of_boundaries',exception)

    def get_shut_of_boundaries(
        self,
        position
        ):
        """Get parameters of the "ShutOfBoundaries" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return: Radius of the boundary neighbourood where sub-pupils must be shut off
        :rtype: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            shutoff_radius_out = ctypes.c_uint()
            self.dll.Imop_SlopesPostProcessorList_GetShutOfBoundariesParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.byref(shutoff_radius_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return shutoff_radius_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_shut_of_boundaries',exception)

    def insert_modulator(
        self,
        position,
        tiltx_modulation,
        tilty_modulation,
        curv_modulation,
        astig0_modulation,
        astig45_modulation,
        others_modulation
        ):
        """Insert a "Modulator" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param tiltx_modulation: Value of x tilt modulation between 0 and 1
        :type tiltx_modulation: float
        :param tilty_modulation: Value of y tilt modulation between 0 and 1
        :type tilty_modulation: float
        :param curv_modulation: Value of curvature modulation between 0 and 1
        :type curv_modulation: float
        :param astig0_modulation: Value of 0 degree astigmatism modulation between 0 and 1
        :type astig0_modulation: float
        :param astig45_modulation: Value of 45 degree astigmatism modulation between 0 and 1
        :type astig45_modulation: float
        :param others_modulation: Value of all other aberrations modulation between 0 and 1
        :type others_modulation: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertModulator(
                message,
                self.slopespostprocessorlist,
                ctypes.c_uint(position),
                ctypes.c_float(tiltx_modulation),
                ctypes.c_float(tilty_modulation),
                ctypes.c_float(curv_modulation),
                ctypes.c_float(astig0_modulation),
                ctypes.c_float(astig45_modulation),
                ctypes.c_float(others_modulation)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_modulator',exception)

    def get_modulator(
        self,
        position
        ):
        """Get parameters of the "Modulator" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return: Value of x , y tilt modulation, Value of curvature modulation, Value of 0 degree astigmatism modulation, Value of 45 degree astigmatism modulation, Value of all other aberrations modulation 
        :rtype: Modulator_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            tiltx_modulation_out = ctypes.c_float()
            tilty_modulation_out = ctypes.c_float()
            curv_modulation_out = ctypes.c_float()
            astig0_modulation_out = ctypes.c_float()
            astig45_modulation_out = ctypes.c_float()
            others_modulation_out = ctypes.c_float()
            self.dll.Imop_SlopesPostProcessorList_GetModulatorParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.byref(tiltx_modulation_out),
                ctypes.byref(tilty_modulation_out),
                ctypes.byref(curv_modulation_out),
                ctypes.byref(astig0_modulation_out),
                ctypes.byref(astig45_modulation_out),
                ctypes.byref(others_modulation_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.Modulator_t(
                tiltx_modulation_out.value,
                tilty_modulation_out.value, 
                curv_modulation_out.value,
                astig0_modulation_out.value,
                astig45_modulation_out.value, 
                others_modulation_out.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_modulator',exception)

    def insert_pupil_from_intensity_applier(
        self,
        position,
        threshold
        ):
        """Insert a "PupilFromIntensityApplier" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :param threshold: Intensity thresholding level
        :type threshold: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_SlopesPostProcessorList_InsertPupilFromIntensityApplier(
                message,
                self.slopespostprocessorlist,
                ctypes.c_uint(position),
                ctypes.c_float(threshold)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : insert_pupil_from_intensity',exception)

    def get_pupil_from_intensity_applier(
        self,
        position
        ):
        """Get parameters of the "PupilFromIntensityApplier" processor
        
        :param position: Position of the processor in the SlopesPostProcessorList
        :type position: uint
        :return: Intensity thresholding level
        :rtype: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            threshold_out = ctypes.c_float()
            self.dll.Imop_SlopesPostProcessorList_GetPupilFromIntensityApplierParameters(
                message,
                self.slopespostprocessorlist,
                ctypes.c_int(position),
                ctypes.byref(threshold_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return threshold_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_pupil_from_intensity',exception)
