#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_hasoslopes as hslp

class SlopesPostProcessor():
    """Class SlopesPostProcessor
    """

    @staticmethod
    def apply_filter(
        hasoslopes_to_modify,
        apply_tiltx_filter,
        apply_tilty_filter,
        apply_curv_filter,
        apply_astig0_filter,
        apply_astig45_filter,
        apply_others_filter
        ):
        """Filter (remove) some HasoSlopes aberrations
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :param apply_tiltx_filter: Activate x tilt removal
        :type apply_tiltx_filter: uchar
        :param apply_tilty_filter: Activate y tilt removal
        :type apply_tilty_filter: uchar
        :param apply_curv_filter: Activate curvature removal
        :type apply_curv_filter: uchar
        :param apply_astig0_filter: Activate 0 degree astigmatism removal
        :type apply_astig0_filter: uchar
        :param apply_astig45_filter: Activate 45 degree astigmatism removal
        :type apply_astig45_filter: uchar
        :param apply_others_filter: Activate all other aberrations removal
        :type apply_others_filter: uchar
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyFilter(
                message,
                hasoslopes_to_modify.hasoslopes,
                ctypes.c_ubyte(apply_tiltx_filter),
                ctypes.c_ubyte(apply_tilty_filter),
                ctypes.c_ubyte(apply_curv_filter),
                ctypes.c_ubyte(apply_astig0_filter),
                ctypes.c_ubyte(apply_astig45_filter),
                ctypes.c_ubyte(apply_others_filter),
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_filter',exception)

    @staticmethod
    def apply_pupil(
        hasoslopes_to_modify,
        pupil
        ):
        """Replace the slopes pupil by the intersection of the current one with the given one
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :param pupil: pupil object
        :type pupil: Pupil
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyPupil(
                message,
                hasoslopes_to_modify.hasoslopes,
                pupil.pupil,
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_pupil',exception)
    
    @staticmethod
    def apply_substractor(
        hasoslopes_to_modify,
        hasoslopes_to_substract
        ):
        """Subtract hasoslopes_to_substract to the input slopes
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :param hasoslopes_to_substract: HasoSlopes to subtract
        :type hasoslopes_to_substract: HasoSlopes
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplySubstractor(
                message,
                hasoslopes_to_modify.hasoslopes,
                hasoslopes_to_substract.hasoslopes,
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_substractor',exception)

    @staticmethod
    def apply_adder(
        hasoslopes_to_modify,
        hasoslopes_to_add
        ):
        """Add hasoslopes_to_add to the input slopes
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :param hasoslopes_to_add: HasoSlopes to add
        :type hasoslopes_to_add: HasoSlopes
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyAdder(
                message,
                hasoslopes_to_modify.hasoslopes,
                hasoslopes_to_add.hasoslopes,
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_adder',exception)

    @staticmethod
    def apply_scaler(
        hasoslopes_to_modify,
        scale_factor
        ):
        """Multiply the input slopes by a float
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :param scale_factor: Scale factor
        :type scale_factor: float
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyScaler(
                message,
                hasoslopes_to_modify.hasoslopes,
                ctypes.c_float(scale_factor),
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_scaler',exception)

    @staticmethod
    def apply_perfect_lens(
        hasoslopes_to_modify,
        focal_lens_m,
        keep_residual_curvature
        ):
        """Add the curvature of a perfect lens to the input slopes
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :param focal_lens_m: Perfect lens focal length (m)
        :type focal_lens_m: float
        :param keep_residual_curvature: Keep the residual curvature
        :type keep_residual_curvature: bool
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyPerfectLens(
                message,
                hasoslopes_to_modify.hasoslopes,
                ctypes.c_float(focal_lens_m),
                ctypes.c_bool(keep_residual_curvature),
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_perfect_lens',exception)

    @staticmethod
    def apply_double_path(
        hasoslopes_to_modify,
        is_active_on_tilt,
        is_active_on_curv
        ):
        """Divide the HasoSlopes by two. May ignore tilt and / or curvature depending on parametrization
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :param is_active_on_tilt: Is Double path active on tilt
        :type is_active_on_tilt: bool
        :param is_active_on_curv: Is Double path is active on curvature
        :type is_active_on_curv: bool
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyDoublePath(
                message,
                hasoslopes_to_modify.hasoslopes,
                ctypes.c_bool(is_active_on_tilt),
                ctypes.c_bool(is_active_on_curv),
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_double_path',exception)

    @staticmethod
    def apply_neighbor_extension(
        hasoslopes_to_modify
        ):
        """Interpolate dark sub-pupils that are surrounded by light sub-pupils
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyNeighborExtension(
                message,
                hasoslopes_to_modify.hasoslopes,
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_neighbor_extension',exception)

    @staticmethod
    def apply_shut_of_boundaries(
        hasoslopes_to_modify,
        shutoff_radius
        ):
        """Shut off the sub-pupils close to boundaries (closing operation with a specified radius)
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyShutOfBoundaries(
                message,
                hasoslopes_to_modify.hasoslopes,
                ctypes.c_uint(shutoff_radius),
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_shut_of_boundaries',exception)

    @staticmethod
    def apply_modulator(
        hasoslopes_to_modify,
        tiltx_modulation,
        tilty_modulation,
        curv_modulation,
        astig0_modulation,
        astig45_modulation,
        others_modulation
        ):
        """Modulate (multiply by a real factor between 0 and 1) some HasoSlopes aberrations
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
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
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyModulator(
                message,
                hasoslopes_to_modify.hasoslopes,
                ctypes.c_float(tiltx_modulation),
                ctypes.c_float(tilty_modulation),
                ctypes.c_float(curv_modulation),
                ctypes.c_float(astig0_modulation),
                ctypes.c_float(astig45_modulation),
                ctypes.c_float(others_modulation),
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_modulator',exception)

    @staticmethod
    def apply_pupil_from_intensity(
        hasoslopes_to_modify,
        threshold
        ):
        """Update Pupil according to Intensity thresholding result
        
        :param hasoslopes_to_modify: HasoSlopes object
        :type hasoslopes_to_modify: HasoSlopes
        :param threshold: Threshold between 0 and 1
        :type threshold: float
        :return: Processed HasoSlopes
        :rtype: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = hslp.HasoSlopes(hasoslopes = hasoslopes_to_modify)
            dll.Imop_SlopesPostProcessor_ApplyPupilFromIntensity(
                message,
                hasoslopes_to_modify.hasoslopes,
                ctypes.c_float(threshold),
                hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_pupil_from_intensity',exception)
