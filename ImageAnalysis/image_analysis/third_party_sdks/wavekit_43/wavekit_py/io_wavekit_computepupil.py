#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_pupil as imop_pupil


class ComputePupil():
    """Class ComputePupil
    """

    @staticmethod
    def apply_neighbor_extension(
        pupil_in
        ):
        """Apply a "NeighborExtension" (Fill) to a Pupil object.
        
        :param pupil_in: Pupil object
        :type pupil_in: Pupil
        :return: processed Pupil object
        :rtype: Pupil
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            pupil_out = imop_pupil.Pupil(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    ),
                value = 1
                )
            dll.Imop_PupilCompute_ApplyNeighborExtension(
                message,
                pupil_in.pupil,
                pupil_out.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return pupil_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_neighbor_extension',exception)

    @staticmethod
    def apply_shut_of_boundaries(
        pupil_in,
        shutoff_radius
        ):
        """Apply a "ShutOfBoundaries" (Erosion) to a Pupil object.
        
        :param pupil_in: Pupil object
        :type pupil_in: Pupil
        :param shutoff_radius: Radius of the boundary neighbourood where sub-pupils must be shut off
        :type shutoff_radius: uint
        :return: processed Pupil object
        :rtype: Pupil
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            pupil_out = imop_pupil.Pupil(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    ),
                value = 1
                )
            dll.Imop_PupilCompute_ApplyShutOfBoundaries(
                message,
                pupil_in.pupil,
                ctypes.c_uint(shutoff_radius),
                pupil_out.pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return pupil_out            
        except Exception as exception:
            raise Exception(__name__+' : apply_shut_of_boundaries',exception)

    @staticmethod
    def fit_zernike_pupil(
        pupil_in,
        detection_mode,
        covering,
        has_central_occultation = False
        ):
        """Compute the geometric parameters of the Zernike pupil from a pupil input.
 
        :param pupil_in: Pupil object
        :type pupil_in: Pupil
        :param detection_mode: Circular pupil detection mode
        :type detection_mode: E_PUPIL_DETECTION
        :param covering: Circular pupil covering mode
        :type covering: E_PUPIL_COVERING
        :param has_central_occultation: If pupil has a central occultation, this boolean must be set to true
        :type has_central_occultation: bool
        :return: Computed Zernike Pupil parameters
        :rtype: ZernikePupil_t
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            center = imop_struct.float2D(0.0, 0.0)
            radius = ctypes.c_float()
            dll.Imop_PupilCompute_FitZernikePupil(
                message,
                pupil_in.pupil,
                ctypes.c_uint(detection_mode),
                ctypes.c_uint(covering),
                ctypes.c_bool(has_central_occultation),
                ctypes.byref(center),
                ctypes.byref(radius)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.ZernikePupil_t(
                center,
                radius.value
                )            
        except Exception as exception:
            raise Exception(__name__+' : fit_zernike_pupil',exception)

    @staticmethod
    def fit_legendre_pupil(
        pupil_in
        ):
        """Compute the geometric parameters of the Legendre pupil from a pupil input.
        This function is not configurable. It takes a user pupil as input and
        detects a square pupil that is as large as possible and entirely inside it.
        If several pupils with the same size suit, one of them is chosen arbitrarily.
        
        :param pupil_in: Pupil object
        :type pupil_in: Pupil
        :return: Computed Legendre Pupil parameters
        :rtype: LegendrePupil_t
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            center = imop_struct.float2D(0.0, 0.0)
            halfsize = imop_struct.float2D(0.0, 0.0)
            dll.Imop_PupilCompute_FitLegendrePupil(
                message,
                pupil_in.pupil,
                ctypes.byref(center),
                ctypes.byref(halfsize)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.LegendrePupil_t(
                center, 
                halfsize
                )
        except Exception as exception:
            raise Exception(__name__+' : fit_legendre_pupil',exception)
