#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_phase as imop_phase
import io_wavekit_structure as imop_struct
import io_wavekit_hasoslopes as imop_hslp

class Compute() : 
    """Class Compute
    """
    
    @staticmethod
    def phase_zonal(
        compute_phase_set,
        hasodata
        ):
        """Compute phase using processed slopes contained in the HasoData object.
        Filter some aberrations of reconstructed phase,\
        depending on the filter given in zonal phase reconstruction parameters provided by ComputePhaseSet.
        
        .. warning:: allowed ComputePhaseSet types : E_COMPUTEPHASESET.ZONAL
        
        :param compute_phase_set: ComputePhaseSet object
        :type compute_phase_set: ComputePhaseSet
        :param hasodata:  HasoData object 
        :type hasodata: HasoData
        :return: Phase Object.
        :rtype: Phase
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            phase_out = imop_phase.Phase(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    )
                )
            dll.Imop_Compute_PhaseZonal(
                message,
                compute_phase_set.computephaseset,
                hasodata.hasodata,
                phase_out.phase
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return phase_out
        except Exception as exception:
            raise Exception(__name__+' : phase_zonal', exception)

    @staticmethod
    def phase_modal(
        compute_phase_set,
        hasodata,
        modal_coef
        ):
        """Compute phase using processed slopes contained in the HasoData object.
        Filter some aberrations of reconstructed phase,
        depending on the filter given in zonal phase reconstruction parameters provided by ComputePhaseSet
        
        .. warning:: allowed ComputePhaseSet types : E_COMPUTEPHASESET.MODAL_LEGENDRE, E_COMPUTEPHASESET.MODAL_ZERNIKE
        
        :param compute_phase_set: ComputePhaseSet object 
        :type compute_phase_set: ComputePhaseSet
        :param hasodata:  HasoData object 
        :type hasodata: HasoData
        :param modal_coef:  ModalCoef object, Preferences are used as input, coefficients values are updated by the processing
        :type modal_coef: ModalCoef
        :return: Phase Object.
        :rtype: Phase
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            phase_out = imop_phase.Phase(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    )
                )
            dll.Imop_Compute_PhaseModal(
                message,
                compute_phase_set.computephaseset,
                hasodata.hasodata,
                modal_coef.modalcoef,
                phase_out.phase
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return phase_out
        except Exception as exception:
            raise Exception(__name__+' : phase_modal',exception)

    @staticmethod
    def phase_modal_zonal(
        compute_phase_set,
        hasodata,
        modal_coef,
        serial_number
        ):
        """Compute phase using processed slopes contained in the HasoData object.
        Filter some aberrations of reconstructed phase,
        depending on the filter given in zonal phase reconstruction parameters provided by **ComputePhaseSet**.
        
        .. warning:: allowed ComputePhaseSet types : E_COMPUTEPHASESET.MODAL_LEGENDRE, E_COMPUTEPHASESET.MODAL_ZERNIKE
        
        :param compute_phase_set: ComputePhaseSet object 
        :type compute_phase_set: ComputePhaseSet
        :param hasodata:  HasoData object 
        :type hasodata: HasoData
        :param modal_coef:  ModalCoef object, Preferences are used as input, coefficients values are updated by the processing
        :type modal_coef: ModalCoef
        :param serial_number:  Serial Number
        :type serial_number: String
        :return: HasoSlopes Residual Object.
        :rtype: Hasoslopes
        :return: Phase Object.
        :rtype: Phase
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = imop_hslp.HasoSlopes(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    ),
                serial_number = serial_number
                )
            phase_out = imop_phase.Phase(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    )
                )
            dll.Imop_Compute_PhaseModalZonal(
                message,
                compute_phase_set.computephaseset,
                hasodata.hasodata,
                modal_coef.modalcoef,
                hasoslopes_out.hasoslopes,
                phase_out.phase
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                hasoslopes_out,
                phase_out
                )
        except Exception as exception:
            raise Exception(__name__+' : phase_modal_zonal',exception)

    @staticmethod
    def phase_from_coef(
        compute_phase_set,
        modal_coef
        ):
        """Compute phase from modal coefficients in a given
        basis of polynomials provided by ComputePhaseSet.
        Modal coeffs must have a projection pupil set :
        
        .. seealso:: ModalCoef.set_zernike_prefs or ModalCoef.set_legendre_prefs for setting preferences
        
        .. seealso:: Compute.zernike_pupil or Compute.legendre_pupil to fit projection pupil to a natural slopes pupil
        
        .. warning:: allowed ComputePhaseSet types : all but E_COMPUTEPHASESET.ZONAL
        
        :param compute_phase_set: ComputePhaseSet object 
        :type compute_phase_set: ComputePhaseSet
        :param modal_coef:  ModalCoef object
        :type modal_coef: ModalCoef
        :return: Phase Object.
        :rtype: Phase
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            phase_out = imop_phase.Phase(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    )
                )
            dll.Imop_Compute_PhaseFromCoef(
                message,
                compute_phase_set.computephaseset,
                modal_coef.modalcoef,
                phase_out.phase
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return phase_out
        except Exception as exception:
            raise Exception(__name__+' : phase_from_coef',exception)

    @staticmethod
    def coef_from_hasodata(
        compute_phase_set,
        hasodata,
        modal_coef
        ):
        """Compute modal coefficients that represent the processed slopes of the input HasoData in a given
        basis of polynomials provided by ComputePhaseSet.
        
        .. warning:: allowed ComputePhaseSet types : all but E_COMPUTEPHASESET.ZONAL
        
        :param compute_phase_set: ComputePhaseSet object 
        :type compute_phase_set: ComputePhaseSet
        :param hasodata:  HasoData object
        :type hasodata: HasoData
        :param modal_coef:  ModalCoef object adress. Preferences are used as input, coefficients values are updated by the processing.
        :type modal_coef: ModalCoef
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            dll.Imop_Compute_CoefFromHasoData(
                message,
                compute_phase_set.computephaseset,
                hasodata.hasodata,
                modal_coef.modalcoef
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : ',exception)