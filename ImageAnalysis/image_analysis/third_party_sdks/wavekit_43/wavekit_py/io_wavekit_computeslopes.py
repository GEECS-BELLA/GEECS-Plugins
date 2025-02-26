#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_phase as imop_phase
import io_wavekit_hasoslopes as imop_hslp

class ComputeSlopes():
    """Class ComputeSlopes
    """

    @staticmethod
    def phase_zonal(
        computephaseset,
        hasoslopes
        ):
        """Compute phase from HasoSlopes.
        Filter some aberrations of reconstructed phase,
        depending on the filter given in zonal phase reconstruction parameters provided by computephaseset.
        
        .. warning:: allowed ComputePhaseSet types : E_COMPUTEPHASESET.ZONAL
        
        :param computephaseset: ComputePhaseSet object
        :type computephaseset: ComputePhaseSet
        :param hasoslopes: HasoSlopes object
        :type hasoslopes: HasoSlopes
        :return: Phase object
        :rtype: Phase
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            phase_out = imop_phase.Phase(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    ),
                )
            dll.Imop_ComputeSlopes_PhaseZonal(
                message,
                computephaseset.computephaseset,
                hasoslopes.hasoslopes,
                phase_out.phase
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return phase_out            
        except Exception as exception:
            raise Exception(__name__+' : phase_zonal',exception)

    @staticmethod
    def phase_modal(
        computephaseset,
        hasoslopes,
        modalcoef
        ):
        """Compute phase from slopes using given modalcoef \
        basis of polynomials provided by computephaseset.
        
        .. warning:: allowed ComputePhaseSet types : E_COMPUTEPHASESET.MODAL_LEGENDRE, E_COMPUTEPHASESET.MODAL_ZERNIKE
        
        :param computephaseset: ComputePhaseSet object
        :type computephaseset: ComputePhaseSet
        :param hasoslopes: HasoSlopes object
        :type hasoslopes: HasoSlopes
        :param modalcoef: ModalCoef object adress. Preferences are used as input, coefficients values are updated by the processing
        :type modalcoef: ModalCoef
        :return: Phase object
        :rtype: Phase
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            phase_out = imop_phase.Phase(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    ),
                )
            dll.Imop_ComputeSlopes_PhaseModal(
                message,
                computephaseset.computephaseset,
                hasoslopes.hasoslopes,
                modalcoef.modalcoef,
                phase_out.phase          
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return phase_out 
        except Exception as exception:
            raise Exception(__name__+' : phase_modal',exception)

    @staticmethod
    def phase_modal_zonal(
        computephaseset,
        hasoslopes,
        modalcoef
        ):
        """Compute phase from slopes using both zonal and modal reconstruction.
        
        .. warning:: allowed ComputePhaseSet types : E_COMPUTEPHASESET.MODAL_ZONAL_ZERNIKE, E_COMPUTEPHASESET.MODAL_ZONAL_LEGENDRE

        Computes phase from slopes thanks to an algorithm that
        combines the results of modal and zonal phase processors.  
        It first computes modal coefficients in the requested polynomial basis using given modal preferences. 
        From those coefficients, it reconstructs slopes.
        The difference between input slopes and reconstructed slopes is called residual slopes.
        Output phase is obtained by summing modal phase obtained from modal
        coefficients using given modal parameters, and zonal phase obtained from
        residual slopes using zonal parameters consisting of given zonal preferences
        and a default-constructed zonal filter.
        Modal coeffs must have a projection pupil set :   
   
        .. seealso:: ModalCoef.set_zernike_prefs or ModalCoef.set_legendre_prefs for setting preferences
        
        .. seealso:: Compute.zernike_pupil or Compute.legendre_pupil to fit projection pupil to a natural slopes pupil.
   
        :param computephaseset: ComputePhaseSet object
        :type computephaseset: ComputePhaseSet
        :param hasoslopes: HasoSlopes object containing the input slopes
        :type hasoslopes: HasoSlopes
        :param modalcoef: ModalCoef object adress. Preferences are used as input, coefficients values are updated by the processing
        :type modalcoef: ModalCoef
        :return: Residual Hasoslopes object, Phase object
        :rtype: tuple(HasoSlopes, Phase)
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            residual_slopes_out = imop_hslp.HasoSlopes(hasoslopes = hasoslopes)
            phase_out = imop_phase.Phase(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    ),
                )
            dll.Imop_ComputeSlopes_PhaseModalZonal(
                message,
                computephaseset.computephaseset,
                hasoslopes.hasoslopes,
                modalcoef.modalcoef,
                residual_slopes_out.hasoslopes,
                phase_out.phase              
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                residual_slopes_out, 
                phase_out     
                )
        except Exception as exception:
            raise Exception(__name__+' : phase_modal_zonal',exception)

    @staticmethod
    def phase_modal_zonal_from_coef(
        computephaseset,
        hasoslopes,
        modalcoef
        ):
        """Compute phase from modal coefficients stored in modalcoef given the preferences provided by computephaseset and adds
        the residual phase computed on input HasoSlopes using the modal + zonal reconstruction.
        Modal coeffs must have a projection pupil set in their preferences.
        
        .. seealso:: ComputeSlopes.phase_modal_zonal for reconstruction process.
        
        .. seealso:: ComputePupil.fit_zernike_pupil to fit a circular pupil to a natural slopes pupil for Zernike projection.
        
        .. seealso:: ComputePupil.fit_legendre_pupil to fit rectangular pupil to a natural slopes pupil for Legendre projection.
        
        .. seealso:: ModalCoef.set_zernike_prefs for setting preferences for a *ModalCoef* of type Zernike.
        
        .. seealso:: ModalCoef.set_legendre_prefs for setting preferences for a *ModalCoef* of type Legendre.
        
        .. warning:: Allowed ComputePhaseSet types : E_COMPUTEPHASESET.MODAL_ZONAL_ZERNIKE, E_COMPUTEPHASESET.MODAL_ZONAL_LEGENDRE.

        :param computephaseset: ComputePhaseSet object
        :type computephaseset: ComputePhaseSet
        :param hasoslopes: HasoSlopes object containing the input zonal residual slopes
        :type hasoslopes: HasoSlopes
        :param modalcoef: ModalCoef object 
        :type modalcoef: ModalCoef
        :return: Residual Hasoslopes object, Phase object
        :rtype: tuple(HasoSlopes, Phase)
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            residual_slopes = imop_hslp.HasoSlopes(hasoslopes = hasoslopes)
            phase_out = imop_phase.Phase(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1),
                    imop_struct.float2D(1.0,1.0)
                    ),
                )
            dll.Imop_ComputeSlopes_PhaseModalZonalFromCoef(
                message,
                computephaseset.computephaseset,
                hasoslopes.hasoslopes,
                modalcoef.modalcoef,
                residual_slopes.hasoslopes,
                phase_out.phase              
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                residual_slopes, 
                phase_out       
                )            
        except Exception as exception:
            raise Exception(__name__+' : phase_modal_zonal_from_coef',exception)

    @staticmethod
    def slopes_from_coef(
        computephaseset,
        modalcoef,
        hasoslopes
        ):
        """Compute slopes from modal coefficients that represent them in
        a given basis of polynomials provided by computephaseset.
        Modal coeffs must have a projection pupil set in their preferences.
        
        .. seealso:: ComputeSlopes.phase_modal_zonal for reconstruction process
        
        .. seealso:: ComputePupil.fit_zernike_pupil to fit a circular pupil to a natural slopes pupil for Zernike projection
        
        .. seealso:: ComputePupil.fit_legendre_pupil to fit rectangular pupil to a natural slopes pupil for Legendre projection
        
        .. seealso:: ModalCoef.set_zernike_prefs for setting preferences for a ModalCoef of type Zernike
        
        .. seealso:: ModalCoef.set_legendre_prefs for setting preferences for a ModalCoef of type Legendre
        
        .. warning:: Allowed ComputePhaseSet types : E_COMPUTEPHASESET.MODAL_ZERNIKE, E_COMPUTEPHASESET.MODAL_LEGENDRE

        :param computephaseset: ComputePhaseSet object
        :type computephaseset: ComputePhaseSet
        :param modalcoef: ModalCoef object 
        :type modalcoef: ModalCoef
        :param hasoslopes: Hasoslopes object
        :type hasoslopes: HasoSlopes
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            dll.Imop_ComputeSlopes_SlopesFromCoef(
                message,
                computephaseset.computephaseset,
                modalcoef.modalcoef,
                hasoslopes.hasoslopes          
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return hasoslopes     
        except Exception as exception:
            raise Exception(__name__+' : slopes_from_coef',exception)

    @staticmethod
    def coef_from_slopes(
        computephaseset,
        hasoslopes,
        modalcoef
        ):
        """Compute modal coefficients that represent the processed slopes of the input Haso Data in a given
        basis of polynomials provided by computephaseset.
        
        .. warning:: allowed ComputePhaseSet types : all but E_COMPUTEPHASESET.ZONAL

        :param computephaseset: ComputePhaseSet object
        :type computephaseset: ComputePhaseSet
        :param hasoslopes: HasoSlopes object 
        :type hasoslopes: HasoSlopes
        :param modalcoef: ModalCoef object 
        :type modalcoef: ModalCoef
        """
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            dll.Imop_ComputeSlopes_CoefFromSlopes(
                message,
                computephaseset.computephaseset,
                hasoslopes.hasoslopes,
                modalcoef.modalcoef            
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)    
        except Exception as exception:
            raise Exception(__name__+' : coef_from_slopes',exception)
            
    @staticmethod
    def spot_diagram_from_slopes(
        hasoslopes,
        radius_of_curvature_mm,
        defocus_mm
    ):
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            
            x_position_mm = ctypes.c_float()
            y_position_mm = ctypes.c_float()
            distances = ctypes.c_float()
            size = ctypes.c_uint()
            dll.Imop_ComputeSlopes_SpotDiagramFromSlopes(
                message,
                hasoslopes.hasoslopes,
                ctypes.c_float(x_position_mm),
                ctypes.c_float(y_position_mm),   
                ctypes.c_float(defocus_mm),   
                ctypes.byref(x_position_mm),
                ctypes.byref(y_position_mm),
                ctypes.byref(distances),
                ctypes.byref(size)                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return(x_position_mm.value, y_position_mm.value, distances.value, size.value)
        except Exception as exception:
            raise Exception(__name__+' : coef_from_slopes',exception)
