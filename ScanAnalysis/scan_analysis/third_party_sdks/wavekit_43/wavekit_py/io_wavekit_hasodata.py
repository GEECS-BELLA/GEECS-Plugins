#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_slopespostprocessorlist as imop_sppl
import io_wavekit_computephaseset as imop_computephaseset
import io_wavekit_modalcoef as imop_modalcoef
import io_wavekit_enum as imop_enum
import io_wavekit_hasoslopes as imop_hslp
import io_wavekit_structure as imop_struct

class HasoData(object):
    """HasoData Class
    
    - Default Constructor (Empty HasoData)
    
    - Constructor from HasoSlopes :
        - **hasoslopes** - HasoSlopes : HasoSlopes object
    
    - Constructor from Copy :
        - **hasodata** - HasoData : HasoData object
    
    - Constructor from Has File :
        - **has_file_path** - string : Absolute path (\*.has)
    """

    def __init_(self):
        """Empty HasoData constructor.
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_NewEmpty(
                message,
                ctypes.pointer(self.hasodata)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init empty',exception)

    def __init_from_hasoslopes(
        self,
        hasoslopes
        ):
        """HasoData constructor from HasoSlopes.
        
        :param hasoslopes: HasoSlopes object
        :type hasoslopes: HasoSlopes
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_New(
                message,
                ctypes.pointer(self.hasodata),
                hasoslopes.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_hasoslopes',exception)

    def __init_from_copy(
        self,
        hasodata
        ):
        """HasoData constructor from copy.
        
        :param hasodata: HasoData object
        :type hasodata: HasoData
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_NewFromCopy(
                message,
                hasodata.hasodata,
                ctypes.pointer(self.hasodata)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_copy',exception)

    def __init_from_file(
        self,
        has_file_path
        ):
        """HasoData constructor from a .has file. If file opening fails, an exception is thrown.
        
        :param has_file_path: Absolute path to .has file
        :type has_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_NewFromFile(
                message,
                ctypes.pointer(self.hasodata),
                ctypes.c_char_p(has_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_file',exception)

    def __init__(self, **kwargs):
        """HasoData constructor
        """
        self.hasodata = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 0):
                entered = 1
                self.__init_()
            if(arg_size == 1):
                if('hasoslopes' in kwargs):
                    entered = 1
                    self.__init_from_hasoslopes(kwargs['hasoslopes'])
                if('hasodata' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['hasodata'])
                if('has_file_path' in kwargs):
                    entered = 1
                    self.__init_from_file(kwargs['has_file_path'])
        except Exception as exception:
            raise Exception(__name__+' : init',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE HASODATA OBJECT---')
                    
    def __del_obj__(self):
        """HasoData destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_HasoData_Delete(message, self.hasodata)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def save(
        self,
        has_file_path,
        comments,
        session_name
        ):
        """Saves a HasoData object to a (\*.has) file.
        If file already exists, it is overwritten, provided that opening
        succeeds. If file opening fails, an exception is thrown.
        
        :param has_file_path: Absolute path to the (\*.has) file
        :type has_file_path: string
        :param comments: Comments
        :type comments: string
        :param session_name: Session name
        :type session_name: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_Save(
                message,
                self.hasodata,
                ctypes.c_char_p(has_file_path.encode('utf-8')),
                ctypes.c_char_p(comments.encode('utf-8')),
                ctypes.c_char_p(session_name.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : save',exception)

    def set_hasoslopes(
        self,
        hasoslopes
        ):
        """Set raw HasoSlopes objects.
        
        :param hasoslopes: HasoSlopes object
        :type hasoslopes: HasoSlopes
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_SetHasoSlopes(
                message,
                self.hasodata,
                hasoslopes.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_hasoslopes',exception)
    
    
    def get_compute_slopes_parameters(self):
        """Get informations from \ref HasoData object.
        
        :return: Wavelength(nm), StartSubPupil, DenoisingStrength, isAutoStartSubPupilDetectionON, isSpotTrackerON, isLiftON, isELSON
        :rtype: tuple(float, uint2D, float, bool, bool, bool, bool)
        """
        try:
            message       = ctypes.create_string_buffer(256)
            wavelength_nm = ctypes.c_double()
            start_subpupil_size = ctypes.c_int()
            denoising_strength  = ctypes.c_double()
            auto_start   = ctypes.c_ubyte()
            spot_tracker = ctypes.c_ubyte()
            lift_algo    = ctypes.c_ubyte()
            els_algo     = ctypes.c_ubyte()
            self.dll.Imop_HasoData_GetComputeSlopesParameters_StartMultiPupilsSize(
                message, 
                self.hasodata,
                ctypes.byref(start_subpupil_size)            
                )
            x_start_subpupil = numpy.zeros(
                start_subpupil_size.value, 
                dtype = numpy.uintc
                )
            y_start_subpupil = numpy.zeros(
                start_subpupil_size.value, 
                dtype = numpy.uintc
                )
            self.dll.Imop_HasoData_GetComputeSlopesParameters.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS") , 
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p         
                ]
            self.dll.Imop_HasoData_GetComputeSlopesParameters(
                message,
                self.hasodata,  
                ctypes.byref(wavelength_nm), 
                x_start_subpupil,  
                y_start_subpupil,
                ctypes.byref(denoising_strength),
                ctypes.byref(auto_start),
                ctypes.byref(spot_tracker),
                ctypes.byref(lift_algo),
                ctypes.byref(els_algo)           
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                wavelength_nm.value,
                x_start_subpupil, 
                y_start_subpupil,
                denoising_strength.value,
                bool(denoising_strength),
                bool(spot_tracker),
                bool(lift_algo),
                bool(els_algo)
                )
        except Exception as exception:
            raise Exception(__name__+' : get_compute_slopes_parameters',exception)

    def set_wavelength(
        self,
        wavelength
        ):
        """Set source wavelength.
        
        :param wavelength: Source wavelength (nm)
        :type wavelength: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_SetWaveLength(
                message,
                self.hasodata,
                ctypes.c_float(wavelength)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_wavelength',exception)

    def get_dimensions(self):
        """Get HasoData dimensions and steps.
        
        :return: HasoData dimensions of the pupil
        :rtype: dimensions
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = imop_struct.uint2D(0, 0)
            steps_out = imop_struct.float2D(0.0, 0.0)
            self.dll.Imop_HasoData_GetDimensionsAndSteps(
                message,
                self.hasodata,
                ctypes.byref(size_out),
                ctypes.byref(steps_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            dim_out = imop_struct.dimensions(size_out, steps_out)
            return dim_out
        except Exception as exception:
            raise Exception(__name__+' : get_dimensions',exception)

    def get_processor_list(self):
        """SlopesPostProcessorList getter.
        
        :return: SlopesPostProcessorList object
        :rtype: SlopesPostProcessorList
        """
        try:
            message = ctypes.create_string_buffer(256)
            proclist = imop_sppl.SlopesPostProcessorList()
            self.dll.Imop_HasoData_GetProcessorList(
                message,
                self.hasodata,
                proclist.slopespostprocessorlist
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return proclist
        except Exception as exception:
            raise Exception(__name__+' : get_processor_list',exception)

    @staticmethod
    def get_info_from_has_file(has_file_path):
        """Get informations from (\*.has) file.
        
        :return: Haso Config : haso_model, haso_serial_number, revision; Haso Dimensions : size, step; Haso Metadata : time_stamp, comments, session_name; Used Wavelength : wavelength_nm, lower_calibration_wl, upper_calibration_wl; Software List
        :rtype: tuple(Config_t, WaveLength_t, dimensions, Metadata_t, str)
        """
        
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            haso_model = ctypes.create_string_buffer(256)
            haso_serial_number = ctypes.create_string_buffer(256)
            revision = ctypes.c_uint()
            size = imop_struct.uint2D(0, 0)
            step = imop_struct.float2D(0.0, 0.0)
            time_stamp = ctypes.c_double()
            comments = ctypes.create_string_buffer(256)
            session_name = ctypes.create_string_buffer(256)
            wavelength_nm = ctypes.c_double()
            lower_calibration_wl = ctypes.c_double()
            upper_calibration_wl = ctypes.c_double()
            softwareInfoList = ctypes.create_string_buffer(256)
            dll.Imop_HasoData_GetInfoFromHasFile(
                message,
                ctypes.c_char_p(has_file_path.encode('utf-8')),
                haso_model,
                haso_serial_number,
                ctypes.byref(revision),
                ctypes.byref(size),
                ctypes.byref(step),
                ctypes.byref(time_stamp),
                comments,
                session_name,
                ctypes.byref(wavelength_nm), 
                ctypes.byref(lower_calibration_wl), 
                ctypes.byref(upper_calibration_wl)              
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            dll.Imop_HasoData_GetSoftwareInfoFromHasFile(
                message,
                ctypes.c_char_p(has_file_path.encode('utf-8')),
                haso_model,
                haso_serial_number,
                comments,
                session_name,
                softwareInfoList             
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                imop_struct.Config_t (
                    str(haso_model.value.decode('utf-8')),
                    str(haso_serial_number.value.decode('utf-8')),
                    revision.value,
                    None
                    ),
                imop_struct.WaveLength_t (
                    wavelength_nm.value,
                    lower_calibration_wl.value,
                    upper_calibration_wl.value
                    ),
                imop_struct.dimensions(
                    size,
                    step
                    ),
                imop_struct.Metadata_t(
                    str(comments.value.decode('utf-8')),
                    str(session_name.value.decode('utf-8')),
                    time_stamp.value,
                    None,
                    None,
                    None,
                    None
                    ),
                str(softwareInfoList.value.decode('utf-8'))
                )
        except Exception as exception:
            raise Exception(__name__+' : get_info_from_has_file',exception)

    def get_hasoslopes(self):
        """Get raw and processed HasoSlopes objects.
        
        :return: Computed HasoSlopes, Raw HasoSlopes
        :rtype: tuple(HasoSlopes, HasoSlopes)
        """
        try:
            message = ctypes.create_string_buffer(256)
            computed_hasoslopes_out = imop_hslp.HasoSlopes(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1, 1), 
                    imop_struct.float2D(1.0, 1.0)
                    ),
                serial_number = ""
                )
            raw_hasoslopes_out = imop_hslp.HasoSlopes(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1, 1), 
                    imop_struct.float2D(1.0, 1.0)
                    ),
                serial_number = ""
                )
            self.dll.Imop_HasoData_GetHasoSlopes(
                message,
                self.hasodata,
                computed_hasoslopes_out.hasoslopes,
                raw_hasoslopes_out.hasoslopes
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                computed_hasoslopes_out,
                raw_hasoslopes_out
                )
        except Exception as exception:
            raise Exception(__name__+' : get_hasoslopes',exception)

    def get_compute_phase_mode(self):
        """Get phase reconstruction mode.
        
        :return: E_COMPUTEPHASESET phase reconstruction mode enumeration.
        :rtype: uchar
        """
        try:
            message = ctypes.create_string_buffer(256)
            type_phase_out = ctypes.c_ubyte()
            self.dll.Imop_HasoData_GetComputePhaseMode(
                message,
                self.hasodata,
                ctypes.byref(type_phase_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return type_phase_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_compute_phase_mode',exception)

    def get_info(self):
        """Get informations from HasoData object.
        
        :return: haso_model, haso_serial_number, camera_serial_number, revision, lower_calibration_wl, upper_calibration_wl, exposure_time_requested, exposure_time_applied, nb_summed_images, time_stamp, is_background_removed, is_smearing_removed, wavelength_nm, session_name, trigger_type, comments, max_level,  start_subpupil, denoising_strength
        :rtype: tuple(Config_t, WaveLength_t, Metadata_t, AcquisitionInfo_t, uint2D, float)
        """
        try:
            message = ctypes.create_string_buffer(256)
            haso_model = ctypes.create_string_buffer(256)
            haso_serial_number = ctypes.create_string_buffer(256)
            camera_serial_number = ctypes.create_string_buffer(256)
            revision = ctypes.c_uint()
            lower_calibration_wl = ctypes.c_double()
            upper_calibration_wl = ctypes.c_double()
            exposure_time_requested = ctypes.c_uint()
            exposure_time_applied = ctypes.c_uint()
            nb_summed_images = ctypes.c_uint()
            time_stamp = ctypes.c_double()
            is_background_removed = ctypes.c_ubyte()
            is_smearing_removed = ctypes.c_ubyte()
            wavelength_nm = ctypes.c_double()
            session_name = ctypes.create_string_buffer(256)
            trigger_type = ctypes.create_string_buffer(256)
            comments = ctypes.create_string_buffer(256)
            max_level = ctypes.c_ulonglong()
            start_subpupil = imop_struct.uint2D(0, 0)
            denoising_strength = ctypes.c_double()            
            self.dll.Imop_HasoData_GetInfo(
                message,
                self.hasodata,
                haso_model,
                haso_serial_number,
                camera_serial_number,
                ctypes.byref(revision),
                ctypes.byref(lower_calibration_wl),
                ctypes.byref(upper_calibration_wl),
                ctypes.byref(exposure_time_requested),
                ctypes.byref(exposure_time_applied),
                ctypes.byref(nb_summed_images),
                ctypes.byref(time_stamp),
                ctypes.byref(is_background_removed),
                ctypes.byref(is_smearing_removed),
                ctypes.byref(wavelength_nm),
                session_name,
                trigger_type,
                comments,
                ctypes.byref(max_level),
                ctypes.byref(start_subpupil),
                ctypes.byref(denoising_strength)                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                imop_struct.Config_t(
                    haso_model.value.decode('utf-8'),
                    haso_serial_number.value.decode('utf-8'),
                    revision.value,
                    None
                    ),
                imop_struct.WaveLength_t(
                    wavelength_nm.value,
                    lower_calibration_wl.value,
                    upper_calibration_wl.value
                    ),
                imop_struct.Metadata_t(
                    comments.value.decode('utf-8'),
                    session_name.value.decode('utf-8'),
                    time_stamp.value,
                    bool(is_background_removed),
                    bool(is_smearing_removed),
                    camera_serial_number.value.decode('utf-8'),
                    max_level.value
                    ),
                imop_struct.AcquisitionInfo_t(
                    exposure_time_requested.value,
                    exposure_time_applied.value,
                    None,
                    nb_summed_images.value,
                    trigger_type.value.decode('utf-8')
                    ),
                start_subpupil,
                denoising_strength.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_info',exception)

    def apply_slopes_post_processor_list(
        self,
        slopespostprocessorlist
        ):
        """Set and apply a SlopesPostProcessorList to the raw HasoSlopes stored in the HasoData object. Results is stored in the HasoData object processed slopes.
        
        :param slopespostprocessorlist: SlopesPostProcessorList object
        :type slopespostprocessorlist: SlopesPostProcessorList
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_ApplySlopesPostProcessorList(
                message,
                self.hasodata,
                slopespostprocessorlist.slopespostprocessorlist
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : apply_slopes_post_processor_list',exception)

    def reset_to_raw_data(
        self
        ):
        """Reset slopes to their raw state and clear process history.
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_ResetToRawData(
                message,
                self.hasodata
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : reset_to_raw_data',exception)

    def get_compute_phase_parameters(self):
        """Get ComputePhaseSet object.
        
        :return: SlopesPostProcessorList object
        :rtype: SlopesPostProcessorList
        """
        try:
            message = ctypes.create_string_buffer(256)
            cps_out = imop_computephaseset.ComputePhaseSet(hasodata = self)
            modal_type = self.get_compute_phase_mode()
            if (modal_type == imop_enum.E_COMPUTEPHASESET.MODAL_LEGENDRE) or (modal_type == imop_enum.E_COMPUTEPHASESET.MODAL_ZONAL_LEGENDRE):
                mc_out = imop_modalcoef.ModalCoef(modal_type = imop_enum.E_MODAL.LEGENDRE)
            else:
                mc_out = imop_modalcoef.ModalCoef(modal_type = imop_enum.E_MODAL.ZERNIKE)
            self.dll.Imop_HasoData_GetComputePhaseParameters(
                message,
                self.hasodata,
                cps_out.computephaseset,
                mc_out.modalcoef
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            if(modal_type == imop_enum.E_COMPUTEPHASESET.ZONAL):
                return (
                    cps_out,
                    None
                    )
            return (
                cps_out,
                mc_out
                )
        except Exception as exception:
            raise Exception(__name__+' : get_compute_phase_parameters',exception)

    def set_compute_phase_parameters(
        self,
        computephaseset,
        modalcoef
        ):
        """Set ComputePhaseSet object.

        .. warning:: allowed value E_COMPUTEPHASESET
        
        :param computephaseset: ComputePhaseSet object
        :type computephaseset: ComputePhaseSet
        :param modalcoef: ModalCoef object
        :type modalcoef: ModalCoef
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoData_SetComputePhaseParameters(
                message,
                self.hasodata,
                computephaseset.computephaseset,
                modalcoef.modalcoef
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_compute_phase_parameters',exception)
