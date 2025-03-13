#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_enum as imop_enum
import io_wavekit_structure as imop_struct

class HasoConfig(object):
    """Class HasoConfig
    """

    @staticmethod
    def get_config(config_file_path):
        """Get Haso configuration information.
        
        :return: Haso configuration
        :rtype: tuple(HasoConfig_t, HasoSpecs_t)
        """
        dll   = imop_library.load_dll()
        try:     
            message = ctypes.create_string_buffer(256)            
            serial_number = ctypes.create_string_buffer(256)
            revision = ctypes.c_uint()
            model = ctypes.create_string_buffer(256)
            nb_subapertures = imop_struct.uint2D(0, 0)
            ulens_step_um = imop_struct.float2D(0.0, 0.0)
            position_for_alignment_pixels = imop_struct.float2D(0.0, 0.0)
            radius_tolerance_pixels = ctypes.c_uint()
            default_start_subaperture_sspp = imop_struct.uint2D(0, 0)
            lower_calibration_wavelength_nm = ctypes.c_double()
            upper_calibration_wavelength_nm = ctypes.c_double()
            black_subaperture_position_sspp = imop_struct.uint2D(0, 0)
            tilt_limit_mrad = ctypes.c_double()
            radius_m = ctypes.c_double()
            micro_lens_focal = ctypes.c_double()
            smearing_limit_wavelength_nm = ctypes.c_double()
            smearing_limit_exposure_duration_s = ctypes.c_double()
            internal_options_list = ctypes.create_string_buffer(256)
            software_info_list = ctypes.create_string_buffer(256)
            sdk_info_list = ctypes.create_string_buffer(256)     
            res = dll.Imop_CoreEngine_GetConfig(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                serial_number,
                ctypes.byref(revision),
                model,
                ctypes.byref(nb_subapertures),
                ctypes.byref(ulens_step_um),
                ctypes.byref(position_for_alignment_pixels),
                ctypes.byref(radius_tolerance_pixels),
                ctypes.byref(default_start_subaperture_sspp),
                ctypes.byref(lower_calibration_wavelength_nm),
                ctypes.byref(upper_calibration_wavelength_nm),
                ctypes.byref(black_subaperture_position_sspp),
                ctypes.byref(tilt_limit_mrad),
                ctypes.byref(radius_m),
                ctypes.byref(micro_lens_focal),
                ctypes.byref(smearing_limit_wavelength_nm),
                ctypes.byref(smearing_limit_exposure_duration_s),
                internal_options_list,
                software_info_list,
                sdk_info_list
                )            
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)        
            return (
                imop_struct.Config_t(
                    model.value.decode('utf-8'),
                    serial_number.value.decode('utf-8'),
                    revision.value,
                    None
                    ),
                imop_struct.HasoSpecs_t(
                    nb_subapertures,
                    ulens_step_um,
                    position_for_alignment_pixels,
                    radius_m.value,
                    radius_tolerance_pixels.value,
                    tilt_limit_mrad.value,
                    default_start_subaperture_sspp,
                    black_subaperture_position_sspp,
                    micro_lens_focal.value,
                    internal_options_list.value.decode('utf-8'),
                    software_info_list.value.decode('utf-8'),
                    sdk_info_list.value.decode('utf-8')
                    ),
                imop_struct.WaveLength_t(
                    None,
                    lower_calibration_wavelength_nm.value,
                    upper_calibration_wavelength_nm.value
                    )
                )
        except Exception as exception:
            raise Exception(__name__+' : get_config',exception)

    @staticmethod
    def is_haso_config_file(config_file_path):
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            is_haso_config_file = ctypes.c_bool()     
            res = dll.Imop_CoreEngine_IsHasoConfigFile(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                ctypes.byref(is_haso_config_file)
                )            
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return is_haso_config_file.value
        except Exception as exception:
            raise Exception(__name__+' : is_haso_config_file',exception)

    @staticmethod
    def get_serial_number(config_file_path):
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            serial_number = ctypes.create_string_buffer(256)
            res = dll.Imop_CoreEngine_GetSerialNumber(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                serial_number
                )            
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return serial_number.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_serial_number',exception)

    @staticmethod
    def get_sdk_config(config_file_path):
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            has_casao_sdk = ctypes.c_bool()
            has_haso_sdk = ctypes.c_bool()
            has_options_sdk = ctypes.c_bool()
            has_x64_traitepente = ctypes.c_bool()
            res = dll.Imop_CoreEngine_GetSDKConfig(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                ctypes.byref(has_haso_sdk),
                ctypes.byref(has_casao_sdk),
                ctypes.byref(has_options_sdk),
                ctypes.byref(has_x64_traitepente)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                has_casao_sdk.value,
                has_haso_sdk.value,
                has_options_sdk.value,
                has_x64_traitepente.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_sdk_config',exception)

    @staticmethod
    def get_software_config(config_file_path):
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            has_haso_soft = ctypes.c_bool()
            has_waveview_lite_soft = ctypes.c_bool()
            has_casao_soft = ctypes.c_bool()
            has_pharao_soft = ctypes.c_bool()
            has_psf_soft = ctypes.c_bool()
            has_mtf_soft = ctypes.c_bool()
            has_msq_soft = ctypes.c_bool()   
            res = dll.Imop_CoreEngine_GetSoftwareConfig(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                ctypes.byref(has_haso_soft),
                ctypes.byref(has_waveview_lite_soft),
                ctypes.byref(has_casao_soft),
                ctypes.byref(has_pharao_soft),
                ctypes.byref(has_psf_soft),
                ctypes.byref(has_mtf_soft),
                ctypes.byref(has_msq_soft)
                )            
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.SoftwareConfig_t(
                has_haso_soft.value, 
                has_waveview_lite_soft.value,
                has_casao_soft.value, 
                has_pharao_soft.value,
                has_psf_soft.value,
                has_mtf_soft.value,
                has_msq_soft.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_software_config',exception)

    @staticmethod
    def get_focal_cam_config(config_file_path):
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            has_focalcam = ctypes.c_bool() 
            res = dll.Imop_CoreEngine_GetFocalCamConfig(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                ctypes.byref(has_focalcam)
                )            
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return has_focalcam.value
        except Exception as exception:
            raise Exception(__name__+' : get_focal_cam_config',exception)

    @staticmethod
    def get_hardware_authorize_config(config_file_path):
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            has_wavefront_sensor_hardware = ctypes.c_bool() 
            has_wavefront_corrector_hardware = ctypes.c_bool() 
            has_camera_hardware = ctypes.c_bool() 
            res = dll.Imop_CoreEngine_GetHardwareAuthorizeConfig(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                ctypes.byref(has_wavefront_sensor_hardware),
                ctypes.byref(has_wavefront_corrector_hardware),
                ctypes.byref(has_camera_hardware)
                )            
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                has_wavefront_sensor_hardware.value, 
                has_wavefront_corrector_hardware.value,
                has_camera_hardware.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_hardware_authorize_config',exception)

    @staticmethod
    def get_camera_plugin_name(config_file_path):
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            plugin_name = ctypes.create_string_buffer(256) 
            res = dll.Imop_CoreEngine_GetCameraPluginName(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                plugin_name
                )            
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return plugin_name.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_camera_plugin_name',exception)

    @staticmethod
    def get_corrector_plugin_name(config_file_path):
        dll   = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            plugin_name = ctypes.create_string_buffer(256) 
            res = dll.Imop_CoreEngine_GetWavefrontCorrectorPluginName(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                plugin_name
                )            
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return plugin_name.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_corrector_plugin_name',exception)
