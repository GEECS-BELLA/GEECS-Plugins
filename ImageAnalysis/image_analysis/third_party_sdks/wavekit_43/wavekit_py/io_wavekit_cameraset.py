#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_enum as imop_enum

class CameraSet(object):
    """Class CameraSet
    
    - Constructor from Config File :
        - **config_file_path** - string : Path to config file (\*.dat)
    
    - Constructor from Image File :
        - **image_file_path** - string : Path to image file (\*.himg)
        
    - Constructor from Camera : 
        - **camera** - Camera : Camera Object
        
    - Constructor from Copy : 
        - **cameraset** - CameraSet : CameraSet Object
    """

    def __init_from_file(
        self,
        config_file_path
        ):
        """CameraSet constructor
        
        :param config_file_path: path to config file
        :type config_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_NewFromConfigFile(
                message,
                ctypes.pointer(self.cameraset),
                ctypes.c_char_p(config_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_file',exception)

    def __init_from_image_file(
        self,
        image_file_path
        ):
        """CameraSet constructor
        
        :param image_file_path: path to image file
        :type image_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_NewFromImageFile(
                message,
                ctypes.c_char_p(image_file_path.encode('utf-8')),
                ctypes.pointer(self.cameraset)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_file',exception)

    def __init_from_camera(
        self,
        camera
        ):
        """CameraSet constructor
        
        :param camera: Camera Object
        :type camera: Camera
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_NewFromCamera(
                message,
                camera.camera,
                ctypes.pointer(self.cameraset)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_camera',exception)

    def __init_from_copy(
        self,
        cameraset
        ):
        """CameraSet constructor from copy
        
        :param cameraset: CameraSet Object
        :type cameraset: CameraSet
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_NewFromCopy(
                message,
                cameraset.cameraset,
                ctypes.pointer(self.cameraset)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_copy',exception)

    def __init__(self,  **kwargs):
        """CameraSet constructor
        """
        self.cameraset = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('config_file_path' in kwargs):
                    entered = 1
                    self.__init_from_file(kwargs['config_file_path'])
                if('image_file_path' in kwargs):
                    entered = 1
                    self.__init_from_image_file(kwargs['image_file_path'])
                if('camera' in kwargs):
                    entered = 1
                    self.__init_from_camera(kwargs['camera'])
                if('cameraset' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['cameraset'])
        except Exception as exception:
            raise Exception(__name__+' : init',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE CAMERASET OBJECT---')

    def __del_obj__(self):
        """CameraSet destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_CameraSet_Delete(
            message, 
            self.cameraset
            )
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_specifications(self):
        """Get Camera specifications
        
        :return: Camera specifications 
        :rtype: Config_t, CameraSpecs_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            driver_name = ctypes.create_string_buffer(256)
            camera_name = ctypes.create_string_buffer(256)
            camera_sn = ctypes.create_string_buffer(256)
            width = ctypes.c_int()
            height = ctypes.c_int()
            pixel_steps_x = ctypes.c_float()
            pixel_steps_y = ctypes.c_float()
            bits_depth = ctypes.c_ubyte()
            is_signed = ctypes.c_bool()
            max_level = ctypes.c_ulonglong()
            max_frame_rate = ctypes.c_float()
            roi_width = ctypes.c_uint()
            roi_height = ctypes.c_uint()
            roi_offset_x = ctypes.c_int()
            roi_offset_y = ctypes.c_int()
            self.dll.Imop_CameraSet_GetSpecifications(
                message,
                self.cameraset,
                driver_name,
                camera_name,
                camera_sn,
                ctypes.byref(width),
                ctypes.byref(height),
                ctypes.byref(pixel_steps_x),
                ctypes.byref(pixel_steps_y),
                ctypes.byref(bits_depth),
                ctypes.byref(is_signed),
                ctypes.byref(max_level),
                ctypes.byref(max_frame_rate),
                ctypes.byref(roi_width),
                ctypes.byref(roi_height),
                ctypes.byref(roi_offset_x),
                ctypes.byref(roi_offset_y)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                imop_struct.Config_t(
                    camera_name.value.decode('utf-8'),
                    camera_sn.value.decode('utf-8'),
                    None,
                    driver_name.value.decode('utf-8')
                    ),
                imop_struct.CameraSpecs_t(
                    imop_struct.dimensions(
                        imop_struct.uint2D(width.value,height.value), 
                        imop_struct.float2D(pixel_steps_x.value, pixel_steps_y.value)
                        ),
                    bits_depth.value,
                    is_signed.value,
                    max_level.value,
                    max_frame_rate.value,
                    imop_struct.uint2D(roi_width.value, roi_height.value), 
                    imop_struct.int2D(roi_offset_x.value, roi_offset_y.value)
                    )
            )   
        except Exception as exception:
            raise Exception(__name__+' : get_driver_info',exception)

    def get_specific_features_list_size(self):
        """Get Camera features list size.
        
        :return: Features list size.
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = ctypes.c_int()
            self.dll.Imop_CameraSet_GetSpecificFeaturesListSize(
                message,
                self.cameraset,
                ctypes.byref(size_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_specific_features_list_size',exception)

    def get_specific_feature_name(
        self,
        index
        ):
        """Get Camera features name at the choosen index.
        
        :param index: Index in the *get_specific_features_list_size* range
        :type index: int
        :return: Feature name
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            feature_name = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_GetSpecificFeatureName(
                message,
                self.cameraset,
                index,
                feature_name
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return feature_name.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_specific_feature_name',exception)

    def get_parameters_list_size(self):
        """Get Camera parameters list size.
        
        :return: Parameters list size.
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = ctypes.c_uint()
            self.dll.Imop_CameraSet_GetParametersListSize(
                message,
                self.cameraset,
                ctypes.byref(size_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_parameters_list_size',exception)

    def get_parameter_name(
        self,
        index
        ):
        """Get Camera parameter name at the choosen index.
        
        :param index: Index in the *get_parameters_list_size* range
        :type index: int
        :return: Parameter name.
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            parameter_name = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_GetParameterName(
                message,
                self.cameraset,
                index,
                parameter_name
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return parameter_name.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_name',exception)

    def get_parameter_type(
        self,
        parameter_name
        ):
        """Get Camera parameter type corresponding to the parameter_name.
        
        :param parameter_name: Parameter Name 
        :type parameter_name: string        
        :return: Parameter type value 
        :rtype: int
        
        .. seealso:: E_TYPES_T        
        """
        try:
            message = ctypes.create_string_buffer(256)
            parameter_type = ctypes.c_int()
            self.dll.Imop_CameraSet_GetParameterType(
                message,
                self.cameraset,
                ctypes.c_char_p(parameter_name.encode('utf-8')),
                ctypes.byref(parameter_type)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return parameter_type.value
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_type',exception)

    def get_parameter_option(
        self,
        parameter_name
        ):
        """Get Camera parameter options corresponding to parameter_name.
        
        :param parameter_name: Parameter Name 
        :type parameter_name: string
        :return: Parameter options : (has_default_value, has_limits, is_connection_parameter, is_gettable, is_gettable_onfly, is_settable, is_settable_onfly, is_string_enum)
        :rtype: ParameterOption_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            is_settable = ctypes.c_bool()
            is_gettable = ctypes.c_bool()
            has_default_value = ctypes.c_bool()
            has_limits = ctypes.c_bool()
            is_string_enum = ctypes.c_bool()
            is_settable_on_fly = ctypes.c_bool()
            is_gettable_on_fly = ctypes.c_bool()
            is_connection_parameter = ctypes.c_bool()
            self.dll.Imop_CameraSet_GetParameterOption(
                message,
                self.cameraset,
                ctypes.c_char_p(parameter_name.encode('utf-8')),
                ctypes.byref(is_settable),
                ctypes.byref(is_gettable),
                ctypes.byref(has_default_value),
                ctypes.byref(has_limits),
                ctypes.byref(is_string_enum),
                ctypes.byref(is_settable_on_fly),
                ctypes.byref(is_gettable_on_fly),
                ctypes.byref(is_connection_parameter)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.ParameterOption_t(
                has_default_value.value,
                has_limits.value,
                is_connection_parameter.value,
                is_gettable.value,
                is_gettable_on_fly.value,
                is_settable.value,
                is_settable_on_fly.value,
                is_string_enum.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_option',exception)

    def get_parameter_limits(
        self,
        parameter_name
        ):
        """Get Camera parameter limits.
        
        :param parameter_name: Parameter Name 
        :type parameter_name: string        
        :return: min, max
        :rtype: tuple(int or double, int or double)
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.INT) :
                min_out = ctypes.c_int()
                max_out = ctypes.c_int()
                self.dll.Imop_CameraSet_GetParameterIntLimits(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(min_out),
                    ctypes.byref(max_out)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return (
                    min_out.value,
                    max_out.value
                    )
            if(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.REAL) :
                min_out = ctypes.c_double()
                max_out = ctypes.c_double()
                self.dll.Imop_CameraSet_GetParameterRealLimits(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(min_out),
                    ctypes.byref(max_out)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return (
                    min_out.value,
                    max_out.value
                    )            
            raise Exception('IO_Error','Parameter type must be int or real)')
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_limits',exception)

    def get_available_values_list_size(
        self,
        parameter_name
        ):
        """Get Parameter available values list size.
        
        .. seealso:: CameraSet.get_parameter_option
                
        :param parameter_name: Parameter Name 
        :type parameter_name: string        
        :return: Size of the list
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = ctypes.c_int()
            self.dll.Imop_CameraSet_GetAvailableValuesListSize(
                message,
                self.cameraset,
                ctypes.c_char_p(parameter_name.encode('utf-8')),
                ctypes.byref(size_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_available_values_list_size',exception)

    def get_available_value(
        self,
        parameter_name,
        index
        ):
        """Get Parameter available value at corresponding index.
        
        .. seealso:: CameraSet.get_parameter_option
        
        :param parameter_name: Parameter Name 
        :type parameter_name: string
        :param index: Index 
        :type index: index        
        :return: Value at index
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            value_out = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_GetAvailableValue(
                message,
                self.cameraset,
                ctypes.c_char_p(parameter_name.encode('utf-8')),
                index,
                value_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return value_out.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_available_value',exception)

    def get_parameter_value(
        self,
        parameter_name,
        get_factory_value = False
        ):
        """Get Camera current parameter value.
        
        :param parameter_name: Parameter Name 
        :type parameter_name: string
        :param get_factory_value: True to get the factory value, False to get current value (default : False)
        :type get_factory_value: bool        
        :return: Value 
        :rtype: int or double or bool or string
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.INT) :
                val_out = ctypes.c_int()
                self.dll.Imop_CameraSet_GetParameterIntValue(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(val_out),
                    ctypes.c_bool(get_factory_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return (val_out.value)
            if(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.REAL) :
                val_out = ctypes.c_double()
                self.dll.Imop_CameraSet_GetParameterRealValue(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(val_out),
                    ctypes.c_bool(get_factory_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return (val_out.value)
            if(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.BOOL) :
                val_out = ctypes.c_bool()
                self.dll.Imop_CameraSet_GetParameterBooleanValue(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(val_out),
                    ctypes.c_bool(get_factory_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return (val_out.value)
            if(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.STRING) :
                val_out = ctypes.create_string_buffer(256)
                self.dll.Imop_CameraSet_GetParameterStringValue(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    val_out,
                    ctypes.c_bool(get_factory_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return (val_out.value.decode('utf-8'))            
            raise Exception('IO_Error','Unknown parameter type')
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_value',exception)

    def reset_parameters(self):
        """Reset parameters to factory values.
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_ResetParameters(
                message,
                self.cameraset
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : reset_parameters',exception)

    def set_parameter_value(
        self,
        parameter_name,
        parameter_value
        ):
        """Set Camera parameter value.
        
        :param parameter_name: Parameter Name 
        :type parameter_name: string
        :param parameter_value: Parameter Value 
        :type parameter_value: int or double or bool or string
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.INT) :
                self.dll.Imop_CameraSet_SetParameterIntValue(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_int(parameter_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            elif(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.REAL) :
                self.dll.Imop_CameraSet_SetParameterRealValue(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_double(parameter_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            elif(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.BOOL) :
                self.dll.Imop_CameraSet_SetParameterBooleanValue(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_bool(parameter_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            elif(self.get_parameter_type(parameter_name) == imop_enum.E_TYPES.STRING) :
                self.dll.Imop_CameraSet_SetParameterStringValue(
                    message,
                    self.cameraset,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_char_p(parameter_value.encode('utf-8'))
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            else:
                raise Exception('IO_Error','Unknown parameter type')
        except Exception as exception:
            raise Exception(__name__+' : set_parameter_value',exception)

    def save(
        self,
        save_file_path
        ):
        """Save CameraSet to a file.
        
        :param save_file_path: Absolute path to the file (\*.xml)
        :type save_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_Save(
                message,
                self.cameraset,
                ctypes.c_char_p(save_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : save',exception)

    def load(
        self,
        load_file_path
        ):
        """Load file content to CameraSet object.
        
        :param load_file_path: Absolute path to the file (\*.xml)
        :type load_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_CameraSet_Load(
                message,
                self.cameraset,
                ctypes.c_char_p(load_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : load',exception)
