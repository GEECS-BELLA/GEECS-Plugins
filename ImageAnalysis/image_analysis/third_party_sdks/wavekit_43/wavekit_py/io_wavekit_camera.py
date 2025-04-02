#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_wavekit_structure as imop_struct
import io_wavekit_enum as imop_enum
import io_wavekit_cameraset as imop_camset
import io_wavekit_image as imop_image

import io_thirdparty_load_library as imop_library

class Camera(object):
    """Class Camera
     
    - Constructor from Config File :    
        - **config_file_path** - string : Path to config file (\*.dat)
    """
    
    def __init_(
        self,
        config_file_path
        ):
        """Camera constructor
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_NewInstanceFromConfigFile(
                message,
                ctypes.pointer(self.camera),
                ctypes.c_char_p(config_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__ + " : init",exception)

    def __init__(self,  **kwargs):
        """Camera constructor
        """
        self.camera = ctypes.c_void_p()
        self.dll    = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1):
                if('config_file_path' in kwargs):
                    entered = 1
                    self.__init_(kwargs['config_file_path'])
        except Exception as exception:
            raise Exception(__name__+' : init',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE CAMERA OBJECT---')

    def __del_obj__(self):
        """Camera Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_Camera_Delete(message, self.camera)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def connect(self):
        """Establish connection to the device
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_Connect(
                message,
                self.camera
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__ + " : connect",exception)

    def disconnect(self):
        """Close connection to the device
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_Disconnect(
                message,
                self.camera
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__ + " : disconnect",exception)

    def snap_processed_image(
        self
        ):
        """Snap Image using new acquisition mode and synchronous synchronization mode.
        There is no need of Starting and Stopping the camera manually with this method.

        .. warning:: Be sure that you have set a background before using this method
        
        :return: Processed image
        :rtype: Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            img_size, img_depth = self.get_size()
            image_out = imop_image.Image(
                size = img_size,
                bit_depth = img_depth
                )
            raw_image_out = imop_image.Image(
                size = img_size,
                bit_depth = img_depth
                )
            self.dll.Imop_Camera_SnapImage(
                message,
                self.camera,
                image_out.image,
                ctypes.c_ubyte(True),
                raw_image_out.image
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return image_out
        except Exception as exception:
            raise Exception(__name__ + " : snap_processed_image",exception)

    def snap_raw_image(
        self
        ):
        """Snap Image using new acquisition mode and synchronous synchronization mode.
        There is no need of Starting and Stopping the camera manually with this method.
        
        :return: Raw image
        :rtype: Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            img_size, img_depth = self.get_size()
            image_out = imop_image.Image(
                size = img_size,
                bit_depth = img_depth
                )
            raw_image_out = imop_image.Image(
                size = img_size,
                bit_depth = img_depth
                )
            self.dll.Imop_Camera_SnapImage(
                message,
                self.camera,
                image_out.image,
                ctypes.c_ubyte(False),
                raw_image_out.image
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return raw_image_out
        except Exception as exception:
            raise Exception(__name__ + " : snap_raw_image",exception)

    def start(
        self,
        acquisition_mode,
        synchronization_mode
        ):
        """Start acquisition.
        
        :param acquisition_mode: Acquisition mode : can be 0 (last_image) or 1 (new_image).
        :type acquisition_mode: E_CAMERA_ACQUISITION_MODE
        :param synchronization_mode: Synchronisation mode : can be 0 (synchronous) or 1 (asynchronous).
        :type synchronization_mode: E_CAMERA_SYNCHRONIZATION_MODE
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_Start(
                message,
                self.camera,
                ctypes.c_ubyte(acquisition_mode),
                ctypes.c_ubyte(synchronization_mode),
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__ + " : start",exception)

    def stop(self):
        """Stop acquisition.
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_Stop(
                message,
                self.camera
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__ + " : stop",exception)

    def get_processed_image(
        self
        ):
        """Retrieve last captured image.
        
        .. seealso:: Camera.async_image_ready

        .. warning:: Be sure that you have set a background before using this method
        
        :return: Processed captured image
        :rtype: Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            img_size, img_depth = self.get_size()
            image_out = imop_image.Image(
                size = img_size,
                bit_depth = img_depth
                )
            raw_image_out = imop_image.Image(
                size = img_size,
                bit_depth = img_depth
                )
            self.dll.Imop_Camera_GetImage(
                message,
                self.camera,
                image_out.image,
                ctypes.c_ubyte(True),
                raw_image_out.image
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return image_out
        except Exception as exception:
            raise Exception(__name__ + " : get_processed_image",exception)

    def get_raw_image(
        self
        ):
        """Retrieve last captured image.
        
        .. seealso:: Camera.async_image_ready
        
        :return: Raw image
        :rtype: Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            img_size, img_depth = self.get_size()
            image_out = imop_image.Image(
                size = img_size,
                bit_depth = img_depth
                )
            raw_image_out = imop_image.Image(
                size = img_size,
                bit_depth = img_depth
                )
            self.dll.Imop_Camera_GetImage(
                message,
                self.camera,
                image_out.image,
                ctypes.c_ubyte(False),
                raw_image_out.image
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return raw_image_out
        except Exception as exception:
            raise Exception(__name__+' : get_raw_image',exception)

    def get_processed_sequence(
        self,
        image_array_size
        ):
        """Capture an image_array_size images array (Synchronous mode - Blocking).

        .. warning:: Be sure that you have set a background before using this method
        
        :param image_array_size: Sequence length (number of images).
        :type image_array_size: ushort
        :return: Processed captured images sequence
        :rtype: Image list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            img_size, img_depth = self.get_size()
            image_list = []
            image_array = (ctypes.c_void_p * image_array_size)()
            for i in range(image_array_size):
                image_list.append(imop_image.Image(size = img_size, bit_depth = img_depth))
                image_array[i] = image_list[i].image
            self.dll.Imop_Camera_GetSequence(
                message,
                self.camera,
                ctypes.c_ushort(image_array_size),
                ctypes.pointer(image_array),
                ctypes.c_ubyte(True)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return image_list
        except Exception as exception:
            raise Exception(__name__+' : get_processed_sequence',exception)

    def get_raw_sequence(
        self,
        image_array_size
        ):
        """Capture an image_array_size images array (Synchronous mode - Blocking).
        
        :param image_array_size: Sequence length (number of images).
        :type image_array_size: ushort
        :return: Captured images sequence
        :rtype: Image list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            img_size, img_depth = self.get_size()
            image_list = []
            image_array = (ctypes.c_void_p * image_array_size)()
            for i in range(image_array_size):
                image_list.append(imop_image.Image(size = img_size, bit_depth = img_depth))
                image_array[i] = image_list[i].image
            self.dll.Imop_Camera_GetSequence(
                message,
                self.camera,
                ctypes.c_ushort(image_array_size),
                ctypes.pointer(image_array),
                ctypes.c_ubyte(False)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return image_list
        except Exception as exception:
            raise Exception(__name__+' : get_raw_sequence',exception)

    def async_image_ready(self):
        """Indicate if a new image is available.
        
        :return: True if image is ready
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            is_ready = ctypes.c_ubyte()
            self.dll.Imop_Camera_AsyncImageReady(
                message,
                self.camera,
                ctypes.byref(is_ready)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return bool(is_ready)
        except Exception as exception:
            raise Exception(__name__+' : async_image_ready',exception)

    def get_state(self):
        """Get camera state : 
        connected or not, 
        running or not, 
        acquisition mode, 
        synchronisation mode.
        
        :return: Is camera connected, Is camera running, Camera acquisition mode, Camera synchronization mode
        :rtype: tuple(bool, bool, E_CAMERA_ACQUISITION_MODE, E_CAMERA_SYNCHRO_MODE)
        """
        try:
            message = ctypes.create_string_buffer(256)
            is_connected         = ctypes.c_ubyte()
            is_running           = ctypes.c_ubyte()
            acquisition_mode     = ctypes.c_ubyte()
            synchronization_mode = ctypes.c_ubyte()
            self.dll.Imop_Camera_GetState(
                message,
                self.camera,
                ctypes.byref(is_connected),
                ctypes.byref(is_running),
                ctypes.byref(acquisition_mode),
                ctypes.byref(synchronization_mode)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                bool(is_connected),
                bool(is_running),
                acquisition_mode.value,
                synchronization_mode.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_state',exception)

    def call_specific_feature(
        self,
        feature_name
        ):
        """Perform an action corresponding to feature_name.
        
        :param feature_name: Name of the feature.
        :type feature_name: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_CallSpecificFeature(
                message,
                self.camera,
                ctypes.c_char_p(feature_name.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : call_specific_feature',exception)

    def is_haso_sensor(self):
        """Get Camera type
        
        :return: True if Camera is an Haso sensor
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            is_haso = ctypes.c_bool()
            self.dll.Imop_Camera_IsHasoSensor(
                message,
                self.camera,
                ctypes.byref(is_haso)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return is_haso.value
        except Exception as exception:
            raise Exception(__name__+' : is_haso_sensor',exception)

    def get_size(self):
        """Get Camera dimensions and depth.
        
        :return: Size, Bit depth
        :rtype: tuple(uint2D, ushort)
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = imop_struct.uint2D(0,0)
            bit_depth = ctypes.c_ushort(0)
            self.dll.Imop_Camera_GetSize(
                message,
                self.camera,
                ctypes.byref(size),
                ctypes.byref(bit_depth)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                size,
                bit_depth.value
                )
        except Exception as exception:
            raise Exception(__name__ + " : get_size",exception)

    def set_background(
        self,
        background_image
        ):
        """Set Camera background image.
        Image with substracted background can be retrieved using
        the get_processed_image, get_processed_sequence or snap_processed_image functions
        
        :param background_image: Background image
        :type background_image: Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_SetBackground(
                message,
                self.camera,
                background_image.image
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_background',exception)

    def get_background(self):
        """Get Camera background image.
        
        :return: Background image as Image object
        :rtype: Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            img_size, img_depth = self.get_size()
            bg_image_out = imop_image.Image(
                size = img_size,
                bit_depth = img_depth
                )
            self.dll.Imop_Camera_GetBackground(
                message,
                self.camera,
                bg_image_out.image
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return bg_image_out
        except Exception as exception:
            raise Exception(__name__+' : get_background',exception)

    def set_nb_images_to_sum(
        self,
        images_count
        ):
        """Set number of images to sum
        Summing images makes the measured wavefront less sensitive to noise (temporal average effect).
        
        :param image_count: Images count.
        :type image_count: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_SetNbImagesToSum(
                message,
                self.camera,
                ctypes.c_uint(images_count)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_nb_images_to_sum',exception)

    def get_nb_images_to_sum(self):
        """Get number of images to sum.
        Summing images makes the measured wavefront less sensitive to noise  (temporal average effect).
        
        :return: Number of images to sum
        :rtype: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            images_count_out = ctypes.c_uint()
            self.dll.Imop_Camera_GetNbImagesToSum(
                message,
                self.camera,
                ctypes.byref(images_count_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return images_count_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_nb_images_to_sum',exception)

    def set_timeout(
        self,
        timeout
        ):
        """Set Camera time out value.
        
        :param timeout:  Time out value (ms).
        :type timeout: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_SetTimeOut(
                message,
                self.camera,
                ctypes.c_uint(timeout)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_timeout',exception)

    def get_timeout(self):
        """Get Camera timeout value.
        
        :return: Timeout value
        :rtype: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            timeout_out = ctypes.c_uint()
            self.dll.Imop_Camera_GetTimeOut(
                message,
                self.camera,
                ctypes.byref(timeout_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return timeout_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_timeout',exception)

    def get_parameter_value(
        self,
        parameter_name
        ):
        """Low level camera parameters getters.
        Get camera parameter value corresponding to parameter_name.
        
        :param parameter_name: Parameter name
        :type parameter_name: string
        :return: Parameter value 
        :rtype: int or double or bool or string
        """
        try:
            message = ctypes.create_string_buffer(256)
            cameraset = imop_camset.CameraSet(camera = self)
            type_ = cameraset.get_parameter_type(parameter_name)
            if(type_ == imop_enum.E_TYPES.INT) :
                int_out = ctypes.c_int()
                self.dll.Imop_Camera_GetParameterInt(
                    message,
                    self.camera,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(int_out)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return int_out.value
            if(type_ == imop_enum.E_TYPES.REAL) :
                float_out = ctypes.c_float()
                self.dll.Imop_Camera_GetParameterReal(
                    message,
                    self.camera,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(float_out)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return float_out.value
            if(type_ == imop_enum.E_TYPES.BOOL) :
                bool_out = ctypes.c_bool()
                self.dll.Imop_Camera_GetParameterBoolean(
                    message,
                    self.camera,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(bool_out)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return bool_out.value
            if(type_ == imop_enum.E_TYPES.STRING) :
                string_out = ctypes.create_string_buffer(256)
                self.dll.Imop_Camera_GetParameterString(
                    message,
                    self.camera,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.byref(string_out)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return string_out.value.decode('utf-8')
            raise Exception('IO_Error','Parameter not found')
        except Exception as exception:
            raise Exception(__name__+' : get_parameter_value',exception)

    def diff_camera_set(
        self,
        cameraset
        ):
        """Check how many parameters differ between Camera parameters and CameraSet given as argument.
        
        :param cameraset: CameraSet Object to compare
        :type cameraset: CameraSet
        :return:  number of differents settable parameters, \
        number of differents onfly settable parameters, \
        number of differents settable connection parameters. 
        :rtype: tuple(int, int, int)
        """
        try:
            message = ctypes.create_string_buffer(256)
            nb_nonequal_settable_params_out = ctypes.c_int()
            nb_nonequals_onfly_settable_params_out = ctypes.c_int()
            nb_nonequals_settable_connection_params_out = ctypes.c_int()
            self.dll.Imop_Camera_DiffCameraSet(
                message,
                self.camera,
                cameraset.cameraset,
                ctypes.byref(nb_nonequal_settable_params_out),
                ctypes.byref(nb_nonequals_onfly_settable_params_out),
                ctypes.byref(nb_nonequals_settable_connection_params_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                nb_nonequal_settable_params_out.value,
                nb_nonequals_onfly_settable_params_out.value,
                nb_nonequals_settable_connection_params_out.value
                )
        except Exception as exception:
            raise Exception(__name__+' : diff_camera_set',exception)

    def reset_parameters(
        self,
        reset_connection_params,
        reset_params
        ):
        """Reset Camera parameters to factory values.
        
        .. warning:: Be sure to disconnect Camera first
        
        :param reset_connection_params: Try to reset connection parameters if True.
        :type reset_connection_params: bool
        :param reset_params: Try to reset parameters if True.
        :type reset_params: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_ResetParameters(
                message,
                self.camera,
                ctypes.c_bool(reset_connection_params),
                ctypes.c_bool(reset_params)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : reset_parameters',exception)

    def set_parameters_from_cameraset(
        self,
        cameraset,
        load_and_apply_connection_params,
        load_and_apply_params
        ):
        """Load and apply camera parameters from a CameraSet object
        If camera is started, only on fly settable parameters will be applied.
        
        :param cameraset: CameraSet Object
        :type cameraset: CameraSet
        :param load_and_apply_connection_params: Try to apply connection parameters if True
        :type load_and_apply_connection_params: bool
        :param load_and_apply_params: Try to apply parameters if True
        :type load_and_apply_params: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Camera_SetCameraSet(
                message,
                self.camera,
                cameraset.cameraset,
                ctypes.c_bool(load_and_apply_connection_params),
                ctypes.c_bool(load_and_apply_params)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_from_cameraset',exception)

    def set_parameter_value(
        self,
        parameter_name,
        parameter_value
        ):
        """Camera parameter setter.
        
        :param parameter_name: Parameter name
        :type parameter_name: string
        :param parameter_value: Parameter value
        :type parameter_value: int or double or bool or string corresponding to the parameter name
        """
        try:
            message = ctypes.create_string_buffer(256)
            cameraset = imop_camset.CameraSet(camera = self)
            type_ = cameraset.get_parameter_type(parameter_name)
            if(type_ == imop_enum.E_TYPES.INT) :
                self.dll.Imop_Camera_SetParameterInt(
                    message,
                    self.camera,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_int(parameter_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return
            if(type_ == imop_enum.E_TYPES.REAL) :
                self.dll.Imop_Camera_SetParameterReal(
                    message,
                    self.camera,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_float(parameter_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return
            if(type_ == imop_enum.E_TYPES.BOOL) :
                self.dll.Imop_Camera_SetParameterBoolean(
                    message,
                    self.camera,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_bool(parameter_value)
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return
            if(type_ == imop_enum.E_TYPES.STRING) :
                self.dll.Imop_Camera_SetParameterString(
                    message,
                    self.camera,
                    ctypes.c_char_p(parameter_name.encode('utf-8')),
                    ctypes.c_char_p(parameter_value.encode('utf-8'))
                    )
                if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
                return
            raise Exception('IO_Error','Unknown parameter type')
        except Exception as exception:
            raise Exception(__name__+' : set_parameter_value',exception)
