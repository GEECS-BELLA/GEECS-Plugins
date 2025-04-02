#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_hasoslopes as imop_hslp

class HasoEngine(object):
    """Class HasoEngine
    
    - Constructor from Config File :
        - **config_file_path** - string : Absolute path to Haso configuration file (*.dat)
    """

    def __init_(
        self,
        config_file_path
        ):
        """HasoEngine constructor from a Haso configuration file.
        
        :param config_file_path: Absolute path to Haso configuration file
        :type config_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoEngine_NewFromConfigFile(
                message,
                ctypes.pointer(self.hasoengine),
                ctypes.c_char_p(config_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init',exception)

    def __init__(self,  **kwargs):
        """HasoEngine constructor
        """
        self.hasoengine = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
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
            raise Exception('IO_Error','---CAN NOT CREATE HASOENGINE OBJECT---')

    def __del_obj__(self):
        """HasoEngine destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_HasoEngine_Delete(message, self.hasoengine)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_barycenter_from_image(
        self,
        image
        ):
        """In the event of large tilt (definition of large depends on the sensor used),
        the spots are shifted by a value greater than the micro-lens pitch, and there is a risk of mismatch.
        The process of pairing (determining the right shift to apply) \
        the start sub-pupil with the appropriate micro-lens is called alignment.
        The HasoEngine provides a set of functions which help to complete this step.
        
        During optical centering process, computes the barycenter coordinates in pixels of the visible spots on the Haso sensor.
        The optical system is aligned when these coordinates are close to the alignment_position_pixels (error < tolerance_radius).
        
        .. seealso:: HasoConfig.get_config
        
        .. warning:: The centering device must be mounted on the Haso sensor
        
        :param image: Image object
        :type image: Image
        :return: Barycenter coordinates
        :rtype: float2D
        """
        try:
            message = ctypes.create_string_buffer(256)
            barycenter = imop_struct.float2D(0.0, 0.0)
            self.dll.Imop_HasoEngine_AlignmentDataFromImage(
                message,
                self.hasoengine,
                image.image,
                ctypes.byref(barycenter)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return barycenter
        except Exception as exception:
            raise Exception(__name__+' : get_barycenter_from_image',exception)

    def do_new_lens(self):
        """Reset spots detection. In particular, reinitialize the spots tracking.
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoEngine_DoNewLens(
                message,
                self.hasoengine
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : do_new_lens',exception)

    def get_start_pixel(self):
        """Compute barycenter coordinates of the spot corresponding to the start sub-pupil in image.
        
        :return: Barycenter coordinates in pixels
        :rtype: int2D
        """
        try:
            message = ctypes.create_string_buffer(256)
            start_pixel_out = imop_struct.int2D(0, 0)
            self.dll.Imop_HasoEngine_GetStartPixel(
                message,
                self.hasoengine,
                ctypes.byref(start_pixel_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return start_pixel_out
        except Exception as exception:
            raise Exception(__name__+' : get_start_pixel',exception)

    def get_start_pixel_multipupils(self):
        """Compute barycenter coordinates of the spot corresponding to the start sub-pupil in multipupils image.
        
        :return: Array of barycenter coordinates in pixels (relative to the HasoImage top-left corner) of the found spot corresponding to the start sub-pupil.
        :rtype: uint2D list[]
        """
        try:
            message = ctypes.create_string_buffer(256)
            x_start_out = numpy.zeros(self.get_multipupils_size(), dtype = numpy.uintc)
            y_start_out = numpy.zeros(self.get_multipupils_size(), dtype = numpy.uintc)
            self.dll.Imop_HasoEngine_GetStartPixel_MultiPupils.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS") 
                ]
            self.dll.Imop_HasoEngine_GetStartPixel_MultiPupils(
                message,
                self.hasoengine,
                x_start_out,
                y_start_out
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            pixels_out = []
            for i in range(x_start_out.size):
                pixels_out.append(imop_struct.uint2D(x_start_out[i], y_start_out[i]))            
            return pixels_out
        except Exception as exception:
            raise Exception(__name__+' : get_start_pixel_multipupils',exception)

    def set_algo_type(
        self,
        algo_type
        ):
        """Set Spots detection algorithm type.
        
        :param algo_type: E_SPOTDETECT_T spots detection algorithm type
        :type algo_type: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoEngine_SetAlgoType(
                message,
                self.hasoengine,
                ctypes.c_int(algo_type)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_algo_type',exception)

    def get_algo_type(self):
        """Get Spots detection algorithm type.
        
        :return: E_SPOTDETECT_T spots detection algorithm type
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            algo_type_out = ctypes.c_int()
            self.dll.Imop_HasoEngine_GetAlgoType(
                message,
                self.hasoengine,
                ctypes.byref(algo_type_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return algo_type_out.value          
        except Exception as exception:
            raise Exception(__name__+' : get_algo_type',exception)

    def set_preferences(
        self,
        start_subpupil,
        denoising_strength,
        auto_start
        ):
        """Set HasoSlopes computation parameters.
        
        :param start_subpupil: Coordinates of the first calculated sub-pupil (index coordinates relative to the top-left corner of the pupil)
        :type start_subpupil: uint2D
        :param denoising_strength: Sensitivity of spot detection on summed images (between 0 and 1)
        :type denoising_strength: float
        :param auto_start: Activation of the auto start subpupil detection in case specified start subpupil is not found
        :type auto_start: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            if type(start_subpupil) is not imop_struct.uint2D :
                raise Exception('IO_Error', 'start_subpupil must be an io_wavekit_structure.uint2D class')
            self.dll.Imop_HasoEngine_SetPreferences(
                message,
                self.hasoengine,
                ctypes.byref(start_subpupil),
                ctypes.c_float(denoising_strength),
                ctypes.c_bool(auto_start)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)    
        except Exception as exception:
            raise Exception(__name__+' : set_preferences',exception)

    def set_spot_tracker(
        self,
        enable_spot_tracker
        ):
        """HasoEngine spot tracker activation.
        
        :param enable_spot_tracker: If True, activates the spot tracker feature that allow absolute tilt measure
        :type enable_spot_tracker: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoEngine_SetSpotTrackerEnabled(
                message,
                self.hasoengine,
                ctypes.c_bool(enable_spot_tracker)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)    
        except Exception as exception:
            raise Exception(__name__+' : set_spot_tracker',exception)

    def get_spot_tracker(self):
        """Get HasoEngine spot tracker activation status.
        
        :return: If True, The spot tracker feature that allow absolute tilt measure is enabled
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            spot_tracker_out = ctypes.c_bool()
            self.dll.Imop_HasoEngine_GetSpotTrackerEnabled(
                message,
                self.hasoengine,
                ctypes.byref(spot_tracker_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return spot_tracker_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_spot_tracker',exception)
            
            
            
    def set_lift_option(
        self,
        enable_lift_option,
        source_wavelength_nm
        ):
        """HasoEngine lift algorithm activation.
        
        :param enable_lift_option: If True, activates the lift algorithm to get slope
        :type enable_lift_option: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoEngine_SetLiftEnabled(
                message,
                self.hasoengine,
                ctypes.c_bool(enable_lift_option),
                ctypes.c_float(source_wavelength_nm)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)    
        except Exception as exception:
            raise Exception(__name__+' : set_lift_option',exception)

    def get_lift_option(self):
        """Get HasoEngine lift option activation status.
        
        :return: If True, The lift option feature is enabled
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            lift_option_out = ctypes.c_bool()
            self.dll.Imop_HasoEngine_GetLiftEnabled(
                message,
                self.hasoengine,
                ctypes.byref(lift_option_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return lift_option_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_lift_option',exception)

    def get_spot_tracker_quality(self):
        """Get HasoEngine spot tracker indicators.
        
        :return: spot_tracker_quality, spot_tracker_quality_threshold
        :rtype: tuple(float, float)
        """
        try:
            message = ctypes.create_string_buffer(256)
            spot_tracker_quality = ctypes.c_float()
            spot_tracker_quality_threshold = ctypes.c_float()
            self.dll.Imop_HasoEngine_GetSpotTrackerQuality(
                message,
                self.hasoengine,
                ctypes.byref(spot_tracker_quality),
                ctypes.byref(spot_tracker_quality_threshold)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                spot_tracker_quality,
                spot_tracker_quality_threshold
                )
        except Exception as exception:
            raise Exception(__name__+' : get_spot_tracker_quality',exception)

    def set_preferences_multipupils(
        self,
        start_subpupil_list,
        denoising_strength,
        auto_start
        ):
        """Set HasoSlopes computation parameters for multipupils image.
        
        :param start_subpupil_list: Array of coordinate of the first calculated sub-pupil (index coordinates relative to the top-left corner of the pupil)
        :type start_subpupil_list: uint2D list[]
        :param denoising_strength: Sensitivity of spot detection on summed images (between 0 and 1)
        :type denoising_strength: float
        :param auto_start: Activation of the auto start subpupil detection in case specified start subpupil is not found
        :type auto_start: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            x_start_out = numpy.zeros((len(start_subpupil_list)), dtype = numpy.uintc)
            y_start_out = numpy.zeros((len(start_subpupil_list)), dtype = numpy.uintc)
            for i in range(len(start_subpupil_list)):
                x_start_out[i] = start_subpupil_list[i].X
                y_start_out[i] = start_subpupil_list[i].Y
            self.dll.Imop_HasoEngine_SetPreferences_MultiPupils.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),  
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_bool
                ]
            self.dll.Imop_HasoEngine_SetPreferences_MultiPupils(
                message,
                self.hasoengine,
                x_start_out,
                y_start_out,
                ctypes.c_int(x_start_out.size),
                ctypes.c_float(denoising_strength),
                ctypes.c_bool(auto_start)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_preferences_multipupils',exception)

    def get_preferences(self):
        """Get HasoSlopes computation parameters.
        
        :return: Coordinates of the first calculated sub-pupil (index coordinates relative to the top-left corner of the pupil), Sensitivity of spot detection on summed images (between 0 and 1), Activation of the auto start subpupil detection in case specified start subpupil is not found
        :rtype: uint2D, float, bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            start_subpupil_out = imop_struct.uint2D(0, 0)
            denoising_strength_out = ctypes.c_float()
            auto_start_out = ctypes.c_bool()            
            self.dll.Imop_HasoEngine_GetPreferences(
                message,
                self.hasoengine,
                ctypes.byref(start_subpupil_out),
                ctypes.byref(denoising_strength_out),
                ctypes.byref(auto_start_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                start_subpupil_out,
                denoising_strength_out.value,
                auto_start_out.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_preferences',exception)

    def get_preferences_multipupils(self):
        """Get HasoSlopes computation parameters.
        
        :return: Array of coordinates of the first calculated sub-pupil, Sensitivity of spot detection on summed images (between 0 and 1), Activation of the auto start subpupil detection in case specified start subpupil is not found
        :rtype: uint2D list[], float, bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            x_start_out = numpy.zeros(self.get_multipupils_size(), dtype = numpy.uintc)
            y_start_out = numpy.zeros(self.get_multipupils_size(), dtype = numpy.uintc)
            denoising_strength_out = ctypes.c_float()
            auto_start_out = ctypes.c_bool()
            self.dll.Imop_HasoEngine_GetPreferences_MultiPupils.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),  
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS"),
                ctypes.c_void_p,
                ctypes.c_void_p
                ]
            self.dll.Imop_HasoEngine_GetPreferences_MultiPupils(
                message,
                self.hasoengine,
                x_start_out,
                y_start_out,
                ctypes.byref(denoising_strength_out),
                ctypes.byref(auto_start_out)               
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            pixels_out = []
            for i in range(x_start_out.size):
                pixels_out.append(imop_struct.uint2D(x_start_out[i], y_start_out[i])) 
            return (
                pixels_out,
                denoising_strength_out.value,
                auto_start_out.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_preferences_multipupils',exception)

    def get_multipupils_size(self):
        """Get HasoSlopes computation multipupil size.
        
        :return:  Size of the array of start subpupils
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = ctypes.c_int()            
            self.dll.Imop_HasoEngine_GetMultiPupilsSize(
                message,
                self.hasoengine,
                ctypes.byref(size_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return size_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_multipupils_size',exception)

    def compute_slopes(
        self,
        image,
        learn_from_trimmer,
        check_calibration_checksum = False
        ):
        """Compute HasoSlopes.
        Please check that parameters have been set, and alignment step is completed.
        
        :param image: Input image object
        :type image: Image
        :param learn_from_trimmer: If True, a possible mis-alignement is corrected by correlating the observed high-frequency aberrations.
        :type learn_from_trimmer: uchar 
        :param check_calibration_checksum: 
        :type check_calibration_checksum: bool
        :return: If True, a possible mis-alignement is corrected by correlating the observed high-frequency aberrations with a pre-calibrated map (trimmer), Computed HasoSlopes
        :rtype: tuple(float, HasoSlopes)
        """
        try:
            message = ctypes.create_string_buffer(256)
            hasoslopes_out = imop_hslp.HasoSlopes(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1, 1), 
                    imop_struct.float2D(1.0, 1.0)
                    ), 
                serial_number = image.get_haso_serial_number()
                )
            trimmer_quality_out = ctypes.c_float()
            self.dll.Imop_HasoEngine_ComputeSlopes(
                message,
                self.hasoengine,
                hasoslopes_out.hasoslopes,
                image.image,
                ctypes.c_ubyte(learn_from_trimmer),
                ctypes.byref(trimmer_quality_out),
                ctypes.c_bool(check_calibration_checksum)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                trimmer_quality_out.value,
                hasoslopes_out
                )
        except Exception as exception:
            raise Exception(__name__+' : compute_slopes',exception)
            
    def get_is_lift_available(
        self
        ):
        """HasoEngine lift availability.
        
        :return: If True, the lift algorithm can be activated.
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            bool_out = ctypes.c_bool
            self.dll.Imop_HasoEngine_GetIsLiftAvailable(
                message, 
                self.hasoengine,
                ctypes.byref(bool_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return bool_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_is_lift_available',exception)
            
    def set_lift_enabled(
        self,
        enable_lift,
        source_wavelength_nm
        ):
        """HasoEngine lift activation.
        
        :param enable_lift: If True, activates the lift algorithm.
        :type enable_lift: bool
        :param source_wavelength_nm: source wavelength (in nm).
        :type source_wavelength_nm: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoEngine_SetLiftEnabled(
                message, 
                self.hasoengine,
                ctypes.c_bool(enable_lift),
                ctypes.c_float(source_wavelength_nm)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_lift_enabled',exception)
            
    def get_is_lift_available(
        self
        ):
        """HasoEngine lift activation status.
        
        :return: Set to true if The lift algorithm is enabled.
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            bool_out = ctypes.c_bool
            self.dll.Imop_HasoEngine_GetLiftEnabled(
                message, 
                self.hasoengine,
                ctypes.byref(bool_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return bool_out.value
        except Exception as exception:
            raise Exception(__name__+' : get_is_lift_available',exception)