#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct

class HasoSlopes(object):
    """Class HasoSlopes
    
    - Constructor from Dimensions : 
        - **dimensions** - dimensions : Dimensions structure containing subpupil number and steps between two subpupil
        - **serial_number** - string : Haso Serial number
    
    - Constructor from Has File : 
        - **has_file_path** - string : Absolute path to (\*.has) file
    
    - Constructor from Copy : 
        - **hasoslopes** - HasoSlopes : HasoSlopes object to copy
    
    - Constructor from Config File : 
        - **config_file_path** - string : Absolute path to Haso configuration file (\*.dat)
    
    - Constructor from HasoData : 
        - **hasodata** - HasoData : HasoData object
        - **apply_processing** - bool : If true, created HasoSlopes will contain the processed slopes of the HasoData, else HasoSlopes will contain the raw slopes of the HasoData
    
    - Constructor from Image : 
        - **image** - Image : Image object
        - **config_file_path** - string : Absolute path to Haso configuration file (\*.dat)
        - **start_subpupil** - uint2D : Coordinates of the first calculated sub-pupil (index coordinates relative to the top-left corner of the pupil)
        - **check_calibration_checksum** - bool : Activation of the calibration checksum verification
    
    - Constructor from Multipupils Image : 
        - **image** - Image : Image object
        - **config_file_path** - string : Absolute path to Haso configuration file (\*.dat)
        - **start_subpupil_x_array** - uint list[] : x Array of coordinates of the first calculated sub-pupil (index coordinates relative to the top-left corner of the pupil)
        - **start_subpupil_y_array** - uint list[] : y Array of coordinates of the first calculated sub-pupil (index coordinates relative to the top-left corner of the pupil)
        - **check_calibration_checksum** - bool : Activation of the calibration checksum verification
    
    - Constructor from ModalCoef : 
        - **modalcoef** - ModalCoef : ModalCoef object
        - **config_file_path** - string : Absolute path to Haso configuration file
    """

    def __init_from_dimensions(
        self,
        dimensions,
        serial_number
        ):
        """HasoSlopes constructor from dimensions and steps
        
        :param dimensions: subpupils number of the HasoSlopes, step of subpupils
        :type dimensions: dimensions
        :param serial_number: Serial number
        :type serial_number: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(type(dimensions)!=imop_struct.dimensions):
                raise Exception('IO_Error', 'dimensions must be an io_wavekit_structure.dimensions class')
            self.dll.Imop_HasoSlopes_NewFromDimensions(
                message,
                ctypes.pointer(self.hasoslopes),
                ctypes.byref(dimensions.size),
                ctypes.byref(dimensions.steps),
                ctypes.c_char_p(serial_number.encode('utf-8'))                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_dimensions',exception)

    def __init_from_has_file(
        self,
        has_file_path
        ):
        """HasoSlopes constructor from (*.has) file
        
        :param has_file_path: Absolute path to (*.has) file
        :type has_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoSlopes_NewFromHasFile(
                message,
                ctypes.pointer(self.hasoslopes),
                ctypes.c_char_p(has_file_path.encode('utf-8'))                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_has_file',exception)

    def __init_from_copy(
        self,
        hasoslopes
        ):
        """HasoSlopes constructor from copy
        
        :param hasoslopes: HasoSlopes object to copy
        :type hasoslopes: HasoSlopes
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoSlopes_NewFromCopy(
                message,
                ctypes.pointer(self.hasoslopes),
                hasoslopes.hasoslopes                
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_copy',exception)

    def __init_from_config_file(
        self,
        config_file_path
        ):
        """HasoSlopes constructor from Haso configuration file
        
        :param config_file_path: Absolute path to Haso configuration file
        :type config_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoSlopes_NewFromConfigFile(
                message,
                ctypes.pointer(self.hasoslopes),
                ctypes.c_char_p(config_file_path.encode('utf-8'))               
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_config_file',exception)

    def __init_from_hasodata(
        self,
        hasodata,
        apply_processing
        ):
        """HasoSlopes constructor from Haso configuration file
        
        :param hasodata: HasoData object
        :type hasodata: HasoData
        :param apply_processing: If true, created HasoSlopes will contain the processed slopes of the HasoData, else HasoSlopes will contain the raw slopes of the HasoData
        :type apply_processing: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoSlopes_NewFromHasoData(
                message,
                ctypes.pointer(self.hasoslopes),
                hasodata.hasodata,
                ctypes.c_bool(apply_processing)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_hasodata',exception)

    def __init_from_image(
        self,
        image,
        config_file_path,
        start_subpupil = None,
        check_calibration_checksum = False
        ):
        """*HasoSlopes* constructor from *Image*
        This constructor computes the slopes on the input Image object
        and creates the corresponding HasoSlopes object.
        The slopes computation core (HasoEngine) parameters are all set to their default values
        and cannot be modified, excepting the start sub-pupil coordinates.
        If no start sub-pupil coordinates are provided, start sub pupil is automatically detected.
        
        :param image: Image object
        :type image: Image
        :param config_file_path: Absolute path to Haso configuration file
        :type config_file_path: string
        :param start_subpupil: Coordinates of the first calculated sub-pupil (index coordinates relative to the top-left corner of the pupil)
        :type start_subpupil: uint2D
        :param check_calibration_checksum: activation of the calibration checksum verification
        :type check_calibration_checksum: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            if start_subpupil is None:
                self.dll.Imop_HasoSlopes_NewFromImage(
                    message,
                    ctypes.pointer(self.hasoslopes),
                    image.image,
                    ctypes.c_char_p(config_file_path.encode('utf-8')),
                    None,
                    ctypes.c_bool(check_calibration_checksum)
                    )
            else :
                if type(start_subpupil) is not imop_struct.uint2D:
                    raise Exception('IO_Error', 'start_subpupil must be an io_wavekit_structure.uint2D class')
                self.dll.Imop_HasoSlopes_NewFromImage(
                    message,
                    ctypes.pointer(self.hasoslopes),
                    image.image,
                    ctypes.c_char_p(config_file_path.encode('utf-8')),
                    ctypes.byref(start_subpupil),
                    ctypes.c_bool(check_calibration_checksum)
                    )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_image',exception)

    def __init_from_multipupils_image(
        self,
        image,
        config_file_path,
        start_subpupil_x_array,
        start_subpupil_y_array,
        check_calibration_checksum = False
        ):
        """*HasoSlopes* constructor from Multipupils *Image*
        This constructor computes the slopes on the input Image object
        and creates the corresponding HasoSlopes object.
        The slopes computation core (HasoEngine) parameters are all set to their default values
        and cannot be modified, excepting the start sub-pupil coordinates.
        
        :param image: Image object
        :type image: Image
        :param config_file_path: Absolute path to Haso configuration file
        :type config_file_path: string
        :param start_subpupil_x_array: x Array of coordinates of the first calculated sub-pupil (index coordinates relative to the top-left corner of the pupil)
        :type start_subpupil_x_array: uint list[]
        :param start_subpupil_y_array: y Array of coordinates of the first calculated sub-pupil (index coordinates relative to the top-left corner of the pupil)
        :type start_subpupil_y_array: uint list[]
        :param check_calibration_checksum: activation of the calibration checksum verification
        :type check_calibration_checksum: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(len(start_subpupil_x_array) != len(start_subpupil_y_array)):
                raise Exception('IO_Error','start_subpupil_x_array and start_subpupil_y_array cannot have different size')
            start_subpupil_array_size = len(start_subpupil_x_array)
            self.dll.Imop_HasoSlopes_NewFromMultiPupilsImage.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                ctypes.c_void_p, 
                ctypes.c_char_p, 
                numpy.ctypeslib.ndpointer(numpy.uint, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.uint, flags="C_CONTIGUOUS"),    
                ctypes.c_int,
                ctypes.c_bool                
                ] 
            self.dll.Imop_HasoSlopes_NewFromMultiPupilsImage(
                message,
                ctypes.pointer(self.hasoslopes),
                image.image,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                numpy.array(start_subpupil_x_array, dtype = numpy.uint),
                numpy.array(start_subpupil_y_array, dtype = numpy.uint),
                ctypes.c_int(start_subpupil_array_size),
                ctypes.c_bool(check_calibration_checksum)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_multipupils_image',exception)

    def __init_from_modalcoef(
        self,
        modalcoef,
        config_file_path
        ):
        """HasoSlopes constructor from ModalCoef
        This constructor computes the slopes on the input modalcoef
        using the type of reconstruction and projection pupil defined in it (Legendre or Zernike)
        and creates the corresponding HasoSlopes object.
        Modal coeffs must have a projection pupil set in their preferences.
        
        .. seealso:: Compute.zernike_pupil to fit a circular pupil to a natural slopes pupil for Zernike projection
        
        .. seealso:: Compute.legendre_pupil to fit rectangular pupil to a natural slopes pupil for Legendre projection
        
        .. seealso:: ModalCoef.set_zernike_prefs for setting preferences for a ModalCoef of type Zernike
        
        .. seealso:: ModalCoef.set_legendre_prefs for setting preferences for a ModalCoef of type Legendre
        
        :param modalcoef: ModalCoef object
        :type modalcoef: ModalCoef
        :param config_file_path: Absolute path to Haso configuration file
        :type config_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoSlopes_NewFromModalCoef(
                message,
                ctypes.pointer(self.hasoslopes),
                modalcoef.modalcoef,
                ctypes.c_char_p(config_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_modalcoef',exception)
            
    def __init__(self, **kwargs):
        """HasoSlopes Constructor
        """
        self.hasoslopes = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 1): 
                if('has_file_path' in kwargs):
                    entered = 1
                    self.__init_from_has_file(kwargs['has_file_path'])   
                if('hasoslopes' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['hasoslopes']) 
                if('config_file_path' in kwargs):
                    entered = 1
                    self.__init_from_config_file(kwargs['config_file_path'])              
            if(arg_size == 2):
                if('dimensions' in kwargs
                    and 'serial_number' in kwargs):
                    entered = 1
                    self.__init_from_dimensions(kwargs['dimensions'], kwargs['serial_number']) 
                if('hasodata' in kwargs
                   and 'apply_processing' in kwargs):
                    entered = 1
                    self.__init_from_hasodata(kwargs['hasodata'], kwargs['apply_processing'])   
                if('image' in kwargs
                   and 'config_file_path' in kwargs):
                    entered = 1
                    self.__init_from_image(kwargs['image'], kwargs['config_file_path'])
                if('modalcoef' in kwargs
                   and 'config_file_path' in kwargs):
                    entered = 1
                    self.__init_from_modalcoef(kwargs['modalcoef'], kwargs['config_file_path'])
            if(arg_size == 3):
                if('image' in kwargs
                   and 'config_file_path' in kwargs
                   and 'start_subpupil' in kwargs):
                    entered = 1
                    self.__init_from_image(kwargs['image'], kwargs['config_file_path'], kwargs['start_subpupil'])
            if(arg_size == 4):
                if('image' in kwargs
                   and 'config_file_path' in kwargs
                   and 'start_subpupil' in kwargs
                   and 'check_calibration_checksum' in kwargs):
                    entered = 1
                    self.__init_from_image(kwargs['image'],kwargs['config_file_path'],kwargs['start_subpupil'],kwargs['check_calibration_checksum'])
                if('image' in kwargs
                   and 'config_file_path' in kwargs
                   and 'start_subpupil_x_array' in kwargs
                   and 'start_subpupil_y_array' in kwargs):
                    entered = 1
                    self.__init_from_multipupils_image(kwargs['image'], kwargs['config_file_path'], kwargs['start_subpupil_x_array'], kwargs['start_subpupil_y_array'])
            if(arg_size == 5):
                if('image' in kwargs
                   and 'config_file_path' in kwargs
                   and 'start_subpupil_x_array' in kwargs
                   and 'start_subpupil_y_array' in kwargs
                   and 'check_calibration_checksum' in kwargs):
                    entered = 1
                    self.__init_from_multipupils_image(kwargs['image'], kwargs['config_file_path'], kwargs['start_subpupil_x_array'], kwargs['start_subpupil_y_array'], kwargs['check_calibration_checksum'])
        
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)
        if(entered == 0):            
            raise Exception('IO_Error','---CAN NOT CREATE HASOSLOPES OBJECT---')
    
    def __del_obj__(self):
        """HasoSlopes destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_HasoSlopes_Delete(message, self.hasoslopes)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_info(self):
        """Get HasoSlopes specifications
        
        :return: Serial Number, Pupil dimensions
        :rtype: tuple(string, dimensions)
        """
        try:
            message = ctypes.create_string_buffer(256)
            serial_out = ctypes.create_string_buffer(256)
            size_out = imop_struct.uint2D(0, 0)
            steps_out = imop_struct.float2D(0.0, 0.0)
            self.dll.Imop_HasoSlopes_GetInfo(
                message,
                self.hasoslopes,
                ctypes.byref(serial_out),
                ctypes.byref(size_out),
                ctypes.byref(steps_out)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                serial_out.value.decode('utf-8'), 
                imop_struct.dimensions(
                    size_out, 
                    steps_out
                    )
                )
        except Exception as exception:
            raise Exception(__name__+' : get_info',exception)

    def set_wavelength(
        self, 
        wavelength
        ):
        """Set HasoSlopes wavelength
        
        :param wavelength: Wavelenght value
        :type wavelength: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoSlopes_SetWaveLength(
                message,
                self.hasoslopes,
                ctypes.c_float(wavelength)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : set_wavelength',exception)

    def get_slopes(self):
        """Get HasoSlopes buffer
        
        :return: Slopes X buffer, Slopes Y buffer
        :rtype: tuple(float 2D numpy.array, float 2D numpy.array)
        """
        try:
            message = ctypes.create_string_buffer(256)
            haso_serial, dimensions = self.get_info()
            sx = numpy.zeros((dimensions.size.Y, dimensions.size.X), dtype = numpy.float32)
            sy = numpy.zeros((dimensions.size.Y, dimensions.size.X), dtype = numpy.float32)
            self.dll.Imop_HasoSlopes_GetSlopesX.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_HasoSlopes_GetSlopesX(
                message, 
                self.hasoslopes, 
                sx
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            self.dll.Imop_HasoSlopes_GetSlopesY.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.float32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_HasoSlopes_GetSlopesY(
                message, 
                self.hasoslopes, 
                sy
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                sx, 
                sy
                )
        except Exception as exception:
            raise Exception(__name__+' : get_slopes',exception)

    def get_intensity(self):
        """Get HasoSlopes intensity buffer
        
        :return: Intensity buffer
        :rtype: int 2D numpy.array
        """
        try:
            message = ctypes.create_string_buffer(256)
            haso_serial, dimensions = self.get_info()
            intensity = numpy.zeros((dimensions.size.Y, dimensions.size.X), dtype = numpy.int32)
            self.dll.Imop_HasoSlopes_GetIntensity.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.int32, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_HasoSlopes_GetIntensity(
                message, 
                self.hasoslopes, 
                intensity
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)        
            return intensity
        except Exception as exception:
            raise Exception(__name__+' : get_intensity',exception)

    def get_pupil_buffer(self):
        """Get HasoSlopes pupil buffer
        
        :return: Pupil buffer
        :rtype: bool 2D numpy.array
        """
        try:
            message = ctypes.create_string_buffer(256)
            haso_serial, dimensions = self.get_info()
            pupil = numpy.zeros((dimensions.size.Y, dimensions.size.X), dtype = numpy.bool)
            self.dll.Imop_HasoSlopes_GetPupilBuffer.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                numpy.ctypeslib.ndpointer(numpy.bool, flags="C_CONTIGUOUS")
                ]
            self.dll.Imop_HasoSlopes_GetPupilBuffer(
                message, 
                self.hasoslopes, 
                pupil
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)        
            return pupil
        except Exception as exception:
            raise Exception(__name__+' : get_pupil_buffer',exception)

    @staticmethod
    def get_info_from_file(has_file_path):
        """Retrieve info from a (\*.has) file
        
        :param has_file_path: Absolute path to the (\*.has) file
        :type has_file_path: string
        :return: Acquisition time, is Background removed, comments, Exposure duration (us), Number summed image, is Smearing removed, Trigger Mode
        :rtype: tuple(Metadata_t, AcquisitionInfo_t)
        """
        dll = imop_library.load_dll()
        try:
        
            message = ctypes.create_string_buffer(256)
            timestamp = ctypes.c_double()
            background_removed = ctypes.c_bool()
            comments = ctypes.create_string_buffer(256)
            exp_duration_us = ctypes.c_uint()
            nb_summed_images = ctypes.c_uint()
            smearing_removed = ctypes.c_bool()
            trigger_mode = ctypes.create_string_buffer(256)
            dll.Imop_HasoSlopes_GetInfo_FromFile(
                message,
                ctypes.c_char_p(has_file_path.encode('utf-8')),
                ctypes.byref(timestamp),
                ctypes.byref(background_removed),
                comments,
                ctypes.byref(exp_duration_us),
                ctypes.byref(nb_summed_images),
                ctypes.byref(smearing_removed),
                trigger_mode
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                imop_struct.Metadata_t( 
                    comments.value.decode('utf-8'),
                    None,
                    timestamp.value,
                    background_removed.value,
                    smearing_removed.value,
                    None,
                    None
                    ),
                imop_struct.AcquisitionInfo_t(
                    None,
                    exp_duration_us.value,
                    None,
                    nb_summed_images.value, 
                    trigger_mode.value.decode('utf-8')
                    )
                )
        except Exception as exception:
            raise Exception(__name__+' : get_info_from_file',exception)

    def save_to_file(
        self, 
        has_file_path, 
        comments, 
        session_name
        ):
        """Save HasoSlopes to a (\*.has) file
        
        :param has_file_path: Absolute path to the (\*.has) file
        :type has_file_path: string
        :param comments: Comments
        :type comments: string
        :param session_name: Session name
        :type session_name: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoSlopes_SaveToFile(
                message,
                self.hasoslopes,
                ctypes.c_char_p(has_file_path.encode('utf-8')),
                ctypes.c_char_p(comments.encode('utf-8')),
                ctypes.c_char_p(session_name.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : save_to_file',exception)

    def get_geometric_properties(self):
        """Compute HasoSlopes geometric properties
        
        :return: Hasoslopes geometric properties : x tilt (mrad), y tilt (mrad), radius (mm), x focus position (mm), y focus position (mm), astgimatism angle (rad), sagittal focal length (mm), tangential focal length (mm)
        :rtype: HasoslopesGeometric_t 
        """
        try:
            message = ctypes.create_string_buffer(256)
            tiltX_mrad = ctypes.c_float()
            tiltY_mrad = ctypes.c_float()
            radius_mm = ctypes.c_float()
            focus_Xpos_mm = ctypes.c_float()
            focus_Ypos_mm = ctypes.c_float()
            astig_angle_rad = ctypes.c_float()
            sagittal_focal_length_mm = ctypes.c_float()
            tangential_focal_length_mm = ctypes.c_float()
            self.dll.Imop_HasoSlopes_GetGeometricProperties(
                message,
                self.hasoslopes,
                ctypes.byref(tiltX_mrad),
                ctypes.byref(tiltY_mrad),
                ctypes.byref(radius_mm),
                ctypes.byref(focus_Xpos_mm),
                ctypes.byref(focus_Ypos_mm),
                ctypes.byref(astig_angle_rad),
                ctypes.byref(sagittal_focal_length_mm),
                ctypes.byref(tangential_focal_length_mm)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.HasoslopesGeometric_t(
                tiltX_mrad.value,
                tiltY_mrad.value,
                radius_mm.value,
                focus_Xpos_mm.value,
                focus_Ypos_mm.value,
                astig_angle_rad.value,
                sagittal_focal_length_mm.value,
                tangential_focal_length_mm.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_geometric_properties',exception)

    def has_aliasing(
        self,
        radius_of_curvature
        ):
        """Test whether HasoSlopes causes aliasing
        When Phase has been reconstructed from slopes, strong local extrema on slopes
        may provoke aliasing.
        This function allows to test whether slopes will
        produce suitable phase for correct gaussian parameters computation. Source
        wavelength must be available in metadata of input Haso slopes. The radius of
        curvature of the slopes, in millimeters, is required to perform the test.
        
        .. seealso:: HasoSlopes.set_waveLength
        
        :param radius_of_curvature: radius of curvature of the slopes (mm)
        :type radius_of_curvature: float
        :return: has aliasing
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            aliasing = ctypes.c_byte()
            self.dll.Imop_HasoSlopes_HasAliasing(
                message,
                self.hasoslopes,
                ctypes.c_float(radius_of_curvature),
                ctypes.byref(aliasing)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return bool(aliasing)
        except Exception as exception:
            raise Exception(__name__+' : has_aliasing',exception)