#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library
import io_wavekit_structure as imop_struct

class Image(object):
    """Class Image
    
    - Constructor from Size and Bit Depth :
        - **size** - uint2D : 2D image size
        - **bit_depth** - ushort : Image bits depth (16 or 32)
    
    - Constructor from Image file :
        - **image_file_path** - string : Absolute path to image file
    
    - Constructor from Copy :
        - **image** - Image : Image object to copy
    """
    
    def __init_(
        self,
        size,
        bit_depth
        ):
        """Single channel Image constructor from dimensions and bit depth
        
        :param size: 2D image size
        :type size: uint2D
        :param bit_depth: Image bits depth (16 or 32)
        :type bit_depth: ushort
        """
        try:
            message = ctypes.create_string_buffer(256)
            if type(size) is not imop_struct.uint2D :
                raise Exception('IO_Error', 'size must be an io_wavekit_structure.uint2D class')
            self.dll.Imop_Image_New(
                message,
                ctypes.pointer(self.image),
                ctypes.byref(size),
                ctypes.c_ushort(bit_depth)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_',exception)
    
    def __init_from_file(
        self,
        image_file_path
        ):
        """Image constructor from file
        
        :param image_file_path: Absolute path to image file (*.himg)
        :type image_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Image_NewFromFile(
                message,
                ctypes.pointer(self.image),
                ctypes.c_char_p(image_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_from_file',exception)
    
    def __init_from_copy(
        self,
        image
        ):
        """Image constructor from copy
        
        :param image: Image object to copy
        :type image: Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Image_NewFromImage(
                message,
                image.image,
                ctypes.pointer(self.image)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init_from_copy',exception)
        

    def __init__(self, **kwargs):
        """Image Constructor
        """
        self.image = ctypes.c_void_p()
        self.dll = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 2):
                if('size' in kwargs
                   and 'bit_depth' in kwargs):
                    entered = 1
                    self.__init_(kwargs['size'], kwargs['bit_depth'])
            if(arg_size == 1):
                if('image_file_path' in kwargs):
                    entered = 1
                    self.__init_from_file(kwargs['image_file_path'])
                if('image' in kwargs):
                    entered = 1
                    self.__init_from_copy(kwargs['image'])
        except Exception as exception:
            raise Exception(__name__+' : init',exception)
        if(entered == 0):
            raise Exception(__name__+' : init','---CAN NOT CREATE IMAGE OBJECT---')
                    
    def __del_obj__(self):
        """Image Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_Image_Delete(message, self.image)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def load(
        self,
        image_file_path
        ):
        """Load Image from file
        
        :param image_file_path: Absolute path to image file (\*.himg)
        :type image_file_path: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Image_Load(
                message,
                self.image,
                ctypes.c_char_p(image_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : load',exception)
            
    def save(
        self,
        saving_file_path,
        comments = "",
        ):
        """Save Image to file
        
        :param image_file_path: Absolute path to image file (\*.himg)
        :type image_file_path: string
        :param comments: Comments to add
        :type comments: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Image_Save(
                message,
                self.image,
                ctypes.c_char_p(comments.encode('utf-8')),
                ctypes.c_char_p(saving_file_path.encode('utf-8'))
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : save',exception)

    def is_haso_image(self):
        """Check if Image has been captured with a Haso sensor
        
        :return: True if Image has been captured with Haso sensor
        :rtype: bool
        """
        try:
            message = ctypes.create_string_buffer(256)
            is_haso_image = ctypes.c_bool()
            self.dll.Imop_Image_IsHasoImage(
                message,
                self.image,
                ctypes.byref(is_haso_image)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return is_haso_image.value
        except Exception as exception:
            raise Exception(__name__+' : is_haso_image',exception)

    def get_haso_serial_number(self):
        """Get Haso serial number from Image object if Image has been captured with a Haso sensor, else return an error
        
        :return: Serial Number
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            haso_serial_number = ctypes.create_string_buffer(256)
            self.dll.Imop_Image_GetHasoSerialNumber(
                message,
                self.image,
                ctypes.byref(haso_serial_number)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return haso_serial_number.value.decode('utf-8')
        except Exception as exception:
            raise Exception(__name__+' : get_haso_serial_number',exception)

    def get_info(self):
        """Get metadata from Image object
        
        :return: Camera Serial Number, Date of the image, Comments, is Smearing removed, is Background removed, Max level value 
        :rtype: tuple(string, double, string, bool, bool, longlong)
        """
        try:
            message = ctypes.create_string_buffer(256)
            camera_serial_number = ctypes.create_string_buffer(256)
            time_stamp = ctypes.c_double()
            comments = ctypes.create_string_buffer(256)
            is_smearing_removed = ctypes.c_bool()
            is_background_removed = ctypes.c_bool()
            max_level = ctypes.c_longlong()
            self.dll.Imop_Image_GetInfo(
                message,
                self.image,
                camera_serial_number,
                ctypes.byref(time_stamp),
                comments,
                ctypes.byref(is_smearing_removed),
                ctypes.byref(is_background_removed),
                ctypes.byref(max_level)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.Metadata_t(
                comments.value.decode('utf-8'),
                None,
                time_stamp.value,
                is_background_removed.value,
                is_smearing_removed.value, 
                camera_serial_number.value.decode('utf-8'),
                max_level.value 
                )
        except Exception as exception:
            raise Exception(__name__+' : get_info',exception)

    def get_acquisition_info(self):
        """Get acquisition information from Image object
        
        :return: Exposure duration requested, Exposure duration applied, Gain, Number summed image, Trigger type
        :rtype: tuple(int, int, float, int, string)
        """
        try:
            message = ctypes.create_string_buffer(256)
            exposure_time_requested = ctypes.c_int()
            exposure_time_applied = ctypes.c_int()
            gain = ctypes.c_float()
            nb_summed_images = ctypes.c_int()
            trigger_type = ctypes.create_string_buffer(256)
            self.dll.Imop_Image_GetAcquisitionInfo(
                message,
                self.image,
                ctypes.byref(exposure_time_requested),
                ctypes.byref(exposure_time_applied),
                ctypes.byref(gain),
                ctypes.byref(nb_summed_images),
                ctypes.byref(trigger_type)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.AcquisitionInfo_t(
                exposure_time_requested.value,
                exposure_time_applied.value, 
                gain.value, 
                nb_summed_images.value, 
                trigger_type.value.decode('utf-8')
                )
        except Exception as exception:
            raise Exception(__name__+' : get_acquisition_info',exception)

    @staticmethod
    def get_info_from_file(image_file_path):
        """Get acquisition information from image file
        
        :param image_file_path: Absolute path to image file (\*.himg)
        :type image_file_path: string
        :return: Bit depth, Image size, Acquisition time, Haso serial number, Camera serial number, Acquisition info (Applied gain, Number summed images, Exposure duration requested, Exposure duration applied)
        :rtype: tuple(AcquisitionInfo_t, Metadata_t, int, uint2D, str)
        """
        dll = imop_library.load_dll()
        try:
            message = ctypes.create_string_buffer(256)
            bit_depth = ctypes.c_ushort()
            size = imop_struct.uint2D(0, 0)
            time_stamp = ctypes.c_double()
            haso_serial_number = ctypes.create_string_buffer(256)
            camera_serial_number = ctypes.create_string_buffer(256)
            gain = ctypes.c_float()
            nb_summed_images = ctypes.c_uint()
            exposure_time_requested = ctypes.c_uint()
            exposure_time_applied = ctypes.c_uint()            
            dll.Imop_Image_GetInfo_FromFile(
                message,
                ctypes.c_char_p(image_file_path.encode('utf-8')),
                ctypes.byref(bit_depth),
                ctypes.byref(size),
                ctypes.byref(time_stamp),
                ctypes.byref(haso_serial_number),
                ctypes.byref(camera_serial_number),
                ctypes.byref(gain),
                ctypes.byref(nb_summed_images),
                ctypes.byref(exposure_time_requested),
                ctypes.byref(exposure_time_applied)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                imop_struct.AcquisitionInfo_t(
                    exposure_time_requested.value, 
                    exposure_time_applied.value,
                    gain.value, 
                    nb_summed_images.value,
                    None
                    ),
                imop_struct.Metadata_t(
                    None,
                    None,
                    time_stamp.value, 
                    None,
                    None, 
                    str(camera_serial_number.value.decode('utf-8')),
                    None
                    ),
                bit_depth.value, 
                size, 
                str(haso_serial_number.value.decode('utf-8')), 
                )
        except Exception as exception:
            raise Exception(__name__+' : get_info_from_file',exception)

    def get_size(self):
        """Get Image size and bit depth
        
        :return: Image size, Bit depth
        :rtype: tuple(uint2D, ushort)
        """
        try:
            message = ctypes.create_string_buffer(256)
            size = imop_struct.uint2D(0, 0)
            bit_depth = ctypes.c_ushort()           
            self.dll.Imop_Image_GetSize(
                message,
                self.image,
                ctypes.byref(size),
                ctypes.byref(bit_depth)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                size,
                bit_depth.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_size',exception)

    def is_processed_image(self):
        """Check whether a background or a filtered version were subtracted from Image
        
        :return: background removed, filtered image removed
        :rtype: tuple(bool, bool)
        """
        try:
            message = ctypes.create_string_buffer(256)
            background_removed = ctypes.c_bool()
            filtered_image_removed = ctypes.c_bool()           
            self.dll.Imop_Image_IsProcessedImage(
                message,
                self.image,
                ctypes.byref(background_removed),
                ctypes.byref(filtered_image_removed)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                background_removed.value, 
                filtered_image_removed.value 
                )
        except Exception as exception:
            raise Exception(__name__+' : is_processed_image',exception)

    def get_data(self):
        """Get pixels buffer as an unsigned int array
        
        :return: Pixel buffer
        :rtype: uintc 2D numpy.array
        """
        try:
            message = ctypes.create_string_buffer(256)
            size, bit_depth = self.get_size()
            values = numpy.zeros(
                (size.Y, size.X), 
                dtype = numpy.uintc
                )
            self.dll.Imop_Image_GetData.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.uintc, flags="C_CONTIGUOUS")  
                ]
            self.dll.Imop_Image_GetData(
                message,
                self.image,
                values
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return values
        except Exception as exception:
            raise Exception(__name__+' : get_data',exception)

    def get_statistics(
        self,
        nb_extreme_pixels
        ):
        """Get differents Statistics from image :
        - The average of the *nb_extreme_pixels* greatest elements.
        - The average of the *nb_extreme_pixels* smallest elements.
        - The percentage of saturated pixels.
        
        :param nb_extreme_pixels: Number of pixels to average for min or max computation
        :type nb_extreme_pixels: uint
        :return: Average of the nb_extreme_pixels greatest elements, Average of the nb_extreme_pixels smallest elements, Percentage of saturated pixels 
        :rtype: tuple(uint, uint, float)
        """
        try:
            message = ctypes.create_string_buffer(256)
            min_ = ctypes.c_uint()
            max_ = ctypes.c_uint()
            fill_percentage = ctypes.c_float()
            self.dll.Imop_Image_GetStatistics(
                message,
                self.image,
                ctypes.c_uint(nb_extreme_pixels),
                ctypes.byref(min_),
                ctypes.byref(max_),
                ctypes.byref(fill_percentage)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                min_.value, 
                max_.value, 
                fill_percentage.value
                )
        except Exception as exception:
            raise Exception(__name__+' : get_statistics',exception)

    def get_saturation(
        self,
        nb_extreme_pixels
        ):
        """Compute the percentage of saturated pixels.
        
        :param nb_extreme_pixels: Number of pixels to average for max computation
        :type nb_extreme_pixels: uint
        :return: Percentage of saturated pixels 
        :rtype: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            fill_percentage = ctypes.c_float()
            self.dll.Imop_Image_GetSaturation(
                message,
                self.image,
                ctypes.c_uint(nb_extreme_pixels),
                ctypes.byref(fill_percentage)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return fill_percentage.value
        except Exception as exception:
            raise Exception(__name__+' : get_saturation',exception)

    def __sub__(
        self,
        image_to_sub
        ):
        """Image Subtraction : Image = self - image_to_sub 
        Images must have the same size, channels count and bits depth.
        
        :return: self - image_to_sub
        :rtype: Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            res = Image(
                size = imop_struct.uint2D(1,1),
                bit_depth = 16
                )
            self.dll.Imop_Image_SubtractionABC(
                message,
                self.image,
                image_to_sub.image,
                res.image
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return res
        except Exception as exception:
            raise Exception(__name__+' : sub',exception)

    def __isub__(
        self,
        image_to_sub
        ):
        """Image Subtraction : self -= image_to_sub 
        Images must have the same size, channels count and bits depth.
        
        :return: self - ImageB 
        :rtype: Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Image_SubtractionAB(
                message,
                self.image,
                image_to_sub.image
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return self
        except Exception as exception:
            raise Exception(__name__+' : isub',exception)