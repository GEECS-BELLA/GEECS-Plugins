#!/usr/bin/python

import os, sys
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_surface as imop_surface
import io_wavekit_structure as imop_struct
import io_wavekit_enum as imop_enum

class HasoField(object):
    """Class HasoField
    
    - Constructor from Parameters :
        - **config_file_path** - string : Absolute path to Haso configuration file (\*.dat)
        - **hasoslopes** - HasoSlopes : HasoSlopes object
        - **phase** - Phase : Phase object
        - **curv_radius** - double : Radius of curvature (mm)
        - **wavelength** - double : Wavelength of the source used to compute the HasoSlopes (nm)
        - **oversampling** - uchar : Increment of oversampling level
    """
    
    def __init_(
        self,
        config_file_path,
        hasoslopes,
        phase,
        curv_radius,
        wavelength,
        oversampling
        ):
        """HasoField constructor from HasoSlopes and phase
        *HasoSlopes* provides slopes pupil, and *Phase* provides a pupil which is expected
        to be equal to slopes pupil or to the greatest common
        pupil between slopes pupil and a projection pupil used for *Phase* computation. 
        This implies that pupil of *Phase* must be included in pupil of
        *HasoSlopes*. After checking this is the case, function uses the pupil
        of *Phase* for resulting *HasoField*. 
        It is possible to oversample the resulting field: each increment of **oversampling_level** doubles the field's size.
        
        :param config_file_path: Absolute path to Haso configuration file
        :type config_file_path: string
        :param hasoslopes: HasoSlopes object
        :type hasoslopes: HasoSlopes
        :param phase: Phase object
        :type phase: Phase
        :param curv_radius: Radius of curvature (mm)
        :type curv_radius: double
        :param wavelength: Wavelength of the source used to compute the HasoSlopes (nm)
        :type wavelength: double
        :param oversampling: Increment of oversampling level
        :type oversampling: uchar
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_HasoField_New(
                message,
                ctypes.pointer(self.hasofield),
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                hasoslopes.hasoslopes,
                phase.phase,
                ctypes.c_double(curv_radius),
                ctypes.byref(ctypes.c_double(wavelength)),
                ctypes.c_ubyte(oversampling)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : init_',exception)

    def __init__(self,  **kwargs):
        """HasoField constructor
        """
        self.hasofield = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 6):
                if('config_file_path' in kwargs
                    and 'hasoslopes' in kwargs
                    and 'phase' in kwargs
                    and 'curv_radius' in kwargs
                    and 'wavelength' in kwargs
                    and 'oversampling' in kwargs
                    ):
                    entered = 1
                    self.__init_(
                        kwargs['config_file_path'],
                        kwargs['hasoslopes'],
                        kwargs['phase'],
                        kwargs['curv_radius'],
                        kwargs['wavelength'],
                        kwargs['oversampling']
                        )
        except Exception as exception:
            raise Exception(__name__+' : init',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE HASOFIELD OBJECT---')

    def __del_obj__(self):
        """Hasofield destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_HasoField_Delete(message, self.hasofield)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_wavelength(self):
        """Get Source wavelength
        
        :return: Wavelength of the source (nm)
        :rtype: double
        """
        try:
            message = ctypes.create_string_buffer(256)
            ret = ctypes.c_double()
            self.dll.Imop_HasoField_GetWaveLength(
                message,
                self.hasofield,
                ctypes.byref(ret)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return ret.value
        except Exception as exception:
            raise Exception(__name__+' : get_wavelength',exception)
            
    def psf(
        self,
        null_phase, 
        flat_intensity,
        defocus,
        config_file_path
        ):
        """Point spread function 
        This function computes point spread function from a HasoField. The
        function allows to choose the defocus which is zero by default. It also
        provides options to use null phase instead of measured phase, and flat
        intensity instead of measured intensity. When using a null phase, phase
        contained in input HasoField is not used. When using a flat intensity,
        intensity contained in input HasoField is used anyway, as well as pupil, to
        compute average intensity.
        
        :param null_phase: Phase is null
        :type null_phase: bool
        :param flat_intensity: Intensity surface is flat
        :type flat_intensity: bool
        :param defocus: Defocus parameter(mm)
        :type defocus: float
        :param config_file_path: Absolute path to the Haso configuration file
        :type config_file_path: string
        :return: PSF stored as Surface - Surface step in um
        :rtype: Surface
        """
        try:
            message = ctypes.create_string_buffer(256)
            ret = imop_surface.Surface(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1), 
                    imop_struct.float2D(1.0,1.0)
                    )
                )
            self.dll.Imop_HasoField_PSF(
                message,
                self.hasofield,
                ctypes.c_bool(null_phase),
                ctypes.c_bool(flat_intensity),
                ctypes.c_float(defocus),
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                ret.surface
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return ret
        except Exception as exception:
            raise Exception(__name__+' : psf',exception)

    def mtf(
        self, 
        psf_surface, 
        config_file_path
        ):
        """Modulation transfer function from point spread function surface
        
        :param psf_surface: PSF stored as Surface
        :type psf_surface: Surface
        :param config_file_path: Absolute path to the Haso configuration file
        :type config_file_path: string
        :return: MTF stored as Surface - Surface step in cycles/mm
        :rtype: Surface
        """
        try:
            message = ctypes.create_string_buffer(256)
            ret = imop_surface.Surface(
                dimensions = imop_struct.dimensions(
                    imop_struct.uint2D(1,1), 
                    imop_struct.float2D(1.0,1.0)
                    )
                )
            self.dll.Imop_HasoField_MTF(
                message,
                psf_surface.surface,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                ret.surface
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return ret
        except Exception as exception:
            raise Exception(__name__+' : mtf',exception)

    def strehl(
        self, 
        config_file_path, 
        flat_experimental_intensity, 
        flat_theoretical_intensity,
        through_focus, 
        oversample, 
        defocus
        ):
        """Strehl ratio from HasoField function
        This function computes Strehl ratio from a HasoField. Input HasoField is
        used to compute experimental and theoretical point spread functions (PSF).
        The function allows to choose the defocus. It also
        provides options to use flat intensity instead of measured intensity for
        computation of experimental or theoretical PSF. Another option is available
        to compute Strehl ratio through focus, in which case theoretical PSF is
        computed in the same place as experimental PSF rather than at focal point.
        There is also an option to oversample PSF surfaces before ratio computation.
        
        :param config_file_path: Absolute path to the Haso configuration file
        :type config_file_path: string
        :param flat_experimental_intensity: Use flat experimental intensity
        :type flat_experimental_intensity: bool
        :param flat_theoretical_intensity: Use flat theoretical intensity
        :type flat_theoretical_intensity: bool
        :param through_focus: Use through focus
        :type through_focus: bool
        :param oversample: Use oversample
        :type oversample: bool
        :param defocus: Defocus parameter(mm)
        :type defocus: float
        :return: Strehl ratio
        :rtype: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            ratio = ctypes.c_float()
            self.dll.Imop_HasoField_Strehl(
                message,
                self.hasofield,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                ctypes.c_bool(flat_experimental_intensity),
                ctypes.c_bool(flat_theoretical_intensity),
                ctypes.c_bool(through_focus),
                ctypes.c_bool(oversample), 
                ctypes.c_float(defocus),
                ctypes.byref(ratio)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return ratio.value
        except Exception as exception:
            raise Exception(__name__+' : strehl',exception)

    def strehl_from_psf(
        self, 
        config_file_path,
        experimental_psf, 
        theoretical_psf, 
        oversample
        ):
        """Strehl ratio from PSF function
        This function Strehl ratio from experimental and theoretical point
        spread functions (PSF).
        There is also an option to oversample both PSF surfaces before ratio computation.
        
        :param config_file_path: Absolute path to the Haso configuration file
        :type config_file_path: string
        :param experimental_psf: Experimental psf values
        :type experimental_psf: Surface
        :param theoretical_psf: Theoretical psf values
        :type theoretical_psf: Surface
        :param oversample: Use oversample
        :type oversample: bool
        :return: Strehl ratio
        :rtype: float
        """
        try:
            message = ctypes.create_string_buffer(256)
            ratio = ctypes.c_float()
            self.dll.Imop_HasoField_StrehlFromPSF(
                message,
                self.hasofield,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                experimental_psf.surface,
                theoretical_psf.surface,
                ctypes.c_bool(oversample), 
                ctypes.byref(ratio)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return ratio.value
        except Exception as exception:
            raise Exception(__name__+' : strehl_from_psf',exception)

    def gaussian_param(
        self,
        config_file_path,
        nb_planes,
        spot_type,
        degree
        ):
        """Gaussian parameters function
        This function computes gaussian parameters of a beam from a HasoField.
        Input HasoField is used to compute values of the point spread function (PSF)
        over different computation planes. The number of computation planes can be
        set by user. The different computation planes constitute the computation
        data, which can be got after processing. In a given computation plane, the
        PSF forms a spot which can be approximated by a circle or a square whose size
        plays a role for gaussian parameters determination. By default, the function
        is configured to use circular spots, but it can be configured to use tilted
        square spots. The square's tilt is defined by the angle it forms with an
        horizontal axis. When the processor is configured to use circular spots, it
        characterizes the circle by its radius. When the processor is configured to
        use square spots, it characterizes the square by the half length of its side.
        In both cases, this quantity is stored in attribute spot_size of
        corresponding ComputationPlane instance.
        
        :param nb_planes: Number of planes
        :type nb_planes: int
        :param config_file_path: Haso Configuration file
        :type config_file_path: string
        :param spot_type: spot shape (circular or square)
        :type spot_type: E_SPOT_SHAPE
        :param degree: Angle of the square spot shape forms with an horizontal axis (degree)
        :type degree: float
        :return: Gaussian Parameters : Waist position (mm), Waist radius (mm), Rayleigh length (mm), Divergence of the beam (radians), m squared parameter
        :rtype: GaussianParameters_t
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(spot_type==imop_enum.E_SPOT_SHAPE.SQUARE):
                spot_type = "square"
            if(spot_type==imop_enum.E_SPOT_SHAPE.CIRCULAR):
                spot_type = "circle"
            waist_position_millimeter = ctypes.c_float()
            waist_radius_millimeter = ctypes.c_float()
            rayleigh_length_millimeter = ctypes.c_float()
            divergence = ctypes.c_float()
            msquared = ctypes.c_float()
            self.dll.Imop_HasoField_GaussianParam(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                self.hasofield,
                ctypes.c_int(nb_planes),
                ctypes.byref(waist_position_millimeter),
                ctypes.byref(waist_radius_millimeter),
                ctypes.byref(rayleigh_length_millimeter),
                ctypes.byref(divergence),
                ctypes.byref(msquared),
                ctypes.c_char_p(spot_type.encode('utf-8')),
                ctypes.c_float(degree)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return imop_struct.GaussianParameters_t(
                    waist_position_millimeter.value, 
                    waist_radius_millimeter.value, 
                    rayleigh_length_millimeter.value,
                    divergence.value, 
                    msquared.value
                    )
        except Exception as exception:
            raise Exception(__name__+' : gaussian_param',exception)

    def complete_gaussian_param(
        self, 
        config_file_path,
        nb_planes,
        spot_type,
        degree
        ):
        """Complete gaussian parameters function
        This function computes gaussian parameters of a beam from a HasoField.
        Input HasoField is used to compute values of the point spread function (PSF)
        over different computation planes. The number of computation planes can be
        set by user. The different computation planes constitute the computation
        data, which can be got after processing. In a given computation plane, the
        PSF forms a spot which can be approximated by a circle or a square whose size
        plays a role for gaussian parameters determination. By default, the function
        is configured to use circular spots, but it can be configured to use tilted
        square spots. The square's tilt is defined by the angle it forms with an
        horizontal axis. When the processor is configured to use circular spots, it
        characterizes the circle by its radius. When the processor is configured to
        use square spots, it characterizes the square by the half length of its side.
        In both cases, this quantity is stored in attribute spot_size of
        corresponding ComputationPlane instance.
        
        :param config_file_path: Absolute path to the Haso configuration file
        :type config_file_path: string
        :param nb_planes: Number of planes
        :type nb_planes: int
        :param spot_type: spot shape (circular or square)
        :type spot_type: E_SPOT_SHAPE
        :param degree: Angle of the square spot shape forms with an horizontal axis (degree)
        :type degree: float
        :param tab_psf: Array of computed psf - psf is a surface with step mm
        :type tab_psf: surface_array
        :return: Gaussian Parameters (Waist position (mm), Waist radius (mm), Rayleigh length (mm), Divergence of the beam (radians), m squared parameter) ; Gaussian Planes Parameters (Array of z cote (mm) of the different planes, Array of spot size (mm) of the different planes, Array of max energy of the different planes) ; Array of computed psf - psf is a surface with step mm ; Indicator of whether the entire beam of last processing could be obtained 
        :rtype: tuple(GaussianParameters_t, GaussianParametersPlanes_t, Surface[], bool)
        """
        try:
            message = ctypes.create_string_buffer(256)
            if(spot_type==imop_enum.E_SPOT_SHAPE.SQUARE):
                spot_type = "square"
            if(spot_type==imop_enum.E_SPOT_SHAPE.CIRCULAR):
                spot_type = "circle"
            waist_position_millimeter = ctypes.c_float()
            waist_radius_millimeter = ctypes.c_float()
            rayleigh_length_millimeter = ctypes.c_float()
            divergence = ctypes.c_float()
            msquared = ctypes.c_float()
            threshold = ctypes.c_float()
            tab_z_cote_mm = numpy.zeros(nb_planes, dtype = numpy.single)
            tab_spot_size_mm = numpy.zeros(nb_planes, dtype = numpy.single)
            threshold_validity = ctypes.c_ubyte()
            tab_max_energy = numpy.zeros(nb_planes, dtype = numpy.single)
            psf_list = []
            psf_array = (ctypes.c_void_p * nb_planes)()
            for i in range(nb_planes):
                psf_list.append(imop_surface.Surface(
                    dimensions = imop_struct.dimensions(
                        imop_struct.uint2D(1,1), 
                        imop_struct.float2D(1.0,1.0)
                        )
                    )
                )
                psf_array[i] = psf_list[i].surface
            self.dll.Imop_HasoField_CompleteGaussianParam.argtypes = [
                ctypes.c_char_p, 
                ctypes.c_char_p, 
                ctypes.c_void_p, 
                ctypes.c_int,    
                ctypes.c_void_p, 
                ctypes.c_void_p, 
                ctypes.c_void_p, 
                ctypes.c_void_p, 
                ctypes.c_void_p, 
                ctypes.c_char_p, 
                ctypes.c_float,  
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"), 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"), 
                ctypes.c_void_p, 
                ctypes.c_void_p, 
                numpy.ctypeslib.ndpointer(numpy.single, flags="C_CONTIGUOUS"),
                ctypes.c_void_p
                ]  
            self.dll.Imop_HasoField_CompleteGaussianParam(
                message,
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                self.hasofield,
                ctypes.c_int(nb_planes),
                ctypes.byref(waist_position_millimeter),
                ctypes.byref(waist_radius_millimeter),
                ctypes.byref(rayleigh_length_millimeter),
                ctypes.byref(divergence),
                ctypes.byref(msquared),
                ctypes.c_char_p(spot_type.encode('utf-8')),
                ctypes.c_float(degree),
                tab_z_cote_mm,
                tab_spot_size_mm,
                ctypes.pointer(psf_array),
                ctypes.byref(threshold_validity),
                tab_max_energy,
                ctypes.byref(threshold)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            return (
                imop_struct.GaussianParameters_t(
                    waist_position_millimeter.value, 
                    waist_radius_millimeter.value, 
                    rayleigh_length_millimeter.value,
                    divergence.value,
                    msquared.value
                    ),
                imop_struct.GaussianParametersPlanes_t(
                    tab_z_cote_mm.tolist(), 
                    tab_spot_size_mm.tolist(),
                    tab_max_energy.tolist()
                    ),
                psf_list,
                bool(threshold_validity),
                threshold.value
                )
        except Exception as exception:
            raise Exception(__name__+' : complete_gaussian_param',exception)
