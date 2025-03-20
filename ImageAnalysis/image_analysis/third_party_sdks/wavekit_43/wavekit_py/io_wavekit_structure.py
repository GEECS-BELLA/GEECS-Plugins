#!/usr/bin/python

import os, sys 
import ctypes
import numpy
from collections import namedtuple


class uint2D(ctypes.Structure):
    """Class uint2D
    
    unsigned int (X,Y) couple
    :attribute X: X value
    :type X: uint
    :attribute Y: Y value
    :type Y: uint
    """
    _fields_ = [
        ('X', ctypes.c_uint),
        ('Y', ctypes.c_uint)
        ]
    def __init__(
        self,
        X, 
        Y
        ):
        super(uint2D, self).__init__(X,Y)
    def __str__(self):
        return (
            'uint2D Object : '+
            'X='+str(self.X)+'; '+
            'Y='+str(self.Y)
            )

class float2D(ctypes.Structure):
    """Class float2D
    
    float (X,Y) couple
    :attribute X: X value
    :type X: float
    :attribute Y: Y value
    :type Y: float
    """
    _fields_ = [
        ('X', ctypes.c_float),
        ('Y', ctypes.c_float)
        ]
    def __init__(
        self, 
        X,
        Y
        ):
        super(float2D, self).__init__(X,Y)
    def __str__(self):
        return (
            'float2D Object : '+
            'X='+str(self.X)+'; '+
            'Y='+str(self.Y)
            )

class int2D(ctypes.Structure):
    """Class int2D
    
    int (X,Y) couple
    :attribute X: X value
    :type X: int
    :attribute Y: Y value
    :type Y: int
    """
    _fields_ = [
        ('X', ctypes.c_int),
        ('Y', ctypes.c_int)
        ]
    def __init__(
        self,
        X,
        Y
        ):
        super(int2D, self).__init__(X,Y)
    def __str__(self):
        return (
            'int2D Object : '+
            'X='+str(self.X)+'; '+
            'Y='+str(self.Y)
            )

class dimensions(ctypes.Structure):
    """Class dimensions
    
    :attribute size: size of the grid
    :type size: uint2D
    :attribute steps: spacing of the grid
    :type steps: float2D
    """
    _fields_ = [
        ('size', uint2D),
        ('steps', float2D)
        ]
    def __init__(
        self, 
        size, 
        steps
        ):
        super(dimensions, self).__init__(size,steps)
    def __str__(self):
        return (
            'dimensions Object : '+
            'size='+str(self.size)+'; '+
            'steps='+str(self.steps)
            )



CorrectorPrefs_t = namedtuple('CorrectorPrefs_t', [
    'min_array',        # float list[]
    'max_array',        # float list[]
    'validity_array',   # int list[]
    'fixed_value_array' # float list[]
    ])
"""Corrector preferences

:param min_array:lowest command value for each actuator
:type min_array: float list[]
:param max_array: highest command value for each actuator
:type max_array: float list[]
:param validity_array: status each actuator
:type validity_array: E_ACTUATOR_CONDITIONS list[]
:param fixed_value_array: fixed value for each actuator
:type fixed_value_array: float list[]
"""

ActuatorPrefs_t = namedtuple('ActuatorPrefs_t', [
    'min_value',         # float
    'max_value',         # float
    'validity',     # int
    'fixed_value'   # float
    ])
"""Actuator preferences

:param min_value: lowest command value
:type min_value: float
:param max_value: highest command value
:type max_value: float
:param validity: status
:type validity: E_ACTUATOR_CONDITIONS
:param fixed_value: fixed value
:type fixed_value: float
"""

ParameterOption_t = namedtuple('ParameterOption_t', [
    'has_default_value',        # bool
    'has_limits',               # bool
    'is_connection_parameter',  # bool
    'is_gettable',              # bool
    'is_gettable_onfly',        # bool    
    'is_settable',              # bool
    'is_settable_onfly',        # bool    
    'is_string_enum'            # bool
    ])
"""Parameter options

:param has_default_value: True if the parameter has a default value
:type has_default_value: bool
:param has_limits: True if the parameter has a min and a max value
:type has_limits: bool
:param is_connection_parameter: True if the parameter is a connection parameter
:type is_connection_parameter: bool
:param is_gettable: True if the parameter value can be read
:type is_gettable: bool
:param is_gettable_onfly: True if the parameter value can be read while device is running
:type is_gettable_onfly: bool
:param is_settable: True if the parameter value can be set
:type is_settable: bool
:param is_settable_onfly: True if the parameter value can be set while device is running
:type is_settable_onfly: bool
:param is_string_enum: True if the parameter is a string enumeration
:type is_string_enum: bool
"""

GaussianParameters_t = namedtuple('GaussianParameters_t', [
    'waist_position',   # float    
    'waist_radius',     # float
    'rayleigh_length',  # float
    'divergence',       # float
    'msquared'          # float
    ])
"""Gaussian parameters

:param waist_position: waist position (mm)
:type waist_position: float
:param waist_radius: waist radius (mm)
:type waist_radius: float
:param rayleigh_length: rayleigh length (mm)
:type rayleigh_length: float
:param divergence: divergence of the beam (radians)
:type divergence: float
:param msquared: m squared parameter
:type waist_position: float
"""

GaussianParametersPlanes_t = namedtuple('GaussianParametersPlanes_t', [
    'z_cote_array',     # float list[]
    'spot_size_array',  # float list[]
    'max_energy_array'  # float list[]
    ])
"""Gaussian parameters planes

:param z_cote_array: z cote of the different planes
:type waist_position: float list[]
:param spot_size_array: spot size of the different planes
:type spot_size_array: float list[]
:param max_energy_array: max energy of the different planes
:type max_energy_array: float list[]
"""

ZernikePupil_t = namedtuple('ZernikePupil_t', [
    'center',   # float2D
    'radius'    # float
    ])
"""Zernike Pupil

:param center: center of the pupil
:type center: float2D
:param radius: radius of the pupil
:type radius: float
"""

LegendrePupil_t = namedtuple('LegendrePupil_t', [
    'center',   # float2D
    'halfsize'  # float2D
    ])
"""Legendre Pupil

:param center: center of the pupil
:type center: float2D
:param halfsize: halfsize of the pupil in each direction
:type radius: float2D
"""

HasoslopesGeometric_t = namedtuple('HasoslopesGeometric_t', [
    'tilt_x',       # float 
    'tilt_y',       # float
    'radius',       # float
    'focus_x_pos',  # float
    'focus_y_pos',  # float
    'astig_angle',  # float
    'sagittal',     # float
    'tangential'    # float
    ])
"""Hasoslopes geometric properties

:param tilt_x: x tilt (mrad)
:type tilt_x: float
:param tilt_y: y tilt (mrad)
:type tilt_y: float
:param radius: radius (mm)
:type radius: float
:param focus_x_pos: x focus pos (mm)
:type focus_x_pos: float
:param focus_y_pos: y focus pos (mm)
:type focus_y_pos: float
:param astig_angle: astigmatism angle (rad)
:type astig_angle: float
:param sagittal: sagittal focal length (mm)
:type sagittal: float
:param tangential: tangential focal length (mm)
:type tangential: float
"""

Config_t = namedtuple('Config_t', [
    'model',            # str
    'serial_number',    # str
    'revision',         # int
    'driver_name',      # str
    ])
"""Configuration information

:param model: Model name
:type model: str
:param serial_number: Serial number
:type serial_number: str
:param revision: Revision number
:type revision: str
:param driver_name: Driver name
:type driver_name: str
"""

HasoSpecs_t = namedtuple('HasoSpecs_t', [
    'nb_subapertures',  # uint2D
    'ulens_step',       # float2D
    'align_pos',        # float2D
    'radius',           # float
    'radius_tolerance', # int
    'tilt_limit',       # float
    'start_subpup',     # uint2D
    'black_subpup',     # uint2D
    'micro_lens_focal', # float
    'internal_options', # str
    'software_info',    # str 
    'sdk_info'          # str      
    ])
""" Haso Specifications

:param nb_subapertures: Sensor size (width, height)
:type nb_subapertures: uint2D
:param ulens_step: Step x and y between the subpupils (um)
:type ulens_step: float2D
:param align_pos: Reference (in pixels), used when aligning sensor
:type align_pos: float2D
:param radius: Not used for the time being.
:type radius: float
:param radius_tolerance: Tolerance (in pixels) that indicates that the sensor is aligned.
:type radius_tolerance: int
:param tilt_limit: Not used for the time being
:type tilt_limit: float
:param start_subpup: Coordinates of the first calculated sub-aperture, in subaperture index
:type start_subpup: uint2D
:param black_subpup: Position of the black subpupil, in subaperture index
:type black_subpup: uint2D
:param micro_lens_focal: Position of the black subpupil, in subaperture index
:type micro_lens_focal: float
:param internal_options: List of the allowed options
:type internal_options: str
:param software_info: List of the allowed specific software functionnalities
:type software_info: str
:param sdk_info: List of the allowed specific SDK functionnalities
:type sdk_info: str
"""

CorrDataManagerSpecs_t = namedtuple('CorrDataManagerSpecs_t', [  
    'haso_serial_number',   # str
    'dimensions',           # dimensions
    'wfc_serial_number',    # str  
    'wfc_name',             # str
    'nb_actuators'          # int
    ])
""" CorrDataManager Specifications

:param haso_serial_number: Sensor size (width, height)
:type haso_serial_number: str
:param dimensions: HasoSlopes size (pixels) and sampling step (um)
:type dimensions: dimensions
:param wfc_serial_number: Wavefront corrector serial number
:type wfc_serial_number: str
:param wfc_name: Wavefront corrector pretty name
:type wfc_name: str
:param nb_actuators: Wavefront corrector actuators count
:type nb_actuators: int
"""

CorrectorSpecs_t = namedtuple('CorrectorSpecs_t', [
    'actuators_count',          # int
    'stabilization_delay',      # int
    'has_internal_memory'       # bool    
    ])
""" WavefrontCorrector Specifications

:param actuators_count: Wavefront corrector actuators count
:type actuators_count: int
:param stabilization_delay: Wavefront corrector actuators stabilization delay
:type stabilization_delay: int
:param has_internal_memory: True if Wavefront corrector electronic can store its position, else false
:type has_internal_memory: bool
"""

CameraSpecs_t = namedtuple('CameraSpecs_t', [
    'dimensions',       # dimensions     
    'bits_depth',       # int
    'is_signed',        # bool
    'max_level',        # int
    'max_frame_rate',   # float
    'roi_size',         # uint2D
    'roi_offset'        # int2D
    ])
""" Camera Specifications

:param dimensions: Sensor size (width, height) in pixels and steps (x,y) between pixels in um 
:type dimensions: dimensions
:param bits_depth: Image dynamic (number of bytes : 1 for 8 bits, 2 for 16 bits ...)
:type bits_depth: int
:param is_signed: Indicates if image data is signed
:type is_signed: bool
:param max_level: Sensor saturation level (grey level)
:type max_level: int
:param max_frame_rate: Camera maximum framerate (image / s)
:type max_frame_rate: float
:param roi_size: Region of interest size (width, height)
:type roi_size: uint2D
:param roi_offset: Region of interest top-left corner (x,y) coordinates
:type roi_offset: int2D
"""

WaveLength_t = namedtuple('WaveLength_t', [
    'wavelength',       # float
    'lower_calib_wl',   # float
    'upper_calib_wl'    # float
    ])
""" Wavelength information

:param wavelength: Source wavelength (nm)
:type wavelength: float
:param lower_calib_wl: Lower calibration wavelength (nm)
:type lower_calib_wl: float
:param upper_calib_w1: Upper calibration wavelength (nm)
:type upper_calib_wl: float
"""

AcquisitionInfo_t = namedtuple('AcquisitionInfo_t', [
    'exposure_requested',   # uint
    'exposure_applied',     # uint
    'gain',                 # float
    'nb_summed_images',     # uint
    'trigger_type'          # str
    ])
""" Acquisition information

:param exposure_requested: Exposure duration requested (us)
:type exposure_requested: uint
:param exposure_applied: Exposure duration applied (us)
:type exposure_applied: uint
:param gain: Applied gain (SI)
:type gain: float
:param nb_summed_images: Number of summed images
:type nb_summed_images: uint
:param trigger_type: Trigger mode
:type trigger_type: str
"""

Metadata_t = namedtuple('Metadata_t', [
    'comments',             # str
    'session_name',         # str
    'time_stamp',           # float
    'background_removed',   # bool
    'smearing_removed',     # bool
    'camera_serial_number', # str
    'max_level'             # int
    ])
""" Acquisition metadata

:param comments: Comments
:type comments: str
:param session_name: Session name
:type session_name: str
:param time_stamp: Timestamp
:type time_stamp: float
:param background_removed: True if the background has been removed
:type background_removed: bool
:param smearing_removed: True if the smearing has been removed
:type smearing_removed: bool
:param camera_serial_number: Camera serial number
:type camera_serial_number: str
:param max_level: Max level
:type max_level: int
"""
    
Filter_t = namedtuple('Filter_t', [
    'tilt_x',       # bool
    'tilt_y',       # bool
    'curvature',    # bool
    'astig0',       # bool
    'astig45',      # bool
    'others'        # bool
    ])
""" Optical aberration filter

:param tilt_x: x tilt filter state
:type tilt_x: bool
:param tilt_y: y tilt filter state
:type tilt_y: bool
:param curvature: curvature filter state
:type curvature: bool
:param astig0: 0deg astigmatism filter state
:type astig0: bool
:param astig45: 45deg astigmatism filter state
:type astig45: bool
:param others: all others aberrations filter state
:type others: bool
"""
    
Modulator_t = namedtuple('Modulator_t', [
    'tilt_x',       # float
    'tilt_y',       # float
    'curvature',    # float
    'astig0',       # float
    'astig45',      # float
    'others'        # float
    ])
""" Optical aberration modulator

:param tilt_x: x tilt modulation
:type tilt_x: float
:param tilt_y: y tilt modulation
:type tilt_y: float
:param curvature: curvature modulation
:type curvature: float
:param astig0: 0deg astigmatism modulation
:type astig0: float
:param astig45: 45deg astigmatism modulation
:type astig45: float
:param others: all others aberrations modulation
:type others: float
"""

Statistics_t = namedtuple('Statistics_t', [
    'rms',  # float
    'pv',   # float
    'max',  # float
    'min'   # float
    ])
""" Statistics

:param rms: root mean square deviation
:type rms: float
:param pv: peak to valley
:type pv: float
:param max: maximum
:type max: float
:param min: minimum
:type min: float
"""

SoftwareConfig_t = namedtuple('SoftwareConfig_t', [
    'has_haso_soft',            # bool
    'has_waveview_lite_soft',   # bool
    'has_casao_soft',           # bool
    'has_pharao_soft',          # bool
    'has_psf_soft',             # bool
    'has_mtf_soft',             # bool
    'has_msq_soft'              # bool
    ])