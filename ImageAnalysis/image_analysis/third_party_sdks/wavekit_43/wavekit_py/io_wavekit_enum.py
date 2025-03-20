#!/usr/bin/python

class E_COMPUTEPHASESET:
    MODAL_LEGENDRE        = 1
    MODAL_ZERNIKE         = 2
    ZONAL                 = 3
    MODAL_ZONAL_ZERNIKE   = 4
    MODAL_ZONAL_LEGENDRE  = 5

class E_ACTUATOR_CONDITIONS:
    VALID       = 0
    FIXED       = 1
    INVALID     = 2

class E_WFC_STATE:
    INIT              = 0
    STANDBY           = 1
    MOVING            = 2

class E_CAMERA_ACQUISITION_MODE:
    LAST       = 0
    NEW        = 1

class E_CAMERA_SYNCHRONIZATION_MODE:
    SYNCHRONOUS    = 0
    ASYNCHRONOUS   = 1

class E_MODAL:
    LEGENDRE    = 0
    ZERNIKE     = 1

class E_ZERNIKE_NORM:
    STD      = 0
    RMS      = 1

class E_PUPIL_DETECTION:
    FIXED_RADIUS    = 0
    FIXED_CENTER    = 1
    AUTOMATIC       = 2
    
class E_PUPIL_COVERING:
    INSCRIBED       = 0
    CIRCUMSCRIBED   = 1

class E_SPOT_DETECT:
    STANDARD  = 0
    FAST      = 1

class E_SPOT_SHAPE:
    CIRCULAR   = 0
    SQUARE     = 1

class E_TYPES:
    BOOL             = 0
    INT              = 1
    REAL             = 2
    STRING           = 3
    SLOPES           = 4
    IMAGE            = 5

class E_STITCH_POSTPROCESS:
    NONE           = 0
    REF_CURV_ERROR = 1
    THERMAL_DRIFT  = 2
