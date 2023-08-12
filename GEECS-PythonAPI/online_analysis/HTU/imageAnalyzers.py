import numpy as np

class BaseAnalysis:
    """
    The base class for all image analysis.

    Parameters:
    -----------
    image : numpy.ndarray
        The image to be analyzed.
    background : numpy.ndarray
        The background to be subtracted or used in analysis.
    """

    def __init__(self, image, background):
        """
        Initializes the base analysis class.

        Parameters:
        -----------
        image : ndarray
            The main image data.
        background : ndarray
            The background data for the image.
        """
        array_2d = np.array(image)
        self.background = np.array(background).astype(np.float32)
        self.array_2d = array_2d.astype(np.float32)
        
        if self.use_background:
            self.working_image_array = self.array_2d - self.background
        else:
            self.working_image_array = self.array_2d

    @property
    def use_background(self):
        """
        Determines whether to use the background in the analysis.

        Returns:
        --------
        bool
            True if the background should be used, otherwise False.
        """
        return True


    def execute(self):
        """
        Executes the analysis on the given image and background. 
        This method should be overridden in child classes.

        Returns:
        --------
        tuple
            Default data as a tuple containing:
            - 2D uint16 array
            - 1D double array
            - 2D double array
        """
        return (
            np.zeros((2, 2), dtype=np.uint16),
            np.zeros(2, dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64)
        )

    def validate_output(self, result):
        """
        Validates and/or provides default values for the output data.

        Parameters:
        -----------
        result : tuple
            The result data to be validated or filled with defaults.

        Returns:
        --------
        tuple
            Validated and possibly filled result data.
        """

        # Default outputs
        default_2d_uint16 = np.zeros((2, 2), dtype=np.uint16)
        default_1d_float64 = np.zeros(2, dtype=np.float64)
        default_2d_float64 = np.zeros((2, 2), dtype=np.float64)

        if not (isinstance(result[0], np.ndarray) and result[0].dtype == np.uint16):
            result = (default_2d_uint16, result[1], result[2])

        if not (isinstance(result[1], np.ndarray) and len(result[1].shape) == 1 and result[1].dtype == np.float64):
            result = (result[0], default_1d_float64, result[2])

        if not (isinstance(result[2], np.ndarray) and len(result[2].shape) == 2 and result[2].dtype == np.float64):
            result = (result[0], result[1], default_2d_float64)

        return result

class UC_ExampleCamAnalysis(BaseAnalysis):
    """
    Analysis class specific for DeviceA.

    Extends:
    --------
    BaseAnalysis
    """
    @property
    def use_background(self):
        return True
            
    def execute(self):
        """
        Executes the device-specific analysis on the given image and background.

        Returns:
        --------
        tuple
            Analysis result data as a tuple containing:
            - 2D uint16 array
            - 1D double array
            - 2D double array
        """
        # ... your device-specific code ...

        result = (
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
        )

        # Ensure the output matches the expected format
        return self.validate_output(result)
  
    
from array import array
import HiResAnalysisModules.HiResMagSpecAnalysis as MagSpecAnalysis
class HiResMagSpecAnalysisClass(BaseAnalysis):
    """
    Analysis class specific for HiResMagSpec.

    Extends:
    --------
    BaseAnalysis
    """

    @property
    def use_background(self):
        return False

    def execute(self):
        """
        Executes the HiResMagSpec specific analysis on the given image.

        Returns:
        --------
        tuple
            Analysis result data as a tuple containing:
            - 2D uint16 array
            - 1D double array
            - 2D double array
        """
        
        # Factor to go from camera counts to pC/MeV
        # Record: July 25th, Scan 24, HiResMagSpec
        normalization_triggerdelay = 15.497208
        normalization_exposure = 0.010000
        normalization_factor = 1.1301095153900242e-06

        inputParams = {
            "Threshold-Value": 230,
            "Saturation-Value": 4095,
            "Normalization-Factor": normalization_factor,
            "Transverse-Calibration": 43,
            "Transverse-Slice-Threshold": 0.02,
            "Transverse-Slice-Binsize": 5
        }

        # You've used the variable name 'image' in the function, 
        # but in the class, you should be using 'self.working_image_array'
        MagSpecDict = MagSpecAnalysis.AnalyzeImage(self.working_image_array, inputParams)
        values = np.array(list(MagSpecDict.values()), dtype=np.float64)

        result = (
            self.working_image_array.astype(np.uint16),  # Ensure the image is uint16 type
            values,
            None  # Here, you can replace None with your 2D double array, if available
        )

        # Ensure the output matches the expected format and fill defaults if necessary
        return self.validate_output(result)

def analyze_image(device_type, image, background):
    """
    Main function to analyze an image based on the device type.

    Parameters:
    -----------
    device_type : str
        Type of the device, e.g., "DeviceA". This is used to 
        select out the specific device from the analyzer
        dictionary
    image : numpy.ndarray
        The image to be analyzed.
    background : numpy.ndarray
        The background to be subtracted or used in analysis.

    Returns:
    --------
    tuple
        Analysis result data as a tuple containing:
        - 2D uint16 array
        - 1D double array
        - 2D double array

    Raises:
    -------
    ValueError
        If the provided device type is not supported.
    """
    analyzers = {
        "UC_TestCam": HiResMagSpecAnalysisClass,
        #"DeviceB": DeviceBAnalysis,
        # ... more devices here
    }

    analyzer_class = analyzers.get(device_type)
    if not analyzer_class:
        raise ValueError(f"Unknown device type: {device_type}")

    analyzer = analyzer_class(image, background)
    return analyzer.execute()
