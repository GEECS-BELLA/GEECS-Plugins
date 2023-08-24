import numpy as np

class LabviewGEECSImageAnalyzer:
    """
    Base class for GEECS Image Analyzers for LabView driver. This should
    only handle things specific to using the labview driver. For example,
    ensuring the input parameters/output parameters conform to how they 
    are implemented on the labview side.
    
    """

    def __init__(self, subtract_background=False):
        self.subtract_background = subtract_background

    def preprocess_image(self, image, background=None):
        """Preprocess the image before analysis."""
        if self.subtract_background and background is not None:
            if image.shape != background.shape:
                raise ValueError("Image and background shapes do not match.")
            return image - background
        return image

    def do_analysis(self, image, background=None):
        """Virtual method to be overridden by specific device implementations."""
        processed_image = self.preprocess_image(image, background)
        raise NotImplementedError("The do_analysis method should be implemented in subclasses.")

    def validate_output(self, result):
        """
        Validates and/or provides default values for the output data.
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

class HiResMagSpec(LabviewGEECSImageAnalyzer):
    """
    Mon 8-7-2023

    A wrapper function to call my Hi Res Mag Spec Analysis in the framework of the Labview code.

    Inputs an image and outputs a list of doubles. For the "_LabView" function.

    The "_Dictionary" function is the same, just the list of doubles is instead a dictionary

    All constants are defined in the function.

    The imports a bit ugly, but haven't had a chance to debug how paths work in the LabView implementation.  Works though.

    @ Chris
    """
    def __init__(self):
        super().__init__(subtract_background=True)

    def do_analysis(self, image, background=None):
        """Perform analysis specific to HiResMagSpec."""
        processed_image = self.preprocess_image(image, background)
        
        # Local import for device-specific analysis
        from analyzers.U_HiResMagCam.U_HiResMagSpec import U_HiResMagSpecImageAnalyzer
        returned_image, MagSpecDict, _ = U_HiResMagSpecImageAnalyzer().analyze_image(processed_image)

        # Define the keys for which values need to be extracted
        keys_of_interest = [
            "Clipped-Percentage",
            "Saturation-Counts",
            "Charge-On-Camera",
            "Peak-Charge",
            "Peak-Charge-Energy",
            "Average-Energy",
            "Energy-Spread",
            "Energy-Spread-Percent",
            "Average-Beam-Size",
            "Projected-Beam-Size",
            "Beam-Tilt",
            "Beam-Intercept",
            "Beam-Intercept-100MeV"
        ]

        values = np.array([MagSpecDict[key] for key in keys_of_interest]).astype(np.float64)
        
        result=(returned_image, values, np.zeros((2, 2), dtype=np.float64))
        
        return result

# Dictionary to map device types to their respective analysis functions
DEVICE_FUNCTIONS = {
    "UC_TestCam": HiResMagSpec(),
    # Add more device types as needed...
}

def execute(device_type, image, background):
    """Main execution function to handle image analysis based on device type."""
    device = DEVICE_FUNCTIONS.get(device_type)
    
    if device:
        result = device.do_analysis(image, background)
        validated_result = device.validate_output(result)
        return validated_result[0], validated_result[1], validated_result[2]
    else:
        raise ValueError(f"Unknown device type: {device_type}")
        
# def execute(device_type, input_tuple):
    # """Main execution function to handle image analysis based on device type."""
    # device = DEVICE_FUNCTIONS.get(device_type)
    
    # if device:
        # result = device.do_analysis(np.array(input_tuple[0]), np.array(input_tuple[1]))
        # validated_result = device.validate_output(result)
        # return validated_result[0], validated_result[1], validated_result[2]
    # else:
        # raise ValueError(f"Unknown device type: {device_type}")