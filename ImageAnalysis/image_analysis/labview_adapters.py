import numpy as np

def HiResMagSpec_LabView(image,background=None):
    """
    Mon 8-7-2023

    A wrapper function to call my Hi Res Mag Spec Analysis in the framework of the Labview code.

    Inputs an image and outputs a list of doubles. For the "_LabView" function.

    The "_Dictionary" function is the same, just the list of doubles is instead a dictionary

    All constants are defined in the function.

    The imports a bit ugly, but haven't had a chance to debug how paths work in the LabView implementation.  Works though.

    @ Chris
    """
    
    #import the analysis class for the specific camera device instance
    # from analyzers.U_HiResMagCam.U_HiResMagSpec import U_HiResMagSpecImageAnalyzer as U_HiResMagSpecImageAnalyzer
    from image_analysis.analyzers.U_HiResMagCam.U_HiResMagSpec import U_HiResMagSpecImageAnalyzer

    returned_image, MagSpecDict, inputParams, lineouts = U_HiResMagSpecImageAnalyzer().analyze_image(image)
   
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
            "Beam-Intercept-100MeV",
            "Optimization-Factor"
        ]

    values = np.array([MagSpecDict[key] for key in keys_of_interest]).astype(np.float64)
    
    # result=(returned_image, values, np.zeros((2, 2), dtype=np.float64))
    result=(returned_image, values, np.zeros((2, 2), dtype=np.float64))

    
    return result
    
# Dictionary to map device types to their respective analysis functions
DEVICE_FUNCTIONS = {
    "UC_TestCam": HiResMagSpec_LabView,
    # Add more device types as needed...
}
 
def analyze_labview_image(device_type, image, background):
    """
    Main function to analyze an image based on the device type.

    Parameters:
    -----------
    device_type : str
        Type of the device, e.g., "UC_TestCam". This is used to 
        select out the specific device 
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

    """
    
    func= DEVICE_FUNCTIONS.get(device_type)
    if func:
        result=func(image,background)
        return result
    else:
        raise ValueError(f"Unknown device type: {device_type}")
