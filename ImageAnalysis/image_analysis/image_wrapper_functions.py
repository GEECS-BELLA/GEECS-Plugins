import numpy as np

def HiResMagSpec_LabView(image):
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
    from analyzers.U_HiResMagCam.U_HiResMagSpec import U_HiResMagSpecImageAnalyzer as U_HiResMagSpecImageAnalyzer
    returned_image, MagSpecDict, inputParams = U_HiResMagSpecImageAnalyzer().analyze_image(image)
   
    values = np.array([
                MagSpecDict["Clipped-Percentage"],
                MagSpecDict["Saturation-Counts"],
                MagSpecDict["Charge-On-Camera"],
                MagSpecDict["Peak-Charge"],
                MagSpecDict["Peak-Charge-Energy"],
                MagSpecDict["Average-Energy"],
                MagSpecDict["Energy-Spread"],
                MagSpecDict["Energy-Spread-Percent"],
                MagSpecDict["Average-Beam-Size"],
                MagSpecDict["Projected-Beam-Size"],
                MagSpecDict["Beam-Tilt"],
                MagSpecDict["Beam-Intercept"],
                MagSpecDict["Beam-Intercept-100MeV"]
            ])
   
    return (returned_image, list(values))
    
    
def execute(device_type, image, background):
    """
    Main function to analyze an image based on the device type.

    Parameters:
    -----------
    device_type : str
        Type of the device, e.g., "UC_TestCam". This is used to 
        select out the specific device using if/else statement.
        to do: switch to use a dictionary
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
    if device_type=="UC_TestCam":
        result=HiResMagSpec_LabView(image)
        return result[0],result[1],np.zeros((2, 2), dtype=np.float64)
