import numpy as np

def HiResMagSpec_LabView(image,background=None):
    from image_analysis.analyzers.U_HiResMagSpec import U_HiResMagSpecImageAnalyzer

    returned_image, mag_spec_dict, input_params, lineouts = U_HiResMagSpecImageAnalyzer().analyze_image(image)
    result = (returned_image, MagSpecDictionaryParse(mag_spec_dict), np.zeros((2, 2), dtype=np.float64))
    return result


def ACaveMagCam3_LabView(image, background=None):
    from image_analysis.analyzers.UC_ACaveMagSpec import UC_ACaveMagSpecImageAnalyzer

    returned_image, mag_spec_dict, input_params, lineouts = UC_ACaveMagSpecImageAnalyzer().analyze_image(image)
    result = (returned_image, MagSpecDictionaryParse(mag_spec_dict), np.zeros((2, 2), dtype=np.float64))
    return result


def MagSpecDictionaryParse(mag_spec_dict):
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
    values = np.array([mag_spec_dict[key] for key in keys_of_interest]).astype(np.float64)
    return values


def UndulatorExitCam_LabView(image, background=None):
    from image_analysis.analyzers.UC_UndulatorExitCam import UC_UndulatorExitCam
    returned_image, mag_spec_dict, input_params, lineouts = UC_UndulatorExitCam().analyze_image(image)

    # Define the keys for which values need to be extracted
    keys_of_interest = [
        "Saturation-Counts",
        "Photon-Counts",
        "Peak-Wavelength",
        "Average-Wavelength",
        "Wavelength-Spread",
        "Optimization-Factor"
    ]
    values = np.array([mag_spec_dict[key] for key in keys_of_interest]).astype(np.float64)
    return_lineouts = lineouts.astype(np.float64)
    result = (returned_image, values, return_lineouts)

    return result

# Dictionary to map device types to their respective analysis functions
DEVICE_FUNCTIONS = {
    "UC_HiResMagCam": HiResMagSpec_LabView,
    "UC_ACaveMagCam3": ACaveMagCam3_LabView,
    "UC_UndulatorExitCam": UndulatorExitCam_LabView,
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
