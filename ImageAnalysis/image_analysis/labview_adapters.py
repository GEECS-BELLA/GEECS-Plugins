import numpy as np

def hi_res_mag_spec_labview(image,background=None):
    from image_analysis.analyzers.UC_HiResMagCam import UC_HiResMagCamImageAnalyzer

    results = UC_HiResMagCamImageAnalyzer().analyze_image(image)
    return parse_mag_spec_results(results)


def acave_cam3_mag_spec_labview(image, background=None):
    from image_analysis.analyzers.UC_ACaveMagCam3 import UC_ACaveMagCam3ImageAnalyzer

    results = UC_ACaveMagCam3ImageAnalyzer().analyze_image(image)
    return parse_mag_spec_results(results)


def parse_mag_spec_results(mag_spec_results):
    returned_image = mag_spec_results['processed_image_uint16']
    mag_spec_dict = mag_spec_results['analyzer_return_dictionary']
    returned_lineouts = mag_spec_results['analyzer_return_lineouts'].astype(np.float64)
    labview_return = (
        returned_image, MagSpecDictionaryParse(mag_spec_dict), returned_lineouts)
    return labview_return


def MagSpecDictionaryParse(mag_spec_dict):
    keys_of_interest = [
        "camera_clipping_factor",
        "camera_saturation_counts",
        "total_charge_pC",
        "peak_charge_pc/MeV",
        "peak_charge_energy_MeV",
        "weighted_average_energy_MeV",
        "energy_spread_weighted_rms_MeV",
        "energy_spread_percent",
        "weighted_average_beam_size_um",
        "projected_beam_size_um",
        "beam_tilt_um/MeV",
        "beam_tilt_intercept_um",
        "beam_tilt_intercept_100MeV_um",
        "optimization_factor",
    ]
    values = np.array([mag_spec_dict[key] for key in keys_of_interest]).astype(np.float64)
    return values


def undulator_exit_cam_labview(image, background=None):
    from image_analysis.analyzers.UC_UndulatorExitCam import UC_UndulatorExitCam

    results = UC_UndulatorExitCam().analyze_image(image)
    returned_image = results['processed_image_uint16']
    spec_dict = results['analyzer_return_dictionary']
    return_lineouts = results['analyzer_return_lineouts']
    # Define the keys for which values need to be extracted
    keys_of_interest = [
        "camera_saturation_counts",
        "camera_total_intensity_counts",
        "peak_wavelength_nm",
        "average_wavelength_nm",
        "wavelength_spread_weighted_rms_nm",
        "optimization_factor",
    ]
    values = np.array([spec_dict[key] for key in keys_of_interest]).astype(np.float64)
    result = (returned_image, values, return_lineouts)

    return result

# Dictionary to map device types to their respective analysis functions
DEVICE_FUNCTIONS = {
    "UC_HiResMagCam": hi_res_mag_spec_labview,
    "UC_ACaveMagCam3": acave_cam3_mag_spec_labview,
    "UC_UndulatorExitCam": undulator_exit_cam_labview,
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
