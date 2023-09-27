import numpy as np


def hi_res_mag_spec_labview(image,background=None):
    # mag_spec_name = 'hires'
    # default_input_config = 'config_files\default_hiresmagcam_settings.ini'
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # config_filename = os.path.join(current_directory, default_input_config)
    # return generic_mag_spec_cam_labview(image, mag_spec_name, config_filename, background)

    # from image_analysis.analyzers.UC_HiResMagCam import UC_HiResMagCamImageAnalyzer
    # results = UC_HiResMagCamImageAnalyzer().analyze_image(image)
    # return parse_mag_spec_results(results)

    import image_analysis.analyzers.UC_GenericMagSpecCam as mag_cam_analyzers
    analyzer = mag_cam_analyzers.return_default_hi_res_mag_cam_analyzer()
    results = analyzer.analyze_image(image)
    return parse_mag_spec_results(results)


def acave_cam3_mag_spec_labview(image, background=None):
    # from image_analysis.analyzers.UC_ACaveMagCam3 import UC_ACaveMagCam3ImageAnalyzer
    # results = UC_ACaveMagCam3ImageAnalyzer().analyze_image(image)
    # return parse_mag_spec_results(results)

    import image_analysis.analyzers.UC_GenericMagSpecCam as mag_cam_analyzers
    analyzer = mag_cam_analyzers.return_default_acave_mag_cam3_analyzer()
    results = analyzer.analyze_image(image)
    return parse_mag_spec_results(results)

"""
def generic_mag_spec_cam_labview(image, mag_spec_name, input_config_filename, background=None):
    from image_analysis.analyzers.UC_GenericMagSpecCam import UC_GenericMagSpecCamAnalyzer

    config = configparser.ConfigParser()
    config.read(input_config_filename)
    analyzer = UC_GenericMagSpecCamAnalyzer(
        mag_spec_name=mag_spec_name,
        roi_top=int(config.get('roi','top')),
        roi_bottom=int(config.get('roi', 'bottom')),
        roi_left=int(config.get('roi', 'left')),
        roi_right=int(config.get('roi', 'right')),
        noise_threshold=int(config.get('settings', 'noise_threshold')),
        edge_pixel_crop=int(config.get('settings', 'edge_pixel_crop')),
        saturation_value=int(config.get('settings', 'saturation_value')),
        normalization_factor=float(config.get('settings', 'normalization_factor')),
        transverse_calibration=float(config.get('settings', 'transverse_calibration')),
        do_transverse_calculation=bool(config.get('settings', 'do_transverse_calculation')),
        transverse_slice_threshold=float(config.get('settings', 'transverse_slice_threshold')),
        transverse_slice_binsize=int(config.get('settings', 'transverse_slice_binsize')),
        optimization_central_energy=float(config.get('settings', 'optimization_central_energy')),
        optimization_bandwidth_energy=float(config.get('settings', 'optimization_bandwidth_energy')))
    results = analyzer.analyze_image(image)

    return parse_mag_spec_results(results)
"""

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
