import numpy as np
import image_analysis.analyzers.default_analyzer_initialization as default_analyzers


def hi_res_mag_spec_labview(image, background=None):
    results = default_analyzers.return_default_hi_res_mag_cam_analyzer().analyze_image(image)
    return parse_mag_spec_results(results)


def acave_cam3_mag_spec_labview(image, background=None):
    results = default_analyzers.return_default_acave_mag_cam3_analyzer().analyze_image(image)
    return parse_mag_spec_results(results)


def parse_mag_spec_results(mag_spec_results):
    returned_image = mag_spec_results['processed_image_uint16']
    mag_spec_dict = mag_spec_results['analyzer_return_dictionary']
    returned_lineouts = mag_spec_results['analyzer_return_lineouts'].astype(np.float64)
    labview_return = (
        returned_image, mag_spec_dictionary_parse(mag_spec_dict), returned_lineouts)
    return labview_return


def mag_spec_dictionary_parse(mag_spec_dict):
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
        "fwhm_percent",
    ]
    values = np.array([mag_spec_dict[key] for key in keys_of_interest]).astype(np.float64)
    return values


def undulator_exit_cam_labview(image, background=None):
    results = default_analyzers.return_default_undulator_exit_cam_analyzer().analyze_image(image)
    return parse_light_spec_results(results)


def undulator_rad2_labview(image, background=None):
    results = default_analyzers.return_default_undulator_rad2_cam_analyzer().analyze_image(image)
    return parse_light_spec_results(results)


def parse_light_spec_results(light_spec_results):
    returned_image = light_spec_results['processed_image_uint16']
    spec_dict = light_spec_results['analyzer_return_dictionary']
    return_lineouts = light_spec_results['analyzer_return_lineouts']
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
    labview_return = (returned_image, values, return_lineouts)
    return labview_return


def amp2_input_labview(image, background=None):

    # run analysis and extract results
    results = default_analyzers.return_default_amp2_input_analyzer().analyze_image(image)
    returned_image = results['processed_image_uint16']
    results_dict = results['analyzer_return_dictionary']

    # parse output and prep for sfile
    keys_of_interest = ['centroidx', 'centroidy']
    values = np.array([results_dict[key]
                       for key in keys_of_interest]).astype(np.float64)
    output = (returned_image, values)

    return output


# Dictionary to map device types to their respective analysis functions
DEVICE_FUNCTIONS = {
    "UC_HiResMagCam": hi_res_mag_spec_labview,
    "UC_ACaveMagCam3": acave_cam3_mag_spec_labview,
    "UC_UndulatorExitCam": undulator_exit_cam_labview,
    "UC_Amp2_IR_input": amp2_input_labview,
    "UC_UndulatorRad2": undulator_rad2_labview,
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

    func = DEVICE_FUNCTIONS.get(device_type)
    if func:
        result = func(image, background)
        return result
    else:
        raise ValueError(f"Unknown device type: {device_type}")
