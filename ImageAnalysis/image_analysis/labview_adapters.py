"""Utility functions for LabVIEW adapters.

Provides functions to analyze LabVIEW images, map device types to analyzer classes,
and handle configuration and result parsing.
"""

import numpy as np
import json
from pathlib import Path
from typing import Type, NamedTuple
from .base import LabviewImageAnalyzer

from .analyzers.UC_GenericMagSpecCam import UC_GenericMagSpecCamAnalyzer as MagSpec
from .analyzers.UC_GenericLightSpecCam import (
    UC_LightSpectrometerCamAnalyzer as LightSpec,
)
from .analyzers.generic_beam_analyzer import BeamSpotAnalyzer as BeamSpot


class DeviceConfig(NamedTuple):
    """
    Configuration for a LabVIEW device.

    Attributes
    ----------
    labview_analyzer_class : Type[LabviewImageAnalyzer]
        Specific image analyzer class for the device.
    key_list_name : str
        Name of the keys dictionary in the .json configuration file.
    default_settings_filename : str
        Filename of the .ini config file in ``image_analysis_configs/``.
    """

    labview_analyzer_class: Type[LabviewImageAnalyzer]  # Specific image_analyzer class
    key_list_name: str  # name of keys dictionary in .json file
    default_settings_filename: str  # .ini config file in "image_analysis_configs/"


DEVICE_FUNCTIONS = {
    "UC_HiResMagCam": DeviceConfig(
        labview_analyzer_class=MagSpec,
        key_list_name="MagSpecCam",
        default_settings_filename="default_hiresmagcam_settings.ini",
    ),
    "UC_ACaveMagCam3": DeviceConfig(
        labview_analyzer_class=MagSpec,
        key_list_name="MagSpecCam",
        default_settings_filename="default_acavemagcam3_settings.ini",
    ),
    "UC_UndulatorExitCam": DeviceConfig(
        labview_analyzer_class=LightSpec,
        key_list_name="LightSpecCam",
        default_settings_filename="default_undulatorexitcam_settings.ini",
    ),
    "UC_UndulatorRad2": DeviceConfig(
        labview_analyzer_class=LightSpec,
        key_list_name="LightSpecCam",
        default_settings_filename="undulatorrad2cam_may2_settings.ini",
    ),
    "UC_PostUndulatorUVSpecCam": DeviceConfig(
        labview_analyzer_class=LightSpec,
        key_list_name="LightSpecCam",
        default_settings_filename="undulatoruvcam_may8_settings.ini",
    ),
    "UC_Amp2_IR_input": DeviceConfig(
        labview_analyzer_class=BeamSpot,
        key_list_name="BeamSpot",
        default_settings_filename="default_amp2input_settings.ini",
    ),
}


def analyze_labview_image(device_type, image, background):
    """
    Main function to analyze an image based on the device type.

    Parameters
    ----------
    device_type : str
        Type of the device, e.g., "UC_TestCam". This is used to
        select out the specific device
    image : numpy.ndarray
        The image to be analyzed.
    background : numpy.ndarray
        The background to be subtracted or used in analysis.

    Returns
    -------
    tuple
        Analysis result data as a tuple containing:
        - 2D uint16 array
        - 1D double array
        - 2D double array

    """
    configuration = DEVICE_FUNCTIONS.get(device_type)
    if configuration:
        analyzer = analyzer_from_device_type(device_type)
        analyzer.apply_background(background=background)
        image_float = image.astype(np.float32)
        result = analyzer.analyze_image(image_float)
        return parse_results_to_labview(result, configuration.key_list_name)
    else:
        raise ValueError(f"Unknown device type: {device_type}")


def analyzer_from_device_type(
    device_type: str, build_path_override=False
) -> LabviewImageAnalyzer:
    """
    Given the device type, return the image_analyzer with default parameters.

    Default parameters as given by the config file in the above dictionary.
    Additionally, this function can be used by outside post-analysis scripts

    Parameters
    ----------
    device_type : str
        Type of the device, e.g., "UC_TestCam". This is used to
        select out the specific device

    Returns
    -------
    LabviewImageAnalyzer
        Instance of the LabviewImageAnalyzer class as implemented for the specified device

    """
    configuration = DEVICE_FUNCTIONS.get(device_type)
    analyzer_class = configuration.labview_analyzer_class
    config_filepath = build_config_path(
        configuration.default_settings_filename, override=build_path_override
    )
    return analyzer_class().apply_config(config_filepath)


def parse_results_to_labview(return_dictionary, key_list_name):
    """
    Parse the return dictionary given by analyze_image into the format expected by Labview.

    Parameters
    ----------
    return_dictionary : dict
        The return dictionary from LabviewImageAnalyzer's analyze_image function
    key_list_name : str
        Name of the key list group as listed in labview_adapters_config.json

    Returns
    -------
    tuple
        Analysis result data as a tuple containing:
        - 2D uint16 array
        - 1D double array
        - 2D double array
    """
    return_image = return_dictionary["processed_image"]
    if return_image is not None:
        validate_uint16_castable(return_image)
        return_image = return_image.astype(np.uint16)

    scalar_dict = return_dictionary["analyzer_return_dictionary"]
    return_lineouts = return_dictionary["analyzer_return_lineouts"]

    keys_of_interest = read_keys_of_interest(key_list_name)
    return_scalars = np.array([scalar_dict[key] for key in keys_of_interest]).astype(
        np.float64
    )

    labview_return = (return_image, return_scalars, return_lineouts)
    return labview_return


def validate_uint16_castable(array):
    """
    Check if a given array could be converted to an unsigned 16 bit array.

    Raises a Type Error if not

    Parameters
    ----------
    array:
        An image that may or may not be formatted to accommodate uint16 conversion
    """
    if array is not None:
        if np.iscomplexobj(array):
            raise TypeError("Image cannot contain complex values")
        if not np.isfinite(array).all():
            raise TypeError("Array contains NaNs or infinite values.")
        if np.any((array < 0) | (array > np.iinfo(np.uint16).max)):
            raise TypeError("Array values out of uint16 range")


def read_keys_of_interest(key_list_name):
    """
    Return the list of keys in the .json file for the given group name.

    Additionally used in unit test scripts

    Parameters
    ----------
    key_list_name : str
        Name of the key list group as listed in labview_adapters_config.json

    Returns
    -------
    list
        Names of all the expected keys of the alysis return

    """
    json_filepath = Path(__file__).resolve().parent / "labview_adapters_config.json"
    with open(json_filepath, "r") as file:
        device_keys = json.load(file)
    return device_keys[key_list_name]["keys_of_interest"]


def build_config_path(config_filename, override=False):
    """
    Take the given config filename and build a Path to this file.

    First, this function checks if the config exists in the Active Version of GEECS-Plugins on the Z: drive.
    If not, the next place it checks is in the current, local version of GEECS-Plugins.  Otherwise,
    assume that the file doesn't exist yet in a proper location and return None.

    Parameters
    ----------
    config_filename : str
        String of the name of the config file

    Returns
    -------
    config filename: Path
        String that points to the config file after determining its location

    """
    active_build_directory = r"Z:\software\control-all-loasis\HTU\Active Version\GEECS-Plugins\image_analysis_configs"
    active_config = Path(active_build_directory, config_filename)

    current_directory = Path(__file__)
    relative_config = (
        current_directory.parents[2] / "image_analysis_configs" / config_filename
    )
    if active_config.exists() and not override:
        return active_config
    elif relative_config.exists():
        return relative_config
    else:
        return None  # Neither worked, this config doesn't exist anywhere...
