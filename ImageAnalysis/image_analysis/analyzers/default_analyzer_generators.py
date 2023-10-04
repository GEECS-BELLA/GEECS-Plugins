from __future__ import annotations
import os

from . import UC_GenericMagSpecCam


def return_default_hi_res_mag_cam_analyzer() -> UC_GenericMagSpecCam.UC_GenericMagSpecCamAnalyzer:
    default_input_config = r'..\config_files\default_hiresmagcam_settings.ini'
    current_directory = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(current_directory, default_input_config)
    return UC_GenericMagSpecCam.return_analyzer_from_config_file(config_filename)


def return_default_acave_mag_cam3_analyzer() -> UC_GenericMagSpecCam.UC_GenericMagSpecCamAnalyzer:
    default_input_config = r'..\config_files\default_acavemagcam3_settings.ini'
    current_directory = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(current_directory, default_input_config)
    return UC_GenericMagSpecCam.return_analyzer_from_config_file(config_filename)
