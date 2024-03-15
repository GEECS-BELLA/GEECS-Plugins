from __future__ import annotations
import os

from . import UC_GenericMagSpecCam as MagSpec
from . import UC_UndulatorExitCam as PhotonSpec
from . import UC_Amp2_IR_input as Amp2Input


def build_config_path(config_filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, (r'..\..\..\image_analysis_configs\\' + config_filename))


def return_default_hi_res_mag_cam_analyzer() -> MagSpec.UC_GenericMagSpecCamAnalyzer:
    return MagSpec.return_analyzer_from_config_file(build_config_path('default_hiresmagcam_settings.ini'))


def return_default_acave_mag_cam3_analyzer() -> MagSpec.UC_GenericMagSpecCamAnalyzer:
    return MagSpec.return_analyzer_from_config_file(build_config_path('default_acavemagcam3_settings.ini'))


def return_default_undulator_exit_cam_analyzer() -> PhotonSpec.UC_UndulatorExitCamAnalyzer:
    return PhotonSpec.return_analyzer_from_config_file(build_config_path('default_undulatorexitcam_settings.ini'))


def return_default_amp2_input_analyzer() -> Amp2Input.AnalyzerAmp2Input:
    return Amp2Input.return_analyzer_from_config_file(build_config_path('default_amp2input_settings.ini'))
