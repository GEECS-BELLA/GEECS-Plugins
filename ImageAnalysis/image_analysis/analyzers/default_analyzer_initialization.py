from __future__ import annotations
from pathlib import Path

from . import UC_GenericMagSpecCam as MagSpec
from . import UC_UndulatorExitCam as PhotonSpec
from . import UC_Amp2_IR_input as Amp2Input
from . import UC_GenericLightSpecCam as PhotonSpec


def build_config_path(config_filename):
    active_build_directory = r'Z:\software\control-all-loasis\HTU\Active Version\GEECS-Plugins\image_analysis_configs\\'
    active_config = Path(active_build_directory + config_filename)

    current_directory = Path(__file__)
    relative_config = current_directory.parents[3] / "image_analysis_configs" / config_filename
    if active_config.exists():
        return active_config
    elif relative_config.exists():
        return relative_config
    else:
        return None  # Neither worked, this config doesn't exist anywhere...


def return_default_hi_res_mag_cam_analyzer() -> MagSpec.UC_GenericMagSpecCamAnalyzer:
    return MagSpec.return_analyzer_from_config_file(build_config_path('default_hiresmagcam_settings.ini'))


def return_default_acave_mag_cam3_analyzer() -> MagSpec.UC_GenericMagSpecCamAnalyzer:
    return MagSpec.return_analyzer_from_config_file(build_config_path('default_acavemagcam3_settings.ini'))


def return_default_undulator_exit_cam_analyzer() -> PhotonSpec.UC_LightSpectrometerCamAnalyzer:
    return PhotonSpec.return_analyzer_from_config_file(build_config_path('default_undulatorexitcam_settings.ini'))


def return_default_amp2_input_analyzer() -> Amp2Input.AnalyzerAmp2Input:
    return Amp2Input.return_analyzer_from_config_file(build_config_path('default_amp2input_settings.ini'))

def return_default_undulator_rad2_cam_analyzer() -> PhotonSpec.UC_LightSpectrometerCamAnalyzer:
    return PhotonSpec.return_analyzer_from_config_file(build_config_path('default_undulatorrad2cam_settings.ini'))
