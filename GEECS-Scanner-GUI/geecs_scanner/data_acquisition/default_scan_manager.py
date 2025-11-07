"""
Moddule designed to be used to enable and run Scan Manager without the GUI.

This is accomplished by loading in default options and timing configurations
specified to each experiment.  An example case is given at the bottom.

TODO Move or copy the `if __name__ == "__main__" block to a test case
"""

from __future__ import annotations
from pathlib import Path
import yaml

from geecs_scanner.data_acquisition import ScanManager
from geecs_scanner.utils import ApplicationPaths


def get_default_scan_manager(experiment: str) -> ScanManager:
    """Returns a default instance of Scan Manager for the given experiment name."""
    defaults = {
        "Undulator": {
            "shot_control_config": "HTU-Normal.yaml",
            "options": {
                "rep_rate_hz": 1,
                "Save Hiatus Period (s)": "",
                "On-Shot TDMS": False,
                "Master Control IP": "192.168.7.203",
            },
        }
    }
    default_settings = defaults["Undulator"]

    app_paths = ApplicationPaths(experiment="Undulator")
    shot_control = app_paths.exp_shot_control / default_settings["shot_control_config"]

    # shot_control = (Path(__file__).parents[2] / "scanner_configs" / "experiments" / experiment
    #                 / "shot_control_configurations" / default_settings["shot_control_config"])

    with open(shot_control) as file:
        shot_control_information = yaml.safe_load(file)
    default_options = default_settings["options"]

    return ScanManager(
        experiment_dir=experiment,
        shot_control_information=shot_control_information,
        options_dict=default_options,
    )


if __name__ == "__main__":
    """This is a demonstration of how to run a scan using only code.

    Here we run a noscan for 5 shot on TC_Phosphor"""

    manager = get_default_scan_manager("Undulator")
    config_filename = (
        Path(__file__).parents[2]
        / "scanner_configs"
        / "experiments"
        / "Undulator"
        / "save_devices"
        / "UC_TC_Phosphor.yaml"
    )
    success = manager.reinitialize(config_path=config_filename)

    if success:
        scan_config = {
            "device_var": "noscan",
            "start": 0,
            "end": 0,
            "step": 1,
            "wait_time": 5.5,
            "additional_description": "Testing out new python data acquisition module",
        }

        manager.start_scan_thread(scan_config=scan_config)

        print("Wait in infinite loop while scan manager works")
        while manager.is_scanning_active():
            pass
        print("Finished!")
