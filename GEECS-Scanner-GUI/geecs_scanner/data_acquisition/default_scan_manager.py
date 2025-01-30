from __future__ import annotations
from pathlib import Path
import yaml
from geecs_scanner.data_acquisition import ScanManager


def get_default_scan_manager(experiment) -> ScanManager:
    defaults = {
        "Undulator": {
            "shot_control_config": "HTU-Normal.yaml",
            "options": {
                "rep_rate_hz": 1,
                "Save Hiatus Period (s)": "",
                "On-Shot TDMS": False,
                "Master Control IP": "192.168.7.203"
            }
        }
    }
    default_settings = defaults["Undulator"]
    shot_control = (Path(__file__).parents[2] / "scanner_configs" / "experiments" / experiment
                    / "shot_control_configurations" / default_settings["shot_control_config"])

    with open(shot_control) as file:
        shot_control_information = yaml.safe_load(file)
    default_options = default_settings["options"]

    return ScanManager(experiment_dir=experiment,
                       shot_control_information=shot_control_information,
                       options_dict=default_options)


if __name__ == "__main__":
    manager = get_default_scan_manager("Undulator")
    config_filename = Path(__file__).parents[2] / "scanner_configs" / "experiments" / "Undulator" / "save_devices" / "UC_TC_Phosphor.yaml"
    manager.reinitialize(config_path=config_filename)

    scan_config = {
        'device_var': 'noscan',
        'start': 0,
        'end': 0,
        'step': 1,
        'wait_time': 5.5,
        'additional_description':'Testing out new python data acquisition module'}

    manager.start_scan_thread(scan_config=scan_config)

    print("Wait in infinite loop while scan manager works")
    while manager.is_scanning_active():
        pass
    print("Finished!")
