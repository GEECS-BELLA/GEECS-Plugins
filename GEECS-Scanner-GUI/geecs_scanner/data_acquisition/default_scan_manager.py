from pathlib import Path
import yaml
from geecs_scanner.data_acquisition import ScanManager


def get_default_scan_manager(experiment):
    defaults = {
        "Undulator": {
            "shot_control_config": "HTU-Normal.yaml",
            "MC_ip": "192.168.7.203",
            "options": {
                "rep_rate_hz": 1,
                "Save Hiatus Period (s)": "",
                "On-Shot TDMS": False
            }
        }
    }
    default_settings = defaults["Undulator"]
    shot_control = (Path(__file__).parents[2] / "scanner_configs" / "experiments" / experiment
                    / "shot_control_configurations" / default_settings["shot_control_config"])

    with open(shot_control) as file:
        shot_control_information = yaml.safe_load(file)
    master_control_ip = default_settings["MC_ip"]
    default_options = default_settings["options"]

    return ScanManager(experiment_dir=experiment, shot_control_information=shot_control_information,
                       options_dict=default_options, MC_ip=master_control_ip)


if __name__ == "__main__":
    get_default_scan_manager("Undulator")
