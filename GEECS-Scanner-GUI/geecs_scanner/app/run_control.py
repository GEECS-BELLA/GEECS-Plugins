import logging
import yaml
from pathlib import Path
from typing import Optional
from geecs_scanner.data_acquisition.scan_manager import ScanManager, get_database_dict


class RunControl:
    """
    Interface class between the GEECS Scanner GUI and the Scan Manager that controls the scan execution
    """
    def __init__(self, experiment_name: str = "",
                 shot_control_configuration: Optional[Path] = None):
        """
        Initializes ScanManager instance using the given experiment information.

        :param experiment_name: Experiment name as in the GEECS Database and Scan Manager file structure
        :param shot_control_configuration: Path to the configuration file with shot control information
        """
        # TODO check if this is still necessary, given GEECSScanner is skipping initialization already if expt is None
        if experiment_name == "" or shot_control_configuration is None:
            logging.warning("Specify experiment name and shot control configuration")
            raise ValueError
        else:
            with open(shot_control_configuration, 'r') as file:
                settings = yaml.safe_load(file)
            self.scan_manager = ScanManager(experiment_dir=experiment_name,
                                            shot_control_information=settings)

        self.is_in_setup = False
        self.is_in_stopping = False

    def get_database_dict(self) -> dict:
        """Returns the dictionary of the entire database, which is stored in Scan Manager"""
        if self.scan_manager is None:
            return {}
        else:
            return get_database_dict()

    def submit_run(self, config_dictionary: dict, scan_config: dict):
        """Submits a scan request to Scan Manager after reinitializing it

        :param config_dictionary: Dictionary of devices to be saved and actions to take
        :param scan_config: Dictionary of parameters used in a 1d scan
        """
        if self.scan_manager is not None:
            self.is_in_setup = True

            success = self.scan_manager.reinitialize(config_path=None, config_dictionary=config_dictionary)
            if success:
                self.scan_manager.start_scan_thread(scan_config=scan_config)

                self.is_in_setup = False

    def get_progress(self) -> int:
        """
        :return:  Current percentage of completion according to Scan Manager
        """
        if self.scan_manager is not None:
            return int(self.scan_manager.estimate_current_completion()*100)
        else:
            return 0

    def is_busy(self) -> bool:
        """ TODO should check if this actually does anything...
        :return: True if it is in setup, false if not.
        """
        return self.is_in_setup

    def is_active(self) -> bool:
        """
        :return: True if a scan is currently active, False otherwise
        """
        if self.scan_manager is not None:
            return self.scan_manager.is_scanning_active()
        else:
            return False

    def stop_scan(self):
        """Sends a request to Scan Manager to stop the current scan and flags that a stop request has been received
        """
        if self.scan_manager is not None:
            if not self.is_stopping():
                self.is_in_stopping = True
                self.scan_manager.stop_scanning_thread()

    def is_stopping(self) -> bool:
        """
        :return: True if a stop scan request has recently been submitted, False otherwise
        """
        return self.is_in_stopping

    def clear_stop_state(self):
        """Reset the state of if Scan Manager is in the process of stopping."""
        self.is_in_stopping = False
