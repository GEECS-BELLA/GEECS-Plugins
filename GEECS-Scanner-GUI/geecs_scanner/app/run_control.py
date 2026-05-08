"""Bridge between the GEECS Scanner GUI and the scan engine."""

import logging
from pathlib import Path
from typing import Optional

from geecs_scanner.app.lib.action_control import ActionControl
from geecs_scanner.app.lib.gui_utilities import read_yaml_file_to_dict
from geecs_scanner.engine.models.scan_execution_config import ScanExecutionConfig
from geecs_scanner.engine.scan_manager import ScanManager, get_database_dict
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceInstantiationError,
)


class RunControl:
    """Interface class between the GEECS Scanner GUI and the scan engine."""

    def __init__(
        self,
        experiment_name: str = "",
        shot_control_configuration: Optional[Path] = None,
    ):
        """Initialize a ScanManager using the given experiment information.

        Parameters
        ----------
        experiment_name : str
            Experiment name as in the GEECS Database and Scan Manager file structure.
        shot_control_configuration : Path, optional
            Path to the configuration file with shot control information.
        """
        # TODO check if this is still necessary, given GEECSScanner is skipping initialization already if expt is None
        if experiment_name == "" or shot_control_configuration is None:
            logging.warning("Specify experiment name and shot control configuration")
            raise ValueError
        else:
            settings = read_yaml_file_to_dict(shot_control_configuration)
            try:
                self.scan_manager = ScanManager(
                    experiment_dir=experiment_name, shot_control_information=settings
                )

                self.action_control = ActionControl(experiment_name=experiment_name)
            except GeecsDeviceInstantiationError as e:
                logging.error(f"GeecsDeviceInstantiationError: {e.message}")
                raise ConnectionError(e.message)

        self.is_in_setup = False
        self.is_in_stopping = False

    def get_database_dict(self) -> dict:
        """Return the dictionary of the entire database stored in Scan Manager."""
        if self.scan_manager is None:
            return {}
        else:
            return get_database_dict()

    def get_action_control(
        self, experiment_name_refresh: Optional[str] = None
    ) -> ActionControl:
        """Return the action control instance associated with the current experiment.

        Parameters
        ----------
        experiment_name_refresh : str, optional
            If provided, reinitialize the action control with this experiment name.

        Returns
        -------
        ActionControl
        """
        if experiment_name_refresh:
            self.action_control = ActionControl(experiment_name=experiment_name_refresh)
        return self.action_control

    def submit_run(self, exec_config: ScanExecutionConfig) -> bool:
        """Submit a scan request to Scan Manager after reinitializing it.

        Parameters
        ----------
        exec_config : ScanExecutionConfig
            Fully validated scan execution config produced by the GUI.

        Returns
        -------
        bool
            True if device reinitialization succeeded.
        """
        success = False
        if self.scan_manager is not None:
            self.is_in_setup = True
            success = self.scan_manager.reinitialize(exec_config=exec_config)
            self.scan_manager.start_scan_thread()
            self.is_in_setup = False
        return success

    def get_progress(self) -> int:
        """Return the current percentage of completion according to Scan Manager."""
        if self.scan_manager is not None:
            return int(self.scan_manager.estimate_current_completion() * 100)
        else:
            return 0

    def is_busy(self) -> bool:
        """Return True if the scan manager is currently in setup.

        Returns
        -------
        bool
        """
        return self.is_in_setup

    def is_active(self) -> bool:
        """Return True if a scan is currently active, False otherwise."""
        if self.scan_manager is not None:
            return self.scan_manager.is_scanning_active()
        else:
            return False

    def stop_scan(self):
        """Send a stop request to Scan Manager and flag that a stop has been received."""
        if self.scan_manager is not None:
            if not self.is_stopping():
                self.is_in_stopping = True
                self.scan_manager.stop_scanning_thread()

    def is_stopping(self) -> bool:
        """Return True if a stop scan request has recently been submitted."""
        return self.is_in_stopping

    def clear_stop_state(self):
        """Reset the state of whether Scan Manager is in the process of stopping."""
        self.is_in_stopping = False
