"""
ScanManager module for GEECS Scanner data acquisition.

This module defines the `ScanManager` class, the central coordinator for
executing scans in the GEECS system. It encapsulates configuration loading,
device orchestration, data logging, trigger control, and optional optimizer
integration. Scan execution is thread-based, enabling pause/resume and safe
stop operations from external GUIs.

Key Responsibilities
--------------------
- Load or receive `ScanConfig` objects and pre-compute scan step sequences
- Configure, synchronize, and restore GEECS devices via `DeviceManager`
- Control shot-trigger hardware through `ScanDevice` interfaces
- Manage data acquisition and file movement via `DataLogger` / `ScanDataManager`
- Support three scan modes:
    * STANDARD      (regular variable sweep)
    * NOSCAN        (single-shot statistics collection)
    * OPTIMIZATION  (feedback-driven scans with a pluggable optimizer)
- Provide thread-safe controls for start, stop, pause, and resume
- Estimate total acquisition time and real-time completion

Dependencies
------------
Standard Library
    threading, time, logging, importlib, warnings, pathlib.Path, typing
Third-Party
    pandas
Internal Modules
    geecs_scanner.data_acquisition:
        • DeviceManager
        • DataLogger
        • ActionManager
        • ScanDataManager
        • ScanStepExecutor
        • utils.ConsoleLogger
    geecs_scanner.optimization.base_optimizer.BaseOptimizer
    geecs_data_utils (ScanConfig, ScanMode)
    geecs_python_api.controls.devices.scan_device.ScanDevice
    geecs_python_api.controls.interface.geecs_errors.GeecsDeviceInstantiationError

Examples
--------
>>> from geecs_scanner.data_acquisition import ScanManager
>>> scan_mgr = ScanManager(
...     experiment_dir=r"C:/Experiments/UndulatorTest",
...     shot_control_information={
...         "device": "MasterControl",
...         "variables": {"Trigger": {"SCAN": 4.0, "OFF": 0.5}}
...     }
... )
>>> my_scan = ScanConfig(
...     device_var="Undulator:gap",
...     start=10,
...     end=15,
...     step=0.5,
...     wait_time=1.0,
...     scan_mode=ScanMode.STANDARD
... )
>>> scan_mgr.reinitialize(config_dictionary={"options": {"rep_rate_hz": 1}})
>>> scan_mgr.initialization_success
True
>>> scan_mgr.start_scan_thread(my_scan)

Notes
-----
All time-critical operations (trigger handling, device polling, optimizer
feedback) are executed in secondary threads to avoid blocking the main GUI
loop. Any method marked as *thread-safe* can be invoked from an external
thread without additional locks.
"""

from __future__ import annotations

# Standard library imports
from typing import Optional, Dict, Any, Union, List
import time
import threading
import logging
import importlib
from pathlib import Path
import warnings

# Third-party library imports
import pandas as pd

# Internal project imports
from . import (
    DeviceManager,
    ActionManager,
    DataLogger,
    DatabaseDictLookup,
    ScanDataManager,
    ScanStepExecutor,
)
from geecs_scanner.optimization.base_optimizer import BaseOptimizer

from geecs_data_utils import ScanConfig, ScanMode  # Adjust the path as needed
from dataclasses import fields

from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceInstantiationError,
)

logger = logging.getLogger(__name__)

database_dict = DatabaseDictLookup()


def get_database_dict():
    """
    Retrieve the current database dictionary.

    Returns
    -------
    dict
        The internal dictionary stored in the `DatabaseDictLookup` instance.
    """
    return database_dict.get_database()


class ScanManager:
    """Manage the execution of scans in the GEECS system.

    This class coordinates all aspects of a scan, including device configuration,
    data acquisition, and logging. It handles different scan modes (standard, noscan,
    optimization) and provides thread-based control for pausing/resuming scans.

    Attributes
    ----------
    device_manager : DeviceManager
        Manager for configuring and controlling devices
    action_manager : ActionManager
        Handles pre/post-scan actions
    optimizer : BaseOptimizer or None
        Optimizer for optimization scans
    initialization_success : bool
        Flag indicating successful initialization
    scan_config : ScanConfig
        Current scan configuration

    Methods
    -------
    start_scan_thread(scan_config)
        Start a new scan in a separate thread
    stop_scanning_thread()
        Stop the currently running scan
    pause_scan()
        Pause the scanning process
    resume_scan()
        Resume the paused scan
    """

    def __init__(
        self,
        experiment_dir: str,
        shot_control_information: dict,
        options_dict: Optional[dict] = None,
        device_manager=None,
        scan_data=None,
    ):
        """Initialize the ScanManager with experiment settings and device configuration.

        Parameters
        ----------
        experiment_dir : str
            Directory where experiment data is stored
        shot_control_information : dict
            Dictionary containing shot control device information
        options_dict : dict, optional
            Additional options for scan configuration
        device_manager : DeviceManager, optional
            Pre-initialized device manager
        scan_data : ScanData, optional
            Pre-initialized scan data manager
        """
        database_dict.reload(experiment_name=experiment_dir)
        self.device_manager = device_manager or DeviceManager(
            experiment_dir=experiment_dir
        )
        self.action_manager = ActionManager(experiment_dir=experiment_dir)
        self.optimizer: Optional[BaseOptimizer] = None
        self.initialization_success = False

        self.MC_ip = ""

        # Initialize ScanDataManager with device_manager and scan_paths
        self.scan_data_manager = ScanDataManager(
            self.device_manager, scan_data, database_dict
        )

        self.data_logger = DataLogger(
            experiment_dir, self.device_manager
        )  # Initialize DataLogger
        self.save_data = True

        self.shot_control: Optional[ScanDevice] = None
        self.shot_control_variables = None
        shot_control_device = shot_control_information.get("device", None)
        if shot_control_device:
            self.shot_control = ScanDevice(shot_control_information["device"])
            self.shot_control_variables = shot_control_information["variables"]

        self.results = {}  # Store results for later processing

        self.stop_scanning_thread_event = (
            threading.Event()
        )  # Event to signal the logging thread to stop

        self.virtual_variable_list = []
        self.virtual_variable_name = None

        self.acquisition_time = 0

        self.scanning_thread = None  # NEW: Separate thread for scanning

        self.scan_step_start_time = 0
        self.scan_step_end_time = 0

        self.initial_state = None
        self.scan_steps = []  # To store the precomputed scan steps

        self.pause_scan_event = threading.Event()  # Event to handle scan pausing
        self.pause_scan_event.set()  # Set to 'running' by default
        self.pause_time = 0

        self.options_dict: dict = {} if options_dict is None else options_dict
        self.save_local = True  # If true, will save locally on device PC before being queued to transfer to network

        self.scan_config: ScanConfig

        self.executor = ScanStepExecutor(
            device_manager=self.device_manager,
            data_logger=self.data_logger,
            scan_data_manager=self.scan_data_manager,
            optimizer=self.optimizer,
            shot_control=self.shot_control,
            options_dict=self.options_dict,
            stop_scanning_thread_event=self.stop_scanning_thread_event,
            pause_scan_event=self.pause_scan_event,
        )
        self.executor.trigger_on_fn = self.trigger_on
        self.executor.trigger_off_fn = self.trigger_off

    def pause_scan(self):
        """
        Pause the scanning process by clearing the pause event.

        Notes
        -----
        Clears the `pause_scan_event` to pause scanning.
        """
        if self.pause_scan_event.is_set():
            self.pause_scan_event.clear()
            logging.info("Scanning paused.")

    def resume_scan(self):
        """
        Resume the scanning process by setting the pause event.

        Notes
        -----
        Sets the `pause_scan_event` to resume scanning.
        """
        if not self.pause_scan_event.is_set():
            self.pause_scan_event.set()
            logging.info("Scanning resumed.")

    def reinitialize(
        self, config_path=None, config_dictionary=None, scan_data=None
    ) -> bool:
        """
        Reinitialize the ScanManager with new configurations and reset the logging system.

        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file.
        config_dictionary : dict, optional
            Dictionary containing configuration settings.
        scan_data : ScanData, optional
            If given, scan_data_manager will use an alternative scan folder.

        Returns
        -------
        bool
            True if successful and all devices connected, False otherwise.

        Raises
        ------
        GeecsDeviceInstantiationError
            If device reinitialization fails during initialization of device manager.
        """
        self.initial_state = None
        self.initialization_success = False

        try:
            self.device_manager.reinitialize(
                config_path=config_path, config_dictionary=config_dictionary
            )
        except GeecsDeviceInstantiationError as e:
            logging.error(
                f"Device reinitialization failed during initialization of device manager. check "
                f"that all devices are on and available: {e}"
            )
            # Optionally, notify the user via a UI pop-up or other mechanism here
            # Then, you can abort reinitialization:
            return False

        self.scan_data_manager = ScanDataManager(
            self.device_manager, scan_data, database_dict
        )

        if config_dictionary is not None and "options" in config_dictionary:
            self.options_dict = config_dictionary["options"]
            self.save_local = not self.options_dict.get("Save Direct on Network", False)

        new_mc_ip = self.options_dict.get("Master Control IP", "")
        if self.shot_control and new_mc_ip and self.MC_ip != new_mc_ip:
            self.MC_ip = new_mc_ip
            self.enable_live_ECS_dump(client_ip=self.MC_ip)

        self.data_logger.reinitialize_sound_player(options=self.options_dict)
        self.data_logger.last_log_time_sync = {}
        self.data_logger.update_repetition_rate(self.options_dict.get("rep_rate_hz", 1))
        self.console_logger.stop_logging()
        self.console_logger.setup_logging()

        self.initialization_success = True
        return self.initialization_success

    def _set_trigger(self, state: str):
        """
        Set the trigger state and update variables accordingly.

        Can be specified using Timing Setup in the GUI.

        Parameters
        ----------
        state : str
            Either 'OFF', 'SCAN', 'STANDBY', or 'SINGLESHOT'. Used for when
            trigger is off, or during/outside a scan.

        Returns
        -------
        list
            List of results from setting trigger variables.

        Notes
        -----
        Valid states are: 'OFF', 'SCAN', 'STANDBY', 'SINGLESHOT'.
        """
        if self.shot_control is None or self.shot_control_variables is None:
            logging.info(f"No shot control device, skipping 'set state {state}'")
            return

        valid_states = ["OFF", "SCAN", "STANDBY", "SINGLESHOT"]
        results = []

        if state in valid_states:
            for variable in self.shot_control_variables.keys():
                variable_settings = self.shot_control_variables[variable]
                if set_value := variable_settings.get(state, ""):
                    results.append(self.shot_control.set(variable, set_value))
                    logging.info(f"Setting {variable} to {set_value}")
            logging.info(f"Trigger turned to state {state}.")
        else:
            logging.error(f"Invalid trigger state: {state}")
        return results

    def trigger_off(self):
        """
        Turn off the trigger by setting state to 'OFF'.

        Notes
        -----
        Calls `_set_trigger` with 'OFF' state to disable the trigger.
        """
        self._set_trigger("OFF")

    def trigger_on(self):
        """
        Turn on the trigger by setting state to 'SCAN'.

        Notes
        -----
        Calls `_set_trigger` with 'SCAN' state to enable the trigger.
        """
        self._set_trigger("SCAN")

    def is_scanning_active(self):
        """
        Check if a scan is currently active.

        Returns
        -------
            bool: True if scanning is active, False otherwise.
        """
        return self.scanning_thread and self.scanning_thread.is_alive()

    def start_scan_thread(self, scan_config: Union[ScanConfig, dict] = None) -> None:
        """Start a new scan in a separate thread.

        This allows the scan to be interrupted externally using the
        stop_scanning_thread method.

        Parameters
        ----------
        scan_config : ScanConfig or dict, optional
            Configuration settings for the scan, including variables, start, end,
            step, and wait times. If a dict is provided, it will be converted to
            a ScanConfig object (with a deprecation warning).

        Notes
        -----
        The scan runs in a separate thread, which allows it to be paused, resumed,
        or stopped externally during execution.
        """
        if not self.initialization_success:
            logging.error(
                "Initialization unsuccessful, cannot start a new scan session"
            )
            return

        if self.is_scanning_active():
            logging.warning(
                "Scanning is already active, cannot start a new scan session."
            )
            return

        # Backward compatibility: allow dict input, with warning
        if isinstance(scan_config, dict):
            valid_keys = {f.name for f in fields(ScanConfig)}
            unknown_keys = set(scan_config) - valid_keys
            if unknown_keys:
                logging.warning(
                    f"Unexpected keys in scan_config dict: {unknown_keys} — they will be ignored."
                )

            warnings.warn(
                "Passing scan_config as a dict is deprecated. Please migrate to using ScanConfig dataclass.",
                DeprecationWarning,
            )
            scan_config = ScanConfig(
                **{k: v for k, v in scan_config.items() if k in valid_keys}
            )

        self.scan_config = scan_config

        # Ensure the stop event is cleared before starting a new session
        self.stop_scanning_thread_event.clear()

        # Start a new thread for logging
        logging.info(f"Scan config: {self.scan_config}")
        self.scanning_thread = threading.Thread(target=self._start_scan)
        self.scanning_thread.start()
        logging.info("Scan thread started.")

    def stop_scanning_thread(self):
        """
        Stop the currently running scanning thread and clean up resources.

        This method sets an internal event to halt the scanning thread, waits for the thread
        to finish, and then disposes of it. If no scanning thread is active, a warning is logged.

        Notes
        -----
        This method is safe to call even if scanning is not active. It performs no action
        in that case except logging a warning.
        """
        if not self.is_scanning_active():
            logging.warning("No active scanning thread to stop.")
            return

        # Stop the scan and wait for the thread to finish
        logging.info("Stopping the scanning thread...")

        # Set the event to signal the logging loop to stop
        logging.info("Stopping the scanning thread...")
        self.stop_scanning_thread_event.set()

        self.scanning_thread.join()  # Wait for the thread to finish
        self.scanning_thread = None  # Clean up the thread
        logging.info("scanning thread stopped and disposed.")

    def _start_scan(self) -> pd.DataFrame:
        """
        Start and execute a scan using the current scan configuration.

        This method performs the full scan lifecycle, including pre-scan setup,
        acquisition time estimation, data logging, device synchronization,
        and execution of the scan steps. Upon completion or error, the scan is
        finalized and cleaned up.

        Requires
        --------
        self.scan_config : ScanConfig
            Must be set before calling this method.
        self.initialization_success : bool
            Must be True to begin the scan.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the logged scan data. If the scan fails early,
            the DataFrame may be empty.

        Raises
        ------
        Exception
            Logs and suppresses any exceptions raised during scan execution.
        """
        if not self.initialization_success:
            logging.error(
                "initialization unsuccessful, cannot start a new scan session"
            )
            return

        log_df = pd.DataFrame()  # Initialize in case of early exit

        try:
            # Pre-logging setup: Trigger devices off, initialize data paths, etc.
            logging.info(
                f"scan config getting sent to pre logging is this: {self.scan_config}"
            )

            self.pre_logging_setup()

            # Estimate acquisition time if necessary
            if self.scan_config:
                self.estimate_acquisition_time()
                logging.info(
                    f"Estimated acquisition time based on scan config: {self.acquisition_time} seconds."
                )

            # Start data logging
            if self.save_data:
                logging.info("add data saving here")
                self.results = self.data_logger.start_logging()
            else:
                logging.info("not doing any data saving")

            in_standby = self.check_devices_in_standby_mode()
            if in_standby:
                pass
            else:
                logging.info("Stopping scanning.")
                log_df = self.stop_scan()
                return log_df

            if (
                self.shot_control is not None
            ):  # TODO Otherwise, can still run scan and save images but will not rename
                self.synchronize_devices()

            # clear source directories of synchronization shots
            self.scan_data_manager.purge_all_local_save_dir()

            # start the acquisition loop
            # self.scan_execution_loop()
            self.executor.execute_scan_loop(self.scan_steps)

        except Exception as e:
            logging.error(f"Error during scanning: {e}")

        logging.info("Stopping scanning.")
        time.sleep(1)
        log_df = self.stop_scan()

        return log_df  # Return the DataFrame with the logged data

    def check_devices_in_standby_mode(self) -> bool:
        """
        Check whether all devices have entered standby mode, with a timeout.

        Returns
        -------
            bool: True if all devices are in standby mode within the timeout; otherwise, False.
        """
        timeout = 8  # timeout in seconds
        start_time = time.time()
        while not self.data_logger.all_devices_in_standby:
            if time.time() - start_time > timeout:
                logging.error(
                    "Timeout reached while waiting for all devices to be go into standby."
                )
                return False
            time.sleep(1)
        return True

    def synchronize_devices(self) -> None:
        """
        Attempt to synchronize all devices by firing test shots and checking their standby status.

        This method repeatedly fires a single-shot trigger and checks whether all devices
        have exited standby mode. If synchronization fails, it resets device status and
        retries until synchronization is successful or a timeout is reached.

        Notes
        -----
        Devices are considered synchronized when all have exited standby mode
        after a test shot and responded appropriately.

        Raises
        ------
        None directly, but logs and stops the scan if timeout is reached.
        """
        timeout = 25.5  # seconds
        start_time = time.time()
        while not self.data_logger.devices_synchronized:
            if time.time() - start_time > timeout:
                logging.error(
                    "Timeout reached while waiting for devices to synchronize."
                )
                logging.info("Stopping scanning.")
                self.stop_scan()
                return
            if self.data_logger.all_devices_in_standby:
                logging.info("Sending single-shot trigger to synchronize devices.")

                res = self._set_trigger("SINGLESHOT")
                logging.info(f"Result of single shot command: {res}")
                # wait 2 seconds after the test fire to allow time for shot to execute and for devices to respond
                time.sleep(2)
                if self.data_logger.devices_synchronized:
                    logging.info("Devices synchronized.")
                    break
                else:
                    logging.warning("Not all devices exited standby after single shot.")
                    devices_still_in_standby = [
                        device
                        for device, status in self.data_logger.standby_mode_device_status.items()
                        if status is True
                    ]
                    logging.warning(
                        f"Devices still in standby: {devices_still_in_standby}"
                    )
                    logging.info("Resetting standby status to none for all devices.")
                    self.data_logger.standby_mode_device_status = {
                        key: None for key in self.data_logger.standby_mode_device_status
                    }
                    logging.info(
                        "Resetting initial timestamp to None for each device to enforce rechecking of stanby mode."
                    )
                    self.data_logger.initial_timestamps = {
                        key: None for key in self.data_logger.initial_timestamps
                    }
                    logging.info("Waiting for devices to re-enter standby mode.")
                    self.data_logger.all_devices_in_standby = False
            # wait 100 ms between checks of device standby status
            time.sleep(0.1)

    def stop_scan(self):
        """
        Stop the scan, save data, and restore initial device states.

        This method finalizes the scan by performing closeout actions, restoring
        devices to their original state, turning the trigger back on, saving
        scan data if enabled, stopping console logging, and resetting internal state.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the processed and saved scan data.
        """
        log_df = pd.DataFrame()

        if self.save_data:
            # Step 1: Stop data logging and handle device saving states
            self._stop_saving_devices()

        # Step 5: Restore the initial state of devices
        if self.initial_state is not None:
            self.restore_initial_state(self.initial_state)

        # Step 4: Turn the trigger back on
        self._set_trigger("STANDBY")

        if self.device_manager.scan_closeout_action is not None:
            logging.info("Attempting to execute closeout actions.")
            logging.info(f"Action list {self.device_manager.scan_closeout_action}")

            self.action_manager.add_action(
                action_name="closeout_action",
                action_seq=self.device_manager.scan_closeout_action,
            )
            self.action_manager.execute_action("closeout_action")

        if self.save_data:
            # Step 6: Process results, save to disk, and log data
            log_df = self.scan_data_manager.process_results(self.results)

            # pass log_df to the post process cleanup method in the file mover of data logger
            if self.save_local:
                self.data_logger.file_mover._post_process_orphaned_files(
                    log_df=log_df,
                    device_save_paths_mapping=self.scan_data_manager.device_save_paths_mapping,
                )
            else:
                self.data_logger.file_mover.scan_is_live = False
                self.data_logger.file_mover._post_process_orphan_task()

            self.data_logger.file_mover.shutdown(wait=True)

            # Step 8: create sfile in analysis folder
            self.scan_data_manager._make_sFile(log_df)

        # Step 8: Stop the console logging
        self.console_logger.stop_logging()

        # Step 9: Move log file if data was saved (use paths from ScanDataManager)
        if self.save_data:
            # Access the path from ScanDataManager
            log_folder_path = Path(self.scan_data_manager.data_txt_path).parent
            self.console_logger.move_log_file(log_folder_path)

        self.scan_step_start_time = 0
        self.scan_step_end_time = 0
        self.data_logger.idle_time = 0

        # Step 10: Reset the device manager to clear up the current subscribers
        self.device_manager.reset()

        # Step 11: Set the initialization flag back to False and require reinitialization to be called again
        self.initialization_success = False

        return log_df

    def _stop_saving_devices(self):
        """
        Stop data logging and reset save paths for non-scalar devices.

        This method disables saving for all non-scalar devices and resets their
        local save paths to a temporary directory. It ensures that all asynchronous
        save commands have time to complete.

        Notes
        -----
        Intended for internal use during scan shutdown or interruption.
        """
        # Stop data logging
        self.data_logger.stop_logging()

        # Handle device saving states
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logging.info(f"Setting save to off for {device_name}")
                device.set("save", "off", sync=False)
                logging.info(f"Setting save to off for {device_name} complete")
                device.set("localsavingpath", "c:\\temp", sync=False)
                logging.info(
                    f"Setting save path back to temp for {device_name} complete"
                )
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

        time.sleep(2)  # Ensure asynchronous commands have time to finish
        logging.info("scanning has stopped for all devices.")

    def save_hiatus(self, hiatus_period: float):
        """
        Temporarily disable and then re-enable saving on non-scalar devices.

        This method turns off data saving for all non-scalar devices, waits for a
        specified hiatus period, and then re-enables saving. It is useful during
        experimental pauses or transitions when saving should be momentarily halted.

        Parameters
        ----------
        hiatus_period : float
            Duration of the hiatus in seconds before re-enabling saving.
        """
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logging.info(f"Setting save to off for {device_name}")
                device.set("save", "off", sync=False)
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

        logging.info(f"All devices with Save OFF for {hiatus_period} s")
        time.sleep(hiatus_period)

        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logging.info(f"Setting save to on for {device_name}")
                device.set("save", "on", sync=False)
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

    def pre_logging_setup(self):
        """Prepare the experimental environment and devices for data acquisition.

        This comprehensive method orchestrates multiple setup steps to ensure
        the experimental system is correctly configured before a scan begins.
        It handles device initialization, trigger management, file preparation,
        and pre-scan configuration.

        Detailed Setup Steps:
        1. Disable trigger to prevent unintended shots
        2. Initialize scan data files and paths
        3. Generate scan steps based on configuration
        4. Configure device save paths (if data saving enabled)
        5. Initialize scan variables through DeviceManager
        6. Capture initial device states
        7. Execute pre-scan setup actions
        8. Generate live experiment configuration dump

        Parameters
        ----------
        None (uses instance attributes)

        Requires
        --------
        self.scan_config : ScanConfig
            Scan configuration object containing sweep parameters
        self.device_manager : DeviceManager
            Manages device configurations and interactions
        self.scan_data_manager : ScanDataManager
            Handles scan data path and file management
        self.save_data : bool
            Flag indicating whether data should be saved during the scan

        Notes
        -----
        - Method is critical for ensuring reproducible and controlled experimental scans
        - Handles both standard and composite scan variables
        - Supports different scan modes (standard, noscan, optimization)
        - Provides flexibility for various experimental configurations

        Raises
        ------
        GeecsDeviceInstantiationError
            If device initialization or configuration fails
        ValueError
            If scan configuration is incomplete or invalid

        Examples
        --------
        >>> scan_mgr = ScanManager(...)
        >>> scan_mgr.scan_config = ScanConfig(...)
        >>> scan_mgr.pre_logging_setup()
        # Prepares devices and environment for the upcoming scan

        See Also
        --------
        start_scan_thread : Initiates the scan after pre-logging setup
        _generate_scan_steps : Generates the sequence of scan steps
        device_manager.handle_scan_variables : Manages device variable initialization
        """
        logging.info("Turning off the trigger.")
        self.trigger_off()

        # initialize a ScanPaths objet and create basic scan files
        self.scan_data_manager.initialize_scan_data_and_output_files()

        # Generate the scan steps
        self.scan_steps = self._generate_scan_steps()
        logging.info(f"steps fro the scan are : {self.scan_steps}")

        time.sleep(2)

        if self.save_data:
            self.scan_data_manager.configure_device_save_paths(
                save_local=self.save_local
            )
            self.data_logger.save_local = self.save_local
            # map information produced in ScanDataManager to the DataLogger to facilitate
            # moving of files etc.
            # self.data_logger.device_save_paths_mapping = self.scan_data_manager.device_save_paths_mapping
            self.data_logger.set_device_save_paths_mapping(
                self.scan_data_manager.device_save_paths_mapping
            )
            self.data_logger.scan_number = (
                self.scan_data_manager.scan_number_int
            )  # TODO replace with a `set` func.

            self.scan_data_manager.write_scan_info_ini(self.scan_config)

        # Handle scan variables and ensure devices are initialized in DeviceManager
        logging.info(f"scan config in pre logging is this: {self.scan_config}")
        try:
            self.device_manager.handle_scan_variables(self.scan_config)
        except GeecsDeviceInstantiationError as e:
            logging.error(
                f"Device instantiation failed during handling of scan devices   {e}"
            )
            raise

        time.sleep(1.5)

        device_var = self.scan_config.device_var
        if device_var:
            # if not self.device_manager.is_statistic_noscan(device_var):
            self.initial_state = self.get_initial_state()

        if self.device_manager.scan_setup_action is not None:
            logging.info("Attempting to execute pre-scan actions.")
            logging.info(f"Action list {self.device_manager.scan_setup_action}")

            self.action_manager.add_action(
                action_name="setup_action",
                action_seq=self.device_manager.scan_setup_action,
            )
            self.action_manager.execute_action("setup_action")

        logging.info(f"attempting to generate ECS live dump using {self.MC_ip}")
        if self.MC_ip is not None and self.shot_control is not None:
            logging.info(f"attempting to generate ECS live dump using {self.MC_ip}")
            self.generate_live_ECS_dump(self.MC_ip)

        logging.info("Pre-logging setup completed.")

    def enable_live_ECS_dump(self, client_ip: str = "192.168.0.1"):
        """
        Enable live ECS dumps on the Master Control (MC) system.

        Sends a sequence of UDP commands to the shot control device to enable remote
        scan ECS dumps and verify the scan path configuration. Intended for setting
        up real-time experiment configuration saving.

        Parameters
        ----------
        client_ip : str, optional
            IP address of the client computer requesting the ECS dump (default is "192.168.0.1").

        Notes
        -----
        If `shot_control` is not configured, the method logs an error and exits.
        """
        if self.shot_control is None:
            logging.error("Cannot enable live ECS dump without shot control device")
            return

        steps = ["enable remote scan ECS dumps", "Main: Check scans path>>None"]

        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(3.5)
            logging.info(f"enable live ecs dumps step {step} complete")
            if not success:
                logging.warning(
                    f"Failed to enable live ECS dumps on MC on computer: {client_ip}"
                )
                break

    def generate_live_ECS_dump(self, client_ip: str = "192.168.0.1"):
        """
        Trigger the generation of a live ECS dump on the Master Control (MC) system.

        Sends a UDP command to save the current experiment device configuration
        at the beginning of a scan.

        Parameters
        ----------
        client_ip : str, optional
            IP address of the client computer requesting the ECS dump (default is "192.168.0.1").

        Notes
        -----
        If `shot_control` is not configured, the method logs an error and exits.
        """
        if self.shot_control is None:
            logging.error("Cannot enable live ECS dump without shot control device")
            return
        logging.info("sending comands to MC to generate ECS live dump")

        steps = [
            # "Main: Check scans path>>None",
            "Save Live Expt Devices Configuration>>ScanStart"
        ]

        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(0.5)
            if not success:
                logging.warning("Failed to generate an ECS live dump")
                break

    def _generate_scan_steps(self) -> List[Dict[str, Any]]:
        """Generate a sequence of scan steps based on the current scan configuration.

        Prepares a list of steps to be executed during the scan, accommodating
        different scan modes and variable configurations. This method handles
        standard linear scans, no-scan modes, and optimization-driven scans.

        Parameters
        ----------
        None (uses instance attributes)

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries representing scan steps, where each dictionary contains:
            - 'variables': A dictionary of device variables and their target values
            - 'wait_time': Duration to wait at each step
            - 'is_composite': Flag indicating if the step involves a composite variable

        Notes
        -----
        Scan Step Generation Strategy:
        - NOSCAN mode: Single step with no variable changes
        - OPTIMIZATION mode: Placeholder steps for dynamic optimization
        - STANDARD mode: Linear sweep of a device variable between start and end values

        Scan Configuration Requirements:
        - scan_config.scan_mode must be set
        - For STANDARD mode, start, end, step, and device_var must be defined
        - For OPTIMIZATION mode, optimizer configuration is required

        Examples
        --------
        >>> scan_mgr = ScanManager(...)
        >>> scan_mgr.scan_config = ScanConfig(
        ...     device_var='Undulator:gap',
        ...     start=10.0,
        ...     end=15.0,
        ...     step=0.5,
        ...     scan_mode=ScanMode.STANDARD
        ... )
        >>> steps = scan_mgr._generate_scan_steps()
        >>> print(steps)
        [
            {'variables': {'Undulator:gap': 10.0}, 'wait_time': 1.0, 'is_composite': False},
            {'variables': {'Undulator:gap': 10.5}, 'wait_time': 1.0, 'is_composite': False},
            ...
        ]

        Raises
        ------
        ValueError
            If scan configuration is incomplete or incompatible with the scan mode

        See Also
        --------
        start_scan_thread : Initiates scan execution using generated steps
        _setup_optimizer_from_config : Configures optimizer for optimization scans
        """
        self.data_logger.bin_num = 0
        steps = []

        mode = self.scan_config.scan_mode
        wait_time = self.scan_config.wait_time

        if mode == ScanMode.NOSCAN:
            steps.append(
                {"variables": {}, "wait_time": wait_time, "is_composite": False}
            )

        elif mode == ScanMode.OPTIMIZATION:
            self._setup_optimizer_from_config()

            num_steps = (
                int(
                    abs(
                        (self.scan_config.end - self.scan_config.start)
                        / self.scan_config.step
                    )
                )
                + 1
            )
            for _ in range(num_steps):
                steps.append(
                    {
                        "variables": {},  # to be filled in dynamically later
                        "wait_time": wait_time,
                        "is_composite": False,
                    }
                )

        elif mode == ScanMode.STANDARD:
            current_value = self.scan_config.start
            end = self.scan_config.end
            step = abs(self.scan_config.step)
            device_var = self.scan_config.device_var

            positive = current_value < end
            while (positive and current_value <= end) or (
                not positive and current_value >= end
            ):
                steps.append(
                    {
                        "variables": {device_var: current_value},
                        "wait_time": wait_time,
                        "is_composite": False,
                    }
                )
                current_value += step if positive else -step
        logging.info(f"scan steps generate: {steps}")
        return steps

    # def _generate_scan_steps(self):
    #     """
    #     Generate the scan steps ahead of time, handling both normal and composite variables.
    #
    #     uses self.scan_config which should be set before executing this method
    #
    #     Returns:
    #         list: A list of scan steps, each containing the variables and their corresponding values.
    #     """
    #
    #     self.data_logger.bin_num = 0
    #     steps = []
    #
    #     device_var = self.scan_config.device_var
    #
    #     if self.device_manager.is_statistic_noscan(device_var):
    #         steps.append({
    #             'variables': device_var,
    #             'wait_time': self.scan_config.wait_time,
    #             'is_composite': False
    #         })
    #
    #     else:
    #         current_value = self.scan_config.start
    #         positive_direction = self.scan_config.start < self.scan_config.end
    #         while (positive_direction and current_value <= self.scan_config.start)\
    #                 or (not positive_direction and current_value >= self.scan_config.start):
    #             steps.append({
    #                 'variables': {device_var: current_value},
    #                 'wait_time': self.scan_config.wait_time,
    #                 'is_composite': False
    #             })
    #             if positive_direction:
    #                 current_value += abs(self.scan_config.step)
    #             else:
    #                 current_value -= abs(self.scan_config.step)
    #
    #     return steps

    def _setup_optimizer_from_config(self):
        """Configure and initialize an optimizer for dynamic scan optimization.

        This method handles the complex process of setting up an optimization
        strategy for experimental scans. It loads an optimizer configuration,
        instantiates the optimizer, configures device requirements, and prepares
        the system for an optimization-driven scan.

        Detailed Setup Process:
        1. Validate optimizer configuration path
        2. Load optimizer configuration
        3. Instantiate BaseOptimizer with required dependencies
        4. Configure device requirements for optimization
        5. Add optimization variables to device manager
        6. Update scan executor with optimizer and logger

        Parameters
        ----------
        None (uses instance attributes)

        Requires
        --------
        self.scan_config : ScanConfig
            Must contain a valid `optimizer_config_path`
        self.scan_data_manager : ScanDataManager
            Used to provide context for optimizer initialization
        self.data_logger : DataLogger
            Used to provide logging capabilities to the optimizer

        Returns
        -------
        None
            Configures the optimizer in-place and updates related components

        Raises
        ------
        ValueError
            If optimizer configuration path is not specified
        ImportError
            If required optimizer modules cannot be imported
        ConfigurationError
            If optimizer configuration is invalid or incomplete

        Notes
        -----
        - Supports dynamic configuration of optimization strategies
        - Handles variable mapping between devices and optimizer
        - Ensures compatibility between optimizer and experimental setup

        Examples
        --------
        >>> scan_mgr = ScanManager(...)
        >>> scan_mgr.scan_config.optimizer_config_path = 'path/to/optimizer_config.yaml'
        >>> scan_mgr._setup_optimizer_from_config()
        # Configures optimizer for the upcoming scan

        See Also
        --------
        BaseOptimizer.from_config_file : Method used to instantiate optimizer
        start_scan_thread : Initiates scan execution with configured optimizer
        device_manager.add_scan_device : Adds optimization variables to device manager
        """
        if not self.scan_config.optimizer_config_path:
            raise ValueError(
                "optimizer_config_path must be set in ScanConfig for optimization scans"
            )

        self.optimizer = BaseOptimizer.from_config_file(
            config_path=self.scan_config.optimizer_config_path,
            scan_data_manager=self.scan_data_manager,
            data_logger=self.data_logger,
        )

        self.device_manager.load_from_dictionary(self.optimizer.device_requirements)

        # for var in self.optimizer.vocs.variables.keys():
        #     self.device_manager.check_then_add_variable(var)
        from collections import defaultdict

        # Step 1: Consolidate variables by device
        device_variables = defaultdict(list)
        for key in self.optimizer.vocs.variables.keys():
            device, variable = key.split(":", 1)
            device_variables[device].append(variable)

        # Step 2: Call add_scan_device for each device
        for device, variables in device_variables.items():
            self.device_manager.add_scan_device(device, variables)

        # Ensure the executor sees the updated optimizer. Maybe it's better to contstruct the executor
        # after this method...
        self.executor.optimizer = self.optimizer
        self.executor.data_logger = self.data_logger

    def estimate_acquisition_time(self):
        """Compute the estimated total duration for a scan based on configuration parameters.

        This method calculates the expected total acquisition time by analyzing
        the scan configuration, taking into account the scan mode, variable range,
        step size, and wait time between steps. The result provides a predictive
        estimate of the scan's total execution time.

        Calculation Strategy:
        - NOSCAN mode: Single wait time period
        - STANDARD mode: Number of steps multiplied by wait time
        - Handles both positive and negative scan directions

        Parameters
        ----------
        None (uses instance attributes)

        Requires
        --------
        self.scan_config : ScanConfig
            Scan configuration object containing:
            - device_var: Scan variable
            - start: Starting value of the scan
            - end: Ending value of the scan
            - step: Increment/decrement between steps
            - wait_time: Duration to wait at each step

        Attributes Modified
        ------------------
        self.acquisition_time : float
            Total estimated scan duration in seconds

        Notes
        -----
        - Adjusts for different scan modes and variable ranges
        - Provides a predictive estimate, not an exact measurement
        - Useful for progress tracking and resource allocation
        - Handles both linear and reverse scan directions

        Examples
        --------
        >>> scan_mgr = ScanManager(...)
        >>> scan_mgr.scan_config = ScanConfig(
        ...     device_var='Undulator:gap',
        ...     start=10.0,
        ...     end=15.0,
        ...     step=0.5,
        ...     wait_time=1.0,
        ...     scan_mode=ScanMode.STANDARD
        ... )
        >>> scan_mgr.estimate_acquisition_time()
        >>> print(scan_mgr.acquisition_time)
        11.0  # 11 steps * 1.0 seconds per step

        Raises
        ------
        AttributeError
            If scan_config is not set before method invocation

        See Also
        --------
        estimate_current_completion : Calculates current scan progress
        _generate_scan_steps : Generates the sequence of scan steps
        """
        total_time = 0

        if self.device_manager.is_statistic_noscan(self.scan_config.device_var):
            total_time += (
                self.scan_config.wait_time - 0.5
            )  # Default to 1 seconds if not provided
        else:
            start = self.scan_config.start
            end = self.scan_config.end
            step = self.scan_config.step
            wait_time = (
                self.scan_config.wait_time
            )  # - 0.5  # Default wait time between steps is 1 second

            # Calculate the number of steps and the total time for this device
            steps = abs((end - start) / step) + 1
            total_time += steps * wait_time

        logging.info(f"Estimated scan time: {total_time}")

        self.acquisition_time = total_time

    def estimate_current_completion(self):
        """
        Estimate the current completion percentage of the scan.

        The estimate is based on the current shot number and the total estimated acquisition time.

        Returns
        -------
        float
            A value between 0.0 and 1.0 indicating the fraction of the scan completed.
            Returns 1.0 if the estimate exceeds 100%.
        """
        if self.acquisition_time == 0:
            return 0
        completion = self.data_logger.get_current_shot() / self.acquisition_time
        return 1 if completion > 1 else completion

    def get_initial_state(self):
        """Capture the initial state of the scan variable before experimental manipulation.

        This method retrieves the current value of the scan variable, handling
        both standard device variables and composite variables. The captured
        state serves as a reference point for restoring the device to its
        original configuration after the scan is complete.

        Detailed State Retrieval Process:
        1. Determine if the scan variable is a composite or standard variable
        2. Access the current value from the appropriate device
        3. Create a dictionary mapping the variable to its initial value

        Parameters
        ----------
        None (uses instance attributes)

        Requires
        --------
        self.scan_config : ScanConfig
            Must contain the `device_var` to be tracked
        self.device_manager : DeviceManager
            Used to access device states and check variable types

        Returns
        -------
        dict
            A dictionary with the following possible formats:
            - For standard variables: {'device_name:variable_name': initial_value}
            - For composite variables: {'device_name:composite_var': initial_value}

        Notes
        -----
        - Critical for maintaining experimental reproducibility
        - Supports both single-device and multi-device composite variables
        - Provides a mechanism for reverting devices to their pre-scan state
        - Handles potential variations in variable naming and device configurations

        Examples
        --------
        >>> scan_mgr = ScanManager(...)
        >>> scan_mgr.scan_config = ScanConfig(device_var='Undulator:gap')
        >>> initial_state = scan_mgr.get_initial_state()
        >>> print(initial_state)
        {'Undulator:gap': 10.5}  # Initial gap value before scan

        >>> # For composite variables
        >>> scan_mgr.scan_config = ScanConfig(device_var='BeamPosition')
        >>> initial_state = scan_mgr.get_initial_state()
        >>> print(initial_state)
        {'BeamPosition:composite_var': {'x': 0.1, 'y': 0.2}}

        Raises
        ------
        KeyError
            If the specified device or variable cannot be found
        AttributeError
            If scan configuration is incomplete

        See Also
        --------
        restore_initial_state : Method used to revert devices to their initial state
        device_manager.is_composite_variable : Checks if a variable is composite
        """
        device_var = self.scan_config.device_var

        if self.device_manager.is_composite_variable(device_var):
            initial_state = {
                f"{device_var}:composite_var": self.device_manager.devices[
                    device_var
                ].state.get("composite_var")
            }
        else:
            device_name, var_name = device_var.split(":")
            initial_state = {
                device_var: self.device_manager.devices[device_name].state.get(var_name)
            }

        logging.info(f"Initial scan variable state: {initial_state}")
        return initial_state

    def restore_initial_state(self, initial_state):
        """Revert devices to their pre-scan configuration after experimental manipulation.

        This method systematically restores devices to their original state by
        setting each device variable back to its initial value. It is a critical
        step in maintaining experimental reproducibility and preventing unintended
        long-term effects from scan procedures.

        Detailed Restoration Process:
        1. Iterate through the provided initial state dictionary
        2. Identify the device and variable for each state entry
        3. Attempt to set the device variable to its original value
        4. Log successful restorations and any encountered errors

        Parameters
        ----------
        initial_state : dict
            A dictionary mapping device variables to their initial values.
            Keys should be in the format "device_name:variable_name",
            with corresponding values representing the original state.

            Example:
            {
                'Undulator:gap': 10.5,
                'BeamPosition:composite_var': {'x': 0.1, 'y': 0.2}
            }

        Notes
        -----
        Restoration Behavior:
        - Supports both standard and composite device variables
        - Gracefully handles missing devices or variables
        - Provides detailed logging for tracking restoration attempts
        - Critical for experimental reproducibility and device safety

        Error Handling:
        - Logs a warning if a device is not found in the device manager
        - Logs an error if setting a variable fails, without interrupting the restoration process
        - Continues attempting to restore other variables even if one fails

        Examples
        --------
        >>> scan_mgr = ScanManager(...)
        >>> initial_state = {
        ...     'Undulator:gap': 10.5,
        ...     'Laser:power': 2.0
        ... }
        >>> scan_mgr.restore_initial_state(initial_state)
        # Restores Undulator gap and Laser power to their initial values

        Raises
        ------
        No explicit exceptions, but logs warnings and errors for:
        - Devices not found in device manager
        - Failures in setting device variables

        See Also
        --------
        get_initial_state : Captures the initial state before scan manipulation
        device_manager.devices : Dictionary of managed devices
        """
        for device_var, value in initial_state.items():
            # Split the key to get the device name and variable name
            device_name, variable_name = device_var.split(":", 1)

            if device_name in self.device_manager.devices:
                device = self.device_manager.devices[device_name]
                try:
                    device.set(variable_name, value)
                    logging.info(f"Restored {device_name}:{variable_name} to {value}.")
                except Exception as e:
                    logging.error(
                        f"Failed to restore {device_name}:{variable_name} to {value}: {e}"
                    )
            else:
                logging.warning(
                    f"Device {device_name} not found. Skipping restoration for {device_var}."
                )


if __name__ == "__main__":
    print("testing")
    module_and_class = (
        "scan_analysis.analyzers.Undulator.CameraImageAnalysis.CameraImageAnalysis"
    )
    if module_and_class:
        module_name, class_name = module_and_class.rsplit(".", 1)
        print(module_name, class_name)
        module = importlib.import_module(module_name)
        analysis_class = getattr(module, class_name)
        print(analysis_class)
