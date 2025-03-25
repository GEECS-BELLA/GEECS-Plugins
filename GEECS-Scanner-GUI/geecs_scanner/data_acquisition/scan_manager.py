from __future__ import annotations

# Standard library imports
from typing import Optional
import time
import threading
import logging
import importlib
from pathlib import Path

# Third-party library imports
import pandas as pd

# Internal project imports
from . import DeviceManager, ActionManager, DataLogger, DatabaseDictLookup, ScanDataManager
from .utils import ConsoleLogger

from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_python_api.controls.interface.geecs_errors import GeecsDeviceInstantiationError

database_dict = DatabaseDictLookup()


def get_database_dict():
    return database_dict.get_database()


class ScanManager:
    """
    Manages the execution of scans, including configuration, device control,
    and data logging. This class handles the interaction between devices,
    data acquisition, and scanning logic. A 'save_devices' config file should be passed
    to the device_manager to initialize the desired saving configuration.
    """
    
    def __init__(self, experiment_dir: str, shot_control_information: dict,
                 options_dict: Optional[dict] = None, device_manager=None, scan_data=None):
        """
        Initialize the ScanManager and its components.

        Args:
            experiment_dir (str): Directory where experiment data is stored.
            device_manager (DeviceManager, optional): DeviceManager instance for managing devices.
            shot_control_information (dict): dict containing shot control information
        """
        database_dict.reload(experiment_name=experiment_dir)
        self.device_manager = device_manager or DeviceManager(experiment_dir=experiment_dir)
        self.action_manager = ActionManager(experiment_dir=experiment_dir)
        self.initialization_success = False

        self.MC_ip = ""
        
        # Initialize ScanDataManager with device_manager and scan_data
        self.scan_data_manager = ScanDataManager(self.device_manager, scan_data, database_dict)

        self.data_logger = DataLogger(experiment_dir, self.device_manager)  # Initialize DataLogger
        self.save_data = True

        self.shot_control: Optional[ScanDevice] = None
        self.shot_control_variables = None
        shot_control_device = shot_control_information.get('device', None)
        if shot_control_device:
            self.shot_control = ScanDevice(shot_control_information['device'])
            self.shot_control_variables = shot_control_information['variables']

        self.results = {}  # Store results for later processing

        self.stop_scanning_thread_event = threading.Event()  # Event to signal the logging thread to stop

        # Use the ConsoleLogger class
        self.console_logger = ConsoleLogger(log_file="scan_execution.log", level=logging.INFO, console=True)
        self.console_logger.setup_logging()

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

    def pause_scan(self):
        """Pause the scanning process by clearing the pause event."""
        if self.pause_scan_event.is_set():
            self.pause_scan_event.clear()
            logging.info("Scanning paused.")

    def resume_scan(self):
        """Resume the scanning process by setting the pause event."""
        if not self.pause_scan_event.is_set():
            self.pause_scan_event.set()
            logging.info("Scanning resumed.")

    def reinitialize(self, config_path=None, config_dictionary=None, scan_data=None) -> bool:
        """
        Reinitialize the ScanManager with new configurations and reset the logging system.

        Args:
            config_path (str, optional): Path to the configuration file.
            config_dictionary (dict, optional): Dictionary containing configuration settings.
            scan_data (ScanData, optional):  If given, scan_data_manager will use an alternative scan folder

        Returns:
            (bool) True if successful and all devices connected.  False otherwise
        """

        self.initial_state = None
        self.initialization_success = False

        try:
            self.device_manager.reinitialize(config_path=config_path, config_dictionary=config_dictionary)
        except GeecsDeviceInstantiationError as e:
            logging.error(f"Device reinitialization failed during initialization of device manager. check "
                          f"that all devices are on and available: {e}")
            # Optionally, notify the user via a UI pop-up or other mechanism here
            # Then, you can abort reinitialization:
            return False

        self.scan_data_manager = ScanDataManager(self.device_manager, scan_data, database_dict)

        if config_dictionary is not None and 'options' in config_dictionary:
            self.options_dict = config_dictionary['options']
            self.save_local = self.options_dict.get('Save Local', True)

        new_mc_ip = self.options_dict.get("Master Control IP", "")
        if self.shot_control and new_mc_ip and self.MC_ip != new_mc_ip:
            self.MC_ip = new_mc_ip
            self.enable_live_ECS_dump(client_ip=self.MC_ip)

        self.data_logger.reinitialize_sound_player(options=self.options_dict)
        self.data_logger.last_log_time_sync = {}
        self.data_logger.update_repetition_rate(self.options_dict.get('rep_rate_hz', 1))
        self.console_logger.stop_logging()
        self.console_logger.setup_logging()

        self.initialization_success = True
        return self.initialization_success

    def _set_trigger(self, state: str):
        """
        Set the trigger state and update variables accordingly.  Can be specified using Timing Setup in the GUI

        Args:
            state (str): Either 'OFF', 'SCAN', or 'STANDBY'.   Used for when trigger is off, or during/outside a scan
        """
        if self.shot_control is None or self.shot_control_variables is None:
            logging.info(f"No shot control device, skipping 'set state {state}'")
            return

        valid_states = ['OFF', 'SCAN', 'STANDBY']

        if state in valid_states:
            for variable in self.shot_control_variables.keys():
                variable_settings = self.shot_control_variables[variable]
                self.shot_control.set(variable, variable_settings[state])
                logging.info(f"Setting {variable} to {variable_settings[state]}")
            logging.info(f"Trigger turned to state {state}.")
        else:
            logging.error(f"Invalid trigger state: {state}")

    def trigger_off(self):
        """Turns off the trigger and sets the amplitude to 0.5."""
        self._set_trigger('OFF')

    def trigger_on(self):
        """Turns on the trigger and sets the amplitude to 4.0."""
        self._set_trigger('SCAN')

    def is_scanning_active(self):
        """
        Check if a scan is currently active.

        Returns:
            bool: True if scanning is active, False otherwise.
        """

        return self.scanning_thread and self.scanning_thread.is_alive()

    def start_scan_thread(self, scan_config=None):
        """
        Start a new scan in a separate thread. Having it in a separate thread allows it to be interupted
        externally using the stop_scan_thread method

        Args:
            scan_config (dict, optional): Configuration settings for the scan, including variables, start, end, step, and wait times.
        """
        if not self.initialization_success:
            logging.error("initialization unsuccessful, cannot start a new scan session")
            return

        if self.is_scanning_active():
            logging.warning("scanning is already active, cannot start a new scan session.")
            return

        # Ensure the stop event is cleared before starting a new session
        self.stop_scanning_thread_event.clear()

        # Start a new thread for logging
        logging.info(f'scan config is this: {scan_config}')
        self.scanning_thread = threading.Thread(target=self.start_scan, args=(scan_config,))
        self.scanning_thread.start()
        logging.info("Scan thread started.")

    def stop_scanning_thread(self):
        """
        Stop the currently running scanning thread and clean up resources.
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

    def start_scan(self, scan_config):

        """
        Start a scan while dynamically performing actions
        or handling 'noscan' (or 'statistics') scans during the acquisition process.

        Args:
            scan_config (dict): Configuration for the scan, including device variables, start, end, step, and wait times.

        Returns:
            pandas.DataFrame: A DataFrame containing the results of the scan.
        """
        if not self.initialization_success:
            logging.error("initialization unsuccessful, cannot start a new scan session")
            return

        log_df = pd.DataFrame()  # Initialize in case of early exit

        try:
            # Pre-logging setup: Trigger devices off, initialize data paths, etc.
            logging.info(f'scan config getting sent to pre logging is this: {scan_config}')

            self.pre_logging_setup(scan_config)

            # Estimate acquisition time if necessary
            if scan_config:
                self.estimate_acquisition_time(scan_config)
                logging.info(f"Estimated acquisition time based on scan config: {self.acquisition_time} seconds.")

            # Start data logging
            if self.save_data:
                logging.info('add data saving here')
                self.results = self.data_logger.start_logging()
            else:
                logging.info('not doing any data saving')

            in_standby = self.check_devices_in_standby_mode()
            if in_standby:
                pass
            else:
                logging.info("Stopping scanning.")
                log_df = self.stop_scan()
                return log_df

            self.synchronize_devices()

            #clear source directories of synchronization shots
            self.scan_data_manager.purge_all_local_save_dir()

            # start the acquisition loop
            self.scan_execution_loop()

        except Exception as e:
            logging.error(f"Error during scanning: {e}")

        logging.info("Stopping scanning.")
        time.sleep(1)
        log_df = self.stop_scan()

        return log_df  # Return the DataFrame with the logged data

    def check_devices_in_standby_mode(self)-> bool:
        """
        Check whether all devices have entered standby mode, with a timeout.

        Returns:
            bool: True if all devices are in standby mode within the timeout; otherwise, False.
        """
        timeout = 8  # timeout in seconds
        start_time = time.time()
        while not self.data_logger.all_devices_in_standby:
            if time.time() - start_time > timeout:
                logging.error("Timeout reached while waiting for all devices to be go into standby.")
                return False
            time.sleep(1)
        return True

    def synchronize_devices(self) -> None:
        """
        Attempt to synchronize all devices but firing a test shot and checking if
        all devices exited standby mode. If all devices do not exit standby mode
        their status is reinitialized to 'none' (i.e. unknown) and then we wait for
        them all to go back into standby mode (i.e. devices each timeout). Once that condition is
        satisified, we can fire another test shot to see if all devices exit standby mode.
        This process repeats until successful or a timeout is reached.
        """
        timeout = 25.5  # seconds
        start_time = time.time()
        while not self.data_logger.devices_synchronized:
            if time.time() - start_time > timeout:
                logging.error("Timeout reached while waiting for devices to synchronize.")
                logging.info("Stopping scanning.")
                self.stop_scan()
                return
            if self.data_logger.all_devices_in_standby:
                logging.info("Sending single-shot trigger to synchronize devices.")

                # TODO: this is hardcoded to fire single shot on a DG645
                res = self.shot_control.set('Trigger.ExecuteSingleShot', 'on')
                logging.info(f"Result of single shot command: {res}")
                #wait 2 seconds after the test fire to allow time for shot to execute and for devices to respond
                time.sleep(2)
                if self.data_logger.devices_synchronized:
                    logging.info("Devices synchronized.")
                    break
                else:
                    logging.warning("Not all devices exited standby after single shot.")
                    devices_still_in_standby = [device for device, status in
                        self.data_logger.standby_mode_device_status.items() if status is True]
                    logging.warning(f"Devices still in standby: {devices_still_in_standby}")
                    logging.info("Resetting standby status to none for all devices.")
                    self.data_logger.standby_mode_device_status = {key: None for key in self.data_logger.standby_mode_device_status}
                    logging.info("Resetting initial timestamp to None for each device to enforce rechecking of stanby mode.")
                    self.data_logger.initial_timestamps = {key: None for key in self.data_logger.initial_timestamps}
                    logging.info("Waiting for devices to re-enter standby mode.")
                    self.data_logger.all_devices_in_standby = False
            #wait 100 ms between checks of device standby status
            time.sleep(0.1)

    def stop_scan(self):
        """
        Stop the scan, save data, and restore the initial device states.

        Returns:
            pandas.DataFrame: DataFrame containing the saved scan data.
        """

        log_df = pd.DataFrame()

        if self.save_data:
            # Step 1: Stop data logging and handle device saving states
            self._stop_saving_devices()

        # Step 5: Restore the initial state of devices
        if self.initial_state is not None:
            self.restore_initial_state(self.initial_state)

        # Step 4: Turn the trigger back on
        self._set_trigger('STANDBY')

        if self.device_manager.scan_closeout_action is not None:
            logging.info("Attempting to execute closeout actions.")
            logging.info(f"Action list {self.device_manager.scan_closeout_action}")

            self.action_manager.add_action(action_name='closeout_action',
                                           action_steps=self.device_manager.scan_closeout_action)
            self.action_manager.execute_action('closeout_action')

        if self.save_data:
            # Step 6: Process results, save to disk, and log data
            log_df = self.scan_data_manager.process_results(self.results)

            # pass log_df to the post process cleanup method in the file mover of data logger
            self.data_logger.file_mover.post_process_orphaned_files(log_df=log_df, device_save_paths_mapping=self.scan_data_manager.device_save_paths_mapping)
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
        Stop the data logger and update the save paths for non-scalar devices.
        NOTE: this method should probably parsed with the device commands migrated elsewhere,
        maybe ScanDataManager or DeviceManager
        """

        # Stop data logging
        self.data_logger.stop_logging()

        # Handle device saving states
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logging.info(f"Setting save to off for {device_name}")
                device.set('save', 'off', sync=False)
                logging.info(f"Setting save to off for {device_name} complete")
                device.set('localsavingpath', 'c:\\temp', sync=False)
                logging.info(f"Setting save path back to temp for {device_name} complete")
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

        time.sleep(2)  # Ensure asynchronous commands have time to finish
        logging.info("scanning has stopped for all devices.")

    def save_hiatus(self, hiatus_period: float):
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logging.info(f"Setting save to off for {device_name}")
                device.set('save', 'off', sync=False)
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

        logging.info(f"All devices with Save OFF for {hiatus_period} s")
        time.sleep(hiatus_period)

        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logging.info(f"Setting save to on for {device_name}")
                device.set('save', 'on', sync=False)
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

    def pre_logging_setup(self, scan_config):
        """
        Precompute all scan steps (including composite and normal variables),
        add scan devices to async_observables, and store the scan steps.
        Execute pre scan setup actions passed through

        Args:
            scan_config (dict): Configuration for the scan, including device variables and steps.
        """

        logging.info("Turning off the trigger.")
        self.trigger_off()

        time.sleep(2)

        if self.save_data:
            self.scan_data_manager.create_and_set_data_paths(save_local=self.save_local)
            # map information produced in ScanDataManager to the DataLogger to facilitate
            # moving of files etc.
            # self.data_logger.device_save_paths_mapping = self.scan_data_manager.device_save_paths_mapping
            self.data_logger.set_device_save_paths_mapping(self.scan_data_manager.device_save_paths_mapping)
            self.data_logger.scan_number = self.scan_data_manager.scan_number_int

            self.scan_data_manager.write_scan_info_ini(scan_config)

        # Handle scan variables and ensure devices are initialized in DeviceManager
        logging.info(f'scan config in pre logging is this: {scan_config}')
        try:
            self.device_manager.handle_scan_variables(scan_config)
        except GeecsDeviceInstantiationError as e:
            logging.error(f"Device instantiation failed during handling of scan devices   {e}")
            raise

        time.sleep(1.5)

        device_var = scan_config['device_var']
        if not self.device_manager.is_statistic_noscan(device_var):
            self.initial_state = self.get_initial_state(scan_config)

        # Generate the scan steps
        self.scan_steps = self._generate_scan_steps(scan_config)

        if self.device_manager.scan_setup_action is not None:
            logging.info("Attempting to execute pre-scan actions.")
            logging.info(f"Action list {self.device_manager.scan_setup_action}")

            self.action_manager.add_action(action_name='setup_action',
                                           action_steps=self.device_manager.scan_setup_action)
            self.action_manager.execute_action('setup_action')

        logging.info(f'attempting to generate ECS live dump using {self.MC_ip}')
        if self.MC_ip is not None and self.shot_control is not None:
            logging.info(f'attempting to generate ECS live dump using {self.MC_ip}')
            self.generate_live_ECS_dump(self.MC_ip)

        logging.info("Pre-logging setup completed.")
  
    def enable_live_ECS_dump(self, client_ip: str = '192.168.0.1'):
        if self.shot_control is None:
            logging.error("Cannot enable live ECS dump without shot control device")
            return

        steps = [
            "enable remote scan ECS dumps",
            "Main: Check scans path>>None"
        ]
        
        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(3.5)
            logging.info(f'enable live ecs dumps step {step} complete')
            if not success:
                logging.warning(f"Failed to enable live ECS dumps on MC on computer: {client_ip}")
                break
    
    def generate_live_ECS_dump(self, client_ip: str = '192.168.0.1'):
        if self.shot_control is None:
            logging.error("Cannot enable live ECS dump without shot control device")
            return
        logging.info('sending comands to MC to generate ECS live dump')

        steps = [
            # "Main: Check scans path>>None",
            "Save Live Expt Devices Configuration>>ScanStart"
        ]
        
        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(.5)
            if not success:
                logging.warning(f"Failed to generate an ECS live dump")
                break

    def _generate_scan_steps(self, scan_config):
        """
        Generate the scan steps ahead of time, handling both normal and composite variables.

        Args:
            scan_config (dict): Configuration for the scan, including variables, start, end, step, and wait times.

        Returns:
            list: A list of scan steps, each containing the variables and their corresponding values.
        """

        self.data_logger.bin_num = 0
        steps = []

        device_var = scan_config['device_var']

        if self.device_manager.is_statistic_noscan(device_var):
            steps.append({
                'variables': device_var,
                'wait_time': scan_config.get('wait_time', 1),
                'is_composite': False
            })

        else:
            current_value = scan_config['start']
            positive_direction = scan_config['start'] < scan_config['end']
            while (positive_direction and current_value <= scan_config['end'])\
                    or (not positive_direction and current_value >= scan_config['end']):
                steps.append({
                    'variables': {device_var: current_value},
                    'wait_time': scan_config.get('wait_time', 1),
                    'is_composite': False
                })
                if positive_direction:
                    current_value += abs(scan_config['step'])
                else:
                    current_value -= abs(scan_config['step'])

        return steps

    def scan_execution_loop(self):

        """
        Execute the precomputed scan steps in a loop, stopping if the stop event is triggered.

        Returns:
            pandas.DataFrame: A DataFrame containing the logged data from the scan.
        """

        log_df = pd.DataFrame()  # Initialize in case of early exit

        counter = 0
        while self.scan_steps:
            # Check if the stop event is set, and exit if so
            if self.stop_scanning_thread_event.is_set():
                logging.info("Scanning has been stopped externally.")
                break
            scan_step = self.scan_steps.pop(0)
            self._execute_step(scan_step['variables'], scan_step['wait_time'], scan_step['is_composite'])
            counter+=1

        logging.info("Stopping logging.")

        return log_df

    def _execute_step(self, component_vars, wait_time, is_composite, max_retries=3, retry_delay=0.5):
        """
        Execute a single step of the scan, handling both composite and normal variables.

        Args:
            component_vars (dict): Dictionary of variables and their values for the scan step.
            wait_time (float): The time to wait after devices have been changed. This is the acquisition time effectively.
            is_composite (bool): Flag indicating whether the step involves composite variables.
            max_retries (int): Maximum number of retries if setting the value is outside the tolerance.
            retry_delay (float): Delay in seconds between retries.
        """
        logging.info("Pausing logging. Turning trigger off before moving devices.")
        if self.data_logger.virtual_variable_name is not None:
            self.data_logger.virtual_variable_value = self.virtual_variable_list[self.data_logger.bin_num]
            logging.info(f"updating virtual value in data_logger from scan_manager to: {self.data_logger.virtual_variable_value}.")

        self.data_logger.bin_num += 1

        self.trigger_off()

        if self.shot_control is not None:
            logging.info(f"shot control state: {self.shot_control.state}")

        if not self.device_manager.is_statistic_noscan(component_vars):
            for device_var, set_val in component_vars.items():
                # Check if the observable has a variable specified (e.g., 'Dev1:var1')
                if ':' in device_var:
                    device_name, var_name = device_var.split(':')
                else:
                    device_name = device_var
                    var_name = 'composite_var'

                device = self.device_manager.devices.get(device_name)

                if device:
                    # Retrieve the tolerance for the variable
                    # TODO Better error handling when tolerance not defined in database editor
                    if device.is_composite:
                        tol=10000 # set for composite vars just returns the set value
                    else:
                        tol = float(ScanDevice.exp_info['devices'][device_name][var_name]['tolerance'])

                    # Retry logic for setting device value
                    success = False
                    attempt = 0

                    while attempt < max_retries:
                        ret_val = device.set(var_name, set_val)  # Send the command to set the value
                        logging.info(f"Attempt {attempt + 1}: Setting {var_name} to {set_val} on {device_name}, returned {ret_val}")

                        # Check if the return value is within tolerance
                        if ret_val - tol <= set_val <= ret_val + tol:
                            logging.info(f"Success: {var_name} set to {ret_val} (within tolerance {tol}) on {device_name}")
                            success = True
                            break
                        else:
                            logging.warning(f"Attempt {attempt + 1}: {var_name} on {device_name} not within tolerance ({ret_val} != {set_val})")
                            attempt += 1
                            time.sleep(retry_delay)  # Wait before retrying

                    if not success:
                        logging.error(f"Failed to set {var_name} on {device_name} after {max_retries} attempts")
                else:
                    logging.warning(f"Device {device_name} not found in device manager.")

        logging.info("Resuming logging. Turning trigger on after all devices have been moved.")

        self.trigger_on()

        #code below used with data_logger to determine if a device is non responsive
        self.scan_step_start_time = time.time()
        self.data_logger.data_recording = True

        # first step of the scan, self.scan_step_end_time is equal to zero.
        # after that, it gets updated. As does the self.scan_step_start_time.
        # idle time should be the time between then end of the "previous" step
        # and the start of the next step.
        if self.scan_step_end_time > 0:
            self.data_logger.idle_time = self.scan_step_start_time - self.scan_step_end_time + self.pause_time
            logging.info(f'idle time between scan steps: {self.data_logger.idle_time}')

        if self.shot_control is not None:
            logging.info(f"shot control state: {self.shot_control.state}")

        # Wait for acquisition time (or until scanning is externally stopped)
        current_time = 0
        start_time = time.time()
        interval_time = 0.1
        self.pause_time = 0
        while current_time < wait_time:
            if self.stop_scanning_thread_event.is_set():
                logging.info("Scanning has been stopped externally.")
                break

            # Check if the scan is paused, and wait until it’s resumed
            if not self.pause_scan_event.is_set():
                self.trigger_off()
                self.data_logger.data_recording = False
                t0 = time.time()
                logging.info("Scan is paused, waiting to resume...")
                self.pause_scan_event.wait()  # Blocks until the event is set (i.e., resumed)
                self.pause_time = time.time() - t0
                self.trigger_on()
                self.data_logger.data_recording = True


            time.sleep(interval_time)
            current_time = time.time() - start_time

            # TODO move this to DataLogger's `_log_device_data` function instead...

            save_on_shot = self.options_dict.get('On-Shot TDMS', False)
            if save_on_shot:
                if current_time % 1 < interval_time:
                    log_df = self.scan_data_manager.convert_to_dataframe(self.results)
                    self.scan_data_manager.dataframe_to_tdms(log_df)

            try:
                hiatus = float(self.options_dict.get("Save Hiatus Period (s)", ""))
            except ValueError:
                hiatus = ""

            if hiatus:
                if self.data_logger.shot_save_event.is_set():
                    self.save_hiatus(hiatus)
                    self.data_logger.shot_save_event.clear()

        # Turn trigger off after waiting
        self.trigger_off()

        self.scan_step_end_time = time.time()
        self.data_logger.data_recording = False

        if self.shot_control is not None:
            logging.info(f"shot control state: {self.shot_control.state}")

    def estimate_acquisition_time(self, scan_config):

        """
        Estimate the total acquisition time based on the scan configuration.

        Args:
            scan_config (dict): Configuration for the scan, including start, end, step, and wait times.
        """

        total_time = 0

        if self.device_manager.is_statistic_noscan(scan_config['device_var']):
            total_time += scan_config.get('wait_time', 1) - 0.5  # Default to 1 seconds if not provided
        else:
            start = scan_config['start']
            end = scan_config['end']
            step = scan_config['step']
            wait_time = scan_config.get('wait_time', 1)# - 0.5  # Default wait time between steps is 1 second

            # Calculate the number of steps and the total time for this device
            steps = abs((end - start) / step) + 1
            total_time += steps * wait_time

        logging.info(f'Estimated scan time: {total_time}')

        self.acquisition_time = total_time

    def estimate_current_completion(self):
        """
        Estimate the current completion percentage of the ongoing scan.

        Returns:
            float: A percentage value (between 0.0 and 1.0) indicating the progress of the scan.
        """

        if self.acquisition_time == 0:
            return 0
        completion = self.data_logger.get_current_shot()/self.acquisition_time
        return 1 if completion > 1 else completion

    def get_initial_state(self, scan_config):

        """
        Retrieve the initial state of the devices involved in the scan.

        Args:
            scan_config (dict): Configuration for the scan, specifying the devices and variables.

        Returns:
            dict: A dictionary mapping each device variable to its initial state.
        """

        device_var = scan_config['device_var']

        if self.device_manager.is_composite_variable(device_var):
            initial_state = {f'{device_var}:composite_var':self.device_manager.devices[device_var].state.get('composite_var')}
        else:
            device_name, var_name = device_var.split(':')
            initial_state = {device_var:self.device_manager.devices[device_name].state.get(var_name)}

        logging.info(f"Initial scan variable state: {initial_state}")
        return initial_state

    def restore_initial_state(self, initial_state):
        """
        Restore the devices to their initial states after the scan has completed.
        NOTE: This could be used more generally to restore any kind of state.

        Args:
            initial_state (dict): A dictionary containing the initial states of the devices,
                                  where keys are in the format "device_name:variable_name",
                                  and values are the states to be restored.
        """
        for device_var, value in initial_state.items():
            # Split the key to get the device name and variable name
            device_name, variable_name = device_var.split(':', 1)

            if device_name in self.device_manager.devices:
                device = self.device_manager.devices[device_name]
                try:
                    device.set(variable_name, value)
                    logging.info(f"Restored {device_name}:{variable_name} to {value}.")
                except Exception as e:
                    logging.error(f"Failed to restore {device_name}:{variable_name} to {value}: {e}")
            else:
                logging.warning(f"Device {device_name} not found. Skipping restoration for {device_var}.")


if __name__ == '__main__':
    print("testing")
    module_and_class = 'scan_analysis.analyzers.Undulator.CameraImageAnalysis.CameraImageAnalysis'
    if module_and_class:
        module_name, class_name = module_and_class.rsplit('.', 1)
        print(module_name, class_name)
        module = importlib.import_module(module_name)
        analysis_class = getattr(module, class_name)
        print(analysis_class)