# Standard library imports
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

from geecs_python_api.controls.devices.geecs_device import GeecsDevice


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
    
    def __init__(self, experiment_dir: str, shot_control_information: dict, device_manager=None, MC_ip = None, scan_data=None):
        """
        Initialize the ScanManager and its components.

        Args:
            experiment_dir (str): Directory where experiment data is stored.
            device_manager (DeviceManager, optional): DeviceManager instance for managing devices.
            shot_control_device (str, optional): GEECS Device that controls the shot timing
        """
        database_dict.reload(experiment_name=experiment_dir)
        self.device_manager = device_manager or DeviceManager(experiment_dir=experiment_dir)
        self.action_manager = ActionManager(experiment_dir=experiment_dir)
        self.MC_ip = MC_ip
        
        # Initialize ScanDataManager with device_manager and scan_data
        self.scan_data_manager = ScanDataManager(self.device_manager, scan_data, database_dict)

        self.data_logger = DataLogger(experiment_dir, self.device_manager)  # Initialize DataLogger
        self.save_data = True

        self.shot_control = GeecsDevice(shot_control_information['device'])
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
        
        self.enable_live_ECS_dump(client_ip = self.MC_ip)

        self.options_dict = {}  # Later initialized in 'reinitialize', but TODO should do it here instead

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

    def reinitialize(self, config_path=None, config_dictionary=None):
        """
        Reinitialize the ScanManager with new configurations and reset the logging system.

        Args:
            config_path (str, optional): Path to the configuration file.
            config_dictionary (dict, optional): Dictionary containing configuration settings.
        """

        self.initial_state = None
        self.device_manager.reinitialize(config_path=config_path, config_dictionary=config_dictionary)
        self.options_dict = config_dictionary['options']
        self.data_logger.reinitialize_sound_player()
        self.data_logger.last_log_time_sync = {}
        self.data_logger.update_repetition_rate(self.options_dict['rep_rate_hz'])
        self.console_logger.stop_logging()
        self.console_logger.setup_logging()

    def _set_trigger(self, state: str):
        """
        Set the trigger state and update variables accordingly.  Can be specified using Timing Setup in the GUI

        Args:
            state (str): Either 'OFF', 'SCAN', or 'STANDBY'.   Used for when trigger is off, or during/outside a scan
        """

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


            # start the acquisition loop
            self.scan_execution_loop()

        except Exception as e:
            logging.error(f"Error during scanning: {e}")

        logging.info("Stopping scanning.")
        time.sleep(1)
        log_df = self.stop_scan()

        return log_df  # Return the DataFrame with the logged data

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

            self.action_manager.add_action({'closeout_action': self.device_manager.scan_closeout_action})
            self.action_manager.execute_action('closeout_action')

        if self.save_data:
            # Step 6: Process results, save to disk, and log data
            log_df = self.scan_data_manager._process_results(self.results)
            
            # Step 7: Process and rename data files
            self.scan_data_manager.process_and_rename()
            
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
            self.scan_data_manager.create_and_set_data_paths()
            self.scan_data_manager.write_scan_info_ini(scan_config)

        # Handle scan variables and ensure devices are initialized in DeviceManager
        logging.info(f'scan config in pre logging is this: {scan_config}')
        self.device_manager.handle_scan_variables(scan_config)

        time.sleep(1.5)

        device_var = scan_config['device_var']
        if not self.device_manager.is_statistic_noscan(device_var):
            self.initial_state = self.get_initial_state(scan_config)

        # Generate the scan steps
        self.scan_steps = self._generate_scan_steps(scan_config)

        if self.device_manager.scan_setup_action is not None:
            logging.info("Attempting to execute pre-scan actions.")
            logging.info(f"Action list {self.device_manager.scan_setup_action}")

            self.action_manager.add_action({'setup_action': self.device_manager.scan_setup_action})
            self.action_manager.execute_action('setup_action')
        
        logging.info(f'attempting to generate ECS live dump using {self.MC_ip}')
        if self.MC_ip is not None:
            logging.info(f'attempting to generate ECS live dump using {self.MC_ip}')            
            self.generate_live_ECS_dump(self.MC_ip)

        logging.info("Pre-logging setup completed.")
  
    def enable_live_ECS_dump(self, client_ip: str = '192.168.0.1'):
        steps = [
            "enable remote scan ECS dumps",
        ]
        
        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(.5)
            if not success:
                logging.warning(f"Failed to enable live ECS dumps on MC on computer: {client_ip}")
                break
    
    def generate_live_ECS_dump(self, client_ip: str = '192.168.0.1'):
        steps = [
            "Main: Check scans path>>None",
            "Save Live Expt Devices Configuration>>ScanStart"
        ]
        
        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(.5)
            if not success:
                logging.warning(f"Failed to generate an ECS live dump")
                break
    
    def _add_scan_devices_to_async_observables(self, scan_config):
        """
        Add the devices and variables involved in the scan to async_observables. This ensures their values
        are logged in the scan files

        Args:
            scan_config (dict): The scan configuration that includes the devices and variables.
        """

        device_var = scan_config['device_var']
        if self.device_manager.is_composite_variable(device_var):
            component_vars = self.device_manager.get_composite_components(device_var, scan_config['start'])
            for device_var in component_vars:
                if device_var not in self.device_manager.async_observables:
                    self.device_manager.async_observables.append(device_var)
        else:
            if device_var not in self.device_manager.async_observables:
                self.device_manager.async_observables.append(device_var)

        logging.info(f"Updated async_observables: {self.device_manager.async_observables}")

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
        elif self.device_manager.is_composite_variable(device_var):
            self.data_logger.virtual_variable_name = device_var
            current_value = scan_config['start']
            while current_value <= scan_config['end']:
                component_vars = self.device_manager.get_composite_components(device_var, current_value)
                steps.append({
                    'variables': component_vars,
                    'wait_time': scan_config.get('wait_time', 1),
                    'is_composite': True
                })
                self.virtual_variable_list.append(current_value)
                current_value += scan_config['step']

        else:
            current_value = scan_config['start']
            while current_value <= scan_config['end']:
                steps.append({
                    'variables': {device_var: current_value},
                    'wait_time': scan_config.get('wait_time', 1),
                    'is_composite': False
                })
                current_value += scan_config['step']

        relative_flag = False
        # Check if it's a composite variable and if it has a relative flag in the YAML
        if self.device_manager.is_composite_variable(device_var):
            composite_variable_info = self.device_manager.composite_variables.get(device_var, {})
            relative_flag = composite_variable_info.get('relative', False)
            # self._apply_relative_adjustment(steps)

        # set relative flag for normal variables
        elif scan_config.get('relative', False):
            relative_flag = True

        if relative_flag and not self.device_manager.is_statistic_noscan(device_var):
            self._apply_relative_adjustment(steps)

        return steps

    def _apply_relative_adjustment(self, scan_steps):

        """
        Adjust the scan steps based on the initial state of the devices if relative scanning is enabled.
        This enables an easy way to start from a current state scan around it.

        Args:
            scan_steps (list): List of scan steps to be adjusted.
        """

        for step in scan_steps:
            for device_var, value in step['variables'].items():
                # Add the initial state value to the step value for each device
                if device_var in self.initial_state:
                    initial_value = self.initial_state[device_var]['value']
                    step['variables'][device_var] += initial_value
                else:
                    logging.warning(f"Initial state for {device_var} not found, skipping relative adjustment.")

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

        logging.info(f"shot control state: {self.shot_control.state}")

        if not self.device_manager.is_statistic_noscan(component_vars):
            for device_var, set_val in component_vars.items():
                device_name, var_name = device_var.split(':', 1)
                device = self.device_manager.devices.get(device_name)

                if device:
                    # Retrieve the tolerance for the variable
                    # TODO Better error handling when tolerance not defined in database editor
                    tol = float(GeecsDevice.exp_info['devices'][device_name][var_name]['tolerance'])

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
            save_on_shot = self.options_dict['On-Shot TDMS']
            if save_on_shot:
                if current_time % 1 < interval_time:
                    log_df = self.scan_data_manager.convert_to_dataframe(self.results)
                    self.scan_data_manager.dataframe_to_tdms(log_df)

        # Turn trigger off after waiting
        self.trigger_off()

        self.scan_step_end_time = time.time()
        self.data_logger.data_recording = False

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
            steps = ((end - start) / step) + 1
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

        scan_variables = []

        if scan_config:
            device_var = scan_config['device_var']

            # Handle composite variables by retrieving each component's state
            if self.device_manager.is_composite_variable(device_var):
                component_vars = self.device_manager.get_composite_components(device_var, scan_config['start'])
                scan_variables.extend(component_vars.keys())
            else:
                scan_variables.append(device_var)

        logging.info(f"Scan variables for initial state: {scan_variables}")
        # Use the existing method to get the current state of all variables
        initial_state = self.device_manager.get_values(scan_variables)

        logging.info(f"Initial scan variable state: {initial_state}")
        return initial_state

    def restore_initial_state(self, initial_state):

        """
        Restore the devices to their initial states after the scan has completed.
        NOTE: this could be used more generally to restore any kind of state

        Args:
            initial_state (dict): A dictionary containing the initial states of the devices.
        """

        for device_var, value_dict in initial_state.items():
            device_name, var_name = device_var.split(':')
            device = self.device_manager.devices.get(device_name)

            if device:
                initial_value = value_dict['value']
                device.set(var_name, initial_value)
                logging.info(f"Restored {device_name}:{var_name} to initial value {initial_value}")
            else:
                logging.warning(f"Device {device_name} not found when trying to restore state.")

        logging.info("All devices restored to their initial states.")



if __name__ == '__main__':
    print("testing")
    module_and_class = 'scan_analysis.analyzers.Undulator.CameraImageAnalysis.CameraImageAnalysis'
    if module_and_class:
        module_name, class_name = module_and_class.rsplit('.', 1)
        print(module_name, class_name)
        module = importlib.import_module(module_name)
        analysis_class = getattr(module, class_name)
        print(analysis_class)