import time
import threading
import logging
import pandas as pd
from pathlib import Path

import numpy as np

from .data_acquisition import DeviceManager, ActionManager, DataInterface, DataLogger
from .utils import ConsoleLogger

from geecs_python_api.controls.interface import load_config
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice

from nptdms import TdmsWriter, ChannelObject

config = load_config()

if config and 'Experiment' in config and 'expt' in config['Experiment']:
    default_experiment = config['Experiment']['expt']
    print(f"default experiment is: {default_experiment}")
else:
    print(
        "Configuration file not found or default experiment not defined. While use Undulator as experiment. Could be a problem for you.")
    default_experiment = 'Undulator'

GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(default_experiment)
device_dict = GeecsDevice.exp_info['devices']

class ScanManager():
    
    """
    Manages the execution of scans. This involves initializing a device_manager if one
    is not. A 'save_devices' config file should be passed to the device_manager to initialize
    the desired saving configuration. 
    """
    
    def __init__(self, experiment_dir=None, shot_control_device=""):
        self.device_manager = DeviceManager(experiment_dir=experiment_dir)
        self.data_interface = DataInterface()
        self.action_manager = ActionManager(experiment_dir=experiment_dir)
        self.data_logger = DataLogger(experiment_dir, self.device_manager)  # Initialize DataLogger

        self.save_data = True

        self.shot_control = GeecsDevice(shot_control_device)
        self.results = {}  # Store results for later processing

        self.stop_scanning_thread_event = threading.Event()  # Event to signal the logging thread to stop
        
        # Use the new ConsoleLogger class
        self.console_logger = ConsoleLogger(log_file="scan_execution.log", level=logging.INFO, console=True)
        self.console_logger.setup_logging()
        
        self.bin_num = 0  # Initialize bin as 0
        self.acquisition_time = 0

        self.scanning_thread = None  # NEW: Separate thread for scanning

        self.tdms_writer = None
        self.initial_state = None

        self.scan_steps = []  # To store the precomputed scan steps

    def reinitialize(self, config_path=None, config_dictionary=None):
        self.initial_state = None
        self.device_manager.reinitialize(config_path=config_path, config_dictionary=config_dictionary)
        self.data_logger.reinitialize_sound_player()
        self.console_logger.stop_logging()
        self.console_logger.setup_logging()

    def _set_trigger(self, state: str, amplitude: float):
        """Helper method to turn the trigger on or off and set amplitude."""
        valid_states = {
            'on': 'External rising edges',
            'off': 'Single shot external rising edges'
        }
        if state in valid_states:
            self.shot_control.set('Trigger.Source', valid_states[state])
            self.shot_control.set('Amplitude.Ch AB', amplitude)
            logging.info(f"Trigger turned {state} with amplitude {amplitude}.")
        else:
            logging.error(f"Invalid trigger state: {state}")

    def trigger_off(self):
        """Turns off the trigger and sets the amplitude to 0.5."""
        self._set_trigger('off', 0.5)

    def trigger_on(self):
        """Turns on the trigger and sets the amplitude to 4.0."""
        self._set_trigger('on', 4.0)

    def is_scanning_active(self):
        return self.scanning_thread and self.scanning_thread.is_alive()

    def start_scan_thread(self, scan_config=None):
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
        if not self.is_scanning_active():
            logging.warning("No active scanning thread to stop.")
            return

        # Stop the scaning and wait for the thread to finish
        logging.info("Stopping the scanning thread...")

        # Set the event to signal the logging loop to stop
        logging.info("Stopping the scanning thread...")
        self.stop_scanning_thread_event.set()  # Set the event here

        self.scanning_thread.join()  # Wait for the thread to finish
        self.scanning_thread = None  # Clean up the thread
        logging.info("scanning thread stopped and disposed.")

    def start_scan(self, scan_config):
        """
        Start a scan while dynamically performing actions 
        or handling 'noscan' (or 'statistics') scans during the acquisition process.
        
        Args:
            scan_config (list of dicts, optional): List of scan configurations for dynamic actions.
                                                   Supports 'noscan' (or 'statistics') for no-action scans.
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

            ###############
            # add in the data loggin portion here
            ##############
            if self.save_data:
                logging.info('add data saving here')
                self.results = self.data_logger.start_logging()
            else:
                logging.info('not doing any data saving')

            # start the acquisition loop
            self.scan_execution_loop()

        except Exception as e:
            logging.error(f"Error during scanning: {e}")

        logging.info("Stopping scaning.")
        time.sleep(1)
        log_df = self.stop_scan()

        return log_df  # Return the DataFrame with the logged data

    def stop_scan(self):
        """
        Stop both event-driven and asynchronous logging, and reset necessary states for reuse.
        """
        log_df = pd.DataFrame()

        if self.save_data:
            # Step 1: Stop data logging and handle device saving states
            self._stop_saving_devices()

        # Step 5: Restore the initial state of devices
        if self.initial_state is not None:
            self.restore_initial_state(self.initial_state)
        
        # Step 4: Turn the trigger back on
        self._set_trigger('on', 0.5)


        if self.save_data:
            # Step 6: Process results, save to disk, and log data
            log_df = self._process_results()

            # Step 7: Process and rename data files
            self.data_interface.process_and_rename()

        # Step 8: Stop the console logging
        self.console_logger.stop_logging()

        # Step 9: Move log file if data was saved
        if self.save_data:
            self.console_logger.move_log_file(self.data_txt_path.parent)

        return log_df

    def _stop_saving_devices(self):
        """
        Stop the data logger, close threads, and update saving paths for non-scalar devices.
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
                
        # Ensure asynchronous commands have time to finish
        time.sleep(2)
       
        logging.info("scanning has stopped for all devices.")

    def _process_results(self):
        """
        Convert results to DataFrame, save data, and write to TDMS.
        """
        if self.results:
            log_df = self.convert_to_dataframe(self.results)
            logging.info("Data logging complete. Returning DataFrame.")

            # Save results to .txt and .h5
            self.save_to_txt_and_h5(log_df, self.data_txt_path, self.data_h5_path, self.sFile_txt_path)

            # Write TDMS files (data and index)
            self.dataframe_to_tdms(log_df)
            self.dataframe_to_tdms(log_df, is_index=True)

            return log_df
        else:
            logging.warning("No data was collected during the logging period.")
            return pd.DataFrame()
     
    def pre_logging_setup(self, scan_config):
        """
        Precompute all scan steps (including composite and normal variables), 
        add scan devices to async_observables, and store the scan steps.
        """

        logging.info("Turning off the trigger.")
        self.trigger_off()

        time.sleep(2)

        if self.save_data:
            self.data_interface.get_next_scan_folder()
            self.create_and_set_data_paths()
            self.write_scan_info_ini(scan_config)

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
            logging.info("attempting to execute pre scan actions.")
            logging.info(f'action list {self.device_manager.scan_setup_action}')

            self.action_manager.add_action({'setup_action': self.device_manager.scan_setup_action})
            self.action_manager.execute_action('setup_action')
            
        logging.info("Pre-logging setup completed.")

    def _add_scan_devices_to_async_observables(self, scan_config):
        """
        Add devices/variables involved in the scan to async_observables if not already present.
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
        Generate a full list of scan steps ahead of time.
        This method handles both normal and composite variables.
        """
        self.bin_num = 0
        steps = []
        
        device_var = scan_config['device_var']

        if self.device_manager.is_statistic_noscan(device_var):
            steps.append({
                'variables': device_var,
                'wait_time': scan_config.get('wait_time', 1),
                'is_composite': False
            })
        elif self.device_manager.is_composite_variable(device_var):
            current_value = scan_config['start']
            while current_value <= scan_config['end']:
                component_vars = self.device_manager.get_composite_components(device_var, current_value)
                steps.append({
                    'variables': component_vars,
                    'wait_time': scan_config.get('wait_time', 1),
                    'is_composite': True
                })
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
        Adjusts the scan steps based on the initial state of the devices if relative scanning is enabled.
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
        Executes the scan loop over precomputed scan steps, stopping if the stop event is triggered.
        """
        log_df = pd.DataFrame()  # Initialize in case of early exit

        while self.scan_steps:
            # Check if the stop event is set, and exit if so
            if self.stop_scanning_thread_event.is_set():
                logging.info("Scanning has been stopped externally.")
                break
                
            scan_step = self.scan_steps.pop(0)
            self._execute_step(scan_step['variables'], scan_step['wait_time'], scan_step['is_composite'])
        


        logging.info("Stopping logging.")

        return log_df

    def _execute_step(self, component_vars, wait_time, is_composite):

        """
        Executes a single step of the scan, handling both composite and normal variables.
        Ensures the trigger is turned off before all moves and turned back on after.
        """

        logging.info("Pausing logging. Turning trigger off before moving devices.")
        self.data_logger.bin_num += 1
        self.trigger_off()

        logging.info(f"shot control state: {self.shot_control.state}")
        if not self.device_manager.is_statistic_noscan(component_vars):
            for device_var, current_value in component_vars.items():
                device_name, var_name = device_var.split(':', 1)
                device = self.device_manager.devices.get(device_name)

                if device:
                    device.set(var_name, current_value)
                    logging.info(f"Set {var_name} to {current_value} for {device_name}")

        logging.info("Resuming logging. Turning trigger on after all devices have been moved.")
        self.trigger_on()
        logging.info(f"shot control state: {self.shot_control.state}")

        current_time = 0
        interval_time = 0.1
        while current_time < wait_time:
            if self.stop_scanning_thread_event.is_set():
                logging.info("Scanning has been stopped externally.")
                break

            time.sleep(interval_time)
            current_time += interval_time

        self.trigger_off()
        logging.info(f"shot control state: {self.shot_control.state}")

    def estimate_acquisition_time(self, scan_config):
        """
        Estimate the total acquisition time based on the scan configuration.
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
        if self.acquisition_time == 0:
            return 0
        completion = self.data_logger.get_current_shot()/self.acquisition_time
        return 1 if completion > 1 else completion

    def get_initial_state(self, scan_config):
        """
        Initialize the state of each scan based on the current values from the subscribers.
        Handles both normal and composite variables.
        Returns a dictionary where each `device_var` is mapped to its current value.
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
        Restore devices to their initial states using the provided initial_state dictionary.
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
        
    def create_and_set_data_paths(self):
        """
        Create data paths for devices that have save_non_scalar=True and set the save data path on those devices.

        Args:
            save_non_scalar_devices (list): List of device names that should save non-scalar data.
        """

        self.data_interface.next_scan_folder = self.data_interface.get_next_scan_folder()

        for device_name in self.device_manager.non_scalar_saving_devices:
            data_path_client_side, data_path_local_side = self.data_interface.build_device_save_paths(device_name)
            self.data_interface.create_device_save_dir(data_path_local_side)

            device = self.device_manager.devices.get(device_name)
            if device:
                save_path = str(data_path_client_side).replace('/', "\\")
                logging.info(f"Setting save data path for {device_name} to {save_path}")
                device.set("localsavingpath", save_path, sync=False)
                time.sleep(.1)
                device.set('save', 'on', sync=False)
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

        analysis_save_path = self.data_interface.build_analysis_save_path()
        self.data_interface.create_device_save_dir(analysis_save_path)

        tdms_output_path, tdms_index_output_path, self.data_txt_path, self.data_h5_path, self.sFile_txt_path = self.data_interface.build_scalar_data_save_paths()
        self.tdms_writer = TdmsWriter(tdms_output_path)
        self.tdms_index_writer = TdmsWriter(tdms_index_output_path)

        time.sleep(1)

    def write_scan_info_ini(self, scan_config):
        """
        Write the scan configuration to an .ini file with the required format.

        Args:
            scan_config (dict): Dictionary containing scan parameters.
        """
        # Check if scan_config is a dictionary
        if not isinstance(scan_config, dict):
            logging.error(f"scan_config is not a dictionary: {type(scan_config)}")
            return

        # Define the file name, replacing 'XXX' with the scan number
        scan_folder = self.data_interface.next_scan_folder
        scan_number = int(scan_folder[-3:])
        filename = f"ScanInfo{scan_folder}.ini"

        scan_var = scan_config.get('device_var', '')
        additional_description = scan_config.get('additional_description', '')

        scan_info = f'{self.device_manager.scan_base_description}. scanning {scan_var}. {additional_description}'

        # Add the Scan Info section
        config_file_contents = [
            "[Scan Info]\n",
            f"Scan No = \"{scan_number}\"\n",
            f"ScanStartInfo = \"{scan_info}\"\n",
            f"Scan Parameter = \"{scan_var}\"\n",
            f"Start = \"{scan_config.get('start', 0)}\"\n",
            f"End = \"{scan_config.get('end', 0)}\"\n",
            f"Step size = \"{scan_config.get('step', 1)}\"\n",
            f"Shots per step = \"{scan_config.get('wait_time', 1)}\"\n",
            f"ScanEndInfo = \"\""
        ]

        # Create the full path for the file
        full_path = Path(self.data_txt_path.parent) / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Attempting to write to {full_path}")

        # Write to the .ini file
        with full_path.open('w') as configfile:
            for line in config_file_contents:
                configfile.write(line)

        logging.info(f"Scan info written to {full_path}")
    
    def fill_async_nans(self, log_df, async_observables, fill_value=0):
        """
        Fill NaN values and empty strings in asynchronous observable columns with the most recent non-NaN value.
        If a column starts with NaN or empty strings, it will backfill with the first non-NaN value.
        After forward and backward filling, remaining NaN or empty entries are filled with `fill_value`.

        Args:
            log_df (pd.DataFrame): The DataFrame containing the logged data.
            async_observables (list): A list of asynchronous observables (columns) to process.
            fill_value (int, float): Value to fill remaining NaN and empty entries (default is 0).
        """
        # Convert empty strings ('') to NaN
        log_df.replace('', pd.NA, inplace=True)

        for async_obs in async_observables:
            if async_obs in log_df.columns:
                # Only process if the entire column is not NaN
                if not log_df[async_obs].isna().all():
                    # First, apply forward fill (ffill) to propagate the last known value
                    log_df[async_obs] = log_df[async_obs].ffill()

                    # Then, apply backward fill (bfill) to fill leading NaNs with the first non-NaN value
                    log_df[async_obs] = log_df[async_obs].bfill()
                else:
                    logging.warning(f"Column {async_obs} consists entirely of NaN values and will be left unchanged.")

        # Finally, fill any remaining NaN values (including converted empty strings) with the specified fill_value
        log_df = log_df.fillna(fill_value)

        # Use infer_objects to downcast the object dtype arrays appropriately
        log_df = log_df.infer_objects(copy=False)

        logging.info(f"Filled remaining NaN and empty values with {fill_value}.")
        return log_df
    
    def save_to_txt_and_h5(self, df, txt_file_path, h5_file_path, sFile_path):
        """
        Save the DataFrame to both .txt (as tab-separated values) and .h5 (HDF5) formats.

        Args:
            df (pd.DataFrame): The DataFrame to be saved.
            txt_file_path (str): Path to save the .txt file.
            h5_file_path (str): Path to save the .h5 file.
        """
        # Save as .txt file (TSV format)
        df.to_csv(txt_file_path, sep='\t', index=False)
        df.to_csv(sFile_path, sep='\t', index=False)
        logging.info(f"Data saved to {txt_file_path}")

        # # Save as .h5 file (HDF5 format)
        # df.to_hdf(h5_file_path, key='data', mode='w')
        # logging.info(f"Data saved to {h5_file_path}")

    def convert_to_dataframe(self, log_entries):
        """
        Convert the synchronized log entries dictionary into a pandas DataFrame.
        """
        log_df = pd.DataFrame.from_dict(log_entries, orient='index')
        log_df = log_df.sort_values(by='Elapsed Time').reset_index(drop=True)

        async_observables = self.device_manager.async_observables
        log_df = self.fill_async_nans(log_df, async_observables)

        # Modify the headers
        new_headers = self.modify_headers(log_df.columns, device_dict)
        log_df.columns = new_headers

        return log_df

    def modify_headers(self, headers, device_dict):
        new_headers = []
        for header in headers:
            if ':' in header:
                device_name, variable = header.split(':')
                new_header = f"{device_name} {variable}"
                # Check if alias exists
                alias = device_dict.get(device_name, {}).get(variable, {}).get('alias')
                if alias:
                    new_header = f"{new_header} Alias:{alias}"
            else:
                new_header = header  # Keep the header as is if no colon
            new_headers.append(new_header)
        return new_headers
           
    
    def dataframe_to_tdms(self, df, is_index=False):
        """
        Convert a DataFrame to a TDMS file or a TDMS index file.
        Groups are created based on the part before the colon in column names,
        and channels are created based on the part after the colon. If no colon exists,
        the group and channel will be the column header.

        Args:
            df (pd.DataFrame): The DataFrame to convert.
            is_index (bool): If True, writes only the structure (groups and channels) without data.
        """

        # Use the correct writer based on whether it's an index file or a data file
        tdms_writer = self.tdms_index_writer if is_index else self.tdms_writer

        with tdms_writer:
            for column in df.columns:
                # Check if the column name contains a colon to split it into group and channel
                if ':' in column:
                    group_name, channel_name = column.split(':', 1)
                else:
                    # If no colon, use the column name for both group and channel
                    group_name = channel_name = column

                # If it's an index file, write an empty array, otherwise write actual data
                data = [] if is_index else df[column].values

                # Create the ChannelObject with the appropriate group, channel, and data
                ch_object = ChannelObject(group_name, channel_name, data)

                # Write the channel data or structure to the TDMS file
                tdms_writer.write_segment([ch_object])

        logging.info(f"TDMS {'index' if is_index else 'data'} file written successfully.")
    
