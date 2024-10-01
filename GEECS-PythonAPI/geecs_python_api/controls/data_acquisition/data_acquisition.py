import numpy as np
import time
import threading
import queue
from datetime import datetime
import logging
import pandas as pd
import yaml

import subprocess
import platform
from pathlib import Path
import os
import re

from nptdms import TdmsWriter, ChannelObject

from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface.geecs_errors import ErrorAPI
import geecs_python_api.controls.interface.message_handling as mh


import asyncio

def setup_logging(log_file="system_log.log", level=logging.INFO, console=False):
    """
    Sets up logging for the module. By default, logs to a file.
    
    Args:
        log_file (str): The file to log to.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        console (bool): If True, also logs to the console.
    """
    # Remove any previously configured handlers to prevent duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging with both file and optional console handlers
    handlers = [logging.FileHandler(log_file)]
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )


# config = load_config()
# if config and 'Experiment' in config and 'expt' in config['Experiment']:
#     default_experiment = config['Experiment']['expt']
#     print(f"default experiment is: {default_experiment}")
# else:
#     print("Configuration file not found or default experiment not defined. While use Undulator as experiment. Could be a problem for you.")
#     default_experiment = 'Undulator'
#
# GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(default_experiment)



class DeviceManager():
    def __init__(self):
        self.devices = {}
        self.event_driven_observables = []  # Store event-driven observables
        self.async_observables = []  # Store asynchronous observables
        self.non_scalar_saving_devices = []  # Store devices that need to save non-scalar data

    def initialize_subscribers(self, variables, clear_devices=True):
        if clear_devices:
            self._clear_existing_devices()
    
        device_map = self.preprocess_observables(variables)
    
        for device_name, var_list in device_map.items():
            if device_name not in self.devices:
                self._subscribe_device(device_name, var_list)

    def _clear_existing_devices(self):
        for device_name, device in self.devices.items():
            try:
                logging.info(f"Attempting to unsubscribe from {device_name}...")
                device.unsubscribe_var_values()
                print('why am i here?')
                device.close()
                logging.info(f"Successfully unsubscribed from {device_name}.")
            except Exception as e:
                logging.error(f"Error unsubscribing from {device_name}: {e}")
        
        self.devices = {}

    def _subscribe_device(self, device_name, var_list):
        device = GeecsDevice(device_name)
        device.use_alias_in_TCP_subscription = False
        logging.info(f'Subscribing {device_name} to variables: {var_list}')
        device.subscribe_var_values(var_list)
        self.devices[device_name] = device

    def close_subscribers(self):
        for device_name, device in self.devices.items():
            try:
                logging.info(f"Attempting to unsubscribe from {device_name}...")
                device.unsubscribe_var_values()
                logging.info(f"Successfully unsubscribed from {device_name}.")
            except Exception as e:
                logging.error(f"Error unsubscribing from {device_name}: {e}")

    def get_values(self, variables):
        results = {}
        for var in variables:
            device_name, _ = var.split(':')
            if device_name not in results:
                results[device_name] = {}        
                var_list = [v.split(':')[1] for v in variables if v.startswith(device_name)]
                state = self.devices[device_name].state
                for device_var in var_list:
                    results[device_name][device_var] = state[device_var]
                results[device_name]['fresh'] = state['fresh']
                results[device_name]['shot number'] = state['shot number']
                self.devices[device_name].state['fresh'] = False

        return self.parse_tcp_states(results)

    def parse_tcp_states(self, get_values_result):
        parsed_dict = {}
        shared_keys = ['fresh', 'shot number']
        for device_name, nested_dict in get_values_result.items():
            common_data = {k: v for k, v in nested_dict.items() if k in shared_keys}
            for variable_name, value in nested_dict.items():
                if variable_name not in shared_keys:
                    new_key = f'{device_name}:{variable_name}'
                    parsed_dict[new_key] = {'value': value, **common_data}
        return parsed_dict

    def preprocess_observables(self, observables):
        device_map = {}
        for observable in observables:
            device_name, var_name = observable.split(':')
            if device_name not in device_map:
                device_map[device_name] = []
            device_map[device_name].append(var_name)
        return device_map

    def load_from_config(self, config_path):
        """
        Load configuration from a YAML file, including scan info, parameters, and device observables.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Load scan info
        self.scan_info = config.get('scan_info', {})
        self.scan_parameters = config.get('scan_parameters', {})
    
        # Initialize devices
        devices = config.get('Devices', {})
        print(devices)
        for device_name, device_config in devices.items():
            variable_list = device_config.get('variable_list', [])
            synchronous = device_config.get('synchronous', False)
            save_non_scalar = device_config.get('save_nonscalar_data', False)
        
            if save_non_scalar:
                self.non_scalar_saving_devices.append(device_name)
        
            if synchronous:
                # Add to event-driven observables
                self.event_driven_observables.extend([f"{device_name}:{var}" for var in variable_list])
            else:
                # Add to asynchronous observables
                self.async_observables.extend([f"{device_name}:{var}" for var in variable_list])
    
        # Initialize the subscribers
        self.initialize_subscribers(self.event_driven_observables + self.async_observables, clear_devices=True)
    
        # Optionally, log the scan info and parameters for reference
        logging.info(f"Loaded scan info: {self.scan_info}")
        logging.info(f"Loaded scan parameters: {self.scan_parameters}")

class DataInterface():
    def __init__(self):
        # Initialize domain, base path, and current date
        self.domain = self.get_domain_windows()
        self.local_base_path = self.get_local_base_path()
        self.client_base_path = Path('Z:/data/Undulator')
        self.year, self.month, self.day = self.get_current_date()
        self.dummy_scan_number = 100
        self.local_scan_dir_base, self.local_analysis_dir_base = self.create_data_path(self.dummy_scan_number)
        self.client_scan_dir_base, self.client_analysis_dir_base = self.create_data_path(self.dummy_scan_number, local=False)
        self.next_scan_folder = self.get_next_scan_folder()

    def get_domain_windows(self):
        """
        Attempt to retrieve the Windows domain from the system info.
        Returns the domain name if found, otherwise returns None.
        """
        try:
            result = subprocess.check_output(['systeminfo'], stderr=subprocess.STDOUT, text=True)
            domain_line = next((line for line in result.splitlines() if "Domain" in line), None)
            if domain_line:
                return domain_line.split(':')[1].strip()
        except Exception as e:
            # Log the error if necessary
            return None

    def get_local_base_path(self):
        """
        Determines the base path for data storage based on domain or hostname.
        """
        domain = self.get_domain_windows()
        hostname = platform.node()

        if domain == 'loasis.gov':
            return Path('Z:/data/Undulator')
        elif hostname == 'Samuels-MacBook-Pro.local':
            return Path('/Volumes/hdna2/data/Undulator')
        else:
            raise ValueError('Unknown computer. Path to data is unknown.')

    def get_current_date(self):
        """
        Returns the current year, month, and day in a specific format.
        """
        month_dict = {
            "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
            "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
            "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
        }

        current_date = datetime.now()
        month = month_dict[current_date.strftime("%m")]
        year = current_date.strftime("%Y")
        day = current_date.strftime("%d")

        return year, month, day

    def create_data_path(self, scan_number, local=True):
        """
        Constructs paths for raw and analysis data based on the date and scan number.
        Returns paths for both scan and analysis directories.
        """
        month_dict = {
            "Jan": "01-Jan", "Feb": "02-Feb", "Mar": "03-Mar", "Apr": "04-Apr",
            "May": "05-May", "Jun": "06-Jun", "Jul": "07-Jul", "Aug": "08-Aug",
            "Sep": "09-Sep", "Oct": "10-Oct", "Nov": "11-Nov", "Dec": "12-Dec"
        }

        # Construct base paths for either local or client side
        base_path = self.local_base_path if local else self.client_base_path
        date_folder = f"{self.year[-2:]}_{month_dict[self.month][:2]}{self.day.zfill(2)}"

        base_folder = base_path / f"Y{self.year}" / month_dict[self.month] / date_folder
        raw_data_path = base_folder / "scans"
        analysis_data_path = base_folder / "analysis"

        return raw_data_path, analysis_data_path

    def reset_date(self, year, month, day):
        """
        Resets the paths using the new date, reinitializing local scan and analysis directories.
        """
        self.year, self.month, self.day = year, month, day
        self.local_scan_dir_base, self.local_analysis_dir_base = self.create_data_path(self.dummy_scan_number)

    def get_last_scan_number(self):
        """
        Retrieves the last scan number by scanning the directory for folders starting with 'Scan' prefix.
        Returns -1 if no valid scan folders are found.
        """
        data_folder = self.local_scan_dir_base
        if not data_folder.is_dir():
            return -1
        
        scan_folders = next(os.walk(data_folder))[1]
        scan_folders = [x for x in scan_folders if re.match(r'^Scan\d{3}$', x)]
        
        if scan_folders:
            return int(scan_folders[-1][-3:])
        else:
            return -1

    def build_device_save_paths(self, device):
        """
        Builds and returns the save paths for both client and local side for a given device.
        """
        device_save_path_client = self.client_scan_dir_base / self.next_scan_folder / device
        device_save_path_local = self.local_scan_dir_base / self.next_scan_folder / device
        
        return device_save_path_client, device_save_path_local

    def build_analysis_save_path(self):
        """
        Builds and returns the save paths for both client and local side for a given device.
        """
        analysis_save_path = self.local_analysis_dir_base / self.next_scan_folder

        return analysis_save_path

    def build_scalar_data_save_paths(self):
        """
        Builds and returns the save path tdms file.
        """
        
        tdms_save_path = self.local_scan_dir_base / self.next_scan_folder / f"{self.next_scan_folder}.tdms"
        tdms_index_save_path = self.local_scan_dir_base / self.next_scan_folder / f"{self.next_scan_folder}.tdms_index"

        data_txt_path = self.local_scan_dir_base / self.next_scan_folder / f"ScanData{self.next_scan_folder}.txt"
        data_h5_path = self.local_scan_dir_base / self.next_scan_folder / f"ScanData{self.next_scan_folder}.h5"

        # sFile_txt_path = self.local_analysis_dir_base / self.next_scan_folder / f"s{int(self.next_scan_folder[-3:])}.txt"
        sFile_txt_path = self.local_analysis_dir_base / f"s{int(self.next_scan_folder[-3:])}.txt"
        
        

        return tdms_save_path, tdms_index_save_path, data_txt_path, data_h5_path, sFile_txt_path

    def create_device_save_dir(self, local_device_save_path):
        """
        Ensures that the directories exist for saving data by creating them if they don't exist.
        """
        local_device_save_path.mkdir(parents=True, exist_ok=True)

    def get_next_scan_folder(self):
        """
        Determines the next scan folder by incrementing the last scan number or starting at 'Scan001'.
        """
        last_scan = self.get_last_scan_number()
        return f'Scan{last_scan + 1:03d}' if last_scan > 0 else 'Scan001'

    def count_files_in_scan_directory(self, scan_dir_base, device):
        """
        Counts the number of files for a specific device in the scan directory.
        """
        device_dir = Path(scan_dir_base) / device
        if device_dir.exists():
            return len(list(device_dir.glob('*.png')))  # Assuming you are counting .png files
        return 0

class DataLogger():
    def __init__(self, device_manager, data_interface):
        self.device_manager = device_manager
        self.data_interface = data_interface
        self.stop_event = threading.Event()  # Event to control polling thread
        self.poll_thread = None  # Placeholder for polling thread
        self.async_t0 = None  # Placeholder for async t0
        self.warning_timeout_sync = 2  # Timeout for synchronous devices (seconds)
        self.warning_timeout_async_factor = 1  # Factor of polling interval for async timeout
        self.last_log_time_sync = {}  # Dictionary to track last log time for synchronous devices
        self.last_log_time_async = {}  # Dictionary to track last log time for async devices
        self.shot_control = GeecsDevice('U_DG645_ShotControl')
        self.polling_interval = .5
        self.results = {}  # Store results for later processing
        self.stop_logging_thread_event = threading.Event()  # Event to signal the logging thread to stop
        
        self.logging_thread = None  # NEW: Separate thread for logging

        self.tdms_writer = None

    def _set_trigger(self, state: str):
        """Helper method to turn the trigger on or off."""
        valid_states = {
            'on': 'External rising edges',
            'off': 'Single shot external rising edges'
        }
        if state in valid_states:
            self.shot_control.set('Trigger.Source', valid_states[state])
            logging.info(f"Trigger turned {state}.")
        else:
            logging.error(f"Invalid trigger state: {state}")

    def trigger_off(self):
        """Turns off the trigger."""
        self._set_trigger('off')

    def trigger_on(self):
        """Turns on the trigger."""
        self._set_trigger('on')

    def create_and_set_data_paths(self):
        """
        Create data paths for devices that have save_non_scalar=True and set the save data path on those devices.
        
        Args:
            save_non_scalar_devices (list): List of device names that should save non-scalar data.
        """
        for device_name in self.device_manager.non_scalar_saving_devices:
            data_path_client_side, data_path_local_side = self.data_interface.build_device_save_paths(device_name)
            self.data_interface.create_device_save_dir(data_path_local_side)
            
            device = self.device_manager.devices.get(device_name)
            if device:
                save_path = str(data_path_client_side).replace('/', "\\")
                logging.info(f"Setting save data path for {device_name} to {save_path}")
                device.set("localsavingpath", save_path)
                device.set('save', 'on')
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

        analysis_save_path = self.data_interface.build_analysis_save_path()
        self.data_interface.create_device_save_dir(analysis_save_path)

        tdms_output_path, tdms_index_output_path, self.data_txt_path, self.data_h5_path, self.sFile_txt_path = self.data_interface.build_scalar_data_save_paths()
        self.tdms_writer = TdmsWriter(tdms_output_path)
        self.tdms_index_writer = TdmsWriter(tdms_index_output_path)

        # Method to check if the logging thread is alive
        
    def is_logging_active(self):
        return self.logging_thread and self.logging_thread.is_alive()

    def start_logging_thread(self, acquisition_time=10, scan_config=None): 
        if self.is_logging_active():
            logging.warning("Logging is already active, cannot start a new logging session.")
            return
        
        # Ensure the stop event is cleared before starting a new session
        self.stop_logging_thread_event.clear()
    
        # Start a new thread for logging
        self.logging_thread = threading.Thread(target=self.start_logging_wait, args=(acquisition_time, scan_config))
        self.logging_thread.start()
        logging.info("Logging thread started.")

    def stop_logging_thread(self):
        if not self.is_logging_active():
            logging.warning("No active logging thread to stop.")
            return

        # Stop the logging and wait for the thread to finish
        logging.info("Stopping the logging thread...")
        self.stop_logging()
        self.logging_thread.join()  # Wait for the thread to finish
        self.logging_thread = None  # Clean up the thread
        logging.info("Logging thread stopped and disposed.")

    def start_logging_wait(self, acquisition_time=None, scan_config=None):
        """
        Start logging for a set amount of time, while dynamically performing actions 
        or handling 'statistics' scans during the acquisition process.
        
        Args:
            acquisition_time (int, optional): Total time to log data (in seconds). If not provided, 
                                              it will be estimated from the scan_config.
            scan_config (list of dicts, optional): List of scan configurations for dynamic actions.
                                                   Supports 'statistics' for no-action scans.
        """
        log_df = pd.DataFrame()  # Initialize in case of early exit
        try:
            # Pre-logging setup: Trigger devices off, initialize data paths, etc.
            self.pre_logging_setup()
    
            # Estimate acquisition time if necessary
            if acquisition_time is None and scan_config:
                acquisition_time = self.estimate_acquisition_time(scan_config)
                logging.info(f"Estimated acquisition time based on scan config: {acquisition_time} seconds.")
    
            # Initialize scan state if scan_config is provided
            scan_state = self.initialize_scan_state(scan_config)
    
            # Start logging and trigger devices on
            self.results = self.start_logging()
            # self.trigger_on()
    
            # Acquisition loop: Handle actions and acquisition time
            log_df = self.acquisition_loop(acquisition_time, scan_config, scan_state)
    
        except Exception as e:
            logging.error(f"Error during logging: {e}")
        finally:
            # Ensure cleanup and exit
            self.trigger_on()
            logging.info("Exiting the logging thread.")
    
        return log_df  # Return the DataFrame with the logged data
        
    def pre_logging_setup(self):
        """Helper method to perform pre-logging setup tasks like initializing data paths."""
        logging.info("Turning off the trigger.")
        self.trigger_off()
    
        time.sleep(3)  # Wait for devices to go to standby
    
        # Initialize the logging process
        self.data_interface.get_next_scan_folder()
        self.create_and_set_data_paths()
        time.sleep(2)  # Wait before starting logging
        logging.info("Pre-logging setup completed.")
       
    def estimate_acquisition_time(self, scan_config):
        """
        Estimate the total acquisition time based on the scan configuration.
        """
        total_time = 0
    
        for scan in scan_config:
            if scan['device_var'] == 'statistics':
                total_time += scan.get('acquisition_time', 10)  # Default to 10 seconds if not provided
            else:
                start = scan['start']
                end = scan['end']
                step = scan['step']
                wait_time = scan.get('wait_time', 1)  # Default wait time between steps is 1 second
    
                # Calculate the number of steps and the total time for this device
                steps = (end - start) / step
                total_time += steps * wait_time
        
        logging.info(f'Estimated scan time: {total_time}')
        
        return total_time
       
    def initialize_scan_state(self, scan_config):
        """
        Initialize the state of each scan based on the start values from scan_config.
        Returns a dictionary where each `device_var` is mapped to its current value.
        """
        scan_state = {}
        if scan_config:
            for scan in scan_config:
                device_var = scan['device_var']
                if device_var != 'statistics':
                    start = scan['start']
                    scan_state[device_var] = start  # Set the current position to the start value
        return scan_state
       
    def acquisition_loop(self, acquisition_time, scan_config, scan_state):
        """
        Main acquisition loop that handles both the acquisition time and actions from scan_config.
        Returns the logged DataFrame.
        """
        elapsed_time = 0
        check_interval = 0.1  # Check every 100ms for abort signal
        action_interval = 1.0  # Interval at which to move stages
        action_timer = 0  # Track time for the next action
    
        # Main acquisition loop
        while elapsed_time < acquisition_time:
            if self.stop_logging_thread_event.is_set():  # If an abort signal is received
                logging.info("Logging has been stopped externally (abort signal).")
                break
    
            # Perform actions at specified intervals (e.g., move stages)
            if scan_config and action_timer >= action_interval:
                self.perform_scan_actions(scan_config, scan_state)
                action_timer = 0  # Reset action timer
    
            # Increment time
            time.sleep(check_interval)
            elapsed_time += check_interval
            action_timer += check_interval
    
        # Stop logging at the end of the acquisition or after an abort
        logging.info("Stopping logging.")
        log_df = self.stop_logging()
    
        return log_df

    def perform_scan_actions(self, scan_config, scan_state):
        """
        Perform the configured actions such as moving stages or logging statistics.
        """
        for scan in scan_config:
            device_var = scan['device_var']  # "DeviceName:Variable"

            if device_var == 'statistics':
                # Turn the trigger on for statistics scans
                logging.info("Performing statistics scan. Turning trigger on.")
                self.trigger_on()  # Ensure the trigger is turned on for data collection

                acquisition_time = scan.get('acquisition_time', 10)  # Default to 10 seconds if not provided
                logging.info(f"Statistics scan: Collecting data for {acquisition_time} seconds.")

                time.sleep(acquisition_time)  # Simulate waiting for the data collection to complete

                logging.info("Statistics scan completed. Turning trigger off.")
                self.trigger_off()  # Turn the trigger off after the data collection
                continue  # Move to the next scan in the config

            # Perform device movements for non-statistics scans
            start = scan['start']
            end = scan['end']
            step = scan['step']
            wait_time = scan.get('wait_time', 0)  # Default to no wait after step

            # Split device_var into device name and variable
            device_name, var_name = device_var.split(':', 1)
            device = self.device_manager.devices.get(device_name)

            if device:
                current_value = scan_state[device_var]

                # Only move the device if it hasn't reached the end
                if current_value <= end:
                    # Turn the trigger off before the move
                    logging.info(f"Pausing logging. Turning trigger off before moving {device_name}.")
                    self.trigger_off()

                    # Perform the device move
                    device.set(var_name, current_value)
                    # time.sleep(1.5)
                    
                    # Turn the trigger back on after the move
                    logging.info(f"Resuming logging. Turning trigger on after moving {device_name}.")
                    self.trigger_on()
                    
                    logging.info(f"Set {var_name} to {current_value} for {device_name}")
                    scan_state[device_var] = current_value + step

                    # Wait for the movement to complete
                    time.sleep(wait_time)
                    self.trigger_off()

    def start_logging(self):
        """
        Start logging data for all devices. Event-driven observables will trigger logs,
        and asynchronous observables will be polled at a regular interval.
        """
        last_timestamps = {}
        initial_timestamps = {}
        standby_mode = {}
        log_entries = {}
        async_t0_set = False  # Track if async t0 has been set

        # Access event-driven and async observables from DeviceManager
        event_driven_observables = self.device_manager.event_driven_observables
        async_observables = self.device_manager.async_observables
    
        def log_update(message, device):
            """
            Handle updates from TCP subscribers and log them when a new timestamp is detected.
            """
            nonlocal async_t0_set  # Keep track if async t0 has been set
            current_timestamp = self._extract_timestamp(message, device)
    
            if current_timestamp is None:
                return
    
            if self._initialize_standby_mode(device, standby_mode, initial_timestamps, current_timestamp):
                return
    
            elapsed_time = self._calculate_elapsed_time(device, initial_timestamps, current_timestamp)
    
            if self._check_duplicate_timestamp(device, last_timestamps, current_timestamp):
                return
    
            self._log_device_data(device, event_driven_observables, log_entries, elapsed_time)
    
            # Set the t0 for asynchronous devices when the first log entry is created
            if not async_t0_set:
                self.async_t0 = time.time()  # Set t0 for async logging
                async_t0_set = True
                logging.info(f"Asynchronous devices t0 set to {self.async_t0}")

            # Update the last log time for this synchronous device
            self.last_log_time_sync[device.get_name()] = time.time()
    
        # Register the logging function for event-driven observables
        self._register_event_logging(event_driven_observables, log_update)
    
        logging.info("Logging has started for all event-driven devices.")
    
        # Start the asynchronous polling in a separate thread
        self._start_async_polling(async_observables, log_entries, timeout = 10)

        # # Start a thread to monitor device warnings
        # self.warning_thread = threading.Thread(target=self._monitor_warnings, args=(event_driven_observables, async_observables))
        # self.warning_thread.start()

        return log_entries
    
    def _extract_timestamp(self, message, device):
        stamp = datetime.now().__str__()
        err = ErrorAPI()
        net_msg = mh.NetworkMessage(tag=device.get_name(), stamp=stamp, msg=message, err=err)
        parsed_data = device.handle_subscription(net_msg)
    
        current_timestamp = parsed_data[2].get('timestamp')
        if current_timestamp is None:
            logging.warning(f"No timestamp found for {device.get_name()}. Using system time instead.")
            current_timestamp = float(stamp)
        return float(current_timestamp)

    def _register_event_logging(self, event_driven_observables, log_update):
        for device_name, device in self.device_manager.devices.items():
            for observable in event_driven_observables:
                if observable.startswith(device_name):
                    logging.info(f"Registering logging for event-driven observable: {observable}")
                    device.event_handler.register('update', 'logger', lambda msg, dev=device: log_update(msg, dev))

    def _start_async_polling(self, async_observables, log_entries, timeout=10):
        """
        Start asynchronous polling in a separate thread, with a check for async_t0 and a timeout mechanism.
        """
        # If there's an existing polling thread, wait for it to finish before starting a new one
        if self.poll_thread and self.poll_thread.is_alive():
            self.poll_thread.join()

        self.poll_thread = threading.Thread(target=self.poll_async_observables, args=(async_observables, log_entries, timeout))
        self.poll_thread.start()

    def poll_async_observables(self, async_observables, log_entries, timeout):
        """
        Poll asynchronous observables at a regular interval and log the data. Wait for async_t0 to be set.
        """
        wait_time = 0
        check_interval = 0.1  # Poll every 0.1 seconds to check if async_t0 is set

        # Wait for async_t0 to be set or until timeout is reached
        while not self.async_t0 and wait_time < timeout:
            time.sleep(check_interval)
            wait_time += check_interval
        
        if not self.async_t0:
            logging.error("Timeout: async_t0 was not set in time.")
            return  # Exit the polling if async_t0 is not set in time

        # Poll each asynchronous observable and log the data
        while not self.stop_event.is_set():
            for observable in async_observables:
                self._poll_single_observable(observable, log_entries)
            time.sleep(self.polling_interval)

    def _poll_single_observable(self, observable, log_entries):
        """
        Poll a single asynchronous observable and log the data. Handle cases where async_t0 is None.
        """
        if not self.async_t0:
            logging.warning("Async t0 is not set. Skipping polling for now.")
            return
        
        device_name, var_name = observable.split(':')
        device = self.device_manager.devices.get(device_name)
    
        if device:
            current_time = time.time()
            elapsed_time = round(current_time - self.async_t0)
    
            value = device.state.get(var_name, 'N/A')
    
            if elapsed_time in log_entries:
                log_entries[elapsed_time].update({f"{device_name}:{var_name}": value})
            else:
                logging.warning(f"No existing row for elapsed time {elapsed_time}. Skipping log for {observable}.")

            # Update the last log time for this asynchronous device
            self.last_log_time_async[device_name] = time.time()

    def fill_async_nans(self, log_df, async_observables):
        """
        Fill NaN values in asynchronous observable columns with the most recent non-NaN value.
        If a column starts with NaN, it will backfill with the first non-NaN value.
        If the entire column is NaN, it will be left unchanged.
    
        Args:
            log_df (pd.DataFrame): The DataFrame containing the logged data.
            async_observables (list): A list of asynchronous observables (columns) to process.
        """
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
        
        logging.info("Filled NaN values in asynchronous variables.")
        return log_df

    def _initialize_standby_mode(self, device, standby_mode, initial_timestamps, current_timestamp):
        if device.get_name() not in standby_mode:
            standby_mode[device.get_name()] = True
        if device.get_name() not in initial_timestamps:
            initial_timestamps[device.get_name()] = current_timestamp
            logging.info(f"Initial dummy timestamp for {device.get_name()} set to {current_timestamp}. Standby mode enabled.")
        t0 = initial_timestamps[device.get_name()]
        if standby_mode[device.get_name()] and current_timestamp != t0:
            standby_mode[device.get_name()] = False
            logging.info(f"Device {device.get_name()} exiting standby mode. First real timestamp: {current_timestamp}")
            initial_timestamps[device.get_name()] = current_timestamp
        elif standby_mode[device.get_name()]:
            logging.info(f"Device {device.get_name()} is still in standby mode.")
            return True
        return False

    def _calculate_elapsed_time(self, device, initial_timestamps, current_timestamp):
        t0 = initial_timestamps[device.get_name()]
        return round(current_timestamp - t0)

    def _check_duplicate_timestamp(self, device, last_timestamps, current_timestamp):
        if device.get_name() in last_timestamps and last_timestamps[device.get_name()] == current_timestamp:
            logging.info(f"Timestamp hasn't changed for {device.get_name()}. Skipping log.")
            return True
        last_timestamps[device.get_name()] = current_timestamp
        return False

    def _log_device_data(self, device, event_driven_observables, log_entries, elapsed_time):
        observables_data = {
            observable.split(':')[1]: device.state.get(observable.split(':')[1], '')
            for observable in event_driven_observables if observable.startswith(device.get_name())
        }
        if elapsed_time not in log_entries:
            log_entries[elapsed_time] = {'Elapsed Time': elapsed_time}
        log_entries[elapsed_time].update({
            f"{device.get_name()}:{key}": value for key, value in observables_data.items()
        })

    def _monitor_warnings(self, event_driven_observables, async_observables):
        """
        Monitor the last log time for each device and issue warnings if a device hasn't updated within the threshold.
        """
        while not self.stop_event.is_set():
            current_time = time.time()

            # Check synchronous devices (event-driven observables)
            for observable in event_driven_observables:
                device_name = observable.split(':')[0]
                last_log_time = self.last_log_time_sync.get(device_name, None)

                if last_log_time and (current_time - last_log_time) > self.warning_timeout_sync:
                    logging.warning(f"Synchronous device {device_name} hasn't updated in over {self.warning_timeout_sync} seconds.")

            # Check asynchronous devices
            async_timeout = self.polling_interval * self.warning_timeout_async_factor
            for observable in async_observables:
                device_name = observable.split(':')[0]
                last_log_time = self.last_log_time_async.get(device_name, None)

                if last_log_time and (current_time - last_log_time) > async_timeout:
                    logging.warning(f"Asynchronous device {device_name} hasn't updated in over {async_timeout} seconds.")

            time.sleep(1)  # Monitor the warnings every second

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
    
        print(f"TDMS {'index' if is_index else 'data'} file written successfully.")

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

    def stop_logging(self):
        """
        Stop both event-driven and asynchronous logging, and reset necessary states for reuse.
        """
        # Unregister all event-driven logging
        for device_name, device in self.device_manager.devices.items():
            device.event_handler.unregister('update', 'logger')

        # Signal to stop the polling thread
        self.stop_event.set()

        if self.poll_thread and self.poll_thread.is_alive():
            self.poll_thread.join()  # Ensure the polling thread has finished
        
        # Reset the stop_event for future logging sessions
        self.stop_event.clear()
        self.poll_thread = None
        self.async_t0 = None
        
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                # turn off saving
                device.set('save','off')
                device.set('localsavingpath','c:\\temp')
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")
        
        logging.info("Logging has stopped for all devices.")
        self.trigger_on()

        # Step 5: Process results and convert to DataFrame
        if self.results:
            log_df = self.convert_to_dataframe(self.results)
            logging.info("Data logging complete. Returning DataFrame.")

            return log_df
        else:
            logging.warning("No data was collected during the logging period.")
            return pd.DataFrame()  # Return an empty DataFrame if no data was collected

    def convert_to_dataframe(self, log_entries):
        """
        Convert the synchronized log entries dictionary into a pandas DataFrame.
        """
        log_df = pd.DataFrame.from_dict(log_entries, orient='index')
        log_df = log_df.sort_values(by='Elapsed Time').reset_index(drop=True)

        async_observables = self.device_manager.async_observables
        log_df = self.fill_async_nans(log_df, async_observables)

        self.save_to_txt_and_h5(log_df, self.data_txt_path, self.data_h5_path, self.sFile_txt_path)

        self.dataframe_to_tdms(log_df)
        self.dataframe_to_tdms(log_df, is_index = True)

        return log_df