import numpy as np
import time
import threading
import queue
from datetime import datetime
import logging
import pandas as pd
import yaml
    
import shutil

import subprocess
import platform
from pathlib import Path
import os
import re

import configparser

from nptdms import TdmsWriter, ChannelObject

from concurrent.futures import ThreadPoolExecutor, as_completed

from geecs_python_api.controls.interface import load_config
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface.geecs_errors import ErrorAPI
import geecs_python_api.controls.interface.message_handling as mh

config = load_config()

if config and 'Experiment' in config and 'expt' in config['Experiment']:
    default_experiment = config['Experiment']['expt']
    print(f"default experiment is: {default_experiment}")
else:
    print("Configuration file not found or default experiment not defined. While use Undulator as experiment. Could be a problem for you.")
    default_experiment = 'Undulator'

GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(default_experiment)
device_dict = GeecsDevice.exp_info['devices']


import asyncio

def setup_console_logging(log_file="system_log.log", level=logging.INFO, console=False):
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
     
def stop_console_logging():
    # Iterate over the root logger's handlers and close each one
    for handler in logging.root.handlers[:]:
        handler.close()  # Close the file or stream
        logging.root.removeHandler(handler)  # Remove the handler

    print("Logging has been stopped and handlers have been removed.")

def move_log_file(src_file, dest_dir):
    """
    Moves the log file to the destination directory using shutil to handle cross-device issues.
    """
    src_path = Path(src_file)
    dest_path = Path(dest_dir) / src_path.name
    
    print(f"Attempting to move {src_path} to {dest_path}")
    
    try:
        shutil.move(str(src_path), str(dest_path))
        print(f"Moved log file to {dest_path}")
    except Exception as e:
        print(f"Failed to move {src_path} to {dest_path}: {e}")

class Sounds:
    def __init__(self):
        # Initialize the threading event and the sound file to be played
        self._play_event = threading.Event()
        self._sound_file = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # Run loop that waits for a play event and plays the specified sound
        while not self._stop_event.is_set():
            if self._play_event.is_set() and self._sound_file:
                # Play the sound (adapt this command based on your OS)
                os.system(f'afplay {self._sound_file}')  # macOS specific, change as necessary
                self._play_event.clear()  # Reset after playing
            time.sleep(0.1)  # Prevent busy-waiting

    def play_beep(self):
        """ Play the 'beep' sound """
        self._sound_file = 'trimmed_tink.aiff'  # Replace with actual path to beep sound
        self._play_event.set()

    def play_toot(self):
        """ Play the 'toot' sound """
        self._sound_file = 'Hero.aiff'  # Replace with actual path to toot sound
        self._play_event.set()

    def stop(self):
        """ Stop the sound thread """
        self._stop_event.set()
        self._thread.join()  # Ensure the thread exits cleanly  

# Example usage of the Sounds class
sounds = Sounds()

class ConfigManager:
    def __init__(self):
        """
        Initialize the ConfigManager with the base directory for configurations.
        The base directory is set to be relative to the directory where this script/module resides.
        """
        # Get the path of the current file (where this class is defined)
        current_dir = Path(__file__).parent

        # Set the base directory to be the 'configs' directory relative to the current directory
        self.base_dir = current_dir / 'configs'
        
        # Ensure base_dir exists
        if not self.base_dir.exists():
            raise FileNotFoundError(f"The base config directory {self.base_dir} does not exist.")

        self.experiment_dir = None

    def set_experiment_dir(self, experiment: str):
        """
        Set the experiment directory based on the subdirectory name.

        Args:
            experiment (str): Name of the subdirectory for the experiment (e.g., 'HTU').
        """
        self.experiment_dir = self.base_dir / experiment
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"The experiment directory {self.experiment_dir} does not exist.")

    def get_config_path(self, config_file: str) -> Path:
        """
        Get the full path of the specified configuration file in the current experiment directory.

        Args:
            config_file (str): The configuration file to retrieve.

        Returns:
            Path: The full path to the config file.
        """
        if not self.experiment_dir:
            raise ValueError("Experiment directory is not set. Call set_experiment_dir() first.")
        
        config_path = self.experiment_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"The config file {config_path} does not exist.")
        
        return config_path
        
        
    def visa_config_generator(self, visa_key, diagnostic_type):
        
        # input_filename = '../../geecs_python_api/controls/data_acquisition/configs/HTU/visa_plunger_lookup.yaml'
        
        input_filename = self.get_config_path('visa_plunger_lookup.yaml')
        
        with open(input_filename, 'r') as file:
            visa_lookup = yaml.safe_load(file)
        
        device_info = visa_lookup[visa_key]
    
        # Define the VisaEBeam camera dynamically based on the visa_key
        visa_ebeam_camera = f"UC_VisaEBeam{visa_key[-1]}"  # Extracts the last number from visa_key (e.g., visa1 -> UC_VisaEBEam1)
    
        if diagnostic_type == 'energy':
            description = f"collecting data on {visa_key}EBeam and U_FELEnergyMeter"
            setup_steps = [
                {'action': 'execute', 'action_name': 'remove_visa_plungers'},
                {'device': device_info['device'], 'variable': device_info['variable'], 'action': 'set', 'value': 'on'},
                {'device': 'U_Velmex', 'variable': 'Position', 'action': 'set', 'value': device_info['energy_meter_position']}
            ]
            devices = {
                visa_ebeam_camera: {
                    'variable_list': ["timestamp"],
                    'synchronous': True,
                    'save_nonscalar_data': True
                },
                'U_FELEnergyMeter': {
                    'variable_list': ["Python Results.ChA", "timestamp"],
                    'synchronous': True,
                    'save_nonscalar_data': True
                }
            }
        
        elif diagnostic_type == 'spectrometer':
            description = f"collecting data on {visa_key}EBeam and U_Spectrometer"
            setup_steps = [
                {'action': 'execute', 'action_name': 'remove_visa_plungers'},
                {'device': device_info['device'], 'variable': device_info['variable'], 'action': 'set', 'value': 'on'},
                {'device': 'U_Velmex', 'variable': 'Position', 'action': 'set', 'value': device_info['spectrometer_position']}
            ]
            devices = {
                visa_ebeam_camera: {
                    'variable_list': ["timestamp"],
                    'synchronous': True,
                    'save_nonscalar_data': True
                },
                'UC_UndulatorRad2': {
                    'variable_list': ["MeanCounts", "timestamp"],
                    'synchronous': True,
                    'save_nonscalar_data': True
                }
            }
    
        # Constructing the YAML structure
        output_data = {
            'Devices': devices,
            'scan_info': {
                'description': description
            },
            'setup_action': {
                'steps': setup_steps
            }
        }
    
        # Writing to a YAML file
        
        output_filename = input_filename.parent / f'{visa_key}_{diagnostic_type}_setup.yaml'
        with open(output_filename, 'w') as outfile:
            yaml.dump(output_data, outfile, default_flow_style=False)
    
        # print(f"YAML file {output_filename} generated successfully!")
        return output_filename      
        
class ActionManager:
    def __init__(self, experiment_dir: str):
        # Initialize the ConfigManager within ActionManager
        self.config_manager = ConfigManager()  # Automatically initialized
        self.config_manager.set_experiment_dir(experiment_dir)  # Set the experiment directory

        # Store the experiment directory path and the full path to actions.yaml
        self.experiment_dir = self.config_manager.experiment_dir
        self.actions_file_path = self.experiment_dir / 'actions.yaml'  # Path to actions.yaml

        # Dictionary to store instantiated GeecsDevices
        self.instantiated_devices = {}
        self.actions = {}

    def load_actions(self):
        """
        Load the master actions from the given YAML file.
        """
        actions_file = str(self.actions_file_path)  # Convert Path object to string
        with open(actions_file, 'r') as file:
            actions = yaml.safe_load(file)
        logging.info(f"Loaded master actions from {actions_file}")
        self.actions = actions['actions']
        return actions['actions']
        
    def add_action(self, action):
        """
        Adds a new action to the actions list.
        action: dict - The complete action dictionary with action name and steps
        """
        # Parse out the action name and steps
        if len(action) != 1:
            raise ValueError("Action must contain exactly one action name")
        
        action_name = list(action.keys())[0]
        steps = action[action_name]['steps']
        
        # Add the action to the actions dictionary
        self.actions[action_name] = {'steps': steps}
    

    def execute_action(self, action_name):
        """
        Execute a single action by its name.
        Handles both standard device actions and nested actions.
        """
        if action_name not in self.actions:
            logging.error(f"Action '{action_name}' is not defined in the available actions.")
            return
        
        action = self.actions[action_name]
        steps = action['steps']

        for step in steps:
            if 'wait' in step:
                self._wait(step['wait'])
            elif 'action_name' in step:
                # Nested action: recursively execute the named action
                nested_action_name = step['action_name']
                logging.info(f"Executing nested action: {nested_action_name}")
                self.execute_action(nested_action_name)
            else:
                # Regular device action
                device_name = step['device']
                variable = step['variable']
                action_type = step['action']
                value = step.get('value')
                expected_value = step.get('expected_value')

                # Instantiate device if it hasn't been done yet
                if device_name not in self.instantiated_devices:
                    self.instantiated_devices[device_name] = GeecsDevice(device_name)

                device = self.instantiated_devices[device_name]

                if action_type == 'set':
                    self._set_device(device, variable, value)
                elif action_type == 'get':
                    self._get_device(device, variable, expected_value)

    def execute_action_list(self, action_list):
        """
        Execute a list of actions, potentially with nested actions.
        """
        for action_name in action_list:
            logging.info(f"Executing action: {action_name}")
            self.execute_action(action_name)

    def _set_device(self, device, variable, value):
        result = device.set(variable, value)
        logging.info(f"Set {device.get_name()}:{variable} to {value}. Result: {result}")

    def _get_device(self, device, variable, expected_value):
        value = device.get(variable)
        if value == expected_value:
            logging.info(f"Get {device.get_name()}:{variable} returned expected value: {value}")
        else:
            logging.warning(f"Get {device.get_name()}:{variable} returned {value}, expected {expected_value}")

    def _wait(self, seconds):
        logging.info(f"Waiting for {seconds} seconds.")
        time.sleep(seconds)

class DeviceManager:
    def __init__(self, experiment_dir: str):
        # Initialize the ConfigManager within DeviceManager
        self.config_manager = ConfigManager()  # Automatically initialized
        self.config_manager.set_experiment_dir(experiment_dir)  # Set the experiment directory
        
        # Initialize variables
        self.devices = {}
        self.event_driven_observables = []  # Store event-driven observables
        self.async_observables = []  # Store asynchronous observables
        self.non_scalar_saving_devices = []  # Store devices that need to save non-scalar data

        # Load paths for required config files using the ConfigManager
        self.base_config_file_path = self.config_manager.get_config_path('base_monitoring_devs.yaml')
        self.composite_variables_file_path = self.config_manager.get_config_path('composite_variables.yaml')

        # Load composite variables from the file
        self.composite_variables = self.load_composite_variables(self.composite_variables_file_path)

        # Placeholder for scan description
        self.scan_base_description = None
        self.scan_parameters = None
        self.scan_setup_action = None

    def load_composite_variables(self, composite_file):
        """
        Load composite variables from the given YAML file.
        """
        try:
            with open(composite_file, 'r') as file:
                composite_variables = yaml.safe_load(file).get('composite_variables', {})
            logging.info(f"Loaded composite variables from {composite_file}")
            return composite_variables
        except FileNotFoundError:
            logging.warning(f"Composite variables file not found: {composite_file}.")
            return {}
    
    def load_base_config(self):
        """
        Load a base configuration of core devices from the base config file.
        If the file does not exist, log a warning and skip the base configuration load.
        """
        try:
            if not self.base_config_file_path.exists():
                logging.warning(f"Base configuration file not found: {self.base_config_file_path}. Skipping base config.")
                return

            with open(self.base_config_file_path, 'r') as file:
                base_config = yaml.safe_load(file)

            logging.info(f"Loaded base configuration from {self.base_config_file_path}")
            self._load_devices_from_config(base_config)

        except Exception as e:
            logging.error(f"Error loading base configuration from {self.base_config_file_path}: {e}")

    def load_from_config(self, config_filename):
        """
        Load configuration from a YAML file, including scan info, parameters, and device observables.
        Also loads the base configuration if necessary.
        """
        # Load base configuration first
        self.load_base_config()

        # Load the specific config for the experiment
        config_path = self.config_manager.get_config_path(config_filename)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Loaded configuration from {config_path}")
        self.load_from_dictionary(config)

    def load_from_dictionary(self, config_dictionary):
        """
        Originally the 2nd half of "load_from_config," this bypasses the need to read the yaml if it is already loaded
        """
        # Load scan info
        self.scan_base_description = config_dictionary.get('scan_info', {}).get('description', '')
        self.scan_parameters = config_dictionary.get('scan_parameters', {})
        self.scan_setup_action = config_dictionary.get('setup_action', None)

        self._load_devices_from_config(config_dictionary)

        # Initialize the subscribers
        self.initialize_subscribers(self.event_driven_observables + self.async_observables, clear_devices=False)

        logging.info(f"Loaded scan info: {self.scan_base_description}")
        logging.info(f"Loaded scan parameters: {self.scan_parameters}")
        
    def _load_devices_from_config(self, config):
        """
        Helper method to load devices from either base_config or config.yaml.
        Appends devices to self.devices, as well as categorizes them into synchronous or asynchronous.
        """
        devices = config.get('Devices', {})
        for device_name, device_config in devices.items():
            variable_list = device_config.get('variable_list', [])
            synchronous = device_config.get('synchronous', False)
            save_non_scalar = device_config.get('save_nonscalar_data', False)

            logging.info(f"{device_name}: Synchronous = {synchronous}, Save_Non_Scalar = {save_non_scalar}")

            # Add to non-scalar saving devices if applicable
            if save_non_scalar:
                self.non_scalar_saving_devices.append(device_name)

            # Categorize as synchronous or asynchronous
            if synchronous:
                self.event_driven_observables.extend([f"{device_name}:{var}" for var in variable_list])
            else:
                self.async_observables.extend([f"{device_name}:{var}" for var in variable_list])

            # Check if device already exists, if not, instantiate it
            if device_name not in self.devices:
                self._subscribe_device(device_name, variable_list)
            else:
                # If device exists, append new variables to its subscription
                self.devices[device_name].subscribe_var_values(variable_list)

        logging.info(f"Devices loaded: {self.devices.keys()}")
                
    def load_composite_variables(self, composite_file):
        with open(composite_file, 'r') as file:
            return yaml.safe_load(file).get('composite_variables', {})

    def is_statistic_noscan(self, variable_name):
        return variable_name in ('noscan', 'statistics')

    def is_composite_variable(self, variable_name):
        return variable_name in self.composite_variables

    def get_composite_components(self, composite_var, value):
        """
        Get the device-variable mappings for a composite variable based on its value.

        Args:
            composite_var (str): The name of the composite variable.
            value (float): The current value of the composite variable.

        Returns:
            dict: A mapping of 'device:variable' -> evaluated value.
        """
        components = self.composite_variables[composite_var]['components']
        variables = {}

        # Iterate over each component and evaluate its relation
        for comp in components:
            relation = comp['relation'].replace("composite_var", str(value))  # Replace the placeholder
            evaluated_value = eval(relation)  # Evaluate the relationship to get the actual value
            variables[f"{comp['device']}:{comp['variable']}"] = evaluated_value

        return variables

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
                time.sleep(0.5)
                device.close()
            except Exception as e:
                logging.error(f"Error unsubscribing from {device_name}: {e}")
                
        # Clear the devices dictionary
        self.devices.clear()
                
    def reset(self):
        """
        Gracefully close the DeviceManager and reset all internal state, making it ready for reinitialization.
        """
        # Step 1: Close all subscribers
        self.close_subscribers()

        # Step 2: Clear internal state (reset lists)
        self.event_driven_observables.clear()
        self.async_observables.clear()
        self.non_scalar_saving_devices.clear()

        logging.info("DeviceManager instance has been reset and is ready for reinitialization.")

    def reinitialize(self, config_path):
        """
        Reinitialize the DeviceManager by loading the configuration file and starting fresh.
        """
        # First, reset the current state
        self.reset()

        # Now load the new configuration and reinitialize the instance
        self.load_from_config(config_path)

        logging.info("DeviceManager instance has been reinitialized.")

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
                   
    def add_scan_device(self, device_name, variable_list):
        """
        Add a new device or append variables to an existing device for scan variables.
        Ensure that default settings are applied for scan-specific devices.
        """
        if device_name not in self.devices:
            logging.info(f"Adding new scan device: {device_name} with default settings.")

            # Default attributes for scan-specific devices (e.g., from scan_config)
            default_device_config = {
                'save_non_scalar_data': False,
                'synchronous': False
            }

            # Create the GeecsDevice for the new scan variable device
            device = GeecsDevice(device_name)
            device.use_alias_in_TCP_subscription = False
            device.subscribe_var_values(variable_list)

            # Store the device with the default configuration in self.devices
            self.devices[device_name] = device
            self.non_scalar_saving_devices.append(device_name) if default_device_config['save_non_scalar_data'] else None

            # Add scan variables to async_observables
            self.async_observables.extend([f"{device_name}:{var}" for var in variable_list])
            logging.info(f"Scan device {device_name} added to async_observables.")
    
        else:
            logging.info(f"Device {device_name} already exists. Adding new variables: {variable_list}")

            # Update the existing device's variable list by subscribing to new variables
            device = self.devices[device_name]
            device.subscribe_var_values(variable_list)

            # Add new variables to async_observables
            self.async_observables.extend([f"{device_name}:{var}" for var in variable_list if f"{device_name}:{var}" not in self.async_observables])
            logging.info(f"Updated async_observables with new variables for {device_name}: {variable_list}")

    def handle_scan_variables(self, scan_config):
        """
        Handle scan variables and instantiate any new devices or append variables to existing ones.
        """
        logging.info("No error yet in handle_scan_variables.")
        logging.info(f"Scan config: {scan_config}.")
    
        for scan in scan_config:
            device_var = scan['device_var']
            logging.info(f"Processing scan device_var: {device_var}")

            # Handle composite variables
            if self.is_statistic_noscan(device_var):
                logging.info("Statistical noscan selected, adding no scan devices.")
            elif self.is_composite_variable(device_var):
                logging.info(f"{device_var} is a composite variable.")
                component_vars = self.get_composite_components(device_var, scan['start'])
                for component_var in component_vars:
                    dev_name, var = component_var.split(':', 1)
                    logging.info(f"Trying to add {dev_name}:{var} to self.devices.")
                    self.add_scan_device(dev_name, [var])  # Add or append the component vars
            else:
                # Normal variables
                logging.info(f"{device_var} is a normal variable.")
                device_name, var_name = device_var.split(':', 1)
                logging.info(f"Trying to add {device_name}:{var_name} to self.devices.")
                self.add_scan_device(device_name, [var_name])  # Add or append the normal variable                   
                    
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
        sFile_info_path = self.local_analysis_dir_base / f"s{int(self.next_scan_folder[-3:])}_info.txt"

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
    def __init__(self, experiment_dir, device_manager=None, data_interface=None):

        if device_manager is None:
            self.device_manager = DeviceManager(experiment_dir)
        else:
            self.device_manager = device_manager

        if data_interface is None:
            self.data_interface = DataInterface()
        else:
            self.data_interface = data_interface
        self.action_manager = ActionManager(experiment_dir=experiment_dir)
        
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
        
        self.console_log_name = "scan_execution.log"
        setup_console_logging(log_file=self.console_log_name, level=logging.INFO, console=True)
        
        self.bin_num = 0  # Initialize bin as 0
        
        self.logging_thread = None  # NEW: Separate thread for logging

        self.tdms_writer = None
        
        self.scan_steps = []  # To store the precomputed scan steps
               
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
        
    def pre_logging_setup(self, scan_config):
        """
        Precompute all scan steps (including composite and normal variables), 
        add scan devices to async_observables, and store the scan steps.
        """
        
        logging.info("Turning off the trigger.")
        self.trigger_off()

        time.sleep(2)
        self.data_interface.get_next_scan_folder()
        self.create_and_set_data_paths()
        
        # Handle scan variables and ensure devices are initialized in DeviceManager
        self.device_manager.handle_scan_variables(scan_config)
        
        self.write_scan_info_ini(scan_config[0])

        # Generate the scan steps
        self.scan_steps = self._generate_scan_steps(scan_config)
        
        if self.device_manager.scan_setup_action is not None:
            logging.info("attempting to execute pre scan actions.")
            logging.info(f'action list {self.device_manager.scan_setup_action}')
            
            self.action_manager.add_action({'setup_action': self.device_manager.scan_setup_action})
            self.action_manager.execute_action('setup_action')
        
        logging.info("Pre-logging setup completed.")
    
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
    
        # Create the configparser object
        config = configparser.ConfigParser()
        scan_var = scan_config.get('device_var', '')
        additional_description = scan_config.get('additional_description', '')
        
        scan_info = f'{self.device_manager.scan_base_description}. scanning {scan_var}. {additional_description}'

        # Add the Scan Info section
        config['Scan Info'] = {
            'Scan No': f'{scan_number}',
            'ScanStartInfo': scan_info,
            'Scan Parameter': scan_var,
            'Start': scan_config.get('start', 0),
            'End': scan_config.get('end', 0),
            'Step size': scan_config.get('step', 1),
            'Shots per step': scan_config.get('wait_time', 1),
            'ScanEndInfo': ''
        }

        # Create the full path for the file
        full_path = Path(self.data_txt_path.parent) / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Attempting to write to {full_path}")

        # Write to the .ini file
        with full_path.open('w') as configfile:
            config.write(configfile)

        logging.info(f"Scan info written to {full_path}")
        
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
        or handling 'noscan' (or 'statistics') scans during the acquisition process.
        
        Args:
            acquisition_time (int, optional): Total time to log data (in seconds). If not provided, 
                                              it will be estimated from the scan_config.
            scan_config (list of dicts, optional): List of scan configurations for dynamic actions.
                                                   Supports 'noscan' (or 'statistics') for no-action scans.
        """
        log_df = pd.DataFrame()  # Initialize in case of early exit
        try:
            # Pre-logging setup: Trigger devices off, initialize data paths, etc.
            self.pre_logging_setup(scan_config)

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
            log_df = self.acquisition_loop(acquisition_time)
    
        except Exception as e:
            logging.error(f"Error during logging: {e}")
    
        return log_df  # Return the DataFrame with the logged data
        
    def _add_scan_devices_to_async_observables(self, scan_config):
        """
        Add devices/variables involved in the scan to async_observables if not already present.
        """
        for scan in scan_config:
            device_var = scan['device_var']
            if self.device_manager.is_composite_variable(device_var):
                component_vars = self.device_manager.get_composite_components(device_var, scan['start'])
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
        for scan in scan_config:
            device_var = scan['device_var']

            if self.device_manager.is_statistic_noscan(device_var):
                steps.append({
                    'variables': device_var,
                    'wait_time': scan.get('wait_time', 1),
                    'is_composite': False
                })
            elif self.device_manager.is_composite_variable(device_var):
                current_value = scan['start']
                while current_value <= scan['end']:
                    component_vars = self.device_manager.get_composite_components(device_var, current_value)
                    steps.append({
                        'variables': component_vars,
                        'wait_time': scan.get('wait_time', 1),
                        'is_composite': True
                    })
                    current_value += scan['step']
            else:
                current_value = scan['start']
                while current_value <= scan['end']:
                    steps.append({
                        'variables': {device_var: current_value},
                        'wait_time': scan.get('wait_time', 1),
                        'is_composite': False
                    })
                    current_value += scan['step']
        return steps

    def acquisition_loop(self, acquisition_time):
        elapsed_time = 0
        check_interval = 0.1  
        step_interval = 1.0  
        step_timer = 0

        while elapsed_time < acquisition_time:
            # Check if logging has been externally stopped
            if self.stop_logging_thread_event.is_set():
                logging.info("Logging has been stopped externally.")
                break

            # Execute scan step if the timer has reached the step interval
            if step_timer >= step_interval:
                if self.scan_steps:
                    # Pop and execute the next scan step
                    scan_step = self.scan_steps.pop(0)
                    self._execute_scan(scan_step['variables'], scan_step['wait_time'], scan_step['is_composite'])
                    step_timer = 0
                else:
                    # Break out of the loop if no more scan steps remain
                    logging.info("No more scan steps to execute. Exiting loop.")
                    break

            # Sleep for the check interval and update timers
            time.sleep(check_interval)
            elapsed_time += check_interval
            step_timer += check_interval

        logging.info("Stopping logging.")
        log_df = self.stop_logging()

        return log_df

    def _execute_scan(self, component_vars, wait_time, is_composite):
        """
        Executes a single step of the scan, handling both composite and normal variables.
        Ensures the trigger is turned off before all moves and turned back on after.
        """
        logging.info("Pausing logging. Turning trigger off before moving devices.")
        self.bin_num += 1
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
        
        time.sleep(wait_time)
        
        self.trigger_off()
        logging.info(f"shot control state: {self.shot_control.state}")
 
    def estimate_acquisition_time(self, scan_config):
        """
        Estimate the total acquisition time based on the scan configuration.
        """
        total_time = 0

        for scan in scan_config:
            if self.device_manager.is_statistic_noscan(scan['device_var']):
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
                if not self.device_manager.is_statistic_noscan(device_var):
                    start = scan['start']
                    scan_state[device_var] = start  # Set the current position to the start value
        return scan_state

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
            # else:
            #     logging.warning(f"No existing row for elapsed time {elapsed_time}. Skipping log for {observable}.")

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
            # Log configuration variables (such as 'bin') only when a new entry is created
            log_entries[elapsed_time]['Bin #'] = self.bin_num
            # os.system('afplay trimmed_tink.aiff')  # This plays a sound on macOS
            # Trigger the beep in the background
            # sounds.play_beep()
            
            
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
        
        # sounds.play_toot()
        sounds.stop()

        # Step 5: Process results and convert to DataFrame
        if self.results:
            log_df = self.convert_to_dataframe(self.results)
            logging.info("Data logging complete. Returning DataFrame.")
                            
            self.save_to_txt_and_h5(log_df, self.data_txt_path, self.data_h5_path, self.sFile_txt_path)

            self.dataframe_to_tdms(log_df)
            self.dataframe_to_tdms(log_df, is_index = True)
        

        
        else:
            logging.warning("No data was collected during the logging period.")
            log_df = pd.DataFrame()
            
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logging.info(f"Setting save to off for {device_name}")
                device.set('save', 'off',sync=False)
                logging.info(f"Setting save to off for {device_name} complete")
                device.set('localsavingpath', 'c:\\temp',sync=False)
                logging.info(f"Setting save path back to temp for {device_name} complete")
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")
        
        stop_console_logging()
        move_log_file(self.console_log_name, self.data_txt_path.parent)

        print("Logging has stopped for all devices.")
        self.shot_control.set('Trigger.Source',"External rising edges")
        


        return log_df

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