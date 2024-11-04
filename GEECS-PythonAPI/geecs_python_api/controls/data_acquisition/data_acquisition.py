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

import concurrent.futures

from nptdms import TdmsWriter, ChannelObject

from concurrent.futures import ThreadPoolExecutor, as_completed

from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface.geecs_errors import ErrorAPI
import geecs_python_api.controls.interface.message_handling as mh

from image_analysis.utils import get_imaq_timestamp_from_png, get_picoscopeV2_timestamp, get_magspecstitcher_timestamp

from .utils import get_full_config_path  # Import the utility function

# For Windows-specific imports
if platform.system() == "Windows":
    import winsound
# For macOS-specific imports
elif platform.system() == "Darwin":
    import simpleaudio as sa

class SoundPlayer:
   
    """
    A class to handle playing sounds (beep and toot) in a background thread.
    """
    
    def __init__(self, beep_frequency=700, beep_duration=0.1, toot_frequency=1200, toot_duration=0.75, sample_rate=44100):
        """
        Initialize the SoundPlayer with default or user-defined frequency, duration, and sample rate.

        Args:
            beep_frequency (int, optional): Frequency of the beep sound in Hz. Default is 500.
            beep_duration (float, optional): Duration of the beep sound in seconds. Default is 0.1.
            toot_frequency (int, optional): Frequency of the toot sound in Hz. Default is 1500.
            toot_duration (float, optional): Duration of the toot sound in seconds. Default is 0.75.
            sample_rate (int, optional): Sample rate for sound generation (used for macOS). Default is 44100.
        """
        
        # Assign the user-defined or default values for frequency and duration
        self.beep_frequency = beep_frequency
        self.beep_duration = beep_duration
        self.toot_frequency = toot_frequency
        self.toot_duration = toot_duration
        self.sample_rate = sample_rate
        
        # Create a queue to hold sound requests
        self.sound_queue = queue.Queue()
        # Create and start the background thread
        self.sound_thread = threading.Thread(target=self._process_queue)
        self.sound_thread.daemon = True  # Mark thread as a daemon so it exits when the main program exits
        self.running = True  # Flag to control thread running
        self.sound_thread.start()

    def play_beep(self):
        """Add a beep sound request to the queue."""
        self.sound_queue.put('beep')

    def play_toot(self):
        """Add a toot sound request to the queue."""
        self.sound_queue.put('toot')

    def stop(self):
        """Stop the sound player by sending a termination signal."""
        self.running = False
        self.sound_queue.put(None)  # Add a termination signal to the queue

    def _process_queue(self):
        
        """
        Continuously process the sound queue and play the appropriate sound 
        based on the request.
        """
        
        while self.running:
            try:
                # Wait for the next sound request (this blocks until a request is added)
                sound_type = self.sound_queue.get()
                
                # Exit the loop if the termination signal is received
                if sound_type is None:
                    break

                # Play the requested sound
                if sound_type == 'beep':
                    self._play_sound(self.beep_frequency, self.beep_duration)
                elif sound_type == 'toot':
                    self._play_sound(self.toot_frequency, self.toot_duration)
                # Mark the task as done
                self.sound_queue.task_done()
            except Exception as e:
                print(f"Error processing sound: {e}")

    def _play_sound(self, frequency, duration):
        """
        Play a sound based on the platform (Windows or macOS).
        
        Args:
            frequency (int): Frequency of the sound in Hz.
            duration (float): Duration of the sound in seconds.
        """
        # Windows: Use winsound.Beep
        if platform.system() == "Windows":
            winsound.Beep(frequency, int(duration * 1000))  # Duration is in milliseconds
        # macOS: Use simpleaudio to play the generated sound
        elif platform.system() == "Darwin":
            audio_data = self._generate_sound(frequency, duration)
            play_obj = sa.play_buffer(audio_data, 1, 2, self.sample_rate)  # 1 channel, 2 bytes per sample
            play_obj.wait_done()
        # Optionally add Linux support or other platforms if needed
        else:
            os.system('printf "\a"')  # Default to terminal bell for unsupported platforms

    def _generate_sound(self, frequency, duration):
        
        """
        Generate a sound for macOS given a frequency and duration.

        Args:
            frequency (int): Frequency of the sound in Hz.
            duration (float): Duration of the sound in seconds.
        
        Returns:
            numpy.ndarray: Array of sound data formatted for playback.
        """
        
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t)
        return (tone * 32767).astype(np.int16)  # Convert to 16-bit PCM format


class ActionManager:
    """
    A class to manage and execute actions, including device actions and nested actions.
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize the ActionManager and load the actions from the specified experiment directory.

        Args:
            experiment_dir (str): The directory where the actions.yaml file is located.
        """

        if experiment_dir is not None:
            # Use the utility function to get the path to the actions.yaml file
            self.actions_file_path = get_full_config_path(experiment_dir, 'aux_configs', 'actions.yaml')

            # Dictionary to store instantiated GeecsDevices
            self.instantiated_devices = {}
            self.actions = {}
            if self.actions_file_path.exists():
                self.load_actions()

    def load_actions(self):
        
        """
        Load the master actions from the given YAML file.

        Returns:
            dict: A dictionary of actions loaded from the YAML file.
        """
        
        actions_file = str(self.actions_file_path)  # Convert Path object to string
        with open(actions_file, 'r') as file:
            actions = yaml.safe_load(file)
        logging.info(f"Loaded master actions from {actions_file}")
        self.actions = actions['actions']
        return actions['actions']

    def add_action(self, action):
        
        """
        Add a new action to the default actions list. NOTE, action is note saved

        Args:
            action (dict): A dictionary containing the action name and steps.
        
        Raises:
            ValueError: If the action dictionary does not contain exactly one action name.
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
        Execute a single action by its name, handling both device actions and nested actions.

        Args:
            action_name (str): The name of the action to execute.
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
                wait_for_execution = step.get('wait_for_execution', True)

                # Instantiate device if it hasn't been done yet
                if device_name not in self.instantiated_devices:
                    self.instantiated_devices[device_name] = GeecsDevice(device_name)

                device = self.instantiated_devices[device_name]

                if action_type == 'set':
                    self._set_device(device, variable, value, sync = wait_for_execution)
                elif action_type == 'get':
                    self._get_device(device, variable, expected_value)

    def _set_device(self, device, variable, value, sync = True):
        """
        Set a device variable to a specified value.

        Args:
            device (GeecsDevice): The device to control.
            variable (str): The variable to set.
            value (any): The value to set for the variable.
        """
        
        result = device.set(variable, value, sync = sync)
        logging.info(f"Set {device.get_name()}:{variable} to {value}. Result: {result}")

    def _get_device(self, device, variable, expected_value):
        
        """
        Get the current value of a device variable and compare it to the expected value.

        Args:
            device (GeecsDevice): The device to query.
            variable (str): The variable to get the value of.
            expected_value (any): The expected value for comparison.
        """
        
        value = device.get(variable)
        if value == expected_value:
            logging.info(f"Get {device.get_name()}:{variable} returned expected value: {value}")
        else:
            logging.warning(f"Get {device.get_name()}:{variable} returned {value}, expected {expected_value}")

    def _wait(self, seconds):
        
        """
        Wait for a specified number of seconds.

        Args:
            seconds (float): The number of seconds to wait.
        """
        
        logging.info(f"Waiting for {seconds} seconds.")
        time.sleep(seconds)


class DeviceManager:
    
    """
    Manages devices for data acquisition operations, including loading configurations, handling scan variables, 
    and subscribing to device updates. Responsible for loading composite variables, initializing 
    device subscriptions, and managing the observables for event-driven and asynchronous data collection.
    """
    
    def __init__(self, experiment_dir: str = None):
       
        """
        Initialize the DeviceManager with optional experiment directory.
        
        Args:
            experiment_dir (str, optional): Path to the directory where experiment configurations are stored.
        """
        self.devices = {}
        self.device_analysis = {} 
        self.event_driven_observables = []  # Store event-driven observables
        self.async_observables = []  # Store asynchronous observables
        self.non_scalar_saving_devices = []  # Store devices that need to save non-scalar data
        self.composite_variables = None
        self.scan_setup_action = {'steps': []}
        self.scan_closeout_action = {'steps': []}
        

        if experiment_dir is not None:
            # Set the experiment directory
            self.experiment_dir = experiment_dir
            
            # Use self.experiment_dir when calling the utility function
            self.base_config_file_path = get_full_config_path(self.experiment_dir, 'save_devices', 'Base_Monitoring_Devices.yaml')
            self.composite_variables_file_path = get_full_config_path(self.experiment_dir, 'aux_configs', 'composite_variables.yaml')
           
            # Load composite variables from the file
            self.composite_variables = self.load_composite_variables(self.composite_variables_file_path)

    def load_composite_variables(self, composite_file):
        
        """
        Load composite variables from the given YAML file.

        Args:
            composite_file (str): Path to the YAML file containing composite variables.

        Returns:
            dict: Dictionary of composite variables.
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
        Load a base configuration of core devices from the base configuration file. 
        If the file does not exist, log a warning and skip the base configuration load.
        NOTE: this method is potentially unnecessary. Doesn't really do anything the load_from_config
        doesn't already do. It was meant to "silently" load a base list of devices to be saved.
        """
        try:
            if not self.base_config_file_path.exists():
                logging.warning(
                    f"Base configuration file not found: {self.base_config_file_path}. Skipping base config.")
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

        Args:
            config_filename (str): The name of the YAML configuration file to load.
        """
        
        # Load base configuration first
        # self.load_base_config()

        # Load the specific config for the experiment
        config_path = get_full_config_path(self.experiment_dir, 'save_devices', config_filename)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Loaded configuration from {config_path}")
        self.load_from_dictionary(config)

    def load_from_dictionary(self, config_dictionary):
        
        """
        Load configuration from a preloaded dictionary, bypassing the need to read a YAML file. Primarily
        used by the GUI, but can enable loading conffigs in a different manner.

        Args:
            config_dictionary (dict): A dictionary containing the experiment configuration.
        """
        
        # Load scan info
        self.scan_base_description = config_dictionary.get('scan_info', {}).get('description', '')
        # # self.scan_parameters = config_dictionary.get('scan_parameters', {})
        # self.scan_setup_action = config_dictionary.get('setup_action', None)
        # self.scan_closeout_action = config_dictionary.get('closeout_action', None)
        
        # Append setup action from config
        setup_actions = config_dictionary.get('setup_action', {}).get('steps', [])
        if setup_actions:
            self.scan_setup_action['steps'].extend(setup_actions)
        
        # Append closeout action from config
        closeout_actions = config_dictionary.get('closeout_action', {}).get('steps', [])
        if closeout_actions:
            self.scan_closeout_action['steps'].extend(closeout_actions)
        
        self._load_devices_from_config(config_dictionary)

        # Initialize the subscribers
        self.initialize_subscribers(self.event_driven_observables + self.async_observables, clear_devices=False)

        logging.info(f"Loaded scan info: {self.scan_base_description}")

    def _load_devices_from_config(self, config):
        """
        Helper method to load devices from the base or custom configuration files.
        Adds devices to the manager and categorizes them as synchronous or asynchronous.

        Args:
            config (dict): A dictionary of devices and their configuration.
        """
        
        devices = config.get('Devices', {})
        for device_name, device_config in devices.items():
            variable_list = device_config.get('variable_list', [])
            synchronous = device_config.get('synchronous', False)
            save_non_scalar = device_config.get('save_nonscalar_data', False)
            post_analysis_class_name = device_config.get('post_analysis_class', None)
            scan_setup = device_config.get('scan_setup', None)
            logging.info(f"{device_name}: Post Analysis = {post_analysis_class_name}")
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
                
            # Append scan setup actions if they exist
            if scan_setup:
                self.append_device_setup_closeout_actions(device_name, scan_setup)
                
            # Store post_analysis_class in device_analysis dictionary
            if post_analysis_class_name:
                self.device_analysis[device_name] = {
                    'post_analysis_class': post_analysis_class_name
                }

        logging.info(f"Devices loaded: {self.devices.keys()}")
        
    def append_device_setup_closeout_actions(self, device_name, scan_setup):
        """
        Append actions to setup_action and closeout_action for the specified device based on scan_setup.

        Args:
            device_name (str): The name of the device.
            scan_setup (dict): Dictionary containing scan setup actions and their corresponding setup/closeout values.
        """
        # Iterate over each key in the 'scan_setup' dictionary
        for analysis_type, values in scan_setup.items():
            # Ensure the setup and closeout values exist in the 'scan_setup'
            if len(values) != 2:
                logging.warning(f"Invalid scan setup actions for {device_name}: {analysis_type} (Expected 2 values, got {len(values)})")
                continue
        
            setup_value, closeout_value = values

            # Append to setup_action
            self.scan_setup_action['steps'].append({
                'action': 'set',
                'device': device_name,
                'value': setup_value,  # setup value
                'variable': analysis_type,
                'wait_for_execution': False
            })
        
            # Append to closeout_action
            self.scan_closeout_action['steps'].append({
                'action': 'set',
                'device': device_name,
                'value': closeout_value,  # closeout value
                'variable': analysis_type,
                'wait_for_execution': False
    
            })

            logging.info(f"Added setup and closeout actions for {device_name}: {analysis_type} (setup={setup_value}, closeout={closeout_value})")

    def is_statistic_noscan(self, variable_name):
        """
        Check if the variable is a 'noscan' or 'statistics' placeholder.

        Args:
            variable_name (str): The variable name to check.

        Returns:
            bool: True if the variable is 'noscan' or 'statistics', False otherwise.
        """
        
        return variable_name in ('noscan', 'statistics')

    def is_composite_variable(self, variable_name):
        """
        Check if the variable is a composite variable.

        Args:
            variable_name (str): The variable name to check.

        Returns:
            bool: True if the variable is a composite variable, False otherwise.
        """
        return self.composite_variables is not None and variable_name in self.composite_variables

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
            logging.info(f"Evaluating relation: {relation}")
            evaluated_value = eval(relation)  # Evaluate the relationship to get the actual value
            variables[f"{comp['device']}:{comp['variable']}"] = evaluated_value

        return variables

    def initialize_subscribers(self, variables, clear_devices=True):
        """
        Initialize subscribers for the specified variables, creating or resetting device subscriptions.

        Args:
            variables (list): A list of device variables to subscribe to.
            clear_devices (bool): If True, clear the existing device subscriptions before initializing new ones.
        """
        
        if clear_devices:
            self._clear_existing_devices()

        device_map = self.preprocess_observables(variables)

        for device_name, var_list in device_map.items():
            if device_name not in self.devices:
                self._subscribe_device(device_name, var_list)

    def _clear_existing_devices(self):
        """
        Clear all existing device subscriptions and reset the devices dictionary.
        """
        
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
        """
        Subscribe to a new device and its associated variables.

        Args:
            device_name (str): The name of the device to subscribe to.
            var_list (list): A list of variables to subscribe to for the device.
        """
        
        device = GeecsDevice(device_name)
        device.use_alias_in_TCP_subscription = False
        logging.info(f'Subscribing {device_name} to variables: {var_list}')
        device.subscribe_var_values(var_list)
        self.devices[device_name] = device

    def reset(self):
        """
        Reset the DeviceManager by closing all subscribers and clearing internal state.
        """
        
        # Step 1: Close all subscribers
        self._clear_existing_devices()

        # Step 2: Clear internal state (reset lists)
        self.event_driven_observables.clear()
        self.async_observables.clear()
        self.non_scalar_saving_devices.clear()
        self.device_analysis.clear()
        
        logging.info(f'synchronous variables after reset: {self.event_driven_observables}')
        logging.info(f'asynchronous variables after reset: {self.async_observables}')
        logging.info(f'non_scalar_saving_devices devices after reset: {self.non_scalar_saving_devices}')
        logging.info(f'devices devices after reset: {self.devices}')
        logging.info("DeviceManager instance has been reset and is ready for reinitialization.")

    def reinitialize(self, config_path=None, config_dictionary=None):
        """
        Reinitialize the DeviceManager by resetting it and loading a new configuration.

        Args:
            config_path (str, optional): Path to the configuration file to load.
            config_dictionary (dict, optional): A dictionary containing the configuration to load.
        """
        
        # First, reset the current state
        self.reset()
        
        self.scan_setup_action['steps'] = []
        self.scan_closeout_action['steps'] =[]

        # Now load the new configuration and reinitialize the instance
        if config_path is not None:
            self.load_from_config(config_path)
        elif config_dictionary is not None:
            self.load_from_dictionary(config_dictionary)
            

        logging.info("DeviceManager instance has been reinitialized.")

    def get_values(self, variables):
        """
        Retrieve the current values of the specified variables from the devices.

        Args:
            variables (list): A list of variables to retrieve values for.

        Returns:
            dict: A dictionary of device names and their corresponding variable values, as 
                    well as the user defined shot number and it's 'fresh' status
        """
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
        """
        Parse the TCP states received from devices and organize them in a structured format.

        Args:
            get_values_result (dict): A dictionary of device states.

        Returns:
            dict: A parsed dictionary of device variable states.
        """
        
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
        """
        Preprocess a list of observables by organizing them into device-variable mappings.

        Args:
            observables (list): A list of device-variable observables.

        Returns:
            dict: A dictionary mapping device names to a list of their variables.
        """
        
        device_map = {}
        for observable in observables:
            device_name, var_name = observable.split(':')
            if device_name not in device_map:
                device_map[device_name] = []
            device_map[device_name].append(var_name)
        return device_map

    def add_scan_device(self, device_name, variable_list):
        """
        Add a new device or append variables to an existing device for scan operations and
        recording their data.

        Args:
            device_name (str): The name of the device to add or update.
            variable_list (list): A list of variables to add for the device.
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
            self.non_scalar_saving_devices.append(device_name) if default_device_config[
                'save_non_scalar_data'] else None

            # Add scan variables to async_observables
            self.async_observables.extend([f"{device_name}:{var}" for var in variable_list])
            logging.info(f"Scan device {device_name} added to async_observables.")

        else:
            logging.info(f"Device {device_name} already exists. Adding new variables: {variable_list}")

            # Update the existing device's variable list by subscribing to new variables
            device = self.devices[device_name]
            device.subscribe_var_values(variable_list)

            # Add new variables to async_observables
            self.async_observables.extend([f"{device_name}:{var}" for var in variable_list if
                                           f"{device_name}:{var}" not in self.async_observables])
            logging.info(f"Updated async_observables with new variables for {device_name}: {variable_list}")

    def handle_scan_variables(self, scan_config):
        """
        Handle the initialization and setup of scan variables, including composite variables.

        Args:
            scan_config (dict): The configuration for the scan, including device and variable information.
        """

        device_var = scan_config['device_var']
        logging.info(f"Processing scan device_var: {device_var}")

        # Handle composite variables
        if self.is_statistic_noscan(device_var):
            logging.info("Statistical noscan selected, adding no scan devices.")
        elif self.is_composite_variable(device_var):
            logging.info(f"{device_var} is a composite variable.")
            component_vars = self.get_composite_components(device_var, scan_config['start'])
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
    DEPENDENT_SUFFIXES = ["-interp", "-interpSpec", "-interpDiv"]
    
    def __init__(self):
        # Initialize domain, base path, and current date
        self.domain = self.get_domain_windows()
        self.local_base_path = self.get_local_base_path()
        self.client_base_path = Path('Z:/data/Undulator')
        self.year, self.month, self.day = self.get_current_date()
        self.dummy_scan_number = 100
        self.local_scan_dir_base, self.local_analysis_dir_base = self.create_data_path(self.dummy_scan_number)
        self.client_scan_dir_base, self.client_analysis_dir_base = self.create_data_path(self.dummy_scan_number,
                                                                                         local=False)
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
    
    def process_and_rename(self, scan_number=None):
        """
        Process the device directories and rename files based on timestamps and scan data.
        The device type is dynamically determined from the directory name, and the appropriate
        timestamp extraction method is used based on the device type.

        Args:
            scan_number (int, optional): Specific scan number to process. If None, uses `next_scan_folder`.
        """
        scan_number_str = f'Scan{scan_number:03d}' if scan_number is not None else self.next_scan_folder
        base = self.local_scan_dir_base
        directory_path = Path(base) / scan_number_str

        logging.info(f"Processing scan folder: {directory_path}")

        # Exclude directories with certain suffixes
        device_directories = [
            d for d in directory_path.iterdir() if d.is_dir() and not any(d.name.endswith(suffix) for suffix in self.DEPENDENT_SUFFIXES)
        ]
        
        # Load scan data
        scan_num = self.get_last_scan_number()
        sPath = Path(self.local_analysis_dir_base / f's{scan_num}.txt')
        logging.info(f"Loading scan data from: {sPath}")

        try:
            df = pd.read_csv(sPath, sep='\t')
        except FileNotFoundError:
            logging.error(f"Scan data file {sPath} not found.")
            return

        # Process each device directory concurrently using threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_device_files, device_dir, df, scan_number_str) for device_dir in device_directories]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error during file processing: {e}")

    def process_device_files(self, device_dir, df, scan_number):
        """
        Generic method to process device files and rename them based on timestamps.
        """
        device_name = device_dir.name
        logging.info(f"Processing device directory: {device_name}")

        # Get the device type and extract relevant data
        device_type = GeecsDatabase.find_device_type(device_name)
        if not device_type:
            logging.warning(f"Device type for {device_name} not found. Skipping.")
            return

        # Collect and match files with timestamps
        device_files = list(device_dir.glob("*"))
        matched_rows = self.process_and_match_files(device_files, df, device_name, device_type)

        # Rename master and dependent files
        self.rename_files(matched_rows, scan_number, device_name)
        self.process_dependent_directories(device_name, device_dir, matched_rows, scan_number)

    def process_and_match_files(self, device_files, df, device_name, device_type):
        """
        Match device files with timestamps from the DataFrame.
        """
        matched_rows = []
        device_timestamp_column = f'{device_name} timestamp'

        if device_timestamp_column not in df.columns:
            logging.warning(f"No matching timestamp column for {device_name} in scan data.")
            return matched_rows
        
        tolerance = 1
        rounded_df_timestamps = df[device_timestamp_column].round(tolerance)
        for device_file in device_files:
            try:
                file_timestamp = self.extract_timestamp_from_file(device_file, device_type)
                file_timestamp_rounded = round(file_timestamp, tolerance)
                match = rounded_df_timestamps[rounded_df_timestamps == file_timestamp_rounded]
                if not match.empty:
                    matched_rows.append((device_file, match.index[0]))
                    logging.info(f"Matched file {device_file} with row {match.index[0]}")
                else:
                    logging.warning(f"No match for {device_file} with timestamp {file_timestamp_rounded}")
            except Exception as e:
                logging.error(f"Error extracting timestamp from {device_file}: {e}")

        return matched_rows

    def extract_timestamp_from_file(self, device_file, device_type):
        """
        Extract timestamp from a device file based on its type.
        """
        device_map = {
            "Point Grey Camera": get_imaq_timestamp_from_png,
            "MagSpecCamera": get_imaq_timestamp_from_png,
            "PicoscopeV2": get_picoscopeV2_timestamp,
            "MagSpecStitcher": get_magspecstitcher_timestamp
        }

        if device_type in device_map:
            return device_map[device_type](device_file)
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

    def process_dependent_directories(self, device_name, device_dir, matched_rows, scan_number):
        """
        Process and rename files in dependent directories.
        """
        for suffix in self.DEPENDENT_SUFFIXES:
            dependent_dir = device_dir.parent / f"{device_name}{suffix}"
            if dependent_dir.exists() and dependent_dir.is_dir():
                logging.info(f"Processing dependent directory: {dependent_dir}")
                dependent_files = list(dependent_dir.glob("*"))
                self.rename_files_in_dependent_directory(dependent_files, matched_rows, scan_number, suffix)

    def rename_files_in_dependent_directory(self, dependent_files, matched_rows, scan_number, suffix):
        """
        Rename dependent files based on matched rows from the master directory.
        """
        for i, (master_file, row_index) in enumerate(matched_rows):
            if i < len(dependent_files):
                dependent_file = dependent_files[i]
                master_new_name = re.sub(r'_\d+$', '', master_file.stem)  # Remove trailing number
                new_name = f"{scan_number}_{master_new_name}{suffix}_{str(row_index + 1).zfill(3)}{dependent_file.suffix}"
                new_path = dependent_file.parent / new_name
                dependent_file.rename(new_path)
                logging.info(f"Renamed {dependent_file} to {new_path}")
            else:
                logging.warning(f"Not enough files in dependent directory to match {master_file}")

    def rename_files(self, matched_rows, scan_number, device_name):
        """
        Rename master files based on scan number, device name, and matched row index.
        """
        for file_path, row_index in matched_rows:
            row_number = str(row_index + 1).zfill(3)
            new_file_name = f"{scan_number}_{device_name}_{row_number}{file_path.suffix}"
            new_file_path = file_path.parent / new_file_name
            if not new_file_path.exists():
                logging.info(f"Renaming {file_path} to {new_file_path}")
                os.rename(file_path, new_file_path)
            else:
                logging.warning(f"File {new_file_path} already exists. Skipping.")
    

class DataLogger():
    
    """
    Handles the logging of data from devices during an scan, supporting both event-driven 
    and asynchronous data acquisition. This class manages polling, logging, and state management 
    for various devices in the experimental setup.
    """
    
    def __init__(self, experiment_dir, device_manager=None):
        
        """
        Initialize the DataLogger with the experiment directory and a device manager.

        Args:
            experiment_dir (str): Directory where the experiment's data is stored.
            device_manager (DeviceManager, optional): The manager responsible for handling devices. 
                                                      If not provided, a new one is initialized.
        """
        

        self.device_manager = device_manager or DeviceManager(experiment_dir)

        self.stop_event = threading.Event()  # Event to control polling thread
        self.poll_thread = None  # Placeholder for polling thread
        self.async_t0 = None  # Placeholder for async t0
        self.warning_timeout_sync = 2  # Timeout for synchronous devices (seconds)
        self.warning_timeout_async_factor = 1  # Factor of polling interval for async timeout
        self.last_log_time_sync = {}  # Dictionary to track last log time for synchronous devices
        self.last_log_time_async = {}  # Dictionary to track last log time for async devices
        self.polling_interval = .5
        self.results = {}  # Store results for later processing

        self.bin_num = 0  # Initialize bin as 0
        
        # Initialize the sound player
        self.sound_player = SoundPlayer()
        self.shot_index = 0
        
        self.virtual_variable_name = None
        self.virtual_variable_value = 0
        
        self.data_recording = False
        self.idle_time = 0
        
    def start_logging(self):
        """
        Start logging data for all devices. Event-driven observables trigger logs, and asynchronous
        observables are polled at regular intervals.
        
        Returns:
            dict: A dictionary storing the logged entries.
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

            Args:
                message (str): Message from the device containing data to be logged.
                device (GeecsDevice): The device object from which the message originated.
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
        self._start_async_polling(async_observables, log_entries, timeout=30)

        # # Start a thread to monitor device warnings
        self.warning_thread = threading.Thread(target=self._monitor_warnings, args=(event_driven_observables, async_observables))
        self.warning_thread.start()

        return log_entries

    def _extract_timestamp(self, message, device):
        
        """
        Extract the timestamp from a device message.

        Args:
            message (str): The message containing device data.
            device (GeecsDevice): The device sending the message.

        Returns:
            float: The extracted timestamp, or the current system time if no timestamp is found.
        """
        
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
        """
        Register event-driven observables for logging.

        Args:
            event_driven_observables (list): A list of event-driven observables to monitor.
            log_update (function): Function to call when an event update occurs.
        """
        
        for device_name, device in self.device_manager.devices.items():
            for observable in event_driven_observables:
                if observable.startswith(device_name):
                    logging.info(f"Registering logging for event-driven observable: {observable}")
                    device.event_handler.register('update', 'logger', lambda msg, dev=device: log_update(msg, dev))

    def _start_async_polling(self, async_observables, log_entries, timeout=10):
        """
        Start polling for asynchronous observables in a separate thread.

        Args:
            async_observables (list): A list of asynchronous observables to poll.
            log_entries (dict): Dictionary to store logged data.
            timeout (int): Timeout in seconds to wait for async_t0 to be set.
        """
        
        # If there's an existing polling thread, wait for it to finish before starting a new one
        if self.poll_thread and self.poll_thread.is_alive():
            self.poll_thread.join()

        self.poll_thread = threading.Thread(target=self.poll_async_observables,
                                            args=(async_observables, log_entries, timeout))
        self.poll_thread.start()

    def poll_async_observables(self, async_observables, log_entries, timeout):
        """
        Poll asynchronous observables at a regular interval and log the data.

        Args:
            async_observables (list): List of observables to poll.
            log_entries (dict): Dictionary to store the logged data.
            timeout (int): Timeout in seconds to wait for async_t0.
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
        Poll a single asynchronous observable and log its data.

        Args:
            observable (str): The observable to poll.
            log_entries (dict): Dictionary to store the logged data.
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
                logging.info(f"updating async var {device_name}:{var_name}: {value}.")
            # else:
            #     logging.warning(f"No existing row for elapsed time {elapsed_time}. Skipping log for {observable}.")

            # Update the last log time for this asynchronous device
            self.last_log_time_async[device_name] = time.time()

    def _initialize_standby_mode(self, device, standby_mode, initial_timestamps, current_timestamp):
        """
        Initialize and manage the standby mode for a device based on its timestamp. This is an
        essential component for getting synchronization correct 

        Args:
            device (GeecsDevice): The device being monitored.
            standby_mode (dict): A dictionary tracking which devices are in standby mode.
            initial_timestamps (dict): A dictionary storing initial timestamps for devices.
            current_timestamp (float): The current timestamp from the device.

        Returns:
            bool: True if the device is still in standby mode, False otherwise.
        """
        if device.get_name() not in standby_mode:
            standby_mode[device.get_name()] = True
        if device.get_name() not in initial_timestamps:
            initial_timestamps[device.get_name()] = current_timestamp
            logging.info(
                f"Initial dummy timestamp for {device.get_name()} set to {current_timestamp}. Standby mode enabled.")
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
        """
        Calculate the elapsed time for a device based on its initial timestamp.

        Args:
            device (GeecsDevice): The device being monitored.
            initial_timestamps (dict): A dictionary storing initial timestamps for devices.
            current_timestamp (float): The current timestamp from the device.

        Returns:
            int: The elapsed time (rounded) since the device's initial timestamp.
        """
        
        t0 = initial_timestamps[device.get_name()]
        return round(current_timestamp - t0)

    def _check_duplicate_timestamp(self, device, last_timestamps, current_timestamp):
        """
        Check if the current timestamp for a device is a duplicate.

        Args:
            device (GeecsDevice): The device being monitored.
            last_timestamps (dict): A dictionary storing the last timestamps for devices.
            current_timestamp (float): The current timestamp from the device.

        Returns:
            bool: True if the timestamp is a duplicate, False otherwise.
        """
        
        if device.get_name() in last_timestamps and last_timestamps[device.get_name()] == current_timestamp:
            logging.info(f"Timestamp hasn't changed for {device.get_name()}. Skipping log.")
            return True
        last_timestamps[device.get_name()] = current_timestamp
        return False

    def _log_device_data(self, device, event_driven_observables, log_entries, elapsed_time):
        
        """
        Log the data for a device during an event-driven observation.

        Args:
            device (GeecsDevice): The device being logged.
            event_driven_observables (list): A list of event-driven observables to monitor.
            log_entries (dict): Dictionary where log data is stored.
            elapsed_time (int): The time elapsed since the logging started.

        Logs:
            - Device variable data and its elapsed time.
            - Plays a beep sound on a new log entry.
        """
        
        observables_data = {
            observable.split(':')[1]: device.state.get(observable.split(':')[1], '')
            for observable in event_driven_observables if observable.startswith(device.get_name())
        }
        if elapsed_time not in log_entries:
            log_entries[elapsed_time] = {'Elapsed Time': elapsed_time}
            # Log configuration variables (such as 'bin') only when a new entry is created
            log_entries[elapsed_time]['Bin #'] = self.bin_num
            if self.virtual_variable_name is not None:
                log_entries[elapsed_time][self.virtual_variable_name] = self.virtual_variable_value
            # os.system('afplay trimmed_tink.aiff')  # This plays a sound on macOS
            # Trigger the beep in the background
            self.sound_player.play_beep()  # Play the beep sound
            self.shot_index += 1

        log_entries[elapsed_time].update({
            f"{device.get_name()}:{key}": value for key, value in observables_data.items()
        })

    def _monitor_warnings(self, event_driven_observables, async_observables):
        """
        Monitor the last log time for each device and issue warnings if a device hasn't updated within the threshold.
        """
        while not self.stop_event.is_set():
            current_time = time.time()
            
            if self.data_recording:
                # Check synchronous devices (event-driven observables)
                for observable in event_driven_observables:
                    device_name = observable.split(':')[0]
                    last_log_time = self.last_log_time_sync.get(device_name, None)

                    if last_log_time and (current_time - (last_log_time + + self.idle_time)) > self.warning_timeout_sync:
                        logging.warning(
                            f"Synchronous device {device_name} hasn't updated in over {self.warning_timeout_sync} seconds.")

                # Check asynchronous devices
                async_timeout = self.polling_interval * self.warning_timeout_async_factor
                for observable in async_observables:
                    device_name = observable.split(':')[0]
                    last_log_time = self.last_log_time_async.get(device_name, None)

                    if last_log_time and (current_time - last_log_time) > async_timeout:
                        logging.warning(
                            f"Asynchronous device {device_name} hasn't updated in over {async_timeout} seconds.")

                time.sleep(1)  # Monitor the warnings every second

    def stop_logging(self):
        """
        Stop both event-driven and asynchronous logging, unregister all event handlers, and reset states.
        """
        # Unregister all event-driven logging
        for device_name, device in self.device_manager.devices.items():
            device.event_handler.unregister('update', 'logger')
            
        self.sound_player.play_toot()

        # Signal to stop the polling thread
        self.stop_event.set()
        
        self.sound_player.stop()

        if self.poll_thread and self.poll_thread.is_alive():
            self.poll_thread.join()  # Ensure the polling thread has finished

        # Reset the stop_event for future logging sessions
        self.stop_event.clear()
        self.poll_thread = None
        self.async_t0 = None

    def reinitialize_sound_player(self):
        """
        Reinitialize the sound player, stopping the current one and creating a new instance.
        """
        
        self.sound_player.stop()
        self.sound_player = SoundPlayer()
        self.shot_index = 0

    def get_current_shot(self):
        """
        Get the current shot index. used for progress bar tracking

        Returns:
            float: The current shot index.
        """
        return float(self.shot_index)


if __name__ == '__main__':
    sound_player = SoundPlayer(beep_frequency=800, toot_frequency=2000)
    sound_player.play_toot()
    sound_player._process_queue()
    time.sleep(1)
    sound_player.stop()
