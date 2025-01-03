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
        self.running = False  # Flag to control thread running

    def start_queue(self):
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
                logging.error(f"Error processing sound: {e}")

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
        # Dictionary to store instantiated GeecsDevices
        self.instantiated_devices = {}
        self.actions = {}

        if experiment_dir is not None:
            # Use the utility function to get the path to the actions.yaml file
            try:
                self.actions_file_path = get_full_config_path(experiment_dir, 'aux_configs', 'actions.yaml')
                self.load_actions()
            except FileNotFoundError:
                logging.warning(f"actions.yaml file not found.")

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
        self.composite_variables = {}
        self.scan_setup_action = {'steps': []}
        self.scan_closeout_action = {'steps': []}

        self.is_reset = False  # Used to determine if a reset is required upon reinitialization

        if experiment_dir is not None:
            # Set the experiment directory
            self.experiment_dir = experiment_dir
            
            try:
                # Use self.experiment_dir when calling the utility function
                self.composite_variables_file_path = get_full_config_path(self.experiment_dir, 'aux_configs', 'composite_variables.yaml')

                # Load composite variables from the file
                self.composite_variables = self.load_composite_variables(self.composite_variables_file_path)
            except FileNotFoundError:
                logging.warning(f"Composite variables file not found.")

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
        self.is_reset = True

    def reinitialize(self, config_path=None, config_dictionary=None):
        """
        Reinitialize the DeviceManager by resetting it and loading a new configuration.

        Args:
            config_path (str, optional): Path to the configuration file to load.
            config_dictionary (dict, optional): A dictionary containing the configuration to load.
        """
        
        # First, reset the current state
        if not self.is_reset:
            self.reset()
        self.is_reset = False
        
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
        self.warning_timeout_sync = 2  # Timeout for synchronous devices (seconds)
        self.warning_timeout_async_factor = 1  # Factor of polling interval for async timeout
        self.last_log_time_sync = {}  # Dictionary to track last log time for synchronous devices
        self.last_log_time_async = {}  # Dictionary to track last log time for async devices
        self.polling_interval = .5
        self.results = {}  # Store results for later processing

        # Initialize the sound player
        self.sound_player = SoundPlayer()
        self.shot_index = 0

        self.bin_num = 0  # Initialize bin as 0

        self.virtual_variable_name = None
        self.virtual_variable_value = 0
        
        self.data_recording = False
        self.idle_time = 0
        
        self.lock = threading.Lock()

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

        # Start the sound player
        self.sound_player.start_queue()

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
            
            current_timestamp = self._extract_timestamp(message, device)

            if current_timestamp is None:
                return

            if self._initialize_standby_mode(device, standby_mode, initial_timestamps, current_timestamp):
                return

            elapsed_time = self._calculate_elapsed_time(device, initial_timestamps, current_timestamp)

            if self._check_duplicate_timestamp(device, last_timestamps, current_timestamp):
                return

            self._log_device_data(device, event_driven_observables, log_entries, elapsed_time)

        # Register the logging function for event-driven observables
        self._register_event_logging(event_driven_observables, log_update)
        
        logging.info('waiting for all devices to go to standby mode. Note, device standby status not checked, just waiting 4 seconds for all devices to timeout')
        time.sleep(4)
        
        logging.info("Logging has started for all event-driven devices.")

        # # Start a thread to monitor device warnings
        # self.warning_thread = threading.Thread(target=self._monitor_warnings, args=(event_driven_observables, async_observables))
        # self.warning_thread.start()

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
        
    def update_async_observables(self, async_observables, log_entries, elapsed_time):
        """
        Update log entries with the latest values for asynchronous observables.

        Args:
            async_observables (list): List of asynchronous observables to update.
            log_entries (dict): Dictionary to store the logged data.
            elapsed_time (int): The time elapsed since the logging started.
        """
        for observable in async_observables:
            device_name, var_name = observable.split(':')
            device = self.device_manager.devices.get(device_name)
            
            if device:
                # Get the latest value from the device state
                value = device.state.get(var_name, 'N/A')
                # Update the log entry for this async variable
                log_entries[elapsed_time][f"{device_name}:{var_name}"] = value
                logging.info(f"Updated async var {device_name}:{var_name} to {value} for elapsed time {elapsed_time}.")

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
        
        with self.lock:
            observables_data = {
                observable.split(':')[1]: device.state.get(observable.split(':')[1], '')
                for observable in event_driven_observables if observable.startswith(device.get_name())
            }
            if elapsed_time not in log_entries:
                logging.info(f'elapsed time in sync devices {elapsed_time}')
                log_entries[elapsed_time] = {'Elapsed Time': elapsed_time}
                # Log configuration variables (such as 'bin') only when a new entry is created
                log_entries[elapsed_time]['Bin #'] = self.bin_num
                if self.virtual_variable_name is not None:
                    log_entries[elapsed_time][self.virtual_variable_name] = self.virtual_variable_value
                    
                # Update with async observable values
                self.update_async_observables(self.device_manager.async_observables, log_entries, elapsed_time)

                # TODO move the on-shot tdms writer functionality from scan manager to here

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

                    if last_log_time and (current_time - (last_log_time + self.idle_time)) > self.warning_timeout_sync:
                        logging.warning(
                            f"Synchronous device {device_name} hasn't updated in over {self.warning_timeout_sync} seconds.")

                # Check asynchronous devices
                async_timeout = self.polling_interval * self.warning_timeout_async_factor
                for observable in async_observables:
                    device_name = observable.split(':')[0]
                    last_log_time = self.last_log_time_async.get(device_name, None)

                    if last_log_time and (current_time - (last_log_time + self.idle_time)) > async_timeout:
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
        # TODO check if this needs to be moved.  It might be cleared before the stop is registered
        # Reset the stop_event for future logging sessions
        self.stop_event.clear()

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
