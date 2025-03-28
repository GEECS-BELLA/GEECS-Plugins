import logging
import yaml
import threading
from pathlib import Path

from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_python_api.controls.interface.geecs_errors import GeecsDeviceInstantiationError

from .utils import get_full_config_path  # Import the utility function


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

        self.fatal_error_event = threading.Event()  # Used to signal a fatal error

        self.is_reset = False  # Used to determine if a reset is required upon reinitialization

        if experiment_dir is not None:
            # Set the experiment directory
            self.experiment_dir = experiment_dir

            try:
                # Use self.experiment_dir when calling the utility function
                self.composite_variables_file_path = get_full_config_path(self.experiment_dir, 'scan_devices',
                                                                          'composite_variables.yaml')

                # Load composite variables from the file
                self.composite_variables = self.load_composite_variables(self.composite_variables_file_path)
            except FileNotFoundError:
                logging.warning(f"Composite variables file not found.")

    def load_composite_variables(self, composite_file: Path):

        """
        Load composite variables from the given YAML file.

        Args:
            composite_file (Path): Path to the YAML file containing composite variables.

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
                logging.warning(
                    f"Invalid scan setup actions for {device_name}: {analysis_type} (Expected 2 values, got {len(values)})")
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

            logging.info(
                f"Added setup and closeout actions for {device_name}: {analysis_type} (setup={setup_value}, closeout={closeout_value})")

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

    def initialize_subscribers(self, variables, clear_devices=True):
        """
        Initialize subscribers for the specified variables, creating or resetting device subscriptions.

        Args:
            variables (list): A list of device variables to subscribe to
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
        If a Geecs device instantiation error occurs, log it and signal a fatal error event.

        Args:
            device_name (str or dict): The name of the device to subscribe to, or a dict of info for composite var
            var_list (list): A list of variables to subscribe to for the device.
        """

        try:
            if self.is_composite_variable(device_name):
                var_dict = self.composite_variables[device_name]
                device = ScanDevice(device_name, var_dict)
            else:
                device = ScanDevice(device_name)
                device.use_alias_in_TCP_subscription = False
                logging.info(f'Subscribing {device_name} to variables: {var_list}')
                device.subscribe_var_values(var_list)

            self.devices[device_name] = device

        except GeecsDeviceInstantiationError as e:
            logging.error(f"Failed to instantiate GEecs device {device_name}: {e}")
            # Signal a fatal error event so that the scan can abort cleanly
            self.fatal_error_event.set()
            raise

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
        self.scan_closeout_action['steps'] = []

        # Now load the new configuration and reinitialize the instance
        if config_path is not None:
            self.load_from_config(config_path)
        elif config_dictionary is not None:
            self.load_from_dictionary(config_dictionary)

        logging.info("DeviceManager instance has been reinitialized.")

    def preprocess_observables(self, observables):
        """
        Preprocess a list of observables by organizing them into device-variable mappings.

        Args:
            observables (list): A list of device-variable observables, e.g. [Dev1:Var1, Dev1:var2, Dev2:var1]

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

    def add_scan_device(self, device_name, variable_list = None):
        """
        Add a new device or append variables to an existing device for scan operations and
        recording their data.

        Args:
            device_name (str or dict): The name of the device to add or update, or dict for composite var
            variable_list (list): A list of variables to add for the device.
        """

        if device_name not in self.devices:
            logging.info(f"Adding new scan device: {device_name} with default settings.")
            self._subscribe_device(device_name, var_list=variable_list)

            # Default attributes for scan-specific devices (e.g., from scan_config)
            default_device_config = {
                'save_non_scalar_data': False,
                'synchronous': False
            }

            self.non_scalar_saving_devices.append(device_name) if default_device_config[
                'save_non_scalar_data'] else None

            # Add scan variables to async_observables
            if self.is_composite_variable(device_name):
                self.async_observables.extend([f"{device_name}"])
            else:
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
            device_name = device_var
            logging.info(f"Trying to add composite device variable {device_var} to self.devices.")
            self.add_scan_device(device_name)  # Add or append the normal variable

        else:
            # Normal variables
            logging.info(f"{device_var} is a normal variable.")
            device_name, var_name = device_var.split(':', 1)
            logging.info(f"Trying to add {device_name}:{var_name} to self.devices.")
            self.add_scan_device(device_name, [var_name])  # Add or append the normal variable
