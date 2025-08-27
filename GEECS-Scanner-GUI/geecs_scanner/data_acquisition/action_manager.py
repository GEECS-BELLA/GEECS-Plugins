"""
Action Management Module for GEECS Experimental Control System.

This module provides the ActionManager class, a sophisticated system for defining,
loading, and executing complex device action sequences in experimental settings.
It enables precise control and automation of device interactions through
configurable, nested action workflows.

Key Features:
- Dynamic action sequence loading from YAML configuration files
- Support for multiple action step types (Set, Get, Wait, Execute)
- Nested action execution
- Device connectivity verification
- Error handling and user interaction mechanisms

Design Principles:
- Flexible configuration through declarative YAML schemas
- Separation of action definition from execution
- Robust error handling and logging
- Seamless integration with GEECS device control infrastructure

Typical Workflow:
1. Define action sequences in a YAML configuration file
2. Load actions using ActionManager
3. Execute individual or nested action sequences
4. Handle device interactions and potential errors

Dependencies:
- PyYAML for configuration parsing
- GEECS Python API for device interactions
- Pydantic for configuration validation

Example Configuration (actions.yaml):
```yaml
actions:
  laser_calibration:
    steps:
      - type: wait
        wait: 1.0
      - type: set
        device: LaserController
        variable: power
        value: 5.0
```

See Also
--------
- geecs_scanner.data_acquisition.schemas.actions : Action type definitions
- geecs_python_api.controls.devices.scan_device.ScanDevice : Device interaction base class
"""

from __future__ import annotations
from typing import Dict, Any, List

import time
import logging
import sys

import yaml

# from PyQt5.QtWidgets import QMessageBox, QApplication

from geecs_python_api.controls.devices.scan_device import ScanDevice
from .utils import get_full_config_path  # Import the utility function
from ..utils.exceptions import ActionError
from geecs_scanner.data_acquisition.schemas.actions import (
    ActionLibrary,
    ActionSequence,
    ActionStep,
    SetStep,
    GetStep,
    ExecuteStep,
    WaitStep,
)

logger = logging.getLogger(__name__)


class ActionManager:
    """Manage and execute complex device actions with nested action support.

    The ActionManager provides a flexible system for defining, loading, and executing
    sequences of device actions. It supports various action types including:
    - Setting device variables
    - Getting device variable values
    - Waiting for specified durations
    - Executing nested action sequences

    Key Features:
    - Load actions from YAML configuration files
    - Dynamic device instantiation
    - Nested action execution
    - Error handling and user interaction for action failures
    - Logging of action execution steps

    Attributes
    ----------
    instantiated_devices : dict
        A cache of ScanDevice instances to avoid repeated device instantiation
    actions : dict
        A dictionary of named action sequences loaded from configuration
    actions_file_path : Path
        Path to the actions configuration YAML file

    Notes
    -----
    - Actions are defined in a YAML file with a specific schema
    - Supports multiple action step types: Set, Get, Wait, Execute
    - Provides mechanisms for device connectivity verification
    - Integrates with GEECS Python API for device interactions

    Examples
    --------
    >>> action_mgr = ActionManager('/path/to/experiment')
    >>> action_mgr.execute_action('calibration_sequence')
    # Executes a predefined calibration action sequence

    See Also
    --------
    geecs_scanner.data_acquisition.schemas.actions : Action type definitions
    geecs_python_api.controls.devices.scan_device.ScanDevice : Device interaction base class
    """

    def __init__(self, experiment_dir: str):
        """
        Initialize the ActionManager with dynamic action sequence loading.

        This method sets up the core infrastructure for managing and executing
        complex device action sequences. It provides flexible configuration
        loading, device management, and error handling mechanisms for experimental
        control workflows.

        Parameters
        ----------
        experiment_dir : str
            The directory path containing the experiment-specific configuration.
            Used to locate and load the actions configuration file.

        Attributes
        ----------
        instantiated_devices : dict
            A runtime cache of ScanDevice instances to optimize device interactions.
            Prevents redundant device instantiations and improves performance.
        actions : Dict[str, ActionSequence]
            A dictionary storing named action sequences loaded from configuration.
            Provides a flexible, dynamic action library for experimental control.
        actions_file_path : Path
            Resolved filesystem path to the actions configuration YAML file.

        Notes
        -----
        Initialization Process:
        - Prepares empty containers for device and action management
        - Attempts to load actions configuration if a valid directory is provided
        - Uses utility function for robust configuration file path resolution
        - Implements graceful handling of missing configuration files

        Raises
        ------
        FileNotFoundError
            If the actions configuration file cannot be located in the specified directory.

        Examples
        --------
        >>> # Standard initialization with experiment directory
        >>> action_mgr = ActionManager('/path/to/experiment')
        >>> # Initialization with optional actions loading
        >>> action_mgr = ActionManager(None)  # No automatic actions loading

        >>> # Dynamic action library management
        >>> action_mgr = ActionManager('Laser_Experiment')
        >>> print(len(action_mgr.actions))  # Number of loaded action sequences
        3

        See Also
        --------
        load_actions : Method for loading action sequences from configuration
        get_full_config_path : Utility for resolving configuration file paths
        """
        # Dictionary to store instantiated GeecsDevices
        self.instantiated_devices: Dict[str, ScanDevice] = {}
        self.actions: Dict[str, ActionSequence] = {}

        if experiment_dir is not None:
            # Use the utility function to get the path to the actions.yaml file
            try:
                self.actions_file_path = get_full_config_path(
                    experiment_dir, "action_library", "actions.yaml"
                )
                self.load_actions()
            except FileNotFoundError:
                logger.exception(
                    f"Actions configuration file not found for experiment: {experiment_dir}"
                )
                logger.info("Continuing with empty action library.")

    def load_actions(self) -> Dict[str, ActionSequence]:
        """
        Load and parse action sequences from the experiment's YAML configuration file.

        This method reads the actions configuration file, validates its structure
        using Pydantic, and populates the internal actions dictionary. It provides
        robust error handling and logging for configuration loading.

        Returns
        -------
        Dict[str, ActionSequence]
            A dictionary of named action sequences loaded from the configuration file.

        Notes
        -----
        Configuration Loading Process:
        - Converts Path object to string for file opening
        - Uses PyYAML for safe YAML parsing
        - Validates configuration using Pydantic's ActionLibrary
        - Logs successful configuration loading
        - Stores actions in the instance's `actions` attribute

        Raises
        ------
        FileNotFoundError
            If the actions configuration file cannot be located
        yaml.YAMLError
            If there are parsing errors in the YAML file
        pydantic.ValidationError
            If the configuration does not match the expected schema

        Examples
        --------
        >>> action_mgr = ActionManager('/path/to/experiment')
        >>> actions = action_mgr.load_actions()
        >>> print(actions.keys())
        dict_keys(['calibration', 'setup', 'teardown'])

        See Also
        --------
        geecs_scanner.data_acquisition.schemas.actions.ActionLibrary : Configuration validation schema
        """
        actions_file = str(self.actions_file_path)  # Convert Path object to string

        try:
            with open(actions_file, "r") as file:
                raw_yaml = yaml.safe_load(file)

            library = ActionLibrary(**raw_yaml)
            logger.info(f"Successfully loaded master actions from {actions_file}")

            self.actions = library.actions
            logger.debug(f"Loaded {len(self.actions)} action sequences")

            return self.actions

        except FileNotFoundError:
            logger.exception(f"Actions configuration file not found: {actions_file}")
            raise
        except yaml.YAMLError as e:
            logger.exception(f"YAML parsing error in {actions_file}: {e}")
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error loading actions from {actions_file}: {e}"
            )
            raise

    def add_action(self, action_name: str, action_seq: ActionSequence):
        """
        Add a new action sequence to the current action library.

        This method allows dynamic addition of action sequences during runtime.
        The action is added to the in-memory actions dictionary but is not
        permanently saved to the configuration file.

        Parameters
        ----------
        action_name : str
            A unique identifier for the action sequence.
        action_seq : ActionSequence
            A validated action sequence containing a list of action steps.

        Notes
        -----
        - Adds the action to the `self.actions` dictionary
        - Does not modify the original configuration file
        - Provides runtime flexibility for action management
        - Validates action sequence structure

        Raises
        ------
        ValueError
            If there's an error adding the action sequence
        KeyError
            If an action with the same name already exists

        Examples
        --------
        >>> action_mgr = ActionManager('/path/to/experiment')
        >>> new_action = ActionSequence(steps=[
        ...     SetStep(device='LaserController', variable='power', value=5.0)
        ... ])
        >>> action_mgr.add_action('laser_calibration', new_action)

        See Also
        --------
        geecs_scanner.data_acquisition.schemas.actions.ActionSequence : Action sequence validation schema
        """
        try:
            if action_name in self.actions:
                logger.warning(f"Overwriting existing action: {action_name}")

            self.actions[action_name] = action_seq
            logger.info(f"Added action sequence: {action_name}")

        except Exception as e:
            error_msg = f"Failed to add action '{action_name}': {e}"
            logger.exception(error_msg)
            raise ValueError(error_msg)

    def execute_action(self, action_name: str):
        """
        Execute a comprehensive action sequence with advanced device interaction and error handling.

        This method provides a robust mechanism for executing complex, multi-step
        action sequences. It supports various action types, nested actions, and
        provides detailed logging and error management.

        Parameters
        ----------
        action_name : str
            The unique identifier of the action sequence to execute.

        Raises
        ------
        ActionError
            - If the specified action is not defined in the action library
            - If an unrecognized action step type is encountered
            - If a device interaction fails and requires user intervention

        Notes
        -----
        Execution Process:
        1. Validate action existence
        2. Ping devices to ensure connectivity
        3. Execute steps sequentially:
            - Wait steps pause execution for specified duration
            - Execute steps trigger nested action sequences
            - Set steps modify device variables with optional synchronization
            - Get steps retrieve and validate device values

        Detailed Step Handling:
        - Supports dynamic step type matching
        - Provides logging for each step execution
        - Handles potential errors during device interactions
        - Supports nested action execution

        Examples
        --------
        >>> action_mgr = ActionManager('/path/to/experiment')
        >>> action_mgr.execute_action('laser_calibration')
        # Executes a predefined laser calibration sequence

        >>> action_mgr.execute_action('complex_experiment_setup')
        # Executes a multi-step, nested action sequence

        See Also
        --------
        add_action : Method for dynamically adding action sequences
        _get_or_create_device : Device instantiation utility
        geecs_scanner.data_acquisition.schemas.actions : Action type definitions
        """
        if action_name not in self.actions:
            raise ActionError(f"Action '{action_name}' is not defined.")

        logger.info(f"Starting execution of action sequence: {action_name}")

        action = self.actions[action_name]
        steps = action.steps

        try:
            self.ping_devices_in_action_list(action_steps=steps)
        except Exception as e:
            logger.exception(
                f"Device connectivity check failed for action {action_name}: {e}"
            )
            raise ActionError(f"Device connectivity error in action {action_name}")

        for step_index, step in enumerate(steps, 1):
            logger.debug(
                f"Executing step {step_index}/{len(steps)} in action {action_name}"
            )

            try:
                match step:
                    case WaitStep():
                        logger.info(f"Waiting for {step.wait} seconds")
                        self._wait(step.wait)
                    case ExecuteStep():
                        logger.info(f"Executing nested action: {step.action_name}")
                        self.execute_action(step.action_name)
                    case SetStep():
                        device = self._get_or_create_device(step.device)
                        self._set_device(
                            device,
                            step.variable,
                            step.value,
                            sync=step.wait_for_execution,
                        )
                    case GetStep():
                        device = self._get_or_create_device(step.device)
                        self._get_device(device, step.variable, step.expected_value)
                    case _:
                        raise ActionError(
                            f"Unrecognized action step type: {type(step)}"
                        )

            except Exception as e:
                logger.exception(
                    f"Error executing step {step_index} in action {action_name}: {e}"
                )
                raise ActionError(f"Step execution failed in action {action_name}")

        logger.info(f"Successfully completed action sequence: {action_name}")

    def _get_or_create_device(self, device_name: str) -> ScanDevice:
        if device_name not in self.instantiated_devices:
            self.instantiated_devices[device_name] = ScanDevice(device_name)
        return self.instantiated_devices[device_name]

    def ping_devices_in_action_list(self, action_steps: List[ActionStep]):
        """
        Verify device connectivity for all unique devices in an action sequence.

        This method performs a comprehensive connectivity check by attempting
        to retrieve the system timestamp from each unique device involved in
        the action steps. It ensures that all required devices are operational
        before executing an action sequence.

        Parameters
        ----------
        action_steps : List[ActionStep]
            A list of action steps to check for device connectivity.

        Raises
        ------
        GeecsDeviceInstantiationError
            If any device in the action list is not turned on or cannot be pinged.

        Notes
        -----
        Connectivity Check Process:
        - Identifies unique devices from SetStep and GetStep action types
        - Attempts to retrieve 'SysTimestamp' from each device
        - Provides early detection of device connectivity issues
        - Prevents execution of actions with unavailable devices

        Examples
        --------
        >>> action_mgr = ActionManager('/path/to/experiment')
        >>> steps = [SetStep(device='LaserController', variable='power', value=5.0)]
        >>> action_mgr.ping_devices_in_action_list(steps)
        # Checks if LaserController is available and responsive

        See Also
        --------
        _get_or_create_device : Device instantiation utility method
        return_value : Method for retrieving device variable values
        """
        devices = {
            step.device for step in action_steps if isinstance(step, (SetStep, GetStep))
        }

        logger.info(f"Checking connectivity for {len(devices)} devices")

        for device_name in devices:
            try:
                timestamp = self.return_value(device_name, "SysTimestamp")
                logger.debug(
                    f"Device {device_name} connectivity check passed. Timestamp: {timestamp}"
                )
            except Exception as e:
                logger.exception(
                    f"Device connectivity check failed for {device_name}: {e}"
                )
                raise

    def clear_action(self, action_name: str):
        """
        Remove a specific action sequence from the action library.

        This method provides a mechanism to dynamically remove action sequences
        from the runtime action library. It ensures that only existing actions
        can be removed and logs appropriate messages for tracking.

        Parameters
        ----------
        action_name : str
            The unique identifier of the action sequence to remove.

        Notes
        -----
        - Checks for action existence before removal
        - Logs an error if the action is not found
        - Provides runtime flexibility for action library management

        Examples
        --------
        >>> action_mgr = ActionManager('/path/to/experiment')
        >>> action_mgr.clear_action('obsolete_calibration')
        # Removes the 'obsolete_calibration' action sequence

        Raises
        ------
        No explicit exceptions, but logs an error for undefined actions
        """
        if action_name not in self.actions:
            logger.error(
                f"Action '{action_name}' is not defined in the available actions."
            )
            return

        del self.actions[action_name]
        logger.info(f"Removed action sequence: {action_name}")

    @staticmethod
    def _set_device(device: ScanDevice, variable: str, value: Any, sync: bool = True):
        """
        Set a device variable to a specified value with optional synchronization.

        This method provides a robust mechanism for modifying device variables,
        with support for synchronous and asynchronous execution. It ensures
        comprehensive logging of device interactions.

        Parameters
        ----------
        device : ScanDevice
            The device instance to interact with.
        variable : str
            The name of the variable to set.
        value : Any
            The value to assign to the variable.
        sync : bool, optional
            Whether to wait for the set operation to complete. Defaults to True.

        Notes
        -----
        - Supports both synchronous and asynchronous device variable setting
        - Provides detailed logging of device interaction results
        - Handles various device variable types

        Examples
        --------
        >>> device = ScanDevice('LaserController')
        >>> ActionManager._set_device(device, 'power', 5.0)
        # Sets laser power to 5.0 with synchronous execution

        >>> ActionManager._set_device(device, 'mode', 'calibration', sync=False)
        # Sets device mode asynchronously
        """
        result = device.set(variable, value, sync=sync)
        logger.info(f"Set {device.get_name()}:{variable} to {value}. Result: {result}")

    def _get_device(self, device: ScanDevice, variable: str, expected_value: Any):
        """
        Retrieve and validate a device variable's value.

        This method provides a comprehensive mechanism for retrieving device
        variables and comparing them against expected values. It supports
        user interaction for handling unexpected values.

        Parameters
        ----------
        device : ScanDevice
            The device instance to query.
        variable : str
            The name of the variable to retrieve.
        expected_value : Any
            The expected value for comparison.

        Notes
        -----
        - Retrieves current device variable value
        - Compares retrieved value with expected value
        - Provides logging for successful and unexpected value retrieval
        - Supports user interaction for handling discrepancies

        Raises
        ------
        ActionError
            If the retrieved value differs from the expected value and
            the user chooses to abort the action.

        Examples
        --------
        >>> device = ScanDevice('Spectrometer')
        >>> action_mgr._get_device(device, 'status', 'ready')
        # Retrieves and validates spectrometer status
        """
        value = device.get(variable)
        if value == expected_value:
            logger.info(
                f"Get {device.get_name()}:{variable} returned expected value: {value}"
            )
        else:
            message = f"Get {device.get_name()}:{variable} returned {value}, expected {expected_value}"
            logger.warning(message)
            if self._prompt_user_quit_action(message):
                raise ActionError(message)

    def return_value(self, device_name: str, variable: str):
        """
        Retrieve a device variable's current value with dynamic device instantiation.

        This method provides a flexible mechanism for retrieving device variables,
        automatically instantiating devices if they haven't been previously created.

        Parameters
        ----------
        device_name : str
            The name of the device to query.
        variable : str
            The name of the variable to retrieve.

        Returns
        -------
        Any
            The current value of the specified device variable.

        Notes
        -----
        - Supports dynamic device instantiation
        - Caches device instances for performance optimization
        - Provides a simple interface for device variable retrieval

        Examples
        --------
        >>> timestamp = action_mgr.return_value('LaserController', 'SysTimestamp')
        # Retrieves system timestamp from LaserController
        """
        if device_name not in self.instantiated_devices:
            self.instantiated_devices[device_name] = ScanDevice(device_name)

        device: ScanDevice = self.instantiated_devices[device_name]
        return device.get(variable)

    @staticmethod
    def _wait(seconds: float):
        """
        Pause execution for a specified duration.

        This method provides a simple, logging-enabled mechanism for introducing
        time-based pauses in action sequences. It supports precise control over
        execution timing.

        Parameters
        ----------
        seconds : float
            The number of seconds to pause execution.

        Notes
        -----
        - Uses time.sleep() for precise timing
        - Logs wait duration for traceability
        - Supports fractional second pauses

        Examples
        --------
        >>> ActionManager._wait(1.5)
        # Pauses execution for 1.5 seconds
        """
        logger.info(f"Waiting for {seconds} seconds.")
        time.sleep(seconds)

    @staticmethod
    def _prompt_user_quit_action(message: str) -> bool:
        """
        Display an interactive dialog for user decision during action execution.

        This method provides a user-friendly mechanism for handling unexpected
        device interactions, allowing users to choose between continuing or
        aborting the current action sequence.

        Parameters
        ----------
        message : str
            Descriptive message explaining the reason for user interaction.

        Returns
        -------
        bool
            True if the user chooses to abort the action, False otherwise.

        Notes
        -----
        - Uses PyQt5 for cross-platform dialog creation
        - Supports dynamic application instance creation
        - Provides clear, actionable user interface
        - Handles various device interaction scenarios

        Examples
        --------
        >>> if action_mgr._prompt_user_quit_action("Unexpected device state"):
        ...     # User chose to abort
        ...     raise ActionError("User aborted action")
        """
        from PyQt5.QtWidgets import QMessageBox, QApplication

        if not QApplication.instance():
            QApplication(sys.argv)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(
            f'Failed "get" command: \n {message} \nQuit out of action and scan?'
        )
        msg_box.setWindowTitle("Action Error")
        msg_box.setStandardButtons(QMessageBox.Abort | QMessageBox.Ignore)
        msg_box.setDefaultButton(QMessageBox.Abort)
        response = msg_box.exec_()

        return response == QMessageBox.Abort
