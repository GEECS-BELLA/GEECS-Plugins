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
from geecs_scanner.data_acquisition.schemas.actions import (ActionLibrary, ActionSequence,
                                                            ActionStep, SetStep, GetStep,
                                                            ExecuteStep, WaitStep)

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
        self.actions: Dict[str, ActionSequence] = {}

        if experiment_dir is not None:
            # Use the utility function to get the path to the actions.yaml file
            try:
                self.actions_file_path = get_full_config_path(experiment_dir, 'action_library', 'actions.yaml')
                self.load_actions()
            except FileNotFoundError:
                logging.warning(f"actions.yaml file not found.")

    def load_actions(self)-> Dict[str, ActionSequence]:

        """
        Load the master actions from the given YAML file.

        Returns:
            dict: A dictionary of NamedActions loaded from the YAML file.
        """

        actions_file = str(self.actions_file_path)  # Convert Path object to string
        with open(actions_file, 'r') as file:
            raw_yaml = yaml.safe_load(file)
        library = ActionLibrary(**raw_yaml)
        logging.info(f"Loaded master actions from {actions_file}")
        self.actions = library.actions
        return self.actions

    def add_action(self, action_name: str, action_steps: dict[str, list]):

        """
        Add a new action to the default actions list. NOTE, action is not saved

        Args:
            action_name (str): Name of the action sequence
            action_steps (dict): A dictionary containing 'steps':list.

        Raises:
            ValueError: If the action steps dictionary does not contain only 'steps'.
        """
        try:
            self.actions[action_name] = ActionSequence(**action_steps)
        except Exception as e:
            raise ValueError(f"Failed to add action '{action_name}': {e}")

    def execute_action(self, action_name: str):

        """
        Execute a single action by its name, handling both device actions and nested actions.

        Args:
            action_name (str): The name of the action to execute.

        Raises:
            ActionError:  if the action name is not defined
        """

        if action_name not in self.actions:
            raise ActionError(f"Action '{action_name}' is not defined.")

        action = self.actions[action_name]
        steps = action.steps
        self.ping_devices_in_action_list(action_steps=steps)

        for step in steps:
            match step:
                case WaitStep():
                    self._wait(step.wait)
                case ExecuteStep():
                    logging.info(f"Executing nested action: {step.action_name}")
                    self.execute_action(step.action_name)
                case SetStep():
                    device = self._get_or_create_device(step.device)
                    self._set_device(device, step.variable, step.value, sync=step.wait_for_execution)
                case GetStep():
                    device = self._get_or_create_device(step.device)
                    self._get_device(device, step.variable, step.expected_value)
                case _:
                    raise ActionError(f"Unrecognized action step type: {type(step)}")

    def _get_or_create_device(self, device_name: str) -> ScanDevice:
        if device_name not in self.instantiated_devices:
            self.instantiated_devices[device_name] = ScanDevice(device_name)
        return self.instantiated_devices[device_name]

    def ping_devices_in_action_list(self, action_steps: List[ActionStep]):
        """
        For each unique device in given list of ActionSteps, ping for the system timestamp.  Raises a
        GeecsDeviceInstantiationError if a device is not turned on
        """
        devices = {
            step.device
            for step in action_steps
            if isinstance(step, (SetStep, GetStep))
        }
        for device_name in devices:
            self.return_value(device_name, 'SysTimestamp')

    def clear_action(self, action_name: str):
        """
        Clears the given action_name from memory

        Args:
            action_name (str): The name of the action to execute.
        """
        if action_name not in self.actions:
            logging.error(f"Action '{action_name}' is not defined in the available actions.")
            return

        del self.actions[action_name]

    @staticmethod
    def _set_device(device: ScanDevice, variable: str, value: Any, sync: bool = True):
        """
        Set a device variable to a specified value.

        Args:
            device (GeecsDevice): The device to control.
            variable (str): The variable to set.
            value (any): The value to set for the variable.
        """

        result = device.set(variable, value, sync = sync)
        logging.info(f"Set {device.get_name()}:{variable} to {value}. Result: {result}")

    def _get_device(self, device: ScanDevice, variable: str, expected_value: Any):

        """
        Get the current value of a device variable and compare it to the expected value.

        Args:
            device (GeecsDevice): The device to query.
            variable (str): The variable to get the value of.
            expected_value (any): The expected value for comparison.

        Raises:
            ActionError: If the 'get' command fails and the user opts to quit out of the action and scan
        """

        value = device.get(variable)
        if value == expected_value:
            logging.info(f"Get {device.get_name()}:{variable} returned expected value: {value}")
        else:
            message = f"Get {device.get_name()}:{variable} returned {value}, expected {expected_value}"
            logging.warning(message)
            if self._prompt_user_quit_action(message):
                raise ActionError(message)

    def return_value(self, device_name: str, variable: str):
        """
        Get the current value of a device variable

        Args:
            device_name (str): The device to query.
            variable (str): The variable to get the value of.
        """

        if device_name not in self.instantiated_devices:
            self.instantiated_devices[device_name] = ScanDevice(device_name)

        device: ScanDevice = self.instantiated_devices[device_name]
        return device.get(variable)

    @staticmethod
    def _wait(seconds):

        """
        Wait for a specified number of seconds.

        Args:
            seconds (float): The number of seconds to wait.
        """

        logging.info(f"Waiting for {seconds} seconds.")
        time.sleep(seconds)

    @staticmethod
    def _prompt_user_quit_action(message: str) -> bool:
        """
        Display a pop-up asking if the user would like to abort the current action or continue.

        :param message: str Message that displays in text box
        :return: bool True if an error should be raised, false otherwise
        """

        from PyQt5.QtWidgets import QMessageBox, QApplication

        if not QApplication.instance():
            QApplication(sys.argv)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(f'Failed "get" command: \n {message} \nQuit out of action and scan?')
        msg_box.setWindowTitle("Action Error")
        msg_box.setStandardButtons(QMessageBox.Abort | QMessageBox.Ignore)
        msg_box.setDefaultButton(QMessageBox.Abort)
        response = msg_box.exec_()

        return response == QMessageBox.Abort
