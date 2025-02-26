from __future__ import annotations

from typing import Any

import time
import logging
import yaml
import sys

from PyQt5.QtWidgets import QMessageBox, QApplication

from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from .utils import get_full_config_path  # Import the utility function
from ..utils.exceptions import ActionError
from ..utils.action_definitions import ActionStep, get_all_action_lists


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
        self.actions: dict[str, list[ActionStep]] = {}

        if experiment_dir is not None:
            # Use the utility function to get the path to the actions.yaml file
            try:
                self.actions_file_path = get_full_config_path(experiment_dir, 'action_library', 'actions.yaml')
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
        self.actions = get_all_action_lists(actions)

        protected_names = ['setup_action', 'closeout_action', 'test_action']
        for protected in protected_names:
            if protected in self.actions.keys():
                logging.warning(f"Protected action name '{protected}' in use in actions.yaml file!")

    def add_action(self, action_name: str, action_steps: list[ActionStep]):

        """
        Add a new action to the default actions list. NOTE, action is not saved

        Args:
            action_name (str): Name of the action sequence
            action_steps (list): A list of ActionStep objects

        """
        self.actions[action_name] = action_steps

    def execute_action(self, action_name: str):

        """
        Execute a single action by its name, handling both device actions and nested actions.

        Args:
            action_name (str): The name of the action to execute.

        Raises:
            ActionError:  if the action name is not defined
        """

        if action_name not in self.actions:
            message = f"Action '{action_name}' is not defined in the available actions."
            logging.error(message)
            raise ActionError(message)

        for step in self.actions[action_name]:
            if step.type == 'wait':
                self._wait(step.time)
            elif step.type == 'execute':
                # Nested action: recursively execute the named action
                logging.info(f"Executing nested action: {step.name}")
                self.execute_action(step.name)

            elif step.type in ['get', 'set']:
                # Regular device action
                device_name = step.device
                action_type = step.type

                # Instantiate device if it hasn't been done yet
                if device_name not in self.instantiated_devices:
                    self.instantiated_devices[device_name] = GeecsDevice(device_name)

                device = self.instantiated_devices[device_name]

                if action_type == 'set':
                    wait_for_execution = True  # TODO not implemented in GUI, but always good to do
                    self._set_device(device, step.variable, step.value, sync=wait_for_execution)
                elif action_type == 'get':
                    self._get_device(device, step.variable, step.expected_value)

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
    def _set_device(device: GeecsDevice, variable: str, value: Any, sync: bool = True):
        """
        Set a device variable to a specified value.

        Args:
            device (GeecsDevice): The device to control.
            variable (str): The variable to set.
            value (any): The value to set for the variable.
        """

        result = device.set(variable, value, sync=sync)
        logging.info(f"Set {device.get_name()}:{variable} to {value}. Result: {result}")

    @staticmethod
    def _get_device(device: GeecsDevice, variable: str, expected_value: Any):

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
            if ActionManager._prompt_user_quit_action(message):
                raise ActionError(message)

    @staticmethod
    def _wait(seconds: float):

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
        if not QApplication.instance():
            QApplication(sys.argv)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(f'Failed "get" command: \n {message} \nQuit out of action and scan?')
        msg_box.setWindowTitle("Action Error")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        response = msg_box.exec_()

        return response == QMessageBox.Yes
