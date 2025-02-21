import time
import logging
import yaml
import sys

from PyQt5.QtWidgets import QMessageBox, QApplication

from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from .utils import get_full_config_path  # Import the utility function
from ..utils.exceptions import ActionError


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
        self.actions = actions['actions']
        return actions['actions']

    def add_action(self, action_name: str, action_steps: dict[str, list]):

        """
        Add a new action to the default actions list. NOTE, action is not saved

        Args:
            action_name (str): Name of the action sequence
            action_steps (dict): A dictionary containing 'steps':list.

        Raises:
            ValueError: If the action steps dictionary does not contain only 'steps'.
        """
        # Check that the steps are formatted correctly
        if 'steps' not in action_steps or len(action_steps) != 1:
            raise ValueError("Action Steps not formatted correctly")

        # Add the action to the actions dictionary
        self.actions[action_name] = action_steps

    def execute_action(self, action_name):

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

    def _wait(self, seconds):

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
