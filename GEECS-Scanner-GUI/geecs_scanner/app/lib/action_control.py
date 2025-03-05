"""
Contains a simpler version of a full "RunControl" that is only capable of performing actions using ActionManager

-Chris
"""
from typing import Union, Any
from geecs_scanner.data_acquisition import ActionManager
import logging

from ...utils.exceptions import ActionError
from geecs_python_api.controls.interface.geecs_errors import GeecsDeviceInstantiationError

from ...utils.sound_player import action_finish_jingle, action_failed_jingle


class ActionControl:
    def __init__(self, experiment_name: str):
        """
        Opens a direct link to the Action Manager within `data_acquisition`

        :param experiment_name: Name of the experiment, used to find the right config files and connect to devices
        """
        self.action_manager = ActionManager(experiment_dir=experiment_name)

    @staticmethod
    def get_new_action(action) -> Union[None, dict[str, str]]:
        """
        Translates a given action keyword to a default dictionary for that action keyword

        :param action: action keyword
        :return: default dictionary for the associated action
        """
        default = None
        if action == 'set':
            default = {
                'action': 'set',
                'device': '',
                'variable': '',
                'value': ''
            }
        elif action == 'get':
            default = {
                'action': 'get',
                'device': '',
                'variable': '',
                'expected_value': ''
            }
        elif action == 'wait':
            default = {
                'wait': ''
            }
        elif action == 'execute':
            default = {
                'action': 'execute',
                'action_name': ''
            }
        elif action == 'run':
            default = {
                'action': 'run',
                'file_name': '',
                'class_name': ''
            }
        return default

    # List of available actions, to be used by the completer for the add action line edit
    list_of_actions = [
        'set',
        'get',
        'wait',
        'execute',
    ]

    @staticmethod
    def generate_action_description(action: dict[str, list]) -> str:
        """ For each action type, generate a string that displays all the information for that action step """
        description = "???"
        if action.get("wait") is not None:
            description = f"wait {action['wait']}"
        elif action['action'] == 'execute':
            description = f"execute {action['action_name']}"
        elif action['action'] == 'set':
            description = f"{action['action']} {action['device']}:{action['variable']} {action.get('value')}"
        elif action['action'] == 'get':
            description = f"{action['action']} {action['device']}:{action['variable']} {action.get('expected_value')}"
        return description

    def perform_action(self, action_list: dict[str, list]):
        """
        Performs the given list of actions using the currently-connected instance of Action Manager

        :param action_list: dict containing steps of the action
        """
        name = 'test_action'
        if 'steps' in action_list and len(action_list['steps']) > 0:
            try:
                self.action_manager.add_action(action_name=name, action_steps=action_list)
                self.action_manager.execute_action(name)
                self.action_manager.clear_action(name)
                action_finish_jingle()
            except ActionError as e:
                logging.error(e.message)
                action_failed_jingle()

    def return_device_value(self, device_name: str, variable: str) -> Any:
        """ Calls TCP get command to a given device for the value of the given variable

        device_name (str): The device to query.
        variable (str): The variable to get the value of.
        :return: Value of device variable, None if variable does not exist

        :raises:
            ActionError if a GeecsDeviceInstantiationError occurred, as ActionError lives outside GEECS-PythonAPI
        """
        try:
            return self.action_manager.return_value(device_name, variable)
        except GeecsDeviceInstantiationError as e:
            raise ActionError(message=e.message)
