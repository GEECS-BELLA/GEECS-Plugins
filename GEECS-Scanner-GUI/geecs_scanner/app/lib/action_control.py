"""
Contains a simpler version of a full "RunControl" that is only capable of performing actions using ActionManager

-Chris
"""
from typing import Union
from geecs_scanner.data_acquisition import ActionManager
import logging

from ...utils.exceptions import ActionError
from ...utils.sound_player import action_finish_jingle, action_faled_jingle
from ...utils.action_definitions import ActionStep


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
    def generate_action_description(action_step: ActionStep) -> str:
        """ For each action type, generate a string that displays all the information for that action step """
        if action_step.type == 'wait':
            description = f"wait {action_step.time}"
        elif action_step.type == 'execute':
            description = f"execute {action_step.name}"
        elif action_step.type == 'set':
            description = f"{action_step.type} {action_step.device}:{action_step.variable} {action_step.value}"
        elif action_step.type == 'get':
            description = f"{action_step.type} {action_step.device}:{action_step.variable} {action_step.expected_value}"
        else:
            raise KeyError(f"Unknown action type '{action_step.type}'")
        return description

    def perform_action(self, action_list: list[ActionStep]):
        """
        Performs the given list of actions using the currently-connected instance of Action Manager

        :param action_list: dict containing steps of the action
        """
        name = 'test_action'
        if len(action_list) > 0:
            try:
                self.action_manager.add_action(action_name=name, action_steps=action_list)
                self.action_manager.execute_action(name)
                self.action_manager.clear_action(name)
                action_finish_jingle()
            except ActionError as e:
                logging.error(e.message)
                action_faled_jingle()
