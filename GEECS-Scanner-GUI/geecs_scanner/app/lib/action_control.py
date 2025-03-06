"""
Contains a simpler version of a full "RunControl" that is only capable of performing actions using ActionManager

-Chris
"""
from typing import Any
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
