"""Simpler RunControl variant that only performs actions via ActionManager."""

from typing import Any

import logging

from geecs_scanner.data_acquisition import ActionManager
from geecs_scanner.data_acquisition.schemas.actions import ActionSequence
from pydantic import ValidationError

from ...utils.exceptions import ActionError
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceInstantiationError,
)

from ...utils.sound_player import action_finish_jingle, action_failed_jingle

from geecs_scanner.app.gui_dialogs import show_action_error_dialog


class ActionControl:
    """Thin wrapper around ActionManager for GUI-triggered action execution."""

    def __init__(self, experiment_name: str):
        """Open a direct link to the ActionManager within data_acquisition.

        :param experiment_name: Name of the experiment, used to find the right config files and connect to devices
        """
        self.action_manager = ActionManager(experiment_dir=experiment_name)

    def perform_action(self, action_list: dict[str, list]):
        """Perform the given list of actions using the currently-connected ActionManager.

        :param action_list: dict containing steps of the action
        """
        name = "test_action"
        if "steps" in action_list and len(action_list["steps"]) > 0:
            try:
                # ✅ Validate structure using ActionSequence model
                validated_sequence = ActionSequence(**action_list)
                self.action_manager.add_action(
                    action_name=name, action_seq=validated_sequence
                )
                self.action_manager.execute_action(name)
                self.action_manager.clear_action(name)
                action_finish_jingle()
            except ValidationError as e:
                logging.error("Invalid action format:\n%s", e)
                show_action_error_dialog(e)
            except (ActionError, GeecsDeviceInstantiationError) as e:
                logging.error("Action failed: %s", e)
                action_failed_jingle()
                show_action_error_dialog(e)

    def return_device_value(self, device_name: str, variable: str) -> Any:
        """Call TCP get command to a given device for the value of the given variable.

        device_name (str): The device to query.
        variable (str): The variable to get the value of.
        :return: Value of device variable, None if variable does not exist

        :raises:
            ActionError if a GeecsDeviceInstantiationError occurred, as ActionError lives outside GEECS-PythonAPI
        """
        try:
            return self.action_manager.return_value(device_name, variable)
        except GeecsDeviceInstantiationError as e:
            raise ActionError(message=str(e))
