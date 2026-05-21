"""Simpler RunControl that performs actions via ActionManager without full scan support."""

from typing import Any
from geecs_scanner.engine import ActionManager, DeviceCommandExecutor
from geecs_scanner.engine.models.actions import ActionSequence
from pydantic import ValidationError
import logging
import sys

from ...utils.exceptions import ActionError
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceInstantiationError,
)

from ...utils.sound_player import action_finish_jingle, action_failed_jingle

from PyQt5.QtWidgets import QMessageBox, QApplication


class ActionControl:
    """Wrapper around ActionManager for GUI-triggered action execution."""

    def __init__(self, experiment_name: str):
        """Open a direct link to the ActionManager within data_acquisition.

        :param experiment_name: Name of the experiment, used to find the right config files and connect to devices
        """
        self.action_manager = ActionManager(experiment_dir=experiment_name)
        self.action_manager.cmd_executor = DeviceCommandExecutor(
            on_escalate=self._on_device_error
        )

    def perform_action(self, action_list: dict[str, list]):
        """Perform the given list of actions using the currently-connected ActionManager.

        :param action_list: dict containing steps of the action
        """
        name = "test_action"
        if "steps" in action_list and len(action_list["steps"]) > 0:
            self.action_manager.on_user_prompt = self._on_device_error
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
                logging.error(f"Invalid action format:\n{e}")
                self._display_error_message(str(e))
            except (ActionError, GeecsDeviceInstantiationError) as e:
                logging.error("Action failed: %s", e)
                action_failed_jingle()
                self._display_error_message(str(e))
            finally:
                self.action_manager.on_user_prompt = None

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

    def _on_device_error(self, exc: Exception, context=None) -> bool:
        """
        Callback wired to action_manager.on_user_prompt during action execution.

        Called when a device command error (Rejected/Failed/Timeout) occurs
        inside the action manager.  Shows the same error dialog as other action
        failures and always returns True (abort) so the caller knows to stop.
        """
        logging.error("Action device error: %s", exc)
        action_failed_jingle()
        msg = str(exc)
        if context:
            msg += f"\n\n{context}"
        self._display_error_message(msg)
        return True

    @staticmethod
    def _display_error_message(message: str):
        """
        Display a message when an action fails.

        :param message: str Message that displays in text box
        :return: bool True if an error should be raised, false otherwise
        """
        if not QApplication.instance():
            QApplication(sys.argv)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(f"Execute Action Failed:\n{message}")
        msg_box.setWindowTitle("Action Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.exec_()
