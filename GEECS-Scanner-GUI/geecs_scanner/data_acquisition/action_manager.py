"""Load and execute named device-action sequences defined in ``actions.yaml``."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import yaml

from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_scanner.data_acquisition.dialog_request import (
    DEVICE_COMMAND_ERRORS,
    escalate_device_error,
)
from geecs_scanner.data_acquisition.schemas.actions import (
    ActionLibrary,
    ActionSequence,
    ActionStep,
    ExecuteStep,
    GetStep,
    SetStep,
    WaitStep,
)

from ..utils.exceptions import ActionError
from .utils import get_full_config_path

logger = logging.getLogger(__name__)


class ActionManager:
    """Load and execute named device-action sequences.

    Supports Set, Get, Wait, and nested Execute steps, loaded from
    ``actions.yaml`` in the experiment config directory.

    Attributes
    ----------
    instantiated_devices : dict
        Cache of ScanDevice instances; avoids repeated TCP connections.
    actions : dict[str, ActionSequence]
    actions_file_path : Path
    on_user_prompt : callable or None
        Injected by ScanManager after construction.
        Signature: ``(exc: Exception) -> bool`` — True means user chose Abort.
        Falls back to auto-abort if None (headless / test contexts).
    """

    def __init__(self, experiment_dir: str):
        self.instantiated_devices: Dict[str, ScanDevice] = {}
        self.actions: Dict[str, ActionSequence] = {}

        # Injected by ScanManager after construction — same pattern as executor callbacks.
        self.on_user_prompt = None

        if experiment_dir is not None:
            try:
                self.actions_file_path = get_full_config_path(
                    experiment_dir, "action_library", "actions.yaml"
                )
                self.load_actions()
            except FileNotFoundError:
                logger.exception(
                    "Actions configuration file not found for experiment: %s",
                    experiment_dir,
                )
                logger.info("Continuing with empty action library.")

    def load_actions(self) -> Dict[str, ActionSequence]:
        """Parse ``actions.yaml`` and populate ``self.actions``."""
        actions_file = str(self.actions_file_path)

        try:
            with open(actions_file, "r") as file:
                raw_yaml = yaml.safe_load(file)

            library = ActionLibrary(**raw_yaml)
            logger.info("Successfully loaded master actions from %s", actions_file)

            self.actions = library.actions
            logger.debug("Loaded %s action sequences", len(self.actions))

            return self.actions

        except FileNotFoundError:
            logger.exception("Actions configuration file not found: %s", actions_file)
            raise

        except yaml.YAMLError:
            logger.exception("YAML parsing error in %s", actions_file)
            raise

        except Exception:
            logger.exception("Unexpected error loading actions from %s", actions_file)
            raise

    def add_action(self, action_name: str, action_seq: ActionSequence):
        """Add or overwrite an action sequence in the in-memory library."""
        try:
            if action_name in self.actions:
                logger.warning("Overwriting existing action: %s", action_name)

            self.actions[action_name] = action_seq
            logger.info("Added action sequence: %s", action_name)

        except Exception:
            logger.exception("Failed to add action %s", action_name)
            raise ValueError(f"Failed to add action {action_name}")

    def execute_action(self, action_name: str):
        """Run all steps of *action_name* sequentially.

        Raises
        ------
        ActionError
            If the action is not defined, a step type is unrecognized, or a
            device interaction fails.
        """
        if action_name not in self.actions:
            raise ActionError(f"Action '{action_name}' is not defined.")

        logger.info("Starting execution of action sequence: %s", action_name)

        action = self.actions[action_name]
        steps = action.steps

        try:
            self.ping_devices_in_action_list(action_steps=steps)
        except Exception:
            logger.exception(
                "Device connectivity check failed for action %s", action_name
            )
            raise ActionError("Device connectivity error in action %s", action_name)

        for step_index, step in enumerate(steps, 1):
            logger.debug(
                "Executing step %s/%s in action %s", step_index, len(steps), action_name
            )

            try:
                match step:
                    case WaitStep():
                        logger.info("Waiting for %s seconds", step.wait)
                        self._wait(step.wait)
                    case ExecuteStep():
                        logger.info("Executing nested action: %s", step.action_name)
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

            except Exception:
                logger.exception(
                    "Error executing step %s in action %s", step_index, action_name
                )
                raise ActionError(f"Step execution failed in action {action_name}")

        logger.info("Successfully completed action sequence: %s", action_name)

    def _get_or_create_device(self, device_name: str) -> ScanDevice:
        if device_name not in self.instantiated_devices:
            self.instantiated_devices[device_name] = ScanDevice(device_name)
        return self.instantiated_devices[device_name]

    def ping_devices_in_action_list(self, action_steps: List[ActionStep]):
        """Instantiate each device in *action_steps* and verify its alive flag."""
        devices = {
            step.device for step in action_steps if isinstance(step, (SetStep, GetStep))
        }

        logger.info(
            "Checking instantiation status for %d unique device(s)", len(devices)
        )

        for device_name in devices:
            try:
                device = self._get_or_create_device(device_name)
                if not device.state.get("Device alive on instantiation", False):
                    raise Exception(
                        f"Device {device_name} failed instantiation connectivity test"
                    )
                logger.debug("Device %s instantiation check passed", device_name)
            except Exception:
                logger.exception(
                    "Device instantiation check failed for %s", device_name
                )
                raise

    def clear_action(self, action_name: str):
        """Remove *action_name* from the in-memory library."""
        if action_name not in self.actions:
            logger.error(
                "Action '%s' is not defined in the available actions.", action_name
            )
            return

        del self.actions[action_name]
        logger.info("Removed action sequence: %s", action_name)

    def _set_device(
        self, device: ScanDevice, variable: str, value: Any, sync: bool = True
    ):
        """Call ``device.set(variable, value)`` and escalate errors to the user dialog."""
        try:
            result = device.set(variable, value, sync=sync)
            logger.info(
                "Set %s:%s to %s. Result: %s",
                device.get_name(),
                variable,
                value,
                result,
            )
        except DEVICE_COMMAND_ERRORS as e:
            logger.error(
                "Set %s:%s to %s failed (%s): %s",
                device.get_name(),
                variable,
                value,
                type(e).__name__,
                e,
            )
            escalate_device_error(e, self.on_user_prompt)

    def _get_device(self, device: ScanDevice, variable: str, expected_value: Any):
        """Retrieve *variable* from *device* and prompt the user if it differs from *expected_value*.

        Raises
        ------
        ActionError
            If the value is unexpected and the user chooses to abort.
        """
        value = device.get(variable)
        if value == expected_value:
            logger.info(
                "Get %s:%s returned expected value: %s",
                device.get_name(),
                variable,
                value,
            )
        else:
            message = f"Get {device.get_name()}:{variable} returned {value}, expected {expected_value}"
            logger.warning(message)
            if self._prompt_user_quit_action(message):
                raise ActionError(message)

    def return_value(self, device_name: str, variable: str):
        """Return the current value of *variable* on *device_name*."""
        if device_name not in self.instantiated_devices:
            self.instantiated_devices[device_name] = ScanDevice(device_name)

        device: ScanDevice = self.instantiated_devices[device_name]
        return device.get(variable)

    @staticmethod
    def _wait(seconds: float):
        logger.info("Waiting for %s seconds.", seconds)
        time.sleep(seconds)

    def _prompt_user_quit_action(self, message: str) -> bool:
        """Show the device-error dialog; auto-aborts in headless / test contexts."""
        if self.on_user_prompt is not None:
            return self.on_user_prompt(ActionError(message))

        # Headless / test fallback: no GUI available, so auto-abort.
        logger.warning(
            "No on_user_prompt callback wired — auto-aborting action on error: %s",
            message,
        )
        return True
