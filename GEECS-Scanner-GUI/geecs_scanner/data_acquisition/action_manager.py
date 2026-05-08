"""Load and execute named device-action sequences defined in ``actions.yaml``."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import yaml

from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_scanner.data_acquisition.dialog_request import (
    DEVICE_COMMAND_ERRORS,
    escalate_device_error,
)
from geecs_scanner.data_acquisition.schemas.actions import (
    ActionLibrary,
    ActionSequence,
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
    actions : dict[str, ActionSequence]
    actions_file_path : Path
    on_user_prompt : callable or None
        Injected by ScanManager after construction.
        Signature: ``(exc: Exception) -> bool`` — True means user chose Abort.
        Falls back to auto-abort if None (headless / test contexts).
    """

    def __init__(self, experiment_dir: str):
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
            logger.debug("Added action sequence: %s", action_name)

        except Exception:
            logger.exception("Failed to add action %s", action_name)
            raise ValueError(f"Failed to add action {action_name}")

    def execute_action(self, action_name: str):
        """Run all steps of *action_name* sequentially.

        Devices are opened fresh at the start of the action and closed in a
        ``finally`` block regardless of success or failure.

        Raises
        ------
        ActionError
            If the action is not defined, a step type is unrecognized, or a
            device interaction fails.
        """
        if action_name not in self.actions:
            raise ActionError(f"Action '{action_name}' is not defined.")

        logger.debug("Starting execution of action sequence: %s", action_name)

        action = self.actions[action_name]
        steps = action.steps

        device_names = {
            step.device for step in steps if isinstance(step, (SetStep, GetStep))
        }
        open_devices: Dict[str, ScanDevice] = {}
        try:
            for name in device_names:
                open_devices[name] = ScanDevice(name)
                if not open_devices[name].state.get(
                    "Device alive on instantiation", False
                ):
                    raise ActionError(
                        f"Device {name} failed connectivity check before action {action_name}"
                    )
                logger.debug("Device %s ready for action %s", name, action_name)

            for step_index, step in enumerate(steps, 1):
                logger.debug(
                    "Executing step %s/%s in action %s",
                    step_index,
                    len(steps),
                    action_name,
                )
                try:
                    match step:
                        case WaitStep():
                            logger.debug("Waiting for %s seconds", step.wait)
                            self._wait(step.wait)
                        case ExecuteStep():
                            logger.debug(
                                "Executing nested action: %s", step.action_name
                            )
                            self.execute_action(step.action_name)
                        case SetStep():
                            self._set_device(
                                open_devices[step.device],
                                step.variable,
                                step.value,
                                sync=step.wait_for_execution,
                            )
                        case GetStep():
                            self._get_device(
                                open_devices[step.device],
                                step.variable,
                                step.expected_value,
                            )
                        case _:
                            raise ActionError(
                                f"Unrecognized action step type: {type(step)}"
                            )
                except Exception as exc:
                    logger.exception(
                        "Error executing step %s in action %s", step_index, action_name
                    )
                    raise ActionError(
                        f"Step execution failed in action {action_name}"
                    ) from exc

        finally:
            for name, device in open_devices.items():
                try:
                    device.close()
                except Exception:
                    logger.warning("Failed to close device %s after action", name)

        logger.debug("Successfully completed action sequence: %s", action_name)

    def clear_action(self, action_name: str):
        """Remove *action_name* from the in-memory library."""
        if action_name not in self.actions:
            logger.error(
                "Action '%s' is not defined in the available actions.", action_name
            )
            return

        del self.actions[action_name]
        logger.debug("Removed action sequence: %s", action_name)

    def _set_device(
        self, device: ScanDevice, variable: str, value: Any, sync: bool = True
    ):
        """Call ``device.set(variable, value)`` and escalate errors to the user dialog."""
        try:
            result = device.set(variable, value, sync=sync)
            logger.debug(
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
            logger.debug(
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
        device = ScanDevice(device_name)
        try:
            return device.get(variable)
        finally:
            try:
                device.close()
            except Exception:
                logger.warning(
                    "Failed to close device %s after return_value", device_name
                )

    @staticmethod
    def _wait(seconds: float):
        logger.debug("Waiting for %s seconds.", seconds)
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
