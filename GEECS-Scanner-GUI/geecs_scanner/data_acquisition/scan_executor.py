"""Step-by-step scan execution for ScanManager."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np

from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_scanner.data_acquisition.dialog_request import (
    DEVICE_COMMAND_ERRORS,
    escalate_device_error,
)
from geecs_scanner.data_acquisition.scan_options import ScanOptions
from geecs_scanner.data_acquisition.trigger_controller import TriggerController
from geecs_scanner.optimization.base_optimizer import BaseOptimizer
from geecs_scanner.utils.exceptions import DeviceCommandError
from geecs_scanner.utils.retry import retry

logger = logging.getLogger(__name__)


class ScanStepExecutor:
    """Execute scan steps: move devices, acquire data, drive the optimizer.

    Owned by :class:`~geecs_scanner.data_acquisition.scan_manager.ScanManager`.

    Attributes
    ----------
    device_manager : DeviceManager
    data_logger : DataLogger
    scan_data_manager : ScanDataManager
    optimizer : BaseOptimizer or None
    trigger_controller : TriggerController or None
    options : ScanOptions
    stop_scanning_thread_event : threading.Event
    pause_scan_event : threading.Event
    on_device_error : callable or None
        Injected by ScanManager. ``(exc) -> bool`` — True means user chose Abort.
    """

    def __init__(
        self,
        device_manager,
        data_logger,
        scan_data_manager,
        options: ScanOptions,
        stop_scanning_thread_event,
        pause_scan_event,
        optimizer: Optional[BaseOptimizer] = None,
        trigger_controller: Optional[TriggerController] = None,
    ):
        self.device_manager = device_manager
        self.data_logger = data_logger
        self.scan_data_manager = scan_data_manager
        self.optimizer = optimizer
        self.trigger_controller = trigger_controller
        self.options = options
        self.stop_scanning_thread_event = stop_scanning_thread_event
        self.pause_scan_event = pause_scan_event

        self.scan_step_start_time = 0
        self.scan_step_end_time = 0
        self.pause_time = 0
        self.results = None
        self.scan_steps = []
        self.on_device_error = None

        logger.debug("Constructing the scan step executor")

    def execute_scan_loop(self, scan_steps: List[Dict[str, Any]]) -> None:
        """Iterate through *scan_steps*, executing each until stopped externally."""
        self.scan_steps = scan_steps
        logger.debug("Attempting to start scan loop with steps: %s", scan_steps)
        step_index = 0

        while step_index < len(self.scan_steps):
            if self.stop_scanning_thread_event.is_set():
                logger.info("Scanning has been stopped externally.")
                break

            scan_step = self.scan_steps[step_index]
            logger.debug("Executing scan step: %s", scan_step)
            self.execute_step(scan_step, step_index)
            step_index += 1

        logger.info("Scan loop completed. Stopping logging.")

    def execute_step(self, step: Dict[str, Any], index: int) -> None:
        """Prepare → move devices → acquire → (evaluate + generate next if optimizing)."""
        logger.debug("Preparing scan step: %s", step)
        self.prepare_for_step()

        logger.debug("Moving devices for step: %s", step["variables"])
        self.move_devices_parallel_by_device(step["variables"], step["is_composite"])

        logger.debug("Waiting for acquisition: %s", step)
        self.wait_for_acquisition(step["wait_time"])

        if self.optimizer:
            logger.debug(
                "Evaluating acquired data and potentially generating next step"
            )
            self.evaluate_acquired_data(index)
            if index + 1 < len(self.scan_steps):
                self.generate_next_step(index + 1)

    def prepare_for_step(self) -> None:
        """Increment bin counter, update virtual variable value, turn trigger off."""
        logger.debug("Pausing logging. Turning trigger off before moving devices.")

        if self.data_logger.virtual_variable_name is not None:
            self.data_logger.virtual_variable_value = (
                self.data_logger.virtual_variable_list[self.data_logger.bin_num]
            )
            logger.debug(
                "updating virtual value in data_logger from scan_manager to: %s.",
                self.data_logger.virtual_variable_value,
            )

        self.data_logger.bin_num += 1
        self.trigger_off()

    def move_devices_parallel_by_device(
        self,
        component_vars: Dict[str, Any],
        is_composite: bool,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        """Set device variables in parallel, grouped by device.

        One thread is launched per device; variables for the same device are
        set sequentially within that thread to avoid conflicts.  Hardware
        communication is retried up to *max_retries* times via
        :func:`~geecs_scanner.utils.retry.retry`.  On exhaustion a
        :class:`~geecs_scanner.utils.exceptions.DeviceCommandError` is raised
        (with exception chaining to the root hardware error) and the user is
        prompted to continue or abort.

        Tolerance failures (device accepted the command but the readback
        is out of spec) are logged as WARNING and do not trigger a retry —
        retrying the same command won't fix a mechanical tuning problem.

        Parameters
        ----------
        component_vars : dict
            ``"device_name:variable_name"`` → target value.
        is_composite : bool
            True when the variables belong to a composite device configuration.
        max_retries : int
            Hardware-exception retry limit per variable.  Default 3.
        retry_delay : float
            Seconds between retries.  Default 0.5.
        """
        if self.device_manager.is_statistic_noscan(component_vars):
            return
        if not component_vars:
            logger.debug("No variables to move for this scan step.")
            return

        vars_by_device: dict[str, list] = defaultdict(list)
        for device_var, set_val in component_vars.items():
            device_name, var_name = (device_var.split(":") + ["composite_var"])[:2]
            vars_by_device[device_name].append((var_name, set_val))

        def _get_tolerance(device, device_name: str, var_name: str) -> float:
            if device.is_composite:
                return 10000.0
            return float(
                GeecsDevice.exp_info["devices"][device_name][var_name]["tolerance"]
            )

        def set_device_variables(device_name: str, var_list: list) -> None:
            """Set all variables for one device; raises DeviceCommandError on failure."""
            device = self.device_manager.devices.get(device_name)
            if not device:
                logger.warning("[%s] Device not found in device manager.", device_name)
                return

            logger.debug("[%s] Setting vars: %s", device_name, var_list)
            for var_name, set_val in var_list:
                tol = _get_tolerance(device, device_name, var_name)

                try:
                    ret_val = retry(
                        lambda vn=var_name, sv=set_val: device.set(vn, sv),
                        attempts=max_retries,
                        delay=retry_delay,
                        catch=DEVICE_COMMAND_ERRORS,
                        on_retry=lambda exc, n, vn=var_name: logger.debug(
                            "[%s] hardware error (attempt %d/%d) setting %s: %s",
                            device_name,
                            n,
                            max_retries,
                            vn,
                            exc,
                        ),
                    )
                except DEVICE_COMMAND_ERRORS as exc:
                    raise DeviceCommandError(
                        device_name,
                        f"set {var_name}",
                        variable=var_name,
                    ) from exc

                if ret_val - tol <= set_val <= ret_val + tol:
                    logger.debug(
                        "[%s] %s=%s within tolerance %s",
                        device_name,
                        var_name,
                        ret_val,
                        tol,
                    )
                else:
                    logger.warning(
                        "[%s] %s=%s not within tolerance of %s",
                        device_name,
                        var_name,
                        ret_val,
                        set_val,
                    )

        with ThreadPoolExecutor(max_workers=len(vars_by_device)) as executor:
            futures = {
                executor.submit(
                    set_device_variables, device_name, var_list
                ): device_name
                for device_name, var_list in vars_by_device.items()
            }
            for future, device_name in futures.items():
                try:
                    future.result()
                except DeviceCommandError as exc:
                    var_list = vars_by_device[device_name]
                    lines = [
                        f"  • {n} → {v}"
                        + (" ← failed here" if n == exc.variable else "")
                        for n, v in var_list
                    ]
                    context = f"All variables queued for {device_name}:\n" + "\n".join(
                        lines
                    )
                    if exc.__cause__ is not None and type(exc.__cause__) is not type(
                        exc
                    ):
                        context += f"\n\nRoot cause: {type(exc.__cause__).__name__}"
                    if escalate_device_error(exc, self.on_device_error, context):
                        self.stop_scanning_thread_event.set()
                    return

    def wait_for_acquisition(self, wait_time: float) -> None:
        """Enable trigger, wait *wait_time* seconds, disable trigger.

        Handles pause/resume events and optional per-shot TDMS saves
        (``options.on_shot_tdms``).
        """
        self.trigger_on()

        self.scan_step_start_time = time.time()
        self.data_logger.data_recording = True

        if self.scan_step_end_time > 0:
            self.data_logger.idle_time = (
                self.scan_step_start_time - self.scan_step_end_time + self.pause_time
            )
            logger.debug("idle time between scan steps: %s", self.data_logger.idle_time)

        current_time = 0
        start_time = time.time()
        interval_time = 0.1
        self.pause_time = 0

        while current_time < wait_time:
            if self.stop_scanning_thread_event.is_set():
                logger.debug("Scanning has been stopped externally.")
                break

            if not self.pause_scan_event.is_set():
                self.trigger_off()
                self.data_logger.data_recording = False
                t0 = time.time()
                logger.debug("Scan is paused, waiting to resume...")
                self.pause_scan_event.wait()
                self.pause_time = time.time() - t0
                self.trigger_on()
                self.data_logger.data_recording = True

            time.sleep(interval_time)
            current_time = time.time() - start_time

            if self.options.on_shot_tdms:
                if current_time % 1 < interval_time:
                    log_df = self.scan_data_manager.convert_to_dataframe(self.results)
                    self.scan_data_manager.dataframe_to_tdms(log_df)

        self.trigger_off()

        self.scan_step_end_time = time.time()
        self.data_logger.data_recording = False

        # trigger_off is asynchronous; brief sleep lets the hardware settle.
        time.sleep(0.5)

    def evaluate_acquired_data(self, index: int = None):
        """Feed current-bin data to the optimizer.

        On the first step (``index == 0``) the logged readback values replace
        the nominal scan-step variables before evaluation, so the optimizer
        learns from actual positions rather than commanded ones.
        """
        if index == 0:
            variables = self.get_control_values_from_log()
            self.scan_steps[index].update({"variables": variables})
            logger.debug("Updated variables for first scan step: %s", variables)

        try:
            self.optimizer.evaluate(inputs=self.scan_steps[index]["variables"])
            logger.debug("Successfully evaluated data for scan step %s", index)
        except Exception:
            logger.exception("Error evaluating data for scan step %s", index)

    def get_control_values_from_log(self) -> Dict[str, float]:
        """Return per-variable mean readback for the current bin.

        Returns
        -------
        dict
            Variable name → mean value. NaN for variables with no logged data.
        """
        current_bin = self.data_logger.bin_num
        variables = {var: [] for var in self.optimizer.vocs.variable_names}

        for entry in self.data_logger.log_entries.values():
            if entry.get("Bin #") == current_bin:
                for var in self.optimizer.vocs.variable_names:
                    if var in entry:
                        variables[var].append(entry[var])

        averaged_values: Dict[str, float] = {}
        for var, values in variables.items():
            if values:
                averaged_values[var] = float(np.mean(values))
                logger.debug("Computed average for %s: %s", var, averaged_values[var])
            else:
                logger.warning(
                    "No readback data found for variable '%s' in bin %s",
                    var,
                    current_bin,
                )
                averaged_values[var] = float("nan")

        return averaged_values

    def generate_next_step(self, next_index: int) -> None:
        """Ask the optimizer to propose the next scan-step variables.

        Uses random inputs for the first two steps (initialization), then
        the optimizer's generation strategy thereafter.
        """
        num_initialization_steps = 2

        try:
            if next_index <= num_initialization_steps:
                logger.debug("Running optimizer initialization")
                next_variables = self.optimizer.vocs.random_inputs(1)[0]
                logger.debug(
                    "Initializer generated random variables: %s", next_variables
                )
            else:
                logger.debug("Running advanced optimizer")
                next_variables = self.optimizer.generate(1)[0]
                logger.debug(
                    "Optimizer generated optimized variables: %s", next_variables
                )

            logger.debug(
                "Next experimental step generated using optimizer: %s", next_variables
            )
        except Exception:
            logger.exception("Failed to update next step via optimizer")
            return

        self.scan_steps[next_index].update({"variables": next_variables})
        logger.debug(
            "Updated scan step %s with new configuration: %s",
            next_index,
            self.scan_steps[next_index],
        )

    def trigger_on(self) -> None:
        """Delegate to trigger_controller if available."""
        if self.trigger_controller is not None:
            self.trigger_controller.trigger_on()

    def trigger_off(self) -> None:
        """Delegate to trigger_controller if available."""
        if self.trigger_controller is not None:
            self.trigger_controller.trigger_off()
