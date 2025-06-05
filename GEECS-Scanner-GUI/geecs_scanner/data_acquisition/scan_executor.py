from __future__ import annotations

# Standard library imports
from typing import Optional, List, Dict, Any

import logging
import time

import pandas as pd
import numpy as np

from geecs_python_api.controls.devices.geecs_device import GeecsDevice

class ScanStepExecutor:
    def __init__(self, device_manager, data_logger, scan_data_manager, optimizer, shot_control, options_dict,
                 stop_scanning_thread_event, pause_scan_event):
        self.device_manager = device_manager
        self.data_logger = data_logger
        self.scan_data_manager = scan_data_manager
        self.optimizer = optimizer
        self.shot_control = shot_control
        self.options_dict = options_dict
        self.stop_scanning_thread_event = stop_scanning_thread_event
        self.pause_scan_event = pause_scan_event

        self.scan_step_start_time = 0
        self.scan_step_end_time = 0
        self.pause_time = 0
        self.results = None
        self.scan_steps = []

        logging.info(f'constructing the scan step executor')

    def execute_scan_loop(self, scan_steps: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Executes a sequence of scan steps in order. Each step moves devices,
        waits for acquisition, and finalizes logging. The loop can be interrupted
        externally via the stop scanning event.

        Args:
            scan_steps (List[Dict[str, Any]]): A list of dictionaries, each containing
                variables to scan, wait time, and a flag for composite scanning.

        Returns:
            pd.DataFrame: A DataFrame representing the collected scan data.
        """

        # make the scan steps an instance variable so that they can be updated later as needed
        self.scan_steps = scan_steps
        logging.info(f'attempting to start loop with : {scan_steps}')

        log_df = pd.DataFrame()
        step_index = 0

        while step_index < len(self.scan_steps):
            if self.stop_scanning_thread_event.is_set():
                logging.info("Scanning has been stopped externally.")
                break

            scan_step = self.scan_steps[step_index]
            logging.info(f'attempting step: {scan_step}')
            self.execute_step(scan_step, step_index)
            step_index += 1

        logging.info("Stopping logging.")
        return log_df

    def execute_step(self, step: Dict[str, Any], index: int) -> None:
        """
        Execute a single scan step by preparing the system, moving the devices,
        waiting for acquisition, and finalizing the step. Allows for updating the
        next step based on acquired data (useful for optimization).

        Args:
            step (Dict[str, Any]): A dictionary containing the scan step configuration.
            index (int): Index of the current step in the scan_steps list.
        """
        logging.info(f'preparing step: {step}')
        self.prepare_for_step()
        self.move_devices(step['variables'], step['is_composite'])
        logging.info(f'waiting for acquisition: {step}')
        self.wait_for_acquisition(step['wait_time'])
        if self.optimizer:
            self.evaluate_acquired_data(index)
            if index + 1 < len(self.scan_steps):
                self.generate_next_step(index+1)


    def prepare_for_step(self) -> None:
        """
        Prepare for a scan step by updating the virtual variable value (if set),
        incrementing the bin number in the data logger, and turning off the trigger.

        """
        logging.info("Pausing logging. Turning trigger off before moving devices.")

        # update variables in data_logger that are not updated via hardware
        if self.data_logger.virtual_variable_name is not None:
            self.data_logger.virtual_variable_value = self.data_logger.virtual_variable_list[self.data_logger.bin_num]
            logging.info(f"updating virtual value in data_logger from scan_manager to: {self.data_logger.virtual_variable_value}.")

        self.data_logger.bin_num += 1

        self.trigger_off()

    def move_devices(self, component_vars: Dict[str, Any], is_composite: bool, max_retries: int = 3, retry_delay: float = 0.5) -> None:
        """
        Set device values for a scan step. Applies retry logic if device value is not within
        specified tolerance. Turns trigger on after devices are moved.

        Args:
            component_vars (Dict[str, Any]): Dictionary of device variables and their target values.
            is_composite (bool): Flag indicating whether variables are composite.
            max_retries (int): Maximum number of retries allowed for each variable setting.
            retry_delay (float): Time in seconds to wait between retries.
        """

        if not self.device_manager.is_statistic_noscan(component_vars):
            for device_var, set_val in component_vars.items():
                device_name, var_name = (device_var.split(':') + ['composite_var'])[:2]
                device = self.device_manager.devices.get(device_name)

                if device:
                    tol = 10000 if device.is_composite else float(GeecsDevice.exp_info['devices'][device_name][var_name]['tolerance'])
                    success = False

                    for attempt in range(max_retries):
                        ret_val = device.set(var_name, set_val)
                        logging.info(f"Attempt {attempt + 1}: Setting {var_name} to {set_val} on {device_name}, returned {ret_val}")
                        if ret_val - tol <= set_val <= ret_val + tol:
                            logging.info(f"Success: {var_name} set to {ret_val} (within tolerance {tol}) on {device_name}")
                            success = True
                            break
                        else:
                            logging.warning(f"Attempt {attempt + 1}: {var_name} on {device_name} not within tolerance ({ret_val} != {set_val})")
                            time.sleep(retry_delay)

                    if not success:
                        logging.error(f"Failed to set {var_name} on {device_name} after {max_retries} attempts")
                else:
                    logging.warning(f"Device {device_name} not found in device manager.")

    def wait_for_acquisition(self, wait_time: float) -> None:
        """
        Wait for the duration of acquisition time during a scan step.
        Handles external stop and pause events, triggers data logging, and optionally
        saves data to TDMS periodically or after a user-defined hiatus.

        Args:
            wait_time (float): Total time to wait during acquisition in seconds.
        """

        self.trigger_on()

        self.scan_step_start_time = time.time()
        self.data_logger.data_recording = True

        if self.scan_step_end_time > 0:
            self.data_logger.idle_time = self.scan_step_start_time - self.scan_step_end_time + self.pause_time
            logging.info(f'idle time between scan steps: {self.data_logger.idle_time}')

        current_time = 0
        start_time = time.time()
        interval_time = 0.1
        self.pause_time = 0

        while current_time < wait_time:
            if self.stop_scanning_thread_event.is_set():
                logging.info("Scanning has been stopped externally.")
                break

            if not self.pause_scan_event.is_set():
                self.trigger_off()
                self.data_logger.data_recording = False
                t0 = time.time()
                logging.info("Scan is paused, waiting to resume...")
                self.pause_scan_event.wait()
                self.pause_time = time.time() - t0
                self.trigger_on()
                self.data_logger.data_recording = True

            time.sleep(interval_time)
            current_time = time.time() - start_time

            if self.options_dict.get('On-Shot TDMS', False):
                if current_time % 1 < interval_time:
                    log_df = self.scan_data_manager.convert_to_dataframe(self.results)
                    self.scan_data_manager.dataframe_to_tdms(log_df)

            try:
                hiatus = float(self.options_dict.get("Save Hiatus Period (s)", ""))
            except ValueError:
                hiatus = ""

            if hiatus and self.data_logger.shot_save_event.is_set():
                self.save_hiatus(hiatus)
                self.data_logger.shot_save_event.clear()

        self.trigger_off()

        self.scan_step_end_time = time.time()
        self.data_logger.data_recording = False

        # adding a sleep, because the trigger_off seems to be asynchronous
        time.sleep(.5)

    def evaluate_acquired_data(self, index: int = None):
        if index == 0:
            variables = self.get_control_values_from_log()
            self.scan_steps[index].update({'variables': variables})

        self.optimizer.evaluate(inputs=self.scan_steps[index]['variables'])

    def get_control_values_from_log(self) -> Dict[str, float]:
        """
        Extracts the average readback values for each variable in the VOCS
        from log_entries for the current bin number.

        Returns:
            Dict[str, float]: Averaged variable values keyed by variable name.
        """
        current_bin = self.data_logger.bin_num
        variables = {var: [] for var in self.optimizer.vocs.variable_names}

        # Filter entries for the current bin
        for entry in self.data_logger.log_entries.values():
            if entry.get("Bin #") == current_bin:
                for var in self.optimizer.vocs.variable_names:
                    if var in entry:
                        variables[var].append(entry[var])

        # Compute averages
        averaged_values = {}
        for var, values in variables.items():
            if values:
                averaged_values[var] = float(np.mean(values))
            else:
                logging.warning(f"No readback data found for variable '{var}' in bin {current_bin}")
                averaged_values[var] = float("nan")

        return averaged_values

    def generate_next_step(self, next_index: int) -> None:
        """
        Update the next scan step using an optimizer if available.
        This executes an optimization process where the next point is determined
        dynamically based on previously logged data.

        Args:
            next_index (int): Index of the scan step to update.
        """

        num_initialization_steps = 1
        try:
            if next_index <= num_initialization_steps:
                # Use generator during initialization as well
                logging.info(f'running optimizer intialzation')
                next_variables = self.optimizer.vocs.random_inputs(1)[0]
                logging.info(f'intializer gave: {next_variables}')
            else:
                logging.info(f'running optimizer for real')
                next_variables = self.optimizer.generate(1)[0]
                logging.info(f'optimizer gave: {next_variables}')

            logging.info(f"Next step generated using optimizer: {next_variables}")
        except Exception as e:
            logging.warning(f"Failed to update next step via optimizer: {e}")
            return

        # Overwrite only the 'variables' key
        self.scan_steps[next_index].update({'variables': next_variables})
        logging.info(f"Next step after update: {self.scan_steps[next_index]}")

    def trigger_on(self) -> None:
        if hasattr(self, 'trigger_on_fn'):
            self.trigger_on_fn()

    def trigger_off(self) -> None:
        if hasattr(self, 'trigger_off_fn'):
            self.trigger_off_fn()

    def save_hiatus(self, duration: float) -> None:
        logging.info(f"Hiatus: Waiting for {duration} seconds before saving data.")
        time.sleep(duration)
