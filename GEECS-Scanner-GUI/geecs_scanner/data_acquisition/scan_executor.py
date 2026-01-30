"""
Advanced Scan Execution Framework for Experimental Control Systems.

This module provides a sophisticated, flexible system for managing complex
experimental workflows, focusing on precise device control, data acquisition,
and adaptive optimization strategies.

Key Features:
- Dynamic scan step generation and execution
- Advanced device interaction and configuration
- Comprehensive data logging and management
- Adaptive optimization strategies
- Robust error handling and event management

The ScanStepExecutor serves as the core orchestration component for experimental
control, enabling:
- Precise device movement and configuration
- Flexible data acquisition timing
- Dynamic workflow adaptation
- Comprehensive experimental logging

Design Principles:
- Modular and extensible architecture
- Support for diverse experimental configurations
- Advanced optimization integration
- Robust error handling and event management

Workflow Components:
- Device management
- Data logging
- Trigger control
- Optimization strategies
- Pause and stop event handling

Dependencies:
- geecs_python_api.controls.devices.geecs_device
- numpy
- logging

See Also
--------
DeviceManager : Manages device configurations and interactions
DataLogger : Handles experimental data logging
ScanDataManager : Manages scan-related data processing
Optimizer : Provides optimization strategies for scan steps

Notes
-----
This module is part of the GEECS (Generalized Experimental Execution Control System)
framework, designed to provide flexible and powerful experimental control capabilities.
"""

from __future__ import annotations

# Standard library imports
from typing import List, Dict, Any, Optional

import logging
import time

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_scanner.optimization.base_optimizer import BaseOptimizer

# -----------------------------------------------------------------------------
# Module-level logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class ScanStepExecutor:
    """
    A sophisticated executor for managing complex scan step sequences.

    This class provides a comprehensive mechanism for executing scan steps, handling
    device interactions, data logging, optimization, and external control events.
    It supports dynamic scan step generation, device movement, acquisition timing,
    and advanced optimization strategies.

    Attributes
    ----------
    device_manager : DeviceManager
        Manages device interactions and configurations.
    data_logger : DataLogger
        Handles logging of experimental data during scan steps.
    scan_data_manager : ScanDataManager
        Manages scan-related data processing and storage.
    optimizer : Optimizer, optional
        Provides optimization strategies for dynamic scan step generation.
    shot_control : ShotControl, optional
        Manages shot-related control mechanisms.
    options_dict : dict
        Configuration options for scan execution.
    stop_scanning_thread_event : threading.Event
        Event to signal external interruption of scanning process.
    pause_scan_event : threading.Event
        Event to manage scan pausing and resuming.

    Scan Execution Attributes
    -------------------------
    scan_step_start_time : float
        Timestamp of the current scan step's start.
    scan_step_end_time : float
        Timestamp of the previous scan step's end.
    pause_time : float
        Cumulative time spent in paused state.
    results : Any
        Stores results from scan step execution.
    scan_steps : List[Dict[str, Any]]
        List of scan steps to be executed.

    Notes
    -----
    - Supports dynamic scan step generation via optimizer
    - Handles device movement with retry mechanisms
    - Provides precise timing control for data acquisition
    - Supports external interruption and pausing of scan sequences
    - Integrates with various experimental control components

    See Also
    --------
    DeviceManager : Manages device configurations and interactions
    DataLogger : Handles experimental data logging
    ScanDataManager : Manages scan-related data processing
    Optimizer : Provides optimization strategies for scan steps
    """

    def __init__(
        self,
        device_manager,
        data_logger,
        scan_data_manager,
        shot_control,
        options_dict,
        stop_scanning_thread_event,
        pause_scan_event,
        optimizer: Optional[BaseOptimizer] = None,
    ):
        """
        Initialize the ScanStepExecutor with experimental control components.

        Parameters
        ----------
        device_manager : DeviceManager
            Manages interactions with experimental devices.
        data_logger : DataLogger
            Handles logging of experimental data.
        scan_data_manager : ScanDataManager
            Manages processing and storage of scan-related data.
        optimizer : Optimizer, optional
            Provides strategies for dynamic scan step generation.
        shot_control : ShotControl, optional
            Manages shot-related control mechanisms.
        options_dict : dict
            Configuration options for scan execution.
        stop_scanning_thread_event : threading.Event
            Event to signal external interruption of scanning.
        pause_scan_event : threading.Event
            Event to manage scan pausing and resuming.

        Notes
        -----
        Initialization Process:
        - Sets up references to experimental control components
        - Prepares timing and state tracking attributes
        - Initializes logging for executor creation

        Logging
        -------
        - Logs the creation of the scan step executor
        """
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

        logger.info("Constructing the scan step executor")

    def execute_scan_loop(self, scan_steps: List[Dict[str, Any]]) -> None:
        """
        Execute a comprehensive sequence of scan steps with advanced control.

        This method manages the entire scan sequence, handling device interactions,
        data acquisition, optimization, and external control events. It provides
        a robust mechanism for executing complex, multi-step experimental scans.

        Parameters
        ----------
        scan_steps : List[Dict[str, Any]]
            A sequence of scan steps, where each step is a dictionary containing:
            - 'variables': Device variables to set
            - 'wait_time': Duration to wait during acquisition
            - 'is_composite': Flag indicating composite scanning mode

        Notes
        -----
        Scan Loop Execution Process:
        1. Store scan steps as an instance variable for dynamic updates
        2. Iterate through scan steps sequentially
        3. Handle external interruption via stop scanning event
        4. Execute individual scan steps
        5. Support dynamic step generation via optimizer
        6. Provide comprehensive logging of scan progression

        External Control Mechanisms:
        - Supports interruption of scan sequence
        - Allows dynamic modification of scan steps
        - Integrates with optimization strategies

        Logging Strategy:
        - Log scan loop initialization
        - Log each step's execution
        - Provide clear termination logging

        Raises
        ------
        No explicit exceptions, but logs potential interruption events

        See Also
        --------
        execute_step : Method for executing individual scan steps
        generate_next_step : Dynamic scan step generation
        """
        # Store scan steps as an instance variable for potential dynamic updates
        self.scan_steps = scan_steps
        logger.info("Attempting to start scan loop with steps: %s", scan_steps)
        step_index = 0

        while step_index < len(self.scan_steps):
            # Check for external interruption
            if self.stop_scanning_thread_event.is_set():
                logger.info("Scanning has been stopped externally.")
                break

            # Execute current scan step
            scan_step = self.scan_steps[step_index]
            logger.info("Executing scan step: %s", scan_step)
            self.execute_step(scan_step, step_index)
            step_index += 1

        logger.info("Scan loop completed. Stopping logging.")

    def execute_step(self, step: Dict[str, Any], index: int) -> None:
        """
        Execute a scan step with advanced device interaction and data acquisition.

        This method manages the entire process of executing a single scan step,
        including system preparation, device movement, data acquisition, and
        potential optimization based on acquired data.

        Parameters
        ----------
        step : Dict[str, Any]
            A configuration dictionary for the current scan step, containing:
            - 'variables': Device variables to set
            - 'is_composite': Flag indicating composite scanning mode
            - 'wait_time': Duration to wait during acquisition

        index : int
            The index of the current step in the scan_steps list.

        Notes
        -----
        Scan Step Execution Process:
        1. Prepare system for the upcoming step
        2. Move devices to specified configuration
        3. Wait for data acquisition
        4. Evaluate acquired data (optional)
        5. Generate next step via optimizer (if available)

        Optimization Integration:
        - Supports dynamic step generation
        - Allows data-driven optimization strategies
        - Provides flexibility in experimental workflows

        Logging Strategy:
        - Log step preparation
        - Log device movement
        - Log acquisition waiting
        - Provide detailed information about step execution

        See Also
        --------
        prepare_for_step : Prepares system before step execution
        move_devices : Moves devices to specified configuration
        wait_for_acquisition : Manages data acquisition timing
        evaluate_acquired_data : Processes data from current step
        generate_next_step : Generates next step via optimizer
        """
        logger.info("Preparing scan step: %s", step)
        self.prepare_for_step()

        logger.info("Moving devices for step: %s", step["variables"])
        # self.move_devices(step["variables"], step["is_composite"])
        self.move_devices_parallel_by_device(step["variables"], step["is_composite"])

        logger.info("Waiting for acquisition: %s", step)
        self.wait_for_acquisition(step["wait_time"])

        if self.optimizer:
            logger.debug(
                "Evaluating acquired data and potentially generating next step"
            )
            self.evaluate_acquired_data(index)
            if index + 1 < len(self.scan_steps):
                self.generate_next_step(index + 1)

    def prepare_for_step(self) -> None:
        """
        Prepare the system for the next scan step by updating virtual variables and managing logging state.

        This method performs several critical pre-scan step operations:
        1. Updates virtual variable value in the data logger if a virtual variable is set
        2. Increments the bin number in the data logger
        3. Turns off the trigger to prepare for device movement

        Notes
        -----
        - Virtual variables are used for tracking scan parameters not directly updated via hardware
        - The bin number is incremented to track the current scan step
        - Turning off the trigger prevents unwanted data collection during device movement

        See Also
        --------
        data_logger : Manages data recording and virtual variable tracking
        trigger_off : Stops data collection trigger
        """
        logger.info("Pausing logging. Turning trigger off before moving devices.")

        # update variables in data_logger that are not updated via hardware
        if self.data_logger.virtual_variable_name is not None:
            self.data_logger.virtual_variable_value = (
                self.data_logger.virtual_variable_list[self.data_logger.bin_num]
            )
            logger.info(
                "updating virtual value in data_logger from scan_manager to: %s.",
                self.data_logger.virtual_variable_value,
            )

        self.data_logger.bin_num += 1

        self.trigger_off()

    def move_devices(
        self,
        component_vars: Dict[str, Any],
        is_composite: bool,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        """
        Set device variables for a scan step with robust error handling and retry mechanism.

        This method attempts to set device variables to specified target values,
        implementing a retry strategy to handle potential setting failures. It supports
        both standard and composite device variables.

        Parameters
        ----------
        component_vars : Dict[str, Any]
            A dictionary mapping device variables to their target values.
            Keys are formatted as 'device_name:variable_name'.
        is_composite : bool
            Flag indicating whether the variables are part of a composite device configuration.
        max_retries : int, optional
            Maximum number of attempts to set a device variable before giving up.
            Defaults to 3.
        retry_delay : float, optional
            Time in seconds to wait between retry attempts.
            Defaults to 0.5 seconds.

        Notes
        -----
        - Skips device setting for statistical no-scan configurations
        - Retrieves device tolerance from experiment info
        - Logs detailed information about setting attempts
        - Handles devices not found in device manager

        Raises
        ------
        Logs warnings and errors for failed device settings, but does not raise exceptions

        See Also
        --------
        device_manager : Manages device configurations and interactions
        GeecsDevice : Base class for device interactions
        """
        if not self.device_manager.is_statistic_noscan(component_vars):
            for device_var, set_val in component_vars.items():
                device_name, var_name = (device_var.split(":") + ["composite_var"])[:2]
                device = self.device_manager.devices.get(device_name)

                if device:
                    tol = (
                        10000
                        if device.is_composite
                        else float(
                            GeecsDevice.exp_info["devices"][device_name][var_name][
                                "tolerance"
                            ]
                        )
                    )
                    success = False

                    for attempt in range(max_retries):
                        ret_val = device.set(var_name, set_val)
                        logger.info(
                            "Attempt %d: Setting %s to %s on %s, returned %s",
                            attempt + 1,
                            var_name,
                            set_val,
                            device_name,
                            ret_val,
                        )
                        if ret_val - tol <= set_val <= ret_val + tol:
                            logger.info(
                                "Success: %s set to %s (within tolerance %s) on %s",
                                var_name,
                                ret_val,
                                tol,
                                device_name,
                            )
                            success = True
                            break
                        else:
                            logger.warning(
                                "Attempt %d: %s on %s not within tolerance (%s != %s)",
                                attempt + 1,
                                var_name,
                                device_name,
                                ret_val,
                                set_val,
                            )
                            time.sleep(retry_delay)

                    if not success:
                        logger.error(
                            "Failed to set %s on %s after %d attempts",
                            var_name,
                            device_name,
                            max_retries,
                        )
                else:
                    logger.warning(
                        "Device %s not found in device manager.", device_name
                    )

    def move_devices_parallel_by_device(
        self,
        component_vars: Dict[str, Any],
        is_composite: bool,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        """
        Set device variables in parallel, grouped by device, with optional retry and tolerance checks.

        This method initiates device variable settings in parallel by assigning one thread per device.
        Variables belonging to the same device are set sequentially in that thread, preserving device-level
        ordering and avoiding conflicts. Threads are launched and the method returns immediately without
        waiting for completion.

        Parameters
        ----------
        component_vars : dict of str to Any
            Dictionary mapping device variables to target values.
            Keys are formatted as "device_name:variable_name".
        is_composite : bool
            Flag indicating whether the variables are part of a composite device configuration.
        max_retries : int, optional
            Maximum number of attempts to set each device variable. Defaults to 3.
        retry_delay : float, optional
            Time (in seconds) to wait between retry attempts. Defaults to 0.5 seconds.

        Notes
        -----
        - This method returns immediately after launching threads; device settings may still be in progress.
        - Device setting for composite variables is performed without tolerance checking.
        - For standard variables, the device-specific tolerance is used to verify success.
        - Device setting is skipped if `component_vars` corresponds to a statistical no-scan configuration.
        - Logs are generated for each attempt, including success, warnings, and failures.
        """
        if self.device_manager.is_statistic_noscan(component_vars):
            return

        if not component_vars:
            logger.info("No variables to move for this scan step.")
            return

        # Step 1: Group variables by device
        vars_by_device = defaultdict(list)
        for device_var, set_val in component_vars.items():
            device_name, var_name = (device_var.split(":") + ["composite_var"])[:2]
            vars_by_device[device_name].append((var_name, set_val))

        # Step 2: Define per-device setting function
        def set_device_variables(device_name, var_list):
            """Helper fucntion to set vars in threads."""
            from geecs_python_api.controls.interface.geecs_errors import (
                GeecsDeviceCommandRejected,
                GeecsDeviceCommandFailed,
            )

            device = self.device_manager.devices.get(device_name)
            if not device:
                logger.warning("Device %s not found.", device_name)
                return

            logger.info("[%s] Preparing to set vars: %s", device_name, var_list)
            for var_name, set_val in var_list:
                tol = (
                    10000
                    if device.is_composite
                    else float(
                        GeecsDevice.exp_info["devices"][device_name][var_name][
                            "tolerance"
                        ]
                    )
                )
                success = False
                for attempt in range(max_retries):
                    try:
                        ret_val = device.set(var_name, set_val)
                        logger.info(
                            "[%s] Attempt %d: Set %s=%s, got %s",
                            device_name,
                            attempt + 1,
                            var_name,
                            set_val,
                            ret_val,
                        )
                        if ret_val - tol <= set_val <= ret_val + tol:
                            logger.info(
                                "[%s] Success: %s=%s within tolerance %s",
                                device_name,
                                var_name,
                                ret_val,
                                tol,
                            )
                            success = True
                            break
                        else:
                            logger.warning(
                                "[%s] %s=%s not within tolerance of %s",
                                device_name,
                                var_name,
                                ret_val,
                                set_val,
                            )
                            time.sleep(retry_delay)

                    except GeecsDeviceCommandRejected as e:
                        logger.error(
                            "[%s] COMMAND REJECTED: %s (attempt %d/%d)",
                            device_name,
                            e,
                            attempt + 1,
                            max_retries,
                        )
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            logger.error(
                                "[%s] Command rejected after all retry attempts. "
                                "Initiating graceful scan termination.",
                                device_name,
                            )
                            self.stop_scanning_thread_event.set()
                            return

                    except GeecsDeviceCommandFailed as e:
                        logger.error(
                            "[%s] COMMAND FAILED: %s (actual value: %s) (attempt %d/%d)",
                            device_name,
                            e.error_detail,
                            e.actual_value,
                            attempt + 1,
                            max_retries,
                        )
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            logger.error(
                                "[%s] Hardware error persists after all retry attempts. "
                                "Initiating graceful scan termination.",
                                device_name,
                            )
                            self.stop_scanning_thread_event.set()
                            return

                if not success:
                    logger.error(
                        "[%s] Failed to set %s after %d attempts.",
                        device_name,
                        var_name,
                        max_retries,
                    )

        # Step 3: Run each device in parallel
        with ThreadPoolExecutor(max_workers=len(vars_by_device)) as executor:
            futures = [
                executor.submit(set_device_variables, device_name, var_list)
                for device_name, var_list in vars_by_device.items()
            ]
            for f in futures:
                f.result()  # propagate exceptions, if any

    def wait_for_acquisition(self, wait_time: float) -> None:
        """
        Manage the acquisition phase of a scan step with comprehensive event handling.

        Handles the entire acquisition process, including:
        - Triggering data recording
        - Managing pause and stop events
        - Optional periodic TDMS data saving
        - Handling user-defined save hiatus periods

        Parameters
        ----------
        wait_time : float
            Total duration of the acquisition phase in seconds.

        Notes
        -----
        - Tracks idle time between scan steps
        - Supports external scanning stop and pause events
        - Optionally saves data to TDMS on each shot or during a hiatus period
        - Manages trigger state and data recording status

        Raises
        ------
        Logs various events and state changes, but does not raise exceptions

        See Also
        --------
        trigger_on : Starts data collection trigger
        trigger_off : Stops data collection trigger
        scan_data_manager : Manages data conversion and saving
        data_logger : Manages data recording state
        """
        self.trigger_on()

        self.scan_step_start_time = time.time()
        self.data_logger.data_recording = True

        if self.scan_step_end_time > 0:
            self.data_logger.idle_time = (
                self.scan_step_start_time - self.scan_step_end_time + self.pause_time
            )
            logger.info("idle time between scan steps: %s", self.data_logger.idle_time)

        current_time = 0
        start_time = time.time()
        interval_time = 0.1
        self.pause_time = 0

        while current_time < wait_time:
            if self.stop_scanning_thread_event.is_set():
                logger.info("Scanning has been stopped externally.")
                break

            if not self.pause_scan_event.is_set():
                self.trigger_off()
                self.data_logger.data_recording = False
                t0 = time.time()
                logger.info("Scan is paused, waiting to resume...")
                self.pause_scan_event.wait()
                self.pause_time = time.time() - t0
                self.trigger_on()
                self.data_logger.data_recording = True

            time.sleep(interval_time)
            current_time = time.time() - start_time

            if self.options_dict.get("On-Shot TDMS", False):
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
        time.sleep(0.5)

    def evaluate_acquired_data(self, index: int = None):
        """
        Evaluate and process data acquired during a scan step, with special handling for initialization.

        This method manages the data evaluation process for each scan step,
        supporting both initial data collection and subsequent optimization
        through the integrated optimizer.

        Parameters
        ----------
        index : int, optional
            The index of the current scan step. Defaults to None.
            Provides special handling for the first scan step (index 0).

        Notes
        -----
        Evaluation Process:
        1. For the first scan step (index 0):
           - Extract control values from logged data
           - Update scan steps with extracted variables
        2. Evaluate acquired data using the optimizer
           - Provides feedback for optimization strategies
           - Supports data-driven experimental refinement

        Optimization Integration:
        - Uses optimizer to process and learn from acquired data
        - Enables dynamic adjustment of experimental parameters
        - Supports iterative improvement of scan strategies

        Logging Strategy:
        - Provides insights into data evaluation process
        - Supports debugging and experimental analysis

        See Also
        --------
        get_control_values_from_log : Extracts control values from logged data
        optimizer.evaluate : Processes and learns from experimental data
        """
        # Special handling for the first scan step
        if index == 0:
            # Extract control values from logged data
            variables = self.get_control_values_from_log()

            # Update scan steps with extracted variables
            self.scan_steps[index].update({"variables": variables})
            logger.debug("Updated variables for first scan step: %s", variables)

        # Evaluate data using the optimizer
        try:
            self.optimizer.evaluate(inputs=self.scan_steps[index]["variables"])
            logger.info("Successfully evaluated data for scan step %s", index)
        except Exception:
            logger.exception("Error evaluating data for scan step %s", index)

    def get_control_values_from_log(self) -> Dict[str, float]:
        """
        Extract and compute average readback values for optimization variables.

        This method provides a robust mechanism for retrieving and processing
        experimental data logged during a scan step, supporting comprehensive
        data analysis and optimization strategies.

        Returns
        -------
        Dict[str, float]
            A dictionary of averaged variable values, keyed by variable name.
            Uses NaN for variables with no logged data.

        Notes
        -----
        Data Extraction and Processing Strategy:
        1. Identify current bin number
        2. Initialize variables dictionary from VOCS
        3. Filter log entries for the current bin
        4. Collect values for each variable
        5. Compute statistical summaries
        6. Handle cases with missing or incomplete data

        Advanced Data Handling:
        - Supports multiple variable tracking
        - Robust to incomplete logging
        - Provides NaN for variables without data
        - Flexible logging mechanism support

        Logging and Debugging:
        - Warns about variables with no logged data
        - Provides detailed insights into data collection
        - Supports experimental workflow analysis

        See Also
        --------
        optimizer.vocs : Variable optimization configuration set
        data_logger.log_entries : Logged experimental data entries
        numpy.mean : Statistical mean computation
        """
        # Get current bin number from data logger
        current_bin = self.data_logger.bin_num

        # Initialize variables dictionary from VOCS
        variables = {var: [] for var in self.optimizer.vocs.variable_names}

        # Collect values for the current bin
        for entry in self.data_logger.log_entries.values():
            if entry.get("Bin #") == current_bin:
                for var in self.optimizer.vocs.variable_names:
                    if var in entry:
                        variables[var].append(entry[var])

        # Compute average values with robust handling
        averaged_values: Dict[str, float] = {}
        for var, values in variables.items():
            if values:
                # Compute mean for variables with logged data
                averaged_values[var] = float(np.mean(values))
                logger.debug("Computed average for %s: %s", var, averaged_values[var])
            else:
                # Use NaN for variables without logged data
                logger.warning(
                    "No readback data found for variable '%s' in bin %s",
                    var,
                    current_bin,
                )
                averaged_values[var] = float("nan")

        return averaged_values

    def generate_next_step(self, next_index: int) -> None:
        """
        Dynamically generate the next experimental scan step using advanced optimization strategies.

        This method provides a sophisticated mechanism for generating experimental
        scan steps, supporting both initialization and iterative optimization
        processes. It leverages the integrated optimizer to determine the most
        promising next experimental configuration.

        Parameters
        ----------
        next_index : int
            The index of the next scan step to be generated.

        Notes
        -----
        Step Generation Process:
        1. Determine generation strategy based on step index
        2. Use random inputs during initialization phase
        3. Apply advanced optimization for subsequent steps
        4. Update scan steps with generated variables

        Optimization Strategies:
        - Supports multi-stage optimization workflow
        - Provides flexible initialization mechanisms
        - Enables data-driven experimental exploration
        - Adapts to different experimental requirements

        Initialization Handling:
        - First 3 steps use random input generation
        - Subsequent steps use advanced optimization algorithms
        - Ensures comprehensive initial experimental space exploration

        Logging and Debugging:
        - Detailed logging of generation process
        - Captures optimizer inputs and outputs
        - Provides insights into optimization decision-making
        - Supports experimental workflow traceability

        Error Handling:
        - Robust error management during step generation
        - Graceful handling of optimization failures
        - Prevents interruption of experimental workflow

        Raises
        ------
        Logs warnings for optimization failures
        Does not raise explicit exceptions to maintain workflow continuity

        See Also
        --------
        optimizer.vocs.random_inputs : Random input generation method
        optimizer.generate : Advanced optimization step generation method
        evaluate_acquired_data : Data evaluation method preceding step generation
        """
        # Define number of initialization steps
        num_initialization_steps = 2

        try:
            # Determine generation strategy based on step index
            if next_index <= num_initialization_steps:
                # Initialization phase: use random inputs for comprehensive exploration
                logger.info("Running optimizer initialization")
                next_variables = self.optimizer.vocs.random_inputs(1)[0]
                logger.debug(
                    "Initializer generated random variables: %s", next_variables
                )
            else:
                # Advanced optimization phase: use sophisticated generation strategy
                logger.info("Running advanced optimizer")
                next_variables = self.optimizer.generate(1)[0]
                logger.debug(
                    "Optimizer generated optimized variables: %s", next_variables
                )

            # Log successful generation
            logger.info(
                "Next experimental step generated using optimizer: %s",
                next_variables,
            )

        except Exception:
            # Robust error handling with detailed logging
            logger.exception("Failed to update next step via optimizer")
            return

        # Update scan steps with generated variables
        self.scan_steps[next_index].update({"variables": next_variables})
        logger.info(
            "Updated scan step %s with new configuration: %s",
            next_index,
            self.scan_steps[next_index],
        )

    def trigger_on(self) -> None:
        """
        Activate the data acquisition trigger mechanism for the experimental workflow.

        This method provides a flexible and dynamic approach to enabling
        data acquisition triggers, supporting various experimental control
        configurations and hardware interaction strategies.

        Notes
        -----
        Trigger Activation Mechanism:
        - Checks for the existence of a custom trigger function
        - Supports optional and configurable trigger implementations
        - Allows for dynamic experimental control setup

        Design Principles:
        - Minimal overhead trigger activation
        - Supports diverse experimental hardware configurations
        - Provides a standardized interface for trigger management

        Trigger Function Requirements:
        - Must be a callable method/function
        - Responsible for hardware-specific trigger activation
        - Should handle any necessary pre-trigger setup

        Experimental Control Integration:
        - Enables precise timing control for data acquisition
        - Supports complex experimental workflows
        - Facilitates synchronization of device interactions

        Error Handling:
        - Silently skips trigger activation if no trigger function is defined
        - Prevents workflow interruption
        - Supports flexible experimental setup


        See Also
        --------
        trigger_off : Deactivate data acquisition trigger
        wait_for_acquisition : Method managing trigger during data collection
        """
        if hasattr(self, "trigger_on_fn"):
            self.trigger_on_fn()

    def trigger_off(self) -> None:
        """
        Deactivate the data acquisition trigger mechanism for the experimental workflow.

        This method provides a flexible and dynamic approach to disabling
        data acquisition triggers, supporting various experimental control
        configurations and hardware interaction strategies.

        Notes
        -----
        Trigger Deactivation Mechanism:
        - Checks for the existence of a custom trigger deactivation function
        - Supports optional and configurable trigger implementations
        - Allows for dynamic experimental control shutdown

        Design Principles:
        - Minimal overhead trigger deactivation
        - Supports diverse experimental hardware configurations
        - Provides a standardized interface for trigger management

        Trigger Function Requirements:
        - Must be a callable method/function
        - Responsible for hardware-specific trigger deactivation
        - Should handle any necessary post-trigger cleanup

        Experimental Control Integration:
        - Enables precise timing control for data acquisition
        - Supports complex experimental workflows
        - Facilitates synchronization of device interactions

        Error Handling:
        - Silently skips trigger deactivation if no trigger function is defined
        - Prevents workflow interruption
        - Supports flexible experimental setup

        See Also
        --------
        trigger_on : Activate data acquisition trigger
        wait_for_acquisition : Method managing trigger during data collection
        """
        if hasattr(self, "trigger_off_fn"):
            self.trigger_off_fn()

    def save_hiatus(self, duration: float) -> None:
        """
        Pause experimental workflow to introduce a controlled delay before data saving.

        This method provides a precise mechanism for introducing a deliberate
        waiting period during the experimental data acquisition process,
        supporting various workflow management and data handling strategies.

        Parameters
        ----------
        duration : float
            The time to wait in seconds before saving experimental data.

        Notes
        -----
        Hiatus Execution Strategy:
        - Uses time.sleep() for precise waiting
        - Supports fractional second pauses
        - Provides flexibility in experimental data management

        Workflow Management:
        - Allows controlled interruption of data acquisition
        - Supports complex experimental protocols
        - Enables fine-grained timing control

        Logging and Debugging:
        - Logs the hiatus duration for traceability
        - Provides insights into experimental workflow
        - Supports detailed experimental process documentation

        Performance Considerations:
        - Minimal computational overhead
        - Non-blocking waiting mechanism
        - Precise time control

        Error Handling:
        - Graceful execution with comprehensive logging
        - Prevents workflow interruption
        - Supports various duration configurations

        See Also
        --------
        time.sleep : Standard Python time-based pause mechanism
        wait_for_acquisition : Primary method managing acquisition timing
        """
        logger.info("Hiatus: Waiting for %s seconds before saving data.", duration)
        time.sleep(duration)
