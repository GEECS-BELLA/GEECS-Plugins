"""
DeviceManager module for GEECS Scanner data acquisition.

This module provides the `DeviceManager` class, which is responsible for managing
devices used in experimental data acquisition. It supports loading configuration files,
initializing device subscriptions, managing synchronous (event-driven) and asynchronous
(polling-based) observables, and handling composite scan variables.

Key Responsibilities
--------------------
- Load scan and device configurations from YAML files or in-memory dictionaries
- Subscribe to GEECS devices and variables for monitoring or data logging
- Manage scan setup and closeout actions using ActionManager-compatible steps
- Track and resolve composite variables from configuration files
- Support both standalone and composite device definitions

Dependencies
------------
- Standard Library:
    - threading: for synchronization and signaling fatal errors
    - logging: for runtime diagnostics and feedback
    - pathlib.Path: for path handling and configuration resolution
- Third-Party:
    - PyYAML: for parsing YAML configuration files
    - pydantic: for validating structured configuration schemas
- Internal Modules:
    - geecs_python_api.controls.devices.scan_device.ScanDevice
    - geecs_python_api.controls.interface.geecs_errors.GeecsDeviceInstantiationError
    - geecs_data_utils.ScanConfig, ScanMode
    - geecs_scanner.data_acquisition.utils.get_full_config_path
    - geecs_scanner.data_acquisition.schemas.save_devices (SaveDeviceConfig, DeviceConfig)
    - geecs_scanner.data_acquisition.schemas.actions (ActionSequence, SetStep)

Usage
-----
This module is typically used by `ScanManager` and `DataLogger` during experiment runtime.
It is responsible for dynamic subscription to device observables and ensures proper
initialization and cleanup of devices based on scan configuration.

"""

from __future__ import annotations
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from geecs_scanner.data_acquisition.schemas.save_devices import (
        SaveDeviceConfig,
        DeviceConfig,
    )

import logging
import yaml
import threading
from pathlib import Path

from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceInstantiationError,
)

from .utils import (
    get_full_config_path,
)  # Import utility function to build paths to config files
from geecs_data_utils import ScanConfig, ScanMode
from .schemas.save_devices import SaveDeviceConfig, DeviceConfig
from geecs_scanner.data_acquisition.schemas.actions import (
    ActionSequence,
    SetStep,
)  # or wherever ActionSequence is defined

from pydantic import ValidationError

# -----------------------------------------------------------------------------
# Module-level logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DeviceManager:
    """
    Manage GEECS device subscriptions and configurations for data acquisition.

    This class handles the loading and parsing of device configurations,
    initialization of device subscriptions (both synchronous and asynchronous),
    and management of composite variables. It provides the main interface for
    acquiring data from multiple devices during scans or optimization routines.

    Attributes
    ----------
    devices : dict
        Dictionary mapping device names to instantiated `ScanDevice` objects.
    event_driven_observables : list of str
        List of observables that are event-triggered (synchronous).
    async_observables : list of str
        List of observables that are polled (asynchronous).
    non_scalar_saving_devices : list of str
        List of device names that produce non-scalar data requiring file renaming.
    composite_variables : dict
        Dictionary of composite scan variables loaded from a config file.
    scan_setup_action : ActionSequence
        Action sequence executed before a scan begins.
    scan_closeout_action : ActionSequence
        Action sequence executed after a scan ends.
    scan_base_description : str
        User-provided description of the scan, used for metadata.
    fatal_error_event : threading.Event
        Event flag to signal a fatal error and halt operations.
    is_reset : bool
        Flag to indicate whether the device manager has been reset.
    experiment_dir : str
        Path to the experiment directory used to resolve config file locations.
    composite_variables_file_path : Path
        Path to the YAML file containing definitions for composite variables.
    """

    def __init__(self, experiment_dir: str | None = None):
        """
        Initialize a new instance of the DeviceManager.

        This constructor sets up the internal data structures used to manage device
        subscriptions, scan actions, and composite variable configurations. If an
        experiment directory is provided, it attempts to load composite variable
        definitions from a YAML configuration file located within that directory.

        Parameters
        ----------
        experiment_dir : str, optional
            Path to the directory containing experiment configuration files. If
            provided, this directory will be used to locate the `composite_variables.yaml`
            file for loading composite scan variables.

        Notes
        -----
        - Initializes internal state including event-driven and asynchronous observable lists,
          setup/closeout scan actions, and device tracking.
        - If the `composite_variables.yaml` file is not found in the provided experiment directory,
          a warning is logged and the `composite_variables` attribute remains empty.
        """
        self.devices: dict[str, ScanDevice] = {}
        self.event_driven_observables: list[str] = []  # Store event-driven observables
        self.async_observables: list[str] = []  # Store asynchronous observables
        self.non_scalar_saving_devices: list[
            str
        ] = []  # Devices that save non-scalar data
        self.composite_variables: dict = {}
        self.scan_setup_action: ActionSequence = ActionSequence(steps=[])
        self.scan_closeout_action: ActionSequence = ActionSequence(steps=[])
        self.scan_base_description: str = ""

        self.fatal_error_event = threading.Event()  # Used to signal a fatal error

        self.is_reset = (
            False  # Used to determine if a reset is required upon reinitialization
        )

        if experiment_dir is not None:
            # Set the experiment directory
            self.experiment_dir = experiment_dir

            try:
                # Use self.experiment_dir when calling the utility function
                self.composite_variables_file_path = get_full_config_path(
                    self.experiment_dir, "scan_devices", "composite_variables.yaml"
                )

                # Load composite variables from the file
                self.composite_variables = self._load_composite_variables(
                    self.composite_variables_file_path
                )
            except FileNotFoundError:
                logger.warning("Composite variables file not found.")

    def _load_composite_variables(self, composite_file: Path) -> dict:
        """Load and parse composite variable definitions from a YAML configuration file.

        Reads a YAML configuration file to extract composite variable definitions,
        which represent complex, multi-device scan parameters that can be controlled
        as a single virtual variable.

        Parameters
        ----------
        composite_file : Path
            Full filesystem path to the composite variables configuration file.
            Expected to be a YAML file with a specific structure for defining
            composite variable mappings.

        Returns
        -------
        dict
            A dictionary containing parsed composite variable definitions.
            Each key represents a composite variable name, and its value contains
            the detailed configuration for that variable.

        Notes
        -----
        Composite Variable YAML Structure:
        - Top-level key must be "composite_variables"
        - Each composite variable is defined with its specific device and variable mappings
        - Supports complex configurations involving multiple devices and variables

        Example YAML Structure:
        ```yaml
        composite_variables:
            BeamPosition:
                device1:
                    variable: x_position
                device2:
                    variable: y_position
        ```

        Raises
        ------
        FileNotFoundError
            If the specified composite variables file does not exist.
        yaml.YAMLError
            If there are parsing errors in the YAML file.

        Examples
        --------
        >>> device_mgr = DeviceManager()
        >>> composite_vars = device_mgr._load_composite_variables(Path('composite_vars.yaml'))
        >>> print(composite_vars)
        {'BeamPosition': {'device1': {...}, 'device2': {...}}}

        See Also
        --------
        is_composite_variable : Method to check if a variable is a composite variable
        load_from_config : Method that uses composite variables during configuration loading
        """
        try:
            with open(composite_file, "r") as file:
                self.composite_variables = yaml.safe_load(file).get(
                    "composite_variables", {}
                )
            logger.info("Loaded composite variables from %s", composite_file)
            return self.composite_variables
        except FileNotFoundError:
            logger.warning("Composite variables file not found: %s.", composite_file)
            return {}

    def load_from_config(self, config_filename: Union[str, Path]):
        """
        Load device configuration and scan metadata from a YAML file.

        This method parses a YAML file that defines the list of devices to be monitored,
        their associated variables, and optional scan setup/closeout actions. The loaded
        configuration is validated and used to initialize device subscriptions.

        Parameters
        ----------
        config_filename : Union[str, Path]
            The path to the YAML configuration file. Can be either a string representing
            the filename within the configured experiment directory or a full `Path` object.

        Notes
        -----
        If `config_filename` is a relative string, it is resolved within the
        experiment directory under the `save_devices` subfolder. If it's a valid
        `Path` object, it is used directly.
        """
        # Load the specific config for the experiment
        if isinstance(config_filename, Path) and config_filename.exists():
            config_path = config_filename
        else:
            config_path = get_full_config_path(
                self.experiment_dir, "save_devices", config_filename
            )

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info("Loaded configuration from %s", config_path)
        self.load_from_dictionary(config)

    def load_from_dictionary(self, config_dictionary):
        """
        Load device and scan configuration from a Python dictionary.

        This method validates a dictionary of experiment configuration data
        (typically loaded from a YAML file or constructed in a GUI), initializes
        device subscriptions, and stores scan setup and closeout actions.

        Parameters
        ----------
        config_dictionary : dict
            A dictionary containing the device configuration, scan info,
            and optional scan actions. It must conform to the SaveDeviceConfig schema.

        Notes
        -----
        - If `scan_info` is present in the dictionary, the `description` field is
          stored for reference.
        - Both synchronous and asynchronous observables are initialized based on the
          provided device definitions.
        - If validation fails, a warning is logged and the method returns early.
        """
        logger.info("config dict is %s", config_dictionary)
        try:
            validated: SaveDeviceConfig = SaveDeviceConfig(**config_dictionary)
        except ValidationError as e:
            logger.error("Invalid save device configuration: %s", e)
            return
        logger.info("validated SaveDeviceConfig is %s", validated)

        # note: there is a bit of mess with all these configs...
        if config_dictionary.get("scan_info", None):
            self.scan_base_description = config_dictionary["scan_info"].get(
                "description"
            )

        self.scan_setup_action = validated.setup_action or ActionSequence(steps=[])
        self.scan_closeout_action = validated.closeout_action or ActionSequence(
            steps=[]
        )

        self._load_devices_from_config(validated.Devices)
        self._initialize_subscribers(
            self.event_driven_observables + self.async_observables, clear_devices=False
        )
        logger.info("Loaded scan info: %s", self.scan_base_description)

    def _load_devices_from_config(self, devices: dict[str, DeviceConfig]):
        """
        Load and configure devices based on provided configuration.

        This method processes a dictionary of device configurations, registers
        each device with the manager, categorizes observables as synchronous
        (event-driven) or asynchronous (polled), and appends any associated scan
        setup actions.

        Parameters
        ----------
        devices : dict of str to DeviceConfig
            A dictionary mapping device names to their configuration objects,
            including variable lists, synchronicity flags, and setup actions.

        Notes
        -----
        - Devices marked as saving non-scalar data are tracked separately.
        - Devices requiring synchronization are automatically appended with the
          'acq_timestamp' variable if not already present.
        - If a device has already been instantiated, new variables are added to its subscription.
        - Setup actions specified in the configuration are added to the overall scan setup sequence.
        """
        for device_name, device_config in devices.items():
            logger.info(
                "%s: Synchronous = %s, Save_Non_Scalar = %s",
                device_name,
                device_config.synchronous,
                device_config.save_nonscalar_data,
            )

            # Add to non-scalar saving devices if applicable
            if device_config.save_nonscalar_data:
                # *NOTE* `acq_timestamp` allows for file renaming of nonscalar data
                if "acq_timestamp" not in device_config.variable_list:
                    device_config.variable_list.append("acq_timestamp")
                self.non_scalar_saving_devices.append(device_name)

            # Categorize as synchronous or asynchronous
            if device_config.synchronous:
                # *NOTE* `acq_timestamp` allows for checking synchronicity
                if "acq_timestamp" not in device_config.variable_list:
                    device_config.variable_list.append("acq_timestamp")
                self.event_driven_observables.extend(
                    [f"{device_name}:{var}" for var in device_config.variable_list]
                )
            else:
                self.async_observables.extend(
                    [f"{device_name}:{var}" for var in device_config.variable_list]
                )

            # Check if device already exists, if not, instantiate it
            if device_name not in self.devices:
                self._subscribe_device(device_name, device_config.variable_list)
            else:
                # If device exists, append new variables to its subscription
                self.devices[device_name].subscribe_var_values(
                    device_config.variable_list
                )

            # Append scan setup actions if they exist
            if device_config.scan_setup:
                self._append_device_setup_closeout_actions(
                    device_name, device_config.scan_setup
                )

        logger.info("Devices loaded: %s", list(self.devices.keys()))

    def _append_device_setup_closeout_actions(self, device_name, scan_setup):
        """
        Append setup and closeout actions for a device based on its scan configuration.

        This method generates `SetStep` actions to initialize and revert device variables
        at the beginning and end of a scan, respectively. It expects the `scan_setup`
        dictionary to contain keys for each variable and a list of two values:
        [setup_value, closeout_value].

        Parameters
        ----------
        device_name : str
            The name of the device for which setup/closeout actions are to be defined.

        scan_setup : dict
            Dictionary where each key is a variable name (str), and each value is a list
            of two elements: [setup_value, closeout_value].

        Notes
        -----
        - Adds the generated `SetStep` objects to the `scan_setup_action` and
          `scan_closeout_action` sequences maintained by the `DeviceManager`.
        - Logs a warning if a variable is missing either the setup or closeout value.
        - Does not wait for execution of each `SetStep` by default (`wait_for_execution=False`).
        """
        # Iterate over each key in the 'scan_setup' dictionary
        for analysis_type, values in scan_setup.items():
            # Ensure the setup and closeout values exist in the 'scan_setup'
            if len(values) != 2:
                logger.warning(
                    "Invalid scan setup actions for %s: %s (Expected 2 values, got %d)",
                    device_name,
                    analysis_type,
                    len(values),
                )
                continue

            setup_value, closeout_value = values

            self.scan_setup_action.steps.append(
                SetStep(
                    action="set",
                    device=device_name,
                    variable=analysis_type,
                    value=setup_value,
                    wait_for_execution=False,
                )
            )

            self.scan_closeout_action.steps.append(
                SetStep(
                    action="set",
                    device=device_name,
                    variable=analysis_type,
                    value=closeout_value,
                    wait_for_execution=False,
                )
            )

            logger.info(
                "Added setup and closeout actions for %s: %s (setup=%s, closeout=%s)",
                device_name,
                analysis_type,
                setup_value,
                closeout_value,
            )

    @staticmethod
    def is_statistic_noscan(variable_name):
        """
        Determine whether a variable name is a special placeholder for 'noscan' or 'statistics'.

        This is used to identify variables that are not part of the scan sweep or
        are intended to store derived statistical data, rather than raw device values.

        Parameters
        ----------
        variable_name : str
            The variable name to check.

        Returns
        -------
        bool
            True if the variable name is 'noscan' or 'statistics'; False otherwise.
        """
        return variable_name in ("noscan", "statistics")

    def is_composite_variable(self, variable_name):
        """
        Determine whether a variable is defined as a composite scan variable.

        Composite variables are virtual scan parameters defined in the configuration
        that represent a coordinated change in multiple device variables.

        Parameters
        ----------
        variable_name : str
            The name of the variable to check.

        Returns
        -------
        bool
            True if the variable is listed in the composite variable definitions; False otherwise.
        """
        return (
            self.composite_variables is not None
            and variable_name in self.composite_variables
        )

    def _initialize_subscribers(self, variables, clear_devices=True):
        """
        Initialize subscribers for the specified variables, creating or resetting device subscriptions.

        Parameters
        ----------
        variables : list of str
            List of device variables to subscribe to, in the format "Device:Variable".
        clear_devices : bool, optional
            If True, clear the existing device subscriptions before initializing new ones. Default is True.

        Returns
        -------
        None

        Notes
        -----
        This method will clear all existing device subscriptions if `clear_devices` is True,
        then subscribe to the provided variables by mapping them to their respective devices.
        Devices not already present in `self.devices` will be instantiated and subscribed.
        """
        if clear_devices:
            self._clear_existing_devices()

        device_map = self._preprocess_observables(variables)

        for device_name, var_list in device_map.items():
            if device_name not in self.devices:
                self._subscribe_device(device_name, var_list)

    def _clear_existing_devices(self):
        """
        Unsubscribe from all currently active devices and reset the device registry.

        This method safely closes all device connections, unsubscribes from variable monitoring,
        and clears the internal device dictionary. It is typically called when reloading a new
        configuration or resetting the scan environment.
        """
        for device_name, device in self.devices.items():
            try:
                logger.info("Attempting to unsubscribe from %s...", device_name)
                device.unsubscribe_var_values()
                device.close()
                logger.info("Successfully unsubscribed from %s.", device_name)
            except Exception:
                logger.exception("Error unsubscribing from %s", device_name)

        self.devices = {}

    def _subscribe_device(self, device_name, var_list):
        """
        Subscribe to a device and register its variable list for monitoring.

        This method attempts to instantiate the specified device and subscribe to the
        provided list of variables. In the case of a composite variable definition (if
        `device_name` is a dictionary), a placeholder can be managed accordingly by the caller.

        Parameters
        ----------
        device_name : str or dict
            The name of the device to subscribe to. If using composite variables, this may
            also be a dictionary containing relevant metadata.
        var_list : list of str
            A list of variable names to subscribe to for this device.

        Notes
        -----
        If device subscription fails (e.g., due to a GeecsDeviceInstantiationError),
        the error is logged and a fatal error event is triggered.
        """
        try:
            if self.is_composite_variable(device_name):
                var_dict = self.composite_variables[device_name]
                device = ScanDevice(device_name, var_dict)
            else:
                device = ScanDevice(device_name)
                device.use_alias_in_TCP_subscription = False
                logger.info("Subscribing %s to variables: %s", device_name, var_list)
                device.subscribe_var_values(var_list)

            self.devices[device_name] = device

        except GeecsDeviceInstantiationError as e:
            logger.error("Failed to instantiate GEecs device %s: %s", device_name, e)
            # Signal a fatal error event so that the scan can abort cleanly
            self.fatal_error_event.set()
            raise

    def reset(self):
        """
        Reset the DeviceManager by clearing all subscriptions and internal state.

        This method performs a full reset of the DeviceManager, which includes:
        - Unsubscribing from all currently subscribed devices
        - Closing device interfaces
        - Clearing internal lists of observables and non-scalar saving devices
        - Marking the instance as reset for safe reuse

        This is typically used before reinitializing the manager with a new configuration.
        """
        # Step 1: Close all subscribers
        self._clear_existing_devices()

        # Step 2: Clear internal state (reset lists)
        self.event_driven_observables.clear()
        self.async_observables.clear()
        self.non_scalar_saving_devices.clear()

        logger.info(
            "synchronous variables after reset: %s", self.event_driven_observables
        )
        logger.info("asynchronous variables after reset: %s", self.async_observables)
        logger.info(
            "non_scalar_saving_devices devices after reset: %s",
            self.non_scalar_saving_devices,
        )
        logger.info("devices devices after reset: %s", self.devices)
        logger.info(
            "DeviceManager instance has been reset and is ready for reinitialization."
        )
        self.is_reset = True

    def reinitialize(
        self, config_path: str | None = None, config_dictionary: dict | None = None
    ):
        """
        Reinitialize the DeviceManager by resetting its state and loading a new configuration.

        This method is used to safely reload device configurations without restarting the full application.
        It resets the internal state (device subscriptions, observables, etc.), clears previous setup and
        closeout actions, and then loads a new configuration either from a file or a dictionary.

        Parameters
        ----------
        config_path : str, optional
            Path to the YAML configuration file. If provided, this will be used to reinitialize devices.
        config_dictionary : dict, optional
            Dictionary representing the configuration. This is typically used by GUI interfaces or
            programmatic setups as an alternative to file-based loading.
        """
        # First, reset the current state
        if not self.is_reset:
            self.reset()
        self.is_reset = False

        self.scan_setup_action = ActionSequence(steps=[])
        self.scan_closeout_action = ActionSequence(steps=[])

        # Now load the new configuration and reinitialize the instance
        if config_path is not None:
            self.load_from_config(config_path)
        elif config_dictionary is not None:
            self.load_from_dictionary(config_dictionary)

        logger.info("DeviceManager instance has been reinitialized.")

    @staticmethod
    def _preprocess_observables(observables: list[str]) -> dict[str, list[str]]:
        """
        Preprocess a list of observable strings into a dictionary mapping devices to variables.

        This utility function takes a list of observables formatted as `"device:variable"` strings,
        and organizes them into a dictionary where each device name maps to a list of its variables.

        Parameters
        ----------
        observables : list of str
            List of observable strings in the format "DeviceName:VariableName".

        Returns
        -------
        dict of str to list of str
            A dictionary mapping device names to their corresponding list of variables.

        Examples
        --------
        >>> observables = ["Dev1:Var1", "Dev1:Var2", "Dev2:Var1"]
        >>> DeviceManager._preprocess_observables(observables)
        {'Dev1': ['Var1', 'Var2'], 'Dev2': ['Var1']}
        """
        device_map: dict[str, list[str]] = {}
        for observable in observables:
            device_name, var_name = observable.split(":")
            if device_name not in device_map:
                device_map[device_name] = []
            device_map[device_name].append(var_name)
        return device_map

    def add_scan_device(self, device_name, variable_list=None):
        """
        Add a scan device and its associated variables for monitoring during scans.

        This method either subscribes to a new device or appends additional variables
        to an existing one. The added observables are tracked as asynchronous (polled)
        variables by default. Composite variables are handled as single-entry observables.

        Parameters
        ----------
        device_name : str or dict
            Name of the device to subscribe to. If the device is a composite variable,
            it may be passed as a dictionary with structure-specific details.
        variable_list : list of str or None, optional
            List of variable names to subscribe to. If `None`, the device is assumed
            to use default settings. Composite variables may not require a variable list.

        Notes
        -----
        - Devices are assumed to be asynchronous unless otherwise configured elsewhere.
        - For new devices, a default configuration is assumed (non-scalar saving = False).
        - Composite variables are tracked using a single observable string without `:var` suffix.
        - If a device is already present, this method extends its variable subscriptions.

        TODO
        ----
        - Determine if `variable_list` should always be a list or a single string.
        - Consider removing logic related to `non_scalar_saving_devices` if unused.
        """
        if device_name not in self.devices:
            logger.info(
                "Adding new scan device: %s with default settings.", device_name
            )
            self._subscribe_device(device_name, var_list=variable_list)

            # TODO can we delete these lines of code for `self.nonscalar_saving_devices`?
            # Default attributes for scan-specific devices (e.g., from scan_config)
            default_device_config = {
                "save_non_scalar_data": False,
                "synchronous": False,
            }

            if default_device_config["save_non_scalar_data"]:
                self.non_scalar_saving_devices.append(device_name)

            # Add scan variables to async_observables
            if self.is_composite_variable(device_name):
                self.async_observables.extend([f"{device_name}"])
            else:
                self.async_observables.extend(
                    [f"{device_name}:{var}" for var in (variable_list or [])]
                )
            logger.info("Scan device %s added to async_observables.", device_name)

        else:
            logger.info(
                "Device %s already exists. Adding new variables: %s",
                device_name,
                variable_list,
            )

            # Update the existing device's variable list by subscribing to new variables
            device = self.devices[device_name]
            device.subscribe_var_values(variable_list)

            # Add new variables to async_observables
            self.async_observables.extend(
                [
                    f"{device_name}:{var}"
                    for var in (variable_list or [])
                    if f"{device_name}:{var}" not in self.async_observables
                ]
            )
            logger.info(
                "Updated async_observables with new variables for %s: %s",
                device_name,
                variable_list,
            )

    def handle_scan_variables(self, scan_config: ScanConfig):
        """
        Process the scan variable specified in the scan configuration.

        This method handles both standard and composite scan variables,
        depending on the scan mode. In NOSCAN or OPTIMIZATION modes, no
        variables are added.

        Parameters
        ----------
        scan_config : ScanConfig
            Configuration object specifying the scan mode and scan variable.
            Includes the device-variable string (e.g., "Device:Variable") or
            a composite variable name.

        Notes
        -----
        - In `ScanMode.NOSCAN`, no variable is added.
        - In `ScanMode.OPTIMIZATION`, device control is expected externally.
        - Otherwise, the scan variable is added using `_check_then_add_variable`.
        """
        logger.info("Handling scan variables with mode: %s", scan_config.scan_mode)

        if scan_config.scan_mode == ScanMode.NOSCAN:
            logger.info("NOSCAN mode: no scan variables to set.")
            return

        if scan_config.scan_mode == ScanMode.OPTIMIZATION:
            logger.info("OPTIMIZATION mode: assume devices will be set dynamically.")
            return

        device_var = scan_config.device_var
        logger.info("Processing scan device_var: %s", device_var)

        self._check_then_add_variable(device_var=device_var)

    def _check_then_add_variable(self, device_var: str):
        """
        Internal helper to register a scan variable for acquisition.

        Determines whether the provided variable is a composite or standard
        variable, and calls `add_scan_device` accordingly.

        Parameters
        ----------
        device_var : str
            The scan variable in the format "Device:Variable" for standard
            variables or a standalone name for composite variables.

        Notes
        -----
        - Composite variables are added without parsing into device/variable pairs.
        - Standard variables are split and added by device name and variable name.
        """
        if self.is_composite_variable(device_var):
            logger.info("%s is a composite variable.", device_var)
            device_name = device_var
            logger.info(
                "Trying to add composite device variable %s to self.devices.",
                device_var,
            )
            self.add_scan_device(device_name)
        else:
            # Normal variable case
            logger.info("%s is a normal variable.", device_var)
            device_name, var_name = device_var.split(":", 1)
            logger.info("Trying to add %s:%s to self.devices.", device_name, var_name)
            self.add_scan_device(device_name, [var_name])
