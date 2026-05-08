"""Device subscription and configuration management for ScanManager and DataLogger."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from geecs_scanner.engine.models.save_devices import (
        DeviceConfig,
        SaveDeviceConfig,
    )

import yaml
from pydantic import ValidationError

from geecs_data_utils import ScanConfig, ScanMode
from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceInstantiationError,
)

from .models.save_devices import DeviceConfig, SaveDeviceConfig
from ..utils.config_utils import get_full_config_path
from geecs_scanner.engine.models.actions import (
    ActionSequence,
    SetStep,
)

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage GEECS device subscriptions and configurations for data acquisition.

    Attributes
    ----------
    devices : dict[str, ScanDevice]
    event_driven_observables : list[str]
        Synchronous observables; logged on TCP event.
    async_observables : list[str]
        Polled observables; logged each shot.
    non_scalar_saving_devices : list[str]
        Devices that produce files requiring timestamp-based renaming.
    composite_variables : dict
    scan_setup_action : ActionSequence
    scan_closeout_action : ActionSequence
    scan_base_description : str
    fatal_error_event : threading.Event
    is_reset : bool
    experiment_dir : str
    composite_variables_file_path : Path
    """

    def __init__(self, experiment_dir: str | None = None):
        self.devices: dict[str, ScanDevice] = {}
        self.event_driven_observables: list[str] = []
        self.async_observables: list[str] = []
        self.non_scalar_saving_devices: list[str] = []
        self.composite_variables: dict = {}
        self.scan_setup_action: ActionSequence = ActionSequence(steps=[])
        self.scan_closeout_action: ActionSequence = ActionSequence(steps=[])
        self.scan_base_description: str = ""

        self.fatal_error_event = threading.Event()

        self.is_reset = (
            False  # Used to determine if a reset is required upon reinitialization
        )

        if experiment_dir is not None:
            self.experiment_dir = experiment_dir

            try:
                self.composite_variables_file_path = get_full_config_path(
                    self.experiment_dir, "scan_devices", "composite_variables.yaml"
                )
                self.composite_variables = self._load_composite_variables(
                    self.composite_variables_file_path
                )
            except FileNotFoundError:
                logger.warning("Composite variables file not found.")

    def _load_composite_variables(self, composite_file: Path) -> dict:
        """Parse ``composite_variables.yaml`` and return the composite variable dict."""
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
        """Load device config from a YAML file path or filename relative to the experiment dir."""
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
        """Validate *config_dictionary* via SaveDeviceConfig and initialize subscriptions."""
        logger.debug("config dict is %s", config_dictionary)
        try:
            validated: SaveDeviceConfig = SaveDeviceConfig(**config_dictionary)
        except ValidationError as e:
            logger.error("Invalid save device configuration: %s", e)
            return
        logger.debug("validated SaveDeviceConfig is %s", validated)

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
        """Categorize devices as sync/async and register their observables."""
        for device_name, device_config in devices.items():
            logger.debug(
                "%s: Synchronous = %s, Save_Non_Scalar = %s",
                device_name,
                device_config.synchronous,
                device_config.save_nonscalar_data,
            )

            if device_config.save_nonscalar_data:
                # *NOTE* `acq_timestamp` allows for file renaming of nonscalar data
                if "acq_timestamp" not in device_config.variable_list:
                    device_config.variable_list.append("acq_timestamp")
                self.non_scalar_saving_devices.append(device_name)

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

            if device_name not in self.devices:
                self._subscribe_device(device_name, device_config.variable_list)
            else:
                self.devices[device_name].subscribe_var_values(
                    device_config.variable_list
                )

            if device_config.scan_setup:
                self._append_device_setup_closeout_actions(
                    device_name, device_config.scan_setup
                )

        logger.info("Devices loaded: %s", list(self.devices.keys()))

    def _append_device_setup_closeout_actions(self, device_name, scan_setup):
        """Build SetStep pairs from *scan_setup* ``{variable: [setup_val, closeout_val]}``."""
        for analysis_type, values in scan_setup.items():
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

            logger.debug(
                "Added setup and closeout actions for %s: %s (setup=%s, closeout=%s)",
                device_name,
                analysis_type,
                setup_value,
                closeout_value,
            )

    @staticmethod
    def is_statistic_noscan(variable_name):
        """Return True if *variable_name* is the ``noscan`` or ``statistics`` placeholder."""
        return variable_name in ("noscan", "statistics")

    def is_composite_variable(self, variable_name):
        """Return True if *variable_name* is defined in ``composite_variables``."""
        return (
            self.composite_variables is not None
            and variable_name in self.composite_variables
        )

    def _initialize_subscribers(self, variables, clear_devices=True):
        """Subscribe to *variables* (``"Device:Variable"`` strings), optionally clearing first."""
        if clear_devices:
            self._clear_existing_devices()

        device_map = self._preprocess_observables(variables)

        for device_name, var_list in device_map.items():
            if device_name not in self.devices:
                self._subscribe_device(device_name, var_list)

    def _clear_existing_devices(self):
        """Unsubscribe and close all devices in parallel, then clear the registry."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _disconnect(device_name, device):
            logger.info("Attempting to unsubscribe from %s...", device_name)
            device.unsubscribe_var_values()
            device.close()
            logger.info("Successfully unsubscribed from %s.", device_name)

        devices_snapshot = dict(self.devices)
        if devices_snapshot:
            with ThreadPoolExecutor(max_workers=len(devices_snapshot)) as executor:
                futures = {
                    executor.submit(_disconnect, name, dev): name
                    for name, dev in devices_snapshot.items()
                }
                for future in as_completed(futures):
                    device_name = futures[future]
                    try:
                        future.result()
                    except Exception:
                        logger.exception("Error unsubscribing from %s", device_name)

        self.devices = {}

    def _subscribe_device(self, device_name, var_list):
        """Instantiate a ScanDevice for *device_name* and subscribe to *var_list*."""
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
            self.fatal_error_event.set()
            raise

    def reset(self):
        """Unsubscribe all devices and clear observable lists; call before reinitialize."""
        self._clear_existing_devices()

        self.event_driven_observables.clear()
        self.async_observables.clear()
        self.non_scalar_saving_devices.clear()

        logger.debug(
            "synchronous variables after reset: %s", self.event_driven_observables
        )
        logger.debug("asynchronous variables after reset: %s", self.async_observables)
        logger.debug(
            "non_scalar_saving_devices devices after reset: %s",
            self.non_scalar_saving_devices,
        )
        logger.debug("devices devices after reset: %s", self.devices)
        logger.debug(
            "DeviceManager instance has been reset and is ready for reinitialization."
        )
        self.is_reset = True

    def reinitialize(
        self, config_path: str | None = None, config_dictionary: dict | None = None
    ):
        """Reset internal state and load a new configuration from file or dict.

        Parameters
        ----------
        config_path : str, optional
        config_dictionary : dict, optional
        """
        if not self.is_reset:
            self.reset()
        self.is_reset = False

        self.scan_setup_action = ActionSequence(steps=[])
        self.scan_closeout_action = ActionSequence(steps=[])

        if config_path is not None:
            self.load_from_config(config_path)
        elif config_dictionary is not None:
            self.load_from_dictionary(config_dictionary)

        logger.info("DeviceManager instance has been reinitialized.")

    @staticmethod
    def _preprocess_observables(observables: list[str]) -> dict[str, list[str]]:
        """Group ``"Device:Variable"`` strings into ``{device: [var, ...]}``."""
        device_map: dict[str, list[str]] = {}
        for observable in observables:
            device_name, var_name = observable.split(":")
            if device_name not in device_map:
                device_map[device_name] = []
            device_map[device_name].append(var_name)
        return device_map

    def add_scan_device(self, device_name, variable_list=None):
        """Subscribe *device_name* to *variable_list* and register it as an async observable.

        If the device already exists, extends its variable subscription instead.

        Todo
        ----
        - Determine if `variable_list` should always be a list or a single string.
        - Consider removing logic related to `non_scalar_saving_devices` if unused.
        """
        if device_name not in self.devices:
            logger.debug(
                "Adding new scan device: %s with default settings.", device_name
            )
            self._subscribe_device(device_name, var_list=variable_list)

            # TODO can we delete these lines of code for `self.nonscalar_saving_devices`?
            default_device_config = {
                "save_non_scalar_data": False,
                "synchronous": False,
            }

            if default_device_config["save_non_scalar_data"]:
                self.non_scalar_saving_devices.append(device_name)

            if self.is_composite_variable(device_name):
                self.async_observables.extend([f"{device_name}"])
            else:
                self.async_observables.extend(
                    [f"{device_name}:{var}" for var in (variable_list or [])]
                )
            logger.debug("Scan device %s added to async_observables.", device_name)

        else:
            logger.debug(
                "Device %s already exists. Adding new variables: %s",
                device_name,
                variable_list,
            )

            device = self.devices[device_name]
            device.subscribe_var_values(variable_list)

            self.async_observables.extend(
                [
                    f"{device_name}:{var}"
                    for var in (variable_list or [])
                    if f"{device_name}:{var}" not in self.async_observables
                ]
            )
            logger.debug(
                "Updated async_observables with new variables for %s: %s",
                device_name,
                variable_list,
            )

    def handle_scan_variables(self, scan_config: ScanConfig):
        """Register the scan variable from *scan_config* unless mode is NOSCAN or OPTIMIZATION."""
        logger.debug("Handling scan variables with mode: %s", scan_config.scan_mode)

        if scan_config.scan_mode == ScanMode.NOSCAN:
            logger.debug("NOSCAN mode: no scan variables to set.")
            return

        if scan_config.scan_mode == ScanMode.OPTIMIZATION:
            logger.debug("OPTIMIZATION mode: assume devices will be set dynamically.")
            return

        device_var = scan_config.device_var
        logger.debug("Processing scan device_var: %s", device_var)

        self._check_then_add_variable(device_var=device_var)

    def _check_then_add_variable(self, device_var: str):
        """Add *device_var* to subscriptions, handling composite vs. standard format."""
        if self.is_composite_variable(device_var):
            logger.debug("%s is a composite variable.", device_var)
            device_name = device_var
            logger.debug(
                "Trying to add composite device variable %s to self.devices.",
                device_var,
            )
            self.add_scan_device(device_name)
        else:
            logger.debug("%s is a normal variable.", device_var)
            device_name, var_name = device_var.split(":", 1)
            logger.debug("Trying to add %s:%s to self.devices.", device_name, var_name)
            self.add_scan_device(device_name, [var_name])
