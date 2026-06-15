"""
Device monitoring group for interlock system.

Manages subscriptions and state access for one or more GeecsDevice instances.
Provides unified interface for condition builders to read device state and timestamps.
Automatically detects stale data and fails-safe (returns unsafe) on timeout.
"""

import logging
import time
from typing import Dict, List, Optional, Any

from geecs_python_api.controls.interface.geecs_database import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice

logger = logging.getLogger(__name__)


class DeviceMonitorGroup:
    """
    Manages subscriptions and state access for multiple GeecsDevice instances.

    Handles device creation, variable subscription, and provides unified interface
    for condition builders to access device state and timestamps.
    """

    def __init__(
        self,
        exp_name: str,
        devices: Optional[Dict[str, str]] = None,
        staleness_timeout_ms: float = 5000,
    ):
        """
        Initialize device monitor group.

        Args:
            exp_name: Experiment name (passed to GeecsDatabase.collect_exp_info())
            devices: Optional dict mapping device alias to device name.
                    Example: {'cam1': 'CAM-PL1-TapeDrivePointing', 'motor1': 'MOTOR-X'}
                    If None, devices can be added dynamically via add_device()
            staleness_timeout_ms: Timeout for data staleness detection in milliseconds.
                                 If device timestamp doesn't update for this duration,
                                 all checks return unsafe (fail-safe). Default 5000ms.
        """
        self.exp_name = exp_name
        self.devices: Dict[str, GeecsDevice] = {}
        self.device_variables: Dict[str, List[str]] = {}
        self.staleness_timeout_s = staleness_timeout_ms / 1000.0

        # Track last seen timestamp and check time for each device (for staleness detection)
        self.last_timestamps: Dict[str, Optional[Any]] = {}
        self.last_check_times: Dict[str, float] = {}

        self.subscribed = False

        # Initialize experiment info
        try:
            GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(exp_name)
            logger.info(f"Initialized experiment info for {exp_name}")
        except Exception as e:
            logger.error(f"Failed to collect experiment info for {exp_name}: {e}")
            raise

        # Add initial devices if provided
        if devices:
            for alias, device_name in devices.items():
                self.add_device(alias, device_name)

    def add_device(
        self, alias: str, device_name: str, variables: Optional[List[str]] = None
    ) -> GeecsDevice:
        """
        Register a device and optionally its variables.

        Args:
            alias: Short name/alias for this device (e.g., 'cam1')
            device_name: Full device name from database (e.g., 'CAM-PL1-TapeDrivePointing')
            variables: Optional list of variables to subscribe to.
                      Will be subscribed when subscribe_all() is called.

        Returns
        -------
            The GeecsDevice instance for additional setup (e.g., use_alias_in_TCP_subscription)

        Raises
        ------
            Exception if device creation fails
        """
        try:
            device = GeecsDevice(device_name)
            self.devices[alias] = device

            if variables:
                self.device_variables[alias] = variables
                logger.debug(
                    f"Registered device '{alias}' ({device_name}) with {len(variables)} variables"
                )
            else:
                self.device_variables[alias] = []
                logger.debug(f"Registered device '{alias}' ({device_name})")

            return device

        except Exception as e:
            logger.error(f"Failed to create device '{alias}' ({device_name}): {e}")
            raise

    def subscribe_all(self) -> bool:
        """
        Subscribe all registered devices to their respective variables.

        Returns
        -------
            True if all subscriptions successful, False otherwise
        """
        success = True

        for alias, device in self.devices.items():
            variables = self.device_variables.get(alias, [])

            if not variables:
                logger.debug(f"Device '{alias}': no variables to subscribe")
                continue

            try:
                result = device.subscribe_var_values(variables)
                if result:
                    logger.info(
                        f"Subscribed device '{alias}' to {len(variables)} variables"
                    )
                else:
                    logger.warning(f"Device '{alias}' subscription returned False")
                    success = False

            except Exception as e:
                logger.error(f"Failed to subscribe device '{alias}': {e}")
                success = False

        self.subscribed = success
        return success

    def unsubscribe_all(self) -> None:
        """
        Unsubscribe all registered devices.
        Called during cleanup/shutdown.
        
        """
        for alias, device in self.devices.items():
            try:
                device.unsubscribe_var_values()
                logger.debug(f"Unsubscribed device '{alias}'")
            except Exception as e:
                logger.error(f"Failed to unsubscribe device '{alias}': {e}")

        self.subscribed = False

    def get_value(self, device_alias: str, variable_name: str) -> Any:
        """
        Read current value of a device variable.

        IMPORTANT: Automatically returns None (triggering fail-safe in condition builders)
        if the device data is stale (hasn't updated within staleness_timeout_ms).

        Args:
            device_alias: Alias used to register device
            variable_name: Name of variable to read

        Returns
        -------
            Variable value, or None if not available or data is stale
        """
        try:
            device = self.devices.get(device_alias)
            if device is None:
                logger.warning(f"Device '{device_alias}' not found")
                return None

            # Check for staleness before returning value
            if self._is_device_stale(device_alias):
                logger.warning(
                    f"Device '{device_alias}' data is stale - returning None (fail-safe)"
                )
                return None

            value = device.state.get(variable_name)
            return value

        except Exception as e:
            logger.error(f"Error reading {device_alias}.{variable_name}: {e}")
            return None

    def get_timestamp(self, device_alias: str) -> Any:
        """
        Get timestamp for a device (for staleness detection).

        Args:
            device_alias: Alias used to register device

        Returns
        -------
            acq_timestamp value, or None if not available.
            Device must have 'acq_timestamp' in subscribed variables.
        """
        try:
            device = self.devices.get(device_alias)
            if device is None:
                logger.warning(f"Device '{device_alias}' not found")
                return None

            timestamp = device.state.get("acq_timestamp")
            return timestamp

        except Exception as e:
            logger.error(f"Error reading {device_alias}.acq_timestamp: {e}")
            return None

    def get_device(self, device_alias: str) -> Optional[GeecsDevice]:
        """
        Get direct access to a GeecsDevice instance.

        Allows advanced usage beyond standard monitoring.

        Args:
            device_alias: Alias used to register device

        Returns
        -------
            GeecsDevice instance, or None if not found
        """
        return self.devices.get(device_alias)

    def get_all_devices(self) -> Dict[str, GeecsDevice]:
        """
        Get all registered devices.

        Returns
        -------
            Dict mapping device alias to GeecsDevice instance
        """
        return self.devices.copy()

    def _is_device_stale(self, device_alias: str) -> bool:
        """
        Check if device data is stale (no update for staleness_timeout_s).

        Internal method called by get_value() to enforce fail-safe behavior.

        Args:
            device_alias: Alias used to register device

        Returns
        -------
            True if data is stale (hasn't updated for timeout duration)
        """
        device = self.devices.get(device_alias)
        if device is None:
            return True  # Device not found = stale

        try:
            current_time = time.time()
            timestamp = device.state.get("acq_timestamp")

            if timestamp is None:
                # First time seeing this device, initialize tracking
                self.last_timestamps[device_alias] = None
                self.last_check_times[device_alias] = current_time
                return False  # Not stale yet

            last_ts = self.last_timestamps.get(device_alias)

            if timestamp == last_ts:
                # Timestamp hasn't changed
                last_check = self.last_check_times.get(device_alias, current_time)
                time_frozen = current_time - last_check

                if time_frozen > self.staleness_timeout_s:
                    logger.warning(
                        f"Device '{device_alias}' data is stale: "
                        f"no update for {time_frozen:.1f}s (timeout: {self.staleness_timeout_s:.1f}s)"
                    )
                    return True
            else:
                # Timestamp updated, reset tracking
                self.last_timestamps[device_alias] = timestamp
                self.last_check_times[device_alias] = current_time

            return False  # Data is fresh

        except Exception as e:
            logger.error(f"Error checking staleness for '{device_alias}': {e}")
            return True  # Fail-safe: consider stale on error
