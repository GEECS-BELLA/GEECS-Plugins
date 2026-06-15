"""
High-level facade for interlock setup and management.

Coordinates device subscriptions, monitor registration, and server lifecycle.
Provides simplified API for creating and controlling interlock servers.
"""

import logging
from typing import Dict, Optional, Callable

from geecs_python_api.controls.interlocks.geecs_interlock_server import InterlockServer
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from .device_monitor_group import DeviceMonitorGroup

logger = logging.getLogger(__name__)


class InterlockConstructor:
    """
    High-level API for interlock setup and management.

    Coordinates device subscriptions, monitor registration, and TCP server lifecycle.
    Provides simplified interface for creating production interlocks.
    """

    def __init__(
        self,
        exp_name: str,
        host: str = "0.0.0.0",
        port: int = 5001,
        staleness_timeout_ms: float = 5000,
    ):
        """
        Initialize interlock constructor.

        Args:
            exp_name: Experiment name
            host: TCP server host (default '0.0.0.0' = all interfaces)
            port: TCP server port (default 5001)
            staleness_timeout_ms: Timeout for data staleness detection (default 5000ms).
                                 All interlocks automatically fail-safe if device data
                                 hasn't updated within this duration.
        """
        self.exp_name = exp_name
        self.host = host
        self.port = port

        # Initialize internal components
        self.device_group = DeviceMonitorGroup(
            exp_name, staleness_timeout_ms=staleness_timeout_ms
        )
        self.server: Optional[InterlockServer] = None
        self.monitor_registrations: Dict[str, Callable[[], bool]] = {}

        logger.info(
            f"Initialized InterlockConstructor for experiment '{exp_name}' "
            f"(staleness timeout: {staleness_timeout_ms}ms)"
        )

    def add_device(
        self, alias: str, device_name: str, variables: list[str]
    ) -> GeecsDevice:
        """
        Register a device and variables to monitor.

        Args:
            alias: Short name/alias for device (e.g., 'cam1')
            device_name: Full device name from database (e.g., 'CAM-PL1-TapeDrivePointing')
            variables: List of variables to subscribe to

        Returns
        -------
            GeecsDevice instance for optional customization
                (e.g., device.use_alias_in_TCP_subscription = False)

        Example:
            cam1 = builder.add_device('cam1', 'CAM-PL1-TapeDrivePointing',
                ['MaxCounts', 'MeanCounts', 'acq_timestamp', 'Target.X', 'Target.Y'])
        """
        return self.device_group.add_device(alias, device_name, variables)

    def add_monitor(
        self, name: str, check_func: Callable[[], bool], interval: float = 0.1
    ) -> None:
        """
        Register a monitor with the interlock server.

        Args:
            name: Human-readable name for this monitor
            check_func: Callable that returns bool (True=unsafe, False=safe).
                       Can be result of any condition builder (ThresholdCheck, MultiCheck, etc.)
            interval: How frequently to run monitor check (seconds, default 0.1s)

        Example:
            from monitor_conditions import ThresholdCheck, MultiCheck

            builder.add_monitor(
                'Camera Multi Check',
                MultiCheck([
                    ThresholdCheck(device_group, 'cam1', 'MaxCounts', 4000, '>'),
                    ThresholdCheck(device_group, 'cam1', 'MeanCounts', 0, '<'),
                ]),
                interval=0.1
            )
        """
        if not callable(check_func):
            raise ValueError(f"check_func must be callable, got {type(check_func)}")

        self.monitor_registrations[name] = check_func

        # Register with server if it exists
        if self.server is not None:
            self.server.register_monitor(name, check_func, interval=interval)
            logger.debug(f"Registered monitor '{name}' with interval {interval}s")
        else:
            logger.debug(
                f"Queued monitor '{name}' for registration (server not started)"
            )

    def start(self) -> None:
        """
        Subscribe all devices and start the TCP server and monitor threads.

        Logs startup status and monitor names.
        Does NOT print to stdout.

        Raises
        ------
            RuntimeError if subscription or server startup fails
        """
        try:
            # Subscribe all devices
            logger.info("Subscribing to devices...")
            if not self.device_group.subscribe_all():
                raise RuntimeError("Device subscription failed")

            # Create and start server
            logger.info(f"Starting interlock server on {self.host}:{self.port}")
            self.server = InterlockServer(host=self.host, port=self.port)

            # Register all monitors
            for name, check_func in self.monitor_registrations.items():
                self.server.register_monitor(name, check_func, interval=0.1)
                logger.debug(f"Registered monitor '{name}'")

            self.server.start()
            logger.info(
                f"Interlock server started on {self.host}:{self.port} "
                f"with {len(self.monitor_registrations)} monitors"
            )

        except Exception as e:
            logger.error(f"Failed to start interlock server: {e}")
            raise

    def stop(self) -> None:
        """
        Stop the TCP server and unsubscribe all devices.

        Logs shutdown status.
        Does NOT print to stdout.
        """
        try:
            if self.server is not None:
                logger.info("Stopping interlock server...")
                self.server.stop()
                self.server = None
                logger.info("Interlock server stopped")

            logger.info("Unsubscribing from devices...")
            self.device_group.unsubscribe_all()
            logger.info("Unsubscribed from all devices")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

    def get_monitor_state(self, name: str) -> Optional[bool]:
        """
        Get current state of a named monitor.

        Args:
            name: Monitor name as registered with add_monitor()

        Returns
        -------
            True (unsafe), False (safe), or None if monitor not found

        Useful for debugging and live inspection.
        """
        if self.server is None:
            logger.warning("Server not running; cannot get monitor state")
            return None

        try:
            return self.server.get_interlock(name)
        except Exception as e:
            logger.error(f"Error getting monitor state for '{name}': {e}")
            return None

    def get_all_monitors(self) -> Dict[str, Optional[bool]]:
        """
        Get state of all monitors.

        Returns
        -------
            Dict mapping monitor name to bool state (True=unsafe, False=safe)
        """
        if self.server is None:
            logger.warning("Server not running; cannot get monitor states")
            return {}

        try:
            return self.server.get_all_interlocks()
        except Exception as e:
            logger.error(f"Error getting all monitor states: {e}")
            return {}

    def __enter__(self):
        """
        Context manager entry: start the server.

        Example:
            with InterlockConstructor('BELLA', host, port) as builder:
                cam1 = builder.add_device(...)
                builder.add_monitor(...)
                # Server auto-starts here
                time.sleep(10)
                # Server auto-stops on exit
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop the server."""
        self.stop()
        return False
