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
from .external_status_listener import ExternalStatusListener, default_decode

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
        port: int | None = None,
        staleness_timeout_ms: float = 5000,
    ):
        """
        Initialize interlock constructor.

        Args:
        ----
            exp_name: Experiment name
            host: TCP server host (default '0.0.0.0' = all interfaces)
            port: TCP server port (required; typically SERVER_PORT from .env)
            staleness_timeout_ms: Timeout for data staleness detection (default 5000ms).
                                 All interlocks automatically fail-safe if device data
                                 hasn't updated within this duration.
        """
        if port is None:
            raise ValueError(
                "InterlockConstructor requires an explicit port "
                "(e.g. SERVER_PORT from .env)"
            )
        self.exp_name = exp_name
        self.host = host
        self.port = port

        # Initialize internal components
        self.device_group = DeviceMonitorGroup(
            exp_name, staleness_timeout_ms=staleness_timeout_ms
        )
        self.server: Optional[InterlockServer] = None
        self.monitor_registrations: Dict[
            str, tuple[Callable[[], bool], float, int]
        ] = {}
        self.external_listeners: list[ExternalStatusListener] = []

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
        ----
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

    def add_external_listener(
        self,
        port: int,
        host: str = "",
        decode: Optional[Callable[[bytes], Optional[bool]]] = None,
        name: Optional[str] = None,
    ) -> ExternalStatusListener:
        """
        Register a UDP listener for a non-GEECS, one-shot status signal.

        Use this for a source that isn't a GEECS device variable - e.g. a
        motion controller on another computer that unicasts a single "safe" or
        "warning" packet whenever its state changes, rather than continuously
        publishing a value. The listener's start()/stop() lifecycle is tied to
        this constructor's, same as subscribed devices. Wrap the returned
        listener in an ExternalStatusCheck to use it as a monitor condition.

        Args:
        ----
            port: UDP port to listen on.
            host: Local interface to bind to (default '' = all interfaces on
                this machine). Must be an address assigned to one of THIS
                machine's own network interfaces, not the sending computer's
                IP - binding to a non-local address fails (WSAEADDRNOTAVAIL /
                WinError 10049 on Windows).
            decode: Function mapping a raw packet payload to True (unsafe),
                False (safe), or None (unrecognized). Defaults to a
                case-insensitive "safe"/"warning" text match - pass your own
                once you know the sender's exact payload format.
            name: Optional label for log messages and diagnostics.

        Returns
        -------
            ExternalStatusListener instance (not yet started; it starts when
            this constructor's start() is called).

        Example:
            motion = interlock.add_external_listener(UDP_PORT, name="stage-motion")
            motion_check = ExternalStatusCheck(motion, label="stage moving")
            interlock.add_monitor("Stage Motion Interlock", motion_check)
        """
        listener = ExternalStatusListener(
            port=port,
            host=host,
            decode=decode or default_decode,
            name=name,
        )
        self.external_listeners.append(listener)
        return listener

    def add_monitor(
        self,
        name: str,
        check_func: Callable[[], bool],
        interval: float = 0.1,
        safe_release_checks: int = 3,
    ) -> None:
        """
        Register a monitor with the interlock server.

        Args:
        ----
            name: Human-readable name for this monitor
            check_func: Callable that returns bool (True=unsafe, False=safe).
                       Can be result of any condition builder (ThresholdCheck, MultiCheck, etc.)
            interval: How frequently to run monitor check (seconds, default 0.1s)
            safe_release_checks: number of consecutive safe polls required before the
                                 interlock clears back to safe

        Example:
        -------
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
        if safe_release_checks < 1:
            raise ValueError(
                f"safe_release_checks must be >= 1, got {safe_release_checks}"
            )

        self.monitor_registrations[name] = (
            check_func,
            interval,
            safe_release_checks,
        )

        # Register with server if it exists
        if self.server is not None:
            diagnostic_func = getattr(check_func, "get_diagnostic_info", None)
            self.server.register_monitor(
                name,
                check_func,
                interval=interval,
                diagnostic_func=diagnostic_func,
                safe_release_checks=safe_release_checks,
            )
            logger.debug(f"Registered monitor '{name}' with interval {interval}s")
        else:
            logger.debug(
                f"Queued monitor '{name}' for registration (server not started)"
            )

    def remove_monitor(self, name: str) -> None:
        """Remove a monitor registration while the server is stopped."""
        if self.server is not None:
            raise RuntimeError("Stop the interlock server before removing a monitor")
        if name not in self.monitor_registrations:
            raise KeyError(f"Monitor '{name}' is not registered")
        del self.monitor_registrations[name]
        logger.info("Removed monitor '%s'", name)

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

            # Start listeners for non-GEECS, externally-signaled conditions
            for listener in self.external_listeners:
                listener.start()

            # Create and start server
            logger.info(f"Starting interlock server on {self.host}:{self.port}")
            self.server = InterlockServer(host=self.host, port=self.port)

            # Register all monitors
            for name, (
                check_func,
                interval,
                safe_release_checks,
            ) in self.monitor_registrations.items():
                diagnostic_func = getattr(check_func, "get_diagnostic_info", None)

                self.server.register_monitor(
                    name,
                    check_func,
                    interval=interval,
                    diagnostic_func=diagnostic_func,
                    safe_release_checks=safe_release_checks,
                )
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

            for listener in self.external_listeners:
                listener.stop()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

    def get_monitor_state(self, name: str) -> Optional[bool]:
        """
        Get current state of a named monitor.

        Args:
        ----
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

    def get_diagnostic_info(self, monitor_name: str) -> str:
        """
        Get diagnostic info for a specific monitor (what variable triggered the unsafe state).

        Args:
        ----
            monitor_name: Monitor name as registered with add_monitor()

        Returns
        -------
            String describing what failed (e.g., "MaxCounts=1500 (threshold: 4000)"),
            or empty string if monitor is safe or not found
        """
        registration = self.monitor_registrations.get(monitor_name)
        check_func = registration[0] if registration else None
        if check_func and hasattr(check_func, "get_diagnostic_info"):
            return check_func.get_diagnostic_info()
        return ""

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
        -------
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
