"""Custom exceptions for GEECS device control errors."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GeecsDeviceInstantiationError(Exception):
    """Raised when a GEECS device fails to instantiate."""

    pass


class GeecsDeviceCommandRejected(Exception):
    """Raised when a device does not acknowledge/accept a UDP command.

    This is a communication-layer failure - the device didn't ACK the command.
    Could indicate:
    - Device offline/unreachable
    - Network issues
    - Device busy
    - Command malformed

    This happens in _process_command() when ack_cmd() returns False after all retries.
    """

    def __init__(
        self, device_name: str, command: str, ipv4: tuple[str, int] | None = None
    ):
        self.device_name = device_name
        self.command = command
        self.ipv4 = ipv4
        super().__init__(
            f"Device '{device_name}' did not acknowledge command '{command}'"
        )


class GeecsDeviceCommandFailed(Exception):
    """Raised when a device accepts a command but fails to execute it.

    This is a hardware/execution-layer failure - device ACK'd but returned err_status=True.
    Could indicate:
    - Value out of range
    - Hardware limit/interlock hit
    - Physical hardware failure (motor stuck, etc.)
    - Invalid state for operation

    This happens in handle_response() when err_status=True.
    """

    def __init__(
        self,
        device_name: str,
        command: str,
        error_detail: str,
        actual_value: str | None = None,
    ):
        self.device_name = device_name
        self.command = command
        self.error_detail = error_detail
        self.actual_value = actual_value
        super().__init__(
            f"Device '{device_name}' command '{command}' failed: {error_detail}"
        )
