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


class GeecsDeviceExeTimeout(Exception):
    """Raised when a device accepts a command but no execution response arrives within timeout.

    This is a communication-layer failure - the command was ACK'd but the device
    never sent back an execution confirmation on the exe port.
    Could indicate:
    - Network packet loss on the exe response
    - Device hung mid-execution
    - UDP pipeline failure

    This happens in _execute() when listen() returns empty after exec_timeout seconds.
    See issue #312 for the planned GUI thread refactor.
    """

    def __init__(self, device_name: str, command: str, timeout: float):
        self.device_name = device_name
        self.command = command
        self.timeout = timeout
        super().__init__(
            f"Device '{device_name}' command '{command}' timed out after {timeout:.1f}s "
            f"waiting for execution response"
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
