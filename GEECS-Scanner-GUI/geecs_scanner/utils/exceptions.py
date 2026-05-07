"""Scanner exception hierarchy and exception utilities.

All scanner-domain exceptions inherit from :class:`ScanError` so callers can
catch the base type when they don't care about sub-category, or catch a
specific subclass when they do.

Hierarchy
---------
::

    ScanError
    ├── ConfigError          user-visible configuration problems
    │   ├── ActionError      wrong action name / failed get command
    │   └── ConflictingScanElements
    ├── DeviceCommandError   hardware device command failed; wraps geecs_python_api errors
    │   └── TriggerError     trigger-device failure; always scan-fatal
    ├── DeviceSynchronizationError
    │   └── DeviceSynchronizationTimeout
    ├── ScanSetupError       pre-logging setup failed; scan cannot proceed
    ├── ScanAbortedError     user requested a stop
    └── DataFileError        file / data I/O failure
        └── OrphanProcessingTimeout

The three geecs_python_api hardware exceptions
(``GeecsDeviceExeTimeout``, ``GeecsDeviceCommandRejected``,
``GeecsDeviceCommandFailed``) are wrapped as :class:`DeviceCommandError`
via Python exception chaining (``raise DeviceCommandError(...) from original``).
Call sites in :mod:`geecs_scanner.data_acquisition` should never import those
API-internal types directly; use :data:`DEVICE_COMMAND_ERRORS` from
:mod:`geecs_scanner.data_acquisition.dialog_request` to catch them at the
boundary, then re-raise as the appropriate scanner-level exception.
"""

from __future__ import annotations

import sys
import traceback


# ---------------------------------------------------------------------------
# Qt exception hook (kept here because main.py imports it via this module)
# ---------------------------------------------------------------------------


def exception_hook(exctype, value, tb):
    """Route uncaught exceptions in PyQt5 windows to the standard logger.

    Installed as ``sys.excepthook`` in ``main.py``.
    """
    print("An error occurred:")
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ScanError(Exception):
    """Base class for all scanner-domain exceptions."""


# ---------------------------------------------------------------------------
# Config / setup
# ---------------------------------------------------------------------------


class ConfigError(ScanError):
    """Raised for user-visible configuration problems that a code fix won't help.

    Catching :class:`ConfigError` is the right pattern for code that wants to
    distinguish "the user configured something wrong" from "the hardware failed".
    """


class ActionError(ConfigError):
    """Wrong action name or failed get command in an action sequence.

    Attributes
    ----------
    message : str
        Human-readable description shown in the GUI message box.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConflictingScanElements(ConfigError):
    """Two save elements share a device but carry incompatible configuration flags.

    Attributes
    ----------
    message : str
        Human-readable description shown in the GUI message box.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ScanSetupError(ScanError):
    """Raised when pre-logging setup fails and the scan cannot proceed."""


# ---------------------------------------------------------------------------
# Device / hardware
# ---------------------------------------------------------------------------


class DeviceCommandError(ScanError):
    """A hardware device command failed after all retry attempts.

    This is the scanner-layer wrapper for the three geecs_python_api hardware
    exceptions.  Call sites should use exception chaining::

        except DEVICE_COMMAND_ERRORS as exc:
            raise DeviceCommandError(
                device_name, operation, variable=var_name
            ) from exc

    Attributes
    ----------
    device_name : str
        Name of the GEECS device that failed (e.g. ``"U_ModeImagerESP"``).
    operation : str
        Short description of what was attempted (e.g. ``"set Position.Axis 1"``).
    variable : str or None
        Variable name being set, if applicable.
    """

    def __init__(
        self,
        device_name: str,
        operation: str,
        *,
        variable: str | None = None,
    ) -> None:
        self.device_name = device_name
        self.operation = operation
        self.variable = variable
        var_clause = f" (variable: {variable})" if variable else ""
        super().__init__(f"Device '{device_name}'{var_clause} failed: {operation}")


class TriggerError(DeviceCommandError):
    """Trigger-device command failed.

    Trigger failures are always scan-fatal — the scan state is undefined if the
    trigger cannot be controlled.  Code that catches
    :class:`DeviceCommandError` and wants to treat trigger failures differently
    can check for this subclass.
    """


# ---------------------------------------------------------------------------
# Device synchronization
# ---------------------------------------------------------------------------


class DeviceSynchronizationError(ScanError):
    """Base class for device synchronization failures."""


class DeviceSynchronizationTimeout(DeviceSynchronizationError):
    """Raised when devices fail to synchronize within the allowed timeout."""


# ---------------------------------------------------------------------------
# Scan lifecycle
# ---------------------------------------------------------------------------


class ScanAbortedError(ScanError):
    """Raised when the user requests a stop during prelogging or before scan execution."""


# ---------------------------------------------------------------------------
# Data / file I/O
# ---------------------------------------------------------------------------


class DataFileError(ScanError):
    """File or data I/O failure.

    Often transient (network share blip, file not yet flushed) and therefore
    retryable with backoff.  Use exception chaining to preserve the original
    ``OSError`` or ``FileNotFoundError``.
    """


class OrphanProcessingTimeout(DataFileError):
    """Raised when orphaned file or task processing exceeds the allowed timeout."""
