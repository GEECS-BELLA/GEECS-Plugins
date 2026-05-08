"""Unified policy point for all device.set() / device.get() calls during a scan."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceCommandFailed,
    GeecsDeviceCommandRejected,
    GeecsDeviceExeTimeout,
)

from geecs_scanner.engine.dialog_request import DEVICE_COMMAND_ERRORS
from geecs_scanner.utils.exceptions import DeviceCommandError
from geecs_scanner.utils.retry import retry

logger = logging.getLogger(__name__)


class DeviceCommandExecutor:
    """Single policy object for every device command issued during a scan.

    Enforces a per-error-type retry policy and routes failures to a user
    escalation callback.  One instance is created per scan run and injected
    into ``ScanStepExecutor``, ``ScanDataManager``, and ``ActionManager``.

    Retry policy
    ------------
    - ``GeecsDeviceCommandRejected`` — retry up to *max_retries* times, then
      escalate.  A rejection means the device accepted the connection but
      refused the command; retrying is reasonable.
    - ``GeecsDeviceExeTimeout`` — escalate immediately.  The device is hung;
      retrying the same command makes it worse.
    - ``GeecsDeviceCommandFailed`` — escalate immediately.  A hardware-level
      failure; retry will not fix it.

    Attributes
    ----------
    on_escalate : callable or None
        ``(exc, context) -> bool`` — True means user chose Abort.
        When None the error is logged and execution continues.
    stop_event : threading.Event or None
        Set automatically when *on_escalate* returns True (abort).
    max_retries : int
    retry_delay : float
    """

    def __init__(
        self,
        on_escalate: Optional[Callable[[Exception, Optional[str]], bool]] = None,
        stop_event: Optional[threading.Event] = None,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        self.on_escalate = on_escalate
        self.stop_event = stop_event
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set(
        self,
        device,
        variable: str,
        value: Any,
        sync: bool = True,
        context: Optional[str] = None,
    ) -> Any:
        """Set *variable* on *device*, applying the retry/escalation policy.

        Parameters
        ----------
        device : ScanDevice
            Device to command.
        variable : str
        value : Any
        sync : bool
            Passed through to ``device.set()``.
        context : str, optional
            Extra information included in escalation dialogs.

        Returns
        -------
        Any
            Return value from ``device.set()``.

        Raises
        ------
        DeviceCommandError
            Raised after retry exhaustion or an immediate-escalation error.
            The stop_event is also set when *on_escalate* returns True.
        """
        device_name = self._name(device)
        try:
            return retry(
                lambda: device.set(variable, value, sync=sync),
                attempts=self.max_retries,
                delay=self.retry_delay,
                catch=(GeecsDeviceCommandRejected,),
                on_retry=lambda exc, n: logger.debug(
                    "[%s] rejected (attempt %d/%d) setting %s: %s",
                    device_name,
                    n,
                    self.max_retries,
                    variable,
                    exc,
                ),
            )
        except GeecsDeviceCommandRejected as exc:
            raise DeviceCommandError(
                device_name, f"set {variable}", variable=variable
            ) from exc
        except (GeecsDeviceExeTimeout, GeecsDeviceCommandFailed) as exc:
            raise DeviceCommandError(
                device_name, f"set {variable}", variable=variable
            ) from exc

    def get(
        self,
        device,
        variable: str,
        context: Optional[str] = None,
    ) -> Any:
        """Get *variable* from *device*, escalating on any hardware error.

        Parameters
        ----------
        device : ScanDevice
        variable : str
        context : str, optional

        Returns
        -------
        Any

        Raises
        ------
        DeviceCommandError
        """
        device_name = self._name(device)
        try:
            return device.get(variable)
        except DEVICE_COMMAND_ERRORS as exc:
            raise DeviceCommandError(
                device_name, f"get {variable}", variable=variable
            ) from exc

    def escalate(
        self,
        exc: Exception,
        context: Optional[str] = None,
    ) -> bool:
        """Route a device failure to the operator escalation callback.

        Sets ``stop_event`` when the user (or the headless fallback) chooses
        Abort.  Always call this after catching a ``DeviceCommandError``.

        Parameters
        ----------
        exc : Exception
        context : str, optional

        Returns
        -------
        bool
            True if the scan should be aborted.
        """
        abort = False
        if self.on_escalate is not None:
            abort = self.on_escalate(exc, context)
        else:
            logger.warning(
                "No escalation callback configured — continuing after error: %s", exc
            )

        if abort and self.stop_event is not None:
            self.stop_event.set()

        return abort

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _name(device) -> str:
        """Return a device name string regardless of whether device has .name or .get_name()."""
        if hasattr(device, "get_name"):
            return device.get_name()
        return getattr(device, "name", repr(device))
