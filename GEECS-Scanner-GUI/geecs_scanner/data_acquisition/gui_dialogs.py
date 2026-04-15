"""GUI dialog helpers for scan data-acquisition error handling.

Dialogs are shown on the main Qt thread via a queue drained by the
200 ms GUI timer in GEECSScannerWindow.  Worker threads submit a
DialogRequest and block on its response_event; the main thread shows
the dialog and sets the event with the result.

Issue #312 tracking note: this module previously called Qt directly
from worker threads (unsafe).  The queue-based pattern implemented here
resolves that.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thread-safe request object
# ---------------------------------------------------------------------------


@dataclass
class DialogRequest:
    """Carries a device error across the worker→main-thread boundary.

    The worker thread puts this on ScanManager's dialog queue and then
    calls ``response_event.wait()``.  The main-thread timer picks it up,
    shows the dialog, writes the result into ``abort``, and calls
    ``response_event.set()``.

    Parameters
    ----------
    exc :
        The device exception that triggered the dialog.
    response_event :
        Set by the main thread once the user has responded.
    abort :
        Single-element list used as a mutable result container.
        ``True`` → user chose Abort; ``False`` → user chose Continue.
    """

    exc: Exception
    response_event: threading.Event = field(default_factory=threading.Event)
    abort: list[bool] = field(default_factory=lambda: [False])


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def _build_device_error_message(exc: Exception) -> tuple[str, str]:
    """Return (title, body) for a device error dialog.

    Parameters
    ----------
    exc :
        One of GeecsDeviceExeTimeout, GeecsDeviceCommandRejected,
        GeecsDeviceCommandFailed, or a generic Exception.

    Returns
    -------
    tuple[str, str]
        Dialog window title and body text.
    """
    from geecs_python_api.controls.interface.geecs_errors import (
        GeecsDeviceCommandFailed,
        GeecsDeviceCommandRejected,
        GeecsDeviceExeTimeout,
    )
    from geecs_scanner.utils.exceptions import ActionError

    if isinstance(exc, GeecsDeviceExeTimeout):
        title = "Device Execution Timeout"
        detail = (
            f"No execution response received within {exc.timeout:.1f}s.\n"
            f"The device may be slow, stuck, or disconnected."
        )
    elif isinstance(exc, GeecsDeviceCommandRejected):
        title = "Device Command Rejected"
        detail = (
            "The device did not acknowledge the command.\n"
            "It may be offline, busy, or the command was malformed."
        )
    elif isinstance(exc, GeecsDeviceCommandFailed):
        title = "Device Command Failed"
        detail = (
            f"The device acknowledged the command but reported an error:\n"
            f"    {exc.error_detail}"
        )
    elif isinstance(exc, ActionError):
        title = "Action Error"
        detail = str(exc)
    else:
        title = "Device Error"
        detail = str(exc)

    device_name = getattr(exc, "device_name", None)
    command = getattr(exc, "command", None)

    if device_name or command:
        header = (
            f"Device:   {device_name or 'unknown'}\n"
            f"Command:  {command or 'unknown'}\n\n"
        )
    else:
        header = ""

    body = (
        f"{header}"
        f"{detail}\n\n"
        f"Please manually verify or correct the hardware state, then:\n"
        f"  • Click  Continue  to proceed (the scan assumes the command succeeded)\n"
        f"  • Click  Abort  to stop the scan"
    )
    return title, body


# ---------------------------------------------------------------------------
# Main-thread dialog display  (call only from the Qt main thread)
# ---------------------------------------------------------------------------


def show_device_error_dialog(request: DialogRequest) -> None:
    """Show the device error dialog and write the result into *request*.

    Must be called from the Qt main thread.  Sets ``request.abort[0]``
    and ``request.response_event`` when the user responds.

    Parameters
    ----------
    request :
        The pending dialog request from the worker thread.
    """
    from PyQt5.QtWidgets import QApplication, QMessageBox

    title, body = _build_device_error_message(request.exc)
    logger.warning(
        "Showing device error dialog to user — %s: %s",
        type(request.exc).__name__,
        getattr(request.exc, "command", str(request.exc)),
    )

    if not QApplication.instance():
        # Headless fallback: auto-abort
        logger.error("No Qt application — auto-aborting on device error.")
        request.abort[0] = True
        request.response_event.set()
        return

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle(title)
    msg_box.setText(body)
    msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Abort)
    msg_box.button(QMessageBox.Ok).setText("Continue")
    msg_box.setDefaultButton(QMessageBox.Abort)

    response = msg_box.exec_()
    abort = response == QMessageBox.Abort

    logger.warning(
        "User responded to device error dialog: %s",
        "Abort" if abort else "Continue",
    )

    request.abort[0] = abort
    request.response_event.set()
