"""Qt dialog helpers for scan data-acquisition error handling.

This module must only be imported from the Qt main thread (or from modules
that are themselves only imported on the main thread).  The worker-thread side
is limited to :mod:`geecs_scanner.data_acquisition.gui_dialogs`, which has no
Qt dependency.
"""

from __future__ import annotations

import logging

from geecs_scanner.data_acquisition.dialog_request import DialogRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def _build_device_error_message(exc: Exception) -> tuple[str, str]:
    """Return (title, body) for a device error dialog.

    Parameters
    ----------
    exc :
        One of GeecsDeviceExeTimeout, GeecsDeviceCommandRejected,
        GeecsDeviceCommandFailed, ActionError, or a generic Exception.

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
        f"  \u2022 Click  Continue  to proceed (the scan assumes the command succeeded)\n"
        f"  \u2022 Click  Abort  to stop the scan"
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
