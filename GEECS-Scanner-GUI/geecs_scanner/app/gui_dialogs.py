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
        One of GeecsDeviceInstantiationError, GeecsDeviceExeTimeout,
        GeecsDeviceCommandRejected, GeecsDeviceCommandFailed, ActionError,
        or a generic Exception.

    Returns
    -------
    tuple[str, str]
        Dialog window title and body text.
    """
    from geecs_python_api.controls.interface.geecs_errors import (
        GeecsDeviceCommandFailed,
        GeecsDeviceCommandRejected,
        GeecsDeviceExeTimeout,
        GeecsDeviceInstantiationError,
    )
    from geecs_scanner.utils.exceptions import ActionError

    if isinstance(exc, GeecsDeviceInstantiationError):
        title = "Device Instantiation Failed"
        detail = (
            f"{exc}\n\n"
            f"The device may be offline, not found in the database, or\n"
            f"unreachable on the network."
        )
    elif isinstance(exc, GeecsDeviceExeTimeout):
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
    if request.context:
        body += f"\n\n{request.context}"
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


def show_action_error_dialog(exc: Exception) -> None:
    """Show a simple warning dialog for an action-execution error.

    Unlike :func:`show_device_error_dialog`, this has no Continue/Abort
    choice — the action has already failed and there is nothing to abort.
    Must be called from the Qt main thread.

    Parameters
    ----------
    exc :
        The exception that caused the action to fail.  ``str(exc)`` is
        shown directly, so the device name is included as long as the
        exception was constructed with it (as all GEECS device errors are).
    """
    from PyQt5.QtWidgets import QApplication, QMessageBox

    logger.warning(
        "Action failed — showing error dialog: %s: %s", type(exc).__name__, exc
    )

    if not QApplication.instance():
        logger.error("No Qt application — cannot show action error dialog.")
        return

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Action Failed")
    msg_box.setText(f"The action could not be completed:\n\n{exc}")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.setDefaultButton(QMessageBox.Ok)
    msg_box.exec_()
