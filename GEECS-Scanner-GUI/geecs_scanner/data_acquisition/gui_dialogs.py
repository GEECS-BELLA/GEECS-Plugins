"""GUI dialog helpers for scan data-acquisition error handling.

These dialogs are currently shown from the scan worker thread, which is
technically unsafe for Qt GUI operations.  See issue #312 for the planned
refactor to emit signals and show dialogs on the main GUI thread instead.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def prompt_user_device_timeout(device_name: str, command: str, timeout: float) -> bool:
    """Show a warning dialog when a device exe response times out after all retries.

    Blocks the calling (scan worker) thread until the user responds.

    Parameters
    ----------
    device_name : str
        Name of the device that timed out.
    command : str
        The command string that was sent (e.g. ``'setTrigger>>SCAN'``).
    timeout : float
        The timeout that elapsed, in seconds.

    Returns
    -------
    bool
        ``True`` if the user chose to abort the scan, ``False`` to continue.

    Notes
    -----
    See issue #312 — this dialog is shown from the scan worker thread, which
    is technically unsafe.  It will be moved to the main GUI thread via signals
    in a future refactor.
    """
    from PyQt5.QtWidgets import QApplication, QMessageBox

    if not QApplication.instance():
        QApplication(sys.argv)

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Device Timeout")
    msg_box.setText(
        f"Device '{device_name}' did not respond to command:\n"
        f"    {command}\n\n"
        f"No execution response received within {timeout:.1f}s.\n\n"
        f"Please fix the hardware manually, then choose Continue to proceed "
        f"with the scan or Abort to stop it."
    )
    msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Abort)
    msg_box.button(QMessageBox.Ok).setText("Continue")
    msg_box.setDefaultButton(QMessageBox.Abort)
    response = msg_box.exec_()
    return response == QMessageBox.Abort
