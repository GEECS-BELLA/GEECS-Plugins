"""Thread-safe dialog request type and device-error escalation helpers.

Worker threads that encounter a device error create a :class:`DialogRequest`,
put it on ``ScanManager.dialog_queue``, and block on ``response_event.wait()``.
The Qt main thread (via the 200 ms QTimer in ``GEECSScannerWindow``) drains the
queue and shows the dialog.  The actual Qt code lives in
``geecs_scanner.app.gui_dialogs`` so this module stays import-safe in headless
environments.

Issue #312 tracking note: the old module called Qt directly from worker threads
(unsafe).  The queue-based pattern implemented here resolves that.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceCommandFailed,
    GeecsDeviceCommandRejected,
    GeecsDeviceExeTimeout,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convenience tuple — use in except clauses to catch any device command error
# ---------------------------------------------------------------------------

DEVICE_COMMAND_ERRORS = (
    GeecsDeviceExeTimeout,
    GeecsDeviceCommandRejected,
    GeecsDeviceCommandFailed,
)


# ---------------------------------------------------------------------------
# Escalation helper
# ---------------------------------------------------------------------------


def escalate_device_error(
    exc: Exception,
    on_escalate: Optional[Callable[[Exception], bool]],
) -> bool:
    """Call *on_escalate* with *exc* and return its result.

    If no callback is wired (headless / test context), the error is logged
    and ``True`` (abort) is returned so the caller can stop the scan safely.

    Parameters
    ----------
    exc :
        The device exception to escalate.
    on_escalate :
        Callable that submits *exc* to the GUI dialog queue and blocks until
        the user responds.  Returns ``True`` → Abort, ``False`` → Continue.

    Returns
    -------
    bool
        ``True`` if the scan should be aborted, ``False`` to continue.
    """
    if on_escalate is not None:
        return on_escalate(exc)
    logger.error("Device error with no escalation callback — auto-aborting: %s", exc)
    return True


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
