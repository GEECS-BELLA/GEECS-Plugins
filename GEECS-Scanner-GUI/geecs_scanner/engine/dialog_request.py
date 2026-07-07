"""Thread-safe dialog request type and device-error escalation helpers.

Worker threads that need an operator decision create a :class:`DialogRequest`,
emit it inside a ``ScanDialogEvent`` through the ``on_event`` callback, and
block on ``response_event.wait()``.  The GUI receives the event on the Qt main
thread (via the ``_scan_event_received`` pyqtSignal), shows the dialog, and
answers by writing ``abort[0]`` and setting ``response_event``.  The actual Qt
code lives in ``geecs_scanner.app.gui_dialogs`` so this module stays
import-safe in headless environments.

Both scan backends use this channel: the legacy engine for device-command
escalation, and ``BlueskyScanner`` (GeecsBluesky) for pre-flight operator
dialogs — the latter imports this module defensively, so it must stay free of
Qt and of heavyweight imports.

Issue #312 tracking note: the old module called Qt directly from worker threads
(unsafe).  The request/response pattern implemented here resolves that.
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
    on_escalate: Optional[Callable[..., bool]],
    context: Optional[str] = None,
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
    context :
        Optional extra information shown in the dialog body — e.g. the full
        list of variables that were being set for a device when the error
        occurred.

    Returns
    -------
    bool
        ``True`` if the scan should be aborted, ``False`` to continue.
    """
    if on_escalate is not None:
        return on_escalate(exc, context=context)
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
        The exception that triggered the dialog.  For requests carrying
        their own ``title``/labels, ``str(exc)`` is the full operator-facing
        dialog body.
    context :
        Optional extra information shown in the dialog body — e.g. the full
        list of variables being set for a device when the error occurred.
    title :
        Optional dialog window title.  ``None`` (legacy device-command
        requests) keeps the title derived from the exception type.
    continue_label :
        Optional text for the non-abort button (e.g. ``"Drop && Continue"``).
        ``None`` keeps the default ``"Continue"``.  When set, the dialog body
        is ``str(exc)`` verbatim — the request author owns the wording of
        what "continue" means.
    abort_label :
        Optional text for the abort button.  ``None`` keeps ``"Abort"``.
    response_event :
        Set by the main thread once the user has responded.
    abort :
        Single-element list used as a mutable result container.
        ``True`` → user chose Abort; ``False`` → user chose Continue.
    """

    exc: Exception
    context: Optional[str] = None
    title: Optional[str] = None
    continue_label: Optional[str] = None
    abort_label: Optional[str] = None
    response_event: threading.Event = field(default_factory=threading.Event)
    abort: list[bool] = field(default_factory=lambda: [False])
