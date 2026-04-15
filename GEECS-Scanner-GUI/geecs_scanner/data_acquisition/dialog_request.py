"""Thread-safe dialog request type for the scan worker → main-thread bridge.

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

import threading
from dataclasses import dataclass, field


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
