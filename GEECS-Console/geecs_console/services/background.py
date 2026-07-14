"""The one blessed daemon-thread → queued-signal worker for one-shot calls.

:class:`BackgroundResult` lived in ``app/main_window.py`` until the Actions
menu needed it from a second window-family module (``app/action_dialog.py``);
this is the shared extraction recorded on issue #510.  The browser's private
twin (``browser/_background.py::BrowserWorker``) is still pending its
mechanical swap onto this class.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class BackgroundResult(QObject):
    """Runs one blocking callable on a daemon thread and reports its result.

    The ``HealthPoller`` shape, generalized: the worker lives on the
    GUI thread, each :meth:`run_async` call spawns a short-lived daemon
    thread, and the result comes back through the queued
    :attr:`result_ready` signal.  Crucially the daemon thread emits on
    *this worker*, never on the main window — emitting a window-owned
    signal from a daemon thread races window teardown and segfaults under
    offscreen pytest (observed with the idle scan-number probe).
    """

    result_ready = Signal(object)
    """Carries the callable's return value, one emission per finished call."""

    def run_async(self, func: Callable[[], object], name: str) -> None:
        """Run *func* on a fresh daemon thread and emit its result.

        Parameters
        ----------
        func : callable
            Zero-argument blocking callable.  Exceptions are logged and
            swallowed (no emission), so wrap the call when a failure result
            should still be delivered.
        name : str
            The daemon thread's name (debugging).
        """
        threading.Thread(target=self._run, args=(func,), name=name, daemon=True).start()

    def _run(self, func: Callable[[], object]) -> None:
        """Call *func* (on the daemon thread) and emit the result."""
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001 — background work is best-effort
            logger.info("background call failed: %s", exc)
            return
        try:
            self.result_ready.emit(result)
        except RuntimeError:
            pass  # the worker was deleted while the call ran
