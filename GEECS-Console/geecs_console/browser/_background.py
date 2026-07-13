"""Thin private worker shim for the scan browser — daemon thread + queued signal.

A minimal twin of the main window's ``BackgroundResult`` (the one blessed
daemon-thread → queued-signal worker), duplicated here because the browser
must not import from ``app/main_window.py`` (another session owns that
file).  **Temporary**: once issue #510's PR lands the shared
``services/background.py``, this module is deleted and the browser imports
that instead — the API here is kept deliberately tiny (one class, one
signal, one method) so the swap is mechanical (see the coordination
comment on #510).

The rules it encodes (the package's no-QThread policy):

- The worker QObject lives on the GUI thread; each call spawns a
  short-lived **daemon** ``threading.Thread`` — never a QThread.
- The daemon thread emits on the *worker*, never on a window-owned signal
  (emitting a window signal from a daemon thread races window teardown and
  segfaults under offscreen pytest).
- Consumers connect ``result_ready`` with an explicit
  ``Qt.ConnectionType.QueuedConnection`` to a ``@Slot(object)`` so results
  are always delivered on the GUI thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class BrowserWorker(QObject):
    """Runs one blocking callable on a daemon thread; reports via a signal.

    Exceptions are not swallowed silently: a failing callable emits
    ``(None, str(exc))`` through :attr:`result_ready` so the window can
    surface catalog errors in the status bar instead of hanging a spinner.
    """

    result_ready = Signal(object)
    """Carries ``(result, error)`` — exactly one is non-None per emission."""

    def run_async(self, func: Callable[[], object], name: str) -> None:
        """Run *func* on a fresh daemon thread and emit its outcome.

        Parameters
        ----------
        func : callable
            Zero-argument blocking callable (a catalog method, typically).
        name : str
            The daemon thread's name (debugging).
        """
        threading.Thread(target=self._run, args=(func,), name=name, daemon=True).start()

    def _run(self, func: Callable[[], object]) -> None:
        """Call *func* (on the daemon thread) and emit ``(result, error)``."""
        try:
            outcome: tuple[object, object] = (func(), None)
        except Exception as exc:  # noqa: BLE001 — reported, never fatal
            logger.info("browser background call failed: %s", exc)
            outcome = (None, str(exc))
        try:
            self.result_ready.emit(outcome)
        except RuntimeError:
            pass  # the worker was deleted while the call ran
