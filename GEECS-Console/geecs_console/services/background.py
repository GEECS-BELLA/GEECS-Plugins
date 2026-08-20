"""The blessed daemon-thread → queued-signal workers (one-shot + interval).

:class:`BackgroundResult` (one-shot) lived in ``app/main_window.py`` until
the Actions menu needed it from a second window-family module
(``app/action_dialog.py``); this is the shared extraction recorded on issue
#510.  The scan browser uses it too (its former private twin,
``browser/_background.py::BrowserWorker``, was deleted once this module
landed).  :class:`HealthPoller` — the interval-polling shape
``BackgroundResult`` generalized — moved here from ``app/main_window.py``
in the issue #534 slimming (step 1).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QObject, Signal, Slot

if TYPE_CHECKING:
    from geecs_console.services.health import HealthProbe

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


class HealthPoller(QObject):
    """Runs ``probe.poll()`` off the GUI thread and reports the result.

    The interval-polling variant of the same daemon-thread → queued-signal
    pattern as :class:`BackgroundResult` (which generalized this shape):
    the poller itself lives on the GUI thread; each :meth:`poll_async` call
    spawns a short-lived daemon thread that runs the (possibly slow)
    blocking ``poll()`` and emits :attr:`report_ready` with the result.  Qt
    marshals the emit back to the GUI-thread slot as a queued delivery, so
    the chips update without ever blocking the event loop — and there is no
    worker Qt event loop or cross-thread QTimer to manage.  Unlike the
    one-shot worker it skips a poll while one is already in flight.

    Parameters
    ----------
    probe :
        The probe to poll; only its ``poll()`` method is used, so it works
        with the real probe (:class:`~geecs_console.services.health.HealthProbe`),
        the stub, or a test fake.
    """

    report_ready = Signal(object)
    """Carries one :class:`~geecs_console.services.health.HealthReport` per poll."""

    def __init__(self, probe: "HealthProbe") -> None:
        super().__init__()
        self._probe = probe
        self._busy = False

    @Slot()
    def poll_async(self) -> None:
        """Kick off one poll in a daemon thread (skipped if one is in flight).

        Called on the GUI thread from the interval timer; returns immediately.
        """
        if self._busy:
            return
        self._busy = True
        threading.Thread(
            target=self._run, name="console-health-poll", daemon=True
        ).start()

    def _run(self) -> None:
        """Poll the probe (on the daemon thread) and emit the report."""
        report = None
        try:
            report = self._probe.poll()
        except Exception:  # noqa: BLE001 — a probe fault must not kill the poller
            report = None
        finally:
            self._busy = False
        if report is not None:
            self.report_ready.emit(report)
