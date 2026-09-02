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

import functools
import logging
import threading
from typing import TYPE_CHECKING, Callable, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

if TYPE_CHECKING:
    from geecs_console.services.health import HealthProbe

logger = logging.getLogger(__name__)


#: Workers with a call in flight, held on the GUI thread until the result
#: has landed there (see ``BackgroundResult._forward``).  The daemon
#: thread's one cross-thread emission targets the worker itself, so the
#: worker must outlive that emission whatever the consumer does meanwhile.
_INFLIGHT: set = set()

#: Sentinel routed through the GUI hop when the callable raised (no
#: ``result_ready`` emission, but the in-flight hold is still released).
_FAILED = object()


class BackgroundResult(QObject):
    """Runs one blocking callable on a daemon thread and reports its result.

    The ``HealthPoller`` shape, generalized: the worker lives on the
    GUI thread, each :meth:`run_async` call spawns a short-lived daemon
    thread, and the result comes back through :attr:`result_ready`,
    **emitted on the GUI thread**.  The daemon thread never emits toward
    the consumer: it hops the result onto the GUI thread through the
    worker's own queued ``_landed`` signal (receiver = this worker, held
    alive in :data:`_INFLIGHT` until the hop completes), and
    :meth:`_forward` re-emits ``result_ready`` there.  Emitting from a
    daemon thread straight at a consumer QObject races the consumer's
    destruction — Qt's C++ connection bookkeeping survives that, PySide's
    Python-slot delivery does not, and it segfaults under offscreen pytest
    (first seen with a window-owned signal and the idle scan-number probe;
    again in 0.28.0 with a controller-owned slot, the queue panel's fetch
    landing while a test window was torn down).  Consumers connect
    ``result_ready`` ``QueuedConnection`` as before; it now merely
    defers the call by one event-loop turn.
    """

    result_ready = Signal(object)
    """Carries the callable's return value, one emission per finished call."""

    _landed = Signal(object)
    """The daemon thread → GUI thread hop (internal; receiver = self)."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._landed.connect(self._forward, Qt.ConnectionType.QueuedConnection)

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
        _INFLIGHT.add(self)
        threading.Thread(target=self._run, args=(func,), name=name, daemon=True).start()

    def _run(self, func: Callable[[], object]) -> None:
        """Call *func* (on the daemon thread) and hop the result to the GUI thread."""
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001 — background work is best-effort
            logger.info("background call failed: %s", exc)
            result = _FAILED
        try:
            self._landed.emit(result)
        except RuntimeError:
            pass  # the worker was deleted while the call ran

    @Slot(object)
    def _forward(self, result: object) -> None:
        """Re-emit on the GUI thread and release the in-flight hold (deferred).

        The hold is released one event-loop turn later rather than here:
        dropping the last reference to a QObject from inside its own slot
        would delete the C++ object under the running metacall.
        """
        QTimer.singleShot(0, functools.partial(_INFLIGHT.discard, self))
        if result is not _FAILED:
            self.result_ready.emit(result)


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
