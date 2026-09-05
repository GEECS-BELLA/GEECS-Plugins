"""The blessed daemon-thread → queued-signal workers (one-shot + interval).

:class:`BackgroundResult` (one-shot) lived in ``app/main_window.py`` until
the Actions menu needed it from a second window-family module
(``app/action_dialog.py``); this is the shared extraction recorded on issue
#510.  The scan browser uses it too (its former private twin,
``browser/_background.py::BrowserWorker``, was deleted once this module
landed).  :class:`HealthPoller` — the interval-polling shape
``BackgroundResult`` generalized — moved here from ``app/main_window.py``
in the issue #534 slimming (step 1).  Both ride the one GUI-thread hop,
:class:`_GuiHopWorker` (0.28.0 for ``BackgroundResult``, 0.28.2 for
``HealthPoller`` — issue #767).  :class:`GuiRelay` (0.30.0, #787) is the
same hop for **long-lived foreign-thread producers** — a CA monitor
callback, a zmq stream loop — that post many payloads over their lifetime
rather than one result per call; the scan monitor's stream workers
subclass the base directly for their typed signals.  Every console
signal that crosses a thread boundary now comes through here.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import threading
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from geecs_console.services.health import HealthReport

if TYPE_CHECKING:
    from geecs_console.services.health import HealthProbe

logger = logging.getLogger(__name__)


#: Workers with calls in flight (worker → count), held on the GUI thread
#: until each result has landed there (see ``_GuiHopWorker._forward``).
#: The daemon thread's one cross-thread emission targets the worker
#: itself, so the worker must outlive that emission whatever the consumer
#: does meanwhile.  A count, not a set: one worker may carry overlapping
#: calls (the now panel's per-experiment probes across an experiment
#: switch), and the first landing must not release the second's hold.
_INFLIGHT: dict = {}

#: The one-shot workers take their hold on the GUI thread (before the
#: thread spawns) and release it there; the long-lived producers
#: (:class:`GuiRelay`, the stream workers) take it *on their own thread*
#: per payload, so the counter's read-modify-write needs a lock.
_INFLIGHT_LOCK = threading.Lock()


def _hold(worker: QObject) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT[worker] = _INFLIGHT.get(worker, 0) + 1


def _release(worker: QObject) -> None:
    with _INFLIGHT_LOCK:
        remaining = _INFLIGHT.get(worker, 0) - 1
        if remaining > 0:
            _INFLIGHT[worker] = remaining
        else:
            _INFLIGHT.pop(worker, None)


def disconnect_quietly(signal: Any, slot: Any = None) -> None:
    """Disconnect *signal* (from *slot*, or from everything) without noise.

    A teardown helper: a never-connected signal's blind ``disconnect()``
    raises nothing in PySide6 but *warns* (``libpyside: Failed to
    disconnect (None) from signal …``, a ``RuntimeWarning``), and a
    signal on a half-torn-down object raises ``RuntimeError``/``TypeError``.
    Dispose paths want neither.

    Parameters
    ----------
    signal : Signal instance
        The bound signal.
    slot : callable, optional
        The receiver to detach; ``None`` detaches every receiver.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            if slot is None:
                signal.disconnect()
            else:
                signal.disconnect(slot)
        except (RuntimeError, TypeError):
            pass


#: Sentinel routed through the GUI hop when the callable raised (no
#: ``result_ready`` emission, but the in-flight hold is still released).
_FAILED = object()


class _GuiHopWorker(QObject):
    """The daemon thread → GUI thread hop both workers share.

    The daemon thread never emits toward the consumer: it calls
    :meth:`_hop`, which emits the worker's own queued ``_landed`` signal
    (receiver = this worker, held alive in :data:`_INFLIGHT` by the
    subclass until the hop completes), and :meth:`_forward` runs on the
    GUI thread, releases the hold and hands the payload to the subclass's
    :meth:`_deliver`.  Emitting from a daemon thread straight at a
    consumer QObject races the consumer's destruction — Qt's C++
    connection bookkeeping survives that, PySide's Python-slot delivery
    does not, and it segfaults under offscreen pytest (first seen with a
    window-owned signal and the idle scan-number probe; again in 0.28.0
    with a controller-owned slot, the queue panel's fetch landing while a
    test window was torn down; then with ``HealthPoller`` → window, the
    busiest emitter, in isolated ``test_main_window.py`` runs — #767;
    the stream workers, the aioca readback callback and the editors'
    completions fetch followed in 0.30.0 — #787).

    Two ways to take the hold: the one-shot workers call :func:`_hold`
    on the GUI thread before spawning their thread and then :meth:`_hop`
    from it; a long-lived producer on a foreign thread calls
    :meth:`_post` per payload, which takes the hold and hops in one step.
    """

    _landed = Signal(object)
    """The daemon thread → GUI thread hop (internal; receiver = self)."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._landed.connect(self._forward, Qt.ConnectionType.QueuedConnection)

    def _hop(self, payload: object) -> None:
        """Send *payload* to the GUI thread (called on the daemon thread)."""
        try:
            self._landed.emit(payload)
        except RuntimeError:
            # A Qt-parented worker was deleted with its parent while the
            # call ran (the hold protects unparented workers only): drop
            # the dead wrapper's hold so the registry does not grow.
            with _INFLIGHT_LOCK:
                _INFLIGHT.pop(self, None)

    def _post(self, payload: object) -> None:
        """Hold + hop in one step — for producers living on a foreign thread.

        A CA monitor callback or a stream loop has no GUI-thread moment
        before each payload to take the hold in, so it is taken here, on
        the producer's thread (:data:`_INFLIGHT_LOCK` makes that safe).
        """
        _hold(self)
        self._hop(payload)

    @Slot(object)
    def _forward(self, payload: object) -> None:
        """Release the in-flight hold (deferred) and deliver on the GUI thread.

        The hold is released one event-loop turn later rather than here:
        dropping the last reference to a QObject from inside its own slot
        would delete the C++ object under the running metacall.
        """
        QTimer.singleShot(0, functools.partial(_release, self))
        self._deliver(payload)

    def _deliver(self, payload: object) -> None:
        """Emit the public signal (GUI thread); subclasses implement."""
        raise NotImplementedError


class BackgroundResult(_GuiHopWorker):
    """Runs one blocking callable on a daemon thread and reports its result.

    The ``HealthPoller`` shape, generalized: the worker lives on the
    GUI thread, each :meth:`run_async` call spawns a short-lived daemon
    thread, and the result comes back through :attr:`result_ready`,
    **emitted on the GUI thread** via the :class:`_GuiHopWorker` hop.
    Consumers connect ``result_ready`` ``QueuedConnection`` as before; it
    now merely defers the call by one event-loop turn.
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
        _hold(self)
        threading.Thread(target=self._run, args=(func,), name=name, daemon=True).start()

    def _run(self, func: Callable[[], object]) -> None:
        """Call *func* (on the daemon thread) and hop the result to the GUI thread."""
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001 — background work is best-effort
            logger.info("background call failed: %s", exc)
            result = _FAILED
        self._hop(result)

    def _deliver(self, payload: object) -> None:
        if payload is not _FAILED:
            self.result_ready.emit(payload)


class HealthPoller(_GuiHopWorker):
    """Runs ``probe.poll()`` off the GUI thread and reports the result.

    The interval-polling variant of the same daemon-thread → GUI-hop
    pattern as :class:`BackgroundResult` (which generalized this shape):
    the poller itself lives on the GUI thread; each :meth:`poll_async` call
    spawns a short-lived daemon thread that runs the (possibly slow)
    blocking ``poll()`` and hops the report onto the GUI thread, where
    :attr:`report_ready` is emitted — so the chips update without ever
    blocking the event loop, and there is no worker Qt event loop or
    cross-thread QTimer to manage.  Unlike the one-shot worker it skips a
    poll while one is already in flight; "in flight" lasts until the
    report has landed on the GUI thread, so the ``_busy`` flag is only
    ever written there and at most one hop per poller is ever pending.

    Every poll gets a 1-based **sequence** (:attr:`polls_started`, GUI
    thread), and a delivered :class:`~geecs_console.services.health.HealthReport`
    is stamped with its poll's sequence — exact, because one poll is in
    flight at a time and the stamp is applied on the GUI thread before
    the next can start.  A consumer that records ``polls_started`` at
    some GUI-thread moment can then tell a report from a poll that began
    before it (the window's gateway-restart narration; review of PR
    #796).  Non-``HealthReport`` payloads (test probes) pass through
    untouched.

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
        self._sequence = 0

    @property
    def polls_started(self) -> int:
        """How many polls have started (GUI-thread state); the last one's sequence."""
        return self._sequence

    @Slot()
    def poll_async(self) -> None:
        """Kick off one poll in a daemon thread (skipped if one is in flight).

        Called on the GUI thread from the interval timer; returns immediately.
        """
        if self._busy:
            return
        self._busy = True
        self._sequence += 1
        _hold(self)
        threading.Thread(
            target=self._run, name="console-health-poll", daemon=True
        ).start()

    def _run(self) -> None:
        """Poll the probe (on the daemon thread) and hop the report to the GUI thread."""
        try:
            report = self._probe.poll()
        except Exception:  # noqa: BLE001 — a probe fault must not kill the poller
            report = None
        self._hop(report)

    def _deliver(self, payload: object) -> None:
        # GUI thread, before _busy clears: the in-flight poll IS the latest
        # started one, so its sequence is exactly self._sequence.
        if isinstance(payload, HealthReport):
            payload = dataclasses.replace(payload, sequence=self._sequence)
        self._busy = False
        if payload is not None:
            self.report_ready.emit(payload)


class GuiRelay(_GuiHopWorker):
    """Hop payloads posted from a foreign thread onto the GUI thread.

    For long-lived producers that call back on their own thread for the
    life of a subscription — the movable panel's aioca readback monitors
    (values arrive on the CA loop thread) — where neither one-shot worker
    fits.  The producer calls :meth:`post` from its thread; the payload
    lands on the GUI thread and :attr:`delivered` is emitted there.  The
    owner connects ``delivered`` (``QueuedConnection`` by convention) and
    calls :meth:`close` in its dispose path: a payload still in flight at
    that moment lands and is dropped rather than emitted, so nothing
    reaches a consumer that is being torn down.
    """

    delivered = Signal(object)
    """One emission per posted payload, on the GUI thread."""

    def __init__(self) -> None:
        super().__init__()
        self._closed = False

    def post(self, payload: object) -> None:
        """Queue *payload* for GUI-thread delivery (any thread)."""
        if self._closed:
            return
        self._post(payload)

    def close(self) -> None:
        """Stop delivering (idempotent; GUI thread) — in-flight payloads drop."""
        self._closed = True

    def _deliver(self, payload: object) -> None:
        if not self._closed:
            self.delivered.emit(payload)
