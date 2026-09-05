"""Scan monitoring for the queueserver path (#648): status poll + two streams.

Owns the three live inputs the R6 now-panel renders once the console is a
manager client:

- **Manager status poll** — ``QueueClient.status()`` on the shared
  :class:`~geecs_console.services.background.HealthPoller` (GUI-thread
  ``QTimer`` + short-lived daemon thread; in-flight skip).  ``re_state``
  drives the state pill.
- **Document stream** — a ``bluesky.callbacks.zmq.RemoteDispatcher`` on
  the worker's document proxy (per-shot progress, totals, scan number,
  terminal states; re-drives the per-shot beeps).
- **Console-output stream** — the manager's captured stdout/stderr text
  (the log tail), watched for the engine's failed-move reason line
  (``FAILED_MOVE_LOG_PREFIX``) so a ``paused`` pill can say *why*.

Controller rules (#534 checklist): ``QObject`` with **no Qt parent**;
injected addresses and callables, never the window; **no daemon thread
emits toward a consumer** — each stream worker is a
``services/background.py::_GuiHopWorker``: the zmq thread posts a tagged
payload through the worker's own queued hop and the public signal
(``document`` / ``line`` / ``pause_reason`` / ``stream_failed``) is
emitted on the GUI thread (#787 — the pre-0.30.0 shape emitted from the
zmq thread at the window, the race class that segfaulted offscreen
pytest); an idempotent :meth:`dispose` the window's ``closeEvent`` calls.
The stream sockets are long-lived daemon threads: ``dispose()`` marks them
stale (posting stops, and a payload already in flight lands and is
dropped), severs the public signals, and never joins.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from geecs_console.services.background import _GuiHopWorker, disconnect_quietly

logger = logging.getLogger(__name__)

#: The engine's failed-move reason line prefix (pause_semantics contract).
#: Imported lazily where used; this fallback keeps the module import-safe
#: and the tests hermetic. Keep in sync with
#: ``geecs_bluesky.plans.pause_semantics.FAILED_MOVE_LOG_PREFIX``.
_FAILED_MOVE_PREFIX_FALLBACK = "FAILED MOVE - pausing for operator"


def _failed_move_prefix() -> str:
    """The engine's prefix when importable, else the pinned fallback."""
    try:
        from geecs_bluesky.plans.pause_semantics import FAILED_MOVE_LOG_PREFIX

        return FAILED_MOVE_LOG_PREFIX
    except Exception:  # pragma: no cover - engine always present in prod
        return _FAILED_MOVE_PREFIX_FALLBACK


class _StreamWorker(_GuiHopWorker):
    """Shared shape of the two stream workers: a gated, abandoned thread.

    ``start()`` spawns the daemon thread (idempotent); :meth:`stop` only
    gates emission and **abandons** the thread: the socket/loop inside
    must never be touched from another thread (pyzmq sockets are not
    thread-safe — a cross-thread close can trip a libzmq assertion, which
    *aborts* the process rather than raising; #653 review finding 3), and
    the thread dies with the app anyway.  The gate is checked twice: on
    the zmq thread before posting, and again in :meth:`_deliver` on the
    GUI thread — so after ``stop()`` nothing is emitted, not even a
    payload that was already in flight.
    """

    #: Public signals a consumer may have connected (severed by
    #: :meth:`sever`; the internal hop connection is left alone).
    PUBLIC_SIGNALS: tuple[str, ...] = ("stream_failed",)

    stream_failed = Signal(str)

    def __init__(self, addr: str, thread_name: str) -> None:
        super().__init__()
        self._addr = addr
        self._thread_name = thread_name
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Spawn the stream thread (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=self._thread_name, daemon=True
        )
        self._thread.start()

    def _run(self) -> None:  # pragma: no cover — subclasses own the socket loop
        raise NotImplementedError

    def stop(self) -> None:
        """Gate further emissions (idempotent; touches nothing else, never joins)."""
        self._stopped = True

    def sever(self) -> None:
        """Detach every consumer from the public signals (GUI thread, idempotent).

        Deliberately **not** ``QObject.disconnect()`` on the whole worker:
        that would also cut the internal ``_landed`` → ``_forward`` hop,
        stranding an in-flight payload's hold forever.
        """
        for name in self.PUBLIC_SIGNALS:
            disconnect_quietly(getattr(self, name))

    def _fail(self, message: str) -> None:
        """Post a setup failure (zmq thread) unless already stopped."""
        if not self._stopped:
            self._post(("failed", message))

    def _deliver(self, payload: object) -> None:
        """Dispatch one tagged payload on the GUI thread (gated on ``stop``)."""
        if self._stopped:
            return
        tag, *args = payload  # type: ignore[misc]
        if tag == "failed":
            self.stream_failed.emit(*args)
        else:
            self._emit(tag, *args)

    def _emit(self, tag: str, *args: Any) -> None:  # pragma: no cover — subclass
        raise NotImplementedError


class DocumentStreamWorker(_StreamWorker):
    """Daemon-thread consumer of the worker's bluesky document stream.

    Emits ``document(name, doc)`` for every document received and
    ``stream_failed(str)`` if the stream cannot be set up (bad address,
    missing import) — post-setup, the zmq SUB socket reconnects on its own
    and needs no supervision.  Both signals are emitted **on the GUI
    thread** (the :class:`_StreamWorker` hop); consumers connect them
    ``QueuedConnection`` as before.
    """

    PUBLIC_SIGNALS = ("document", "stream_failed")

    document = Signal(str, object)

    def __init__(self, doc_addr: str) -> None:
        super().__init__(doc_addr, "qs-doc-stream")

    def _run(self) -> None:
        try:
            from bluesky.callbacks.zmq import RemoteDispatcher

            dispatcher = RemoteDispatcher(self._addr)

            def _forward(name: str, doc: dict) -> None:
                if not self._stopped:
                    self._post(("document", name, dict(doc)))

            dispatcher.subscribe(_forward)
            dispatcher.start()  # blocks for the thread's lifetime
        except Exception as exc:
            if not self._stopped:
                logger.warning("document stream at %s ended", self._addr, exc_info=True)
                self._fail(
                    f"document stream unavailable ({exc}) — live "
                    "per-shot progress is off"
                )

    def _emit(self, tag: str, *args: Any) -> None:
        self.document.emit(*args)


class ConsoleStreamWorker(_StreamWorker):
    """Daemon-thread consumer of the manager's console-output text stream.

    Emits ``line(str)`` per received line, ``pause_reason(str)`` when a
    line carries the engine's failed-move prefix — the paused pill's "why"
    — and ``stream_failed(str)`` if the monitor cannot be set up.  All on
    the GUI thread (the :class:`_StreamWorker` hop).  Text only: this
    stream never carries documents (see qserver/README.md).
    """

    PUBLIC_SIGNALS = ("line", "pause_reason", "stream_failed")

    line = Signal(str)
    pause_reason = Signal(str)

    def __init__(self, info_addr: str) -> None:
        super().__init__(info_addr, "qs-console-stream")

    def _run(self) -> None:
        try:
            from bluesky_queueserver_api.console_monitor import (
                ConsoleMonitor_ZMQ_Threads,
            )

            monitor = ConsoleMonitor_ZMQ_Threads(
                zmq_info_addr=self._addr,
                zmq_encoding="json",
                poll_timeout=0.5,
                max_msgs=10_000,
                max_lines=1_000,
            )
            monitor.enable()
            prefix = _failed_move_prefix()
            try:
                while not self._stopped:
                    try:
                        msg = monitor.next_msg(timeout=0.5)
                    except Exception:  # timeout — poll the stop flag
                        continue
                    self._handle_text(str(msg.get("msg", "")), prefix)
            finally:
                monitor.disable()
        except Exception as exc:
            if not self._stopped:
                logger.warning("console stream at %s ended", self._addr, exc_info=True)
                self._fail(
                    f"console-output stream unavailable ({exc}) — the "
                    "log tail and pause reasons are off"
                )

    def _handle_text(self, text: str, prefix: str) -> None:
        """Split *text* into lines and post each (plus any pause reason)."""
        for raw_line in text.splitlines():
            stripped = raw_line.rstrip()
            if not stripped or self._stopped:
                continue
            self._post(("line", stripped))
            if prefix in stripped:
                # Everything after the prefix is the engine's reason text.
                reason = stripped.split(prefix, 1)[1].lstrip(" :-")
                self._post(("pause_reason", reason or stripped))

    def _emit(self, tag: str, *args: Any) -> None:
        if tag == "line":
            self.line.emit(*args)
        elif tag == "pause_reason":
            self.pause_reason.emit(*args)


class _StatusProbe:
    """Adapter: ``QueueClient.status`` as the ``HealthPoller`` probe shape."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def poll(self) -> Any:
        """One manager status snapshot (never raises — client contract)."""
        return self._client.status()


class ScanMonitorController(QObject):
    """Owns the poller and both stream workers; the window consumes signals.

    Parameters
    ----------
    queue_client :
        The :class:`~geecs_bluesky.qs_client.QueueClient` whose
        ``status()`` the poller runs.
    info_addr, doc_addr :
        Stream addresses (``None`` disables that stream — the stub/offline
        case).
    poll_interval_ms :
        Manager status poll cadence. 1 s default: snappy enough for the
        state pill, far below the manager's cost floor.
    """

    def __init__(
        self,
        queue_client: Any,
        *,
        info_addr: Optional[str] = None,
        doc_addr: Optional[str] = None,
        poll_interval_ms: int = 1_000,
    ) -> None:
        super().__init__()
        from geecs_console.services.background import HealthPoller

        self._poller = HealthPoller(_StatusProbe(queue_client))
        self._poll_interval_ms = poll_interval_ms
        self._timer: Any = None
        self.documents: Optional[DocumentStreamWorker] = (
            DocumentStreamWorker(doc_addr) if doc_addr else None
        )
        self.console: Optional[ConsoleStreamWorker] = (
            ConsoleStreamWorker(info_addr) if info_addr else None
        )
        self._disposed = False

    @property
    def status_ready(self) -> Any:
        """The poller's queued result signal (``report_ready(object)``)."""
        return self._poller.report_ready

    def start(self, timer_parent: Any) -> None:
        """Begin polling (QTimer on *timer_parent*, the window) and streaming.

        The timer lives on the GUI thread via *timer_parent* — the health
        poller pattern; controllers themselves have no Qt parent.
        """
        from PySide6.QtCore import QTimer

        if self._disposed or self._timer is not None:
            return
        self._timer = QTimer(timer_parent)
        self._timer.timeout.connect(self._poller.poll_async)
        self._timer.start(self._poll_interval_ms)
        self._poller.poll_async()
        if self.documents is not None:
            self.documents.start()
        if self.console is not None:
            self.console.start()

    def dispose(self) -> None:
        """Stop the timer and streams; sever consumers (idempotent)."""
        if self._disposed:
            return
        self._disposed = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        for worker in (self.documents, self.console):
            if worker is not None:
                worker.stop()
                worker.sever()
        # Quietly: a monitor whose status_ready nobody connected (tests)
        # would otherwise warn "Failed to disconnect (None)".
        disconnect_quietly(self._poller.report_ready)
