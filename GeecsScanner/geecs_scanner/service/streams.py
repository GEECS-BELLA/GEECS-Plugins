"""The latest-run picture, fed by the two 0MQ streams a browser cannot read.

The worker's document stream on the proxy out-port is **pickled Python**
(qserver DEPLOYMENT.md: a non-Python subscriber "is not offered today"),
and the manager's console-output stream is 0MQ too.  So this process
consumes both, reduces them to one small picture, and the web layer hands
that picture to the page over Server-Sent Events as JSON.  The pickled
wire format never leaves the process.

The reduction is the console's, live-verified: the start document seeds
the scan number and the planned total (``num_points × shots_per_step``,
falling back to ``max_iterations`` for adaptive plans — the Scan013
lesson: never inherit the previous run's total), primary-stream events
advance ``shots_done``, the stop document records the exit.  A console
line carrying :data:`~geecs_bluesky.qs_client.FAILED_MOVE_LOG_PREFIX`
becomes the paused run's *why*.

Best-effort by design, with the console's threading rules (#653): the
consumer threads are daemons, ``stop`` does not exist — a zmq socket must
never be touched from another thread (a cross-thread close can abort the
process) — so the threads live for the process lifetime and emission is
gated by the cache's lock.  A stream that cannot be set up marks the
picture unavailable with a reason; the manager poll stays authoritative.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Optional

from geecs_scanner.service.models import ConsoleLine, ProgressOut

logger = logging.getLogger("geecs_scanner.streams")

#: Console lines kept for late-joining pages.
_CONSOLE_KEEP = 400

#: The streams whose events are shots.  A strict run's rows are ``primary``
#: events; a gated run's ``primary`` is datum-only and its per-shot rows are
#: the sampler's ``shots`` stream (phase 2, ``plans/gated.py``).  The same
#: pair the s-file writer counts (``geecs_bluesky.callbacks.ROW_STREAMS``);
#: ``tests/test_streams.py`` pins the two together.  Spelled here rather
#: than imported because ``callbacks`` drags the engine in.
ROW_STREAMS: tuple[str, ...] = ("primary", "shots")


class ProgressCache:
    """Lock-protected latest-run picture plus a ring of console lines.

    Feed it from the real streams with :meth:`ensure_started`, or directly
    with :meth:`on_document` / :meth:`push_console_line` (tests, the demo
    backend).  :meth:`version` changes whenever anything changed, so a
    poller can skip unchanged rounds.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._started = False
        self._doc_addr: Optional[str] = None
        self._available = False
        self._detail = "stream cache not started"
        self._state: dict[str, Any] = {}
        self._rows: set[str] = set()
        self._console: deque[ConsoleLine] = deque(maxlen=_CONSOLE_KEEP)
        self._seq = 0
        #: Distinguishes this process's sequence numbers from a restarted
        #: one's — a reconnecting page must not take a low seq for "seen".
        self.epoch = f"{int(clock() * 1000):x}"
        self._version = 0

    # ---------------------------------------------------------------- read

    def version(self) -> int:
        """A counter that changes with every update."""
        with self._lock:
            return self._version

    def snapshot(self) -> ProgressOut:
        """The picture as the page reads it."""
        with self._lock:
            return ProgressOut(
                available=self._available, detail=self._detail, **self._state
            )

    def console_since(self, seq: int, limit: int = 200) -> list[ConsoleLine]:
        """Console lines with ``seq`` greater than *seq*, oldest first."""
        with self._lock:
            return [ln for ln in self._console if ln.seq > seq][:limit]

    # ---------------------------------------------------------------- feed

    def mark_available(self, available: bool, detail: str = "") -> None:
        """Record whether the document stream is being consumed."""
        with self._lock:
            self._available = available
            self._detail = detail
            self._version += 1

    def on_document(self, name: str, doc: dict) -> None:
        """Reduce one bluesky document into the picture."""
        with self._lock:
            now = self._clock()
            if name == "start":
                self._rows = set()
                num_points = doc.get("num_points") or doc.get("max_iterations")
                shots = doc.get("shots_per_step")
                total: Optional[int] = None
                try:
                    if num_points and shots:
                        total = int(num_points) * int(shots)
                    elif doc.get("plan_name") == "count" and num_points:
                        total = int(num_points)
                except (TypeError, ValueError):
                    total = None
                tag = (
                    doc.get("scan_tag") if isinstance(doc.get("scan_tag"), dict) else {}
                )
                day = None
                try:
                    if tag.get("year") and tag.get("month") and tag.get("day"):
                        day = f"{int(tag['year']):04d}-{int(tag['month']):02d}-{int(tag['day']):02d}"
                except (TypeError, ValueError):
                    day = None
                self._state = {
                    "scan_number": _as_int(doc.get("scan_number")),
                    "scan_folder": str(doc["scan_folder"])
                    if doc.get("scan_folder")
                    else None,
                    "day": day,
                    "plan_name": doc.get("plan_name"),
                    "planned_total": total,
                    "shots_done": 0,
                    "state": "running",
                    "exit_status": None,
                    "paused_reason": None,
                    "updated_at": now,
                }
            elif name == "descriptor":
                if doc.get("name") in ROW_STREAMS and doc.get("uid"):
                    self._rows.add(str(doc["uid"]))
            elif name == "event":
                if str(doc.get("descriptor")) in self._rows:
                    seq = _as_int(doc.get("seq_num")) or 0
                    self._state["shots_done"] = max(
                        seq, int(self._state.get("shots_done") or 0)
                    )
                    # Progress proves the resume: a row after a failed-move
                    # pause means the operator resumed, so the reason and
                    # the paused word go (the MCP's finding #683-1).
                    if self._state.get("state") == "paused":
                        self._state["state"] = "running"
                        self._state["paused_reason"] = None
                    self._state["updated_at"] = now
            elif name == "stop":
                status = str(doc.get("exit_status") or "")
                self._state["exit_status"] = status
                self._state["state"] = "done" if status == "success" else "aborted"
                self._state["updated_at"] = now
            self._version += 1

    def push_console_line(
        self, text: str, failed_move_prefix: Optional[str] = None
    ) -> None:
        """Append one console line; a failed-move line sets the paused reason."""
        with self._lock:
            self._seq += 1
            self._console.append(
                ConsoleLine(seq=self._seq, text=text, at=self._clock())
            )
            if failed_move_prefix and failed_move_prefix in text:
                reason = text.split(failed_move_prefix, 1)[1].lstrip(" :-")
                self._state["paused_reason"] = reason or text
                self._state["state"] = "paused"
            self._version += 1

    # ----------------------------------------------------------- real streams

    def ensure_started(self, doc_addr: Optional[str], info_addr: Optional[str]) -> None:
        """Start both consumer threads once (idempotent, never raises).

        Once-only by design: the first call's addresses latch for the
        process lifetime; there is no way to retire a zmq consumer thread
        safely.  A later call with different addresses warns and is ignored.
        """
        with self._lock:
            if self._started:
                if doc_addr and doc_addr != self._doc_addr:
                    logger.warning(
                        "streams already consuming %r — new address %r ignored "
                        "(restart the service to re-point streams)",
                        self._doc_addr,
                        doc_addr,
                    )
                return
            self._started = True
            self._doc_addr = doc_addr
            if not doc_addr:
                self._detail = "no document-stream address configured"
                return
        threading.Thread(
            target=self._run_documents,
            args=(doc_addr,),
            name="geecs-scanner-docs",
            daemon=True,
        ).start()
        if info_addr:
            threading.Thread(
                target=self._run_console,
                args=(info_addr,),
                name="geecs-scanner-console",
                daemon=True,
            ).start()

    def _run_documents(self, doc_addr: str) -> None:
        try:
            from bluesky.callbacks.zmq import RemoteDispatcher

            dispatcher = RemoteDispatcher(doc_addr)
            dispatcher.subscribe(self.on_document)
            self.mark_available(True, "")
            dispatcher.start()  # blocks for the thread's lifetime
        except Exception as exc:  # noqa: BLE001 — best-effort by design
            logger.warning("document stream at %s unavailable: %s", doc_addr, exc)
            self.mark_available(False, f"document stream unavailable: {exc}")

    def _run_console(self, info_addr: str) -> None:
        try:
            from bluesky_queueserver_api.console_monitor import (
                ConsoleMonitor_ZMQ_Threads,
            )

            from geecs_bluesky.qs_client import FAILED_MOVE_LOG_PREFIX

            monitor = ConsoleMonitor_ZMQ_Threads(
                zmq_info_addr=info_addr,
                zmq_encoding="json",
                poll_timeout=0.5,
                max_msgs=10_000,
                max_lines=1_000,
            )
            monitor.enable()
            while True:
                try:
                    msg = monitor.next_msg(timeout=0.5)
                except Exception:  # noqa: BLE001 — a timeout; poll again
                    continue
                for raw in str(msg.get("msg", "")).splitlines():
                    line = raw.rstrip()
                    if line:
                        self.push_console_line(line, FAILED_MOVE_LOG_PREFIX)
        except Exception as exc:  # noqa: BLE001 — best-effort by design
            logger.warning("console stream at %s unavailable: %s", info_addr, exc)


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
