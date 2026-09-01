"""The best-effort progress cache: document + console-text stream consumers.

The v2 ``scan_progress`` upgrade (#676): one lazily-started, process-wide
cache fed by two daemon threads —

- the worker's bluesky **document stream** (``RemoteDispatcher`` on the
  proxy out-port): the start document seeds scan number and planned
  totals (``num_points × shots_per_step``, falling back to
  ``max_iterations`` for optimize runs — the console's live-verified
  Scan013 lesson), primary-stream events advance ``shots_done``, the
  stop document records the exit status;
- the manager's **console-output text stream**: lines carrying the
  engine's failed-move prefix (``FAILED_MOVE_LOG_PREFIX``, via the
  qs_client re-export — never a ``plans/*`` import) become the paused
  scan's *why*.

Best-effort BY DESIGN: setup failure marks the cache unavailable with a
reason and ``scan_progress`` degrades to its poll answer.  The honesty
boundary: ``available=true`` means the dispatcher is consuming, not that
the address is *right* — zmq connects lazily, so a wrong/unreachable
``doc_addr`` reads as an available-but-forever-empty picture (console
parity; the manager poll stays the authoritative answer either way), and
a one-time setup failure stays down until process restart.  The threading
rules are the console's, inherited from the #653 review: threads are
daemons, ``stop`` never exists — a zmq socket must never be touched from
another thread (a cross-thread close can trip a libzmq assertion that
ABORTS the process), so the threads live for the process lifetime and
emission is simply gated by the cache's own lock.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("geecs_mcp.scans.progress_stream")


class ProgressCache:
    """Lock-protected latest-run picture built from the two streams."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._doc_addr: Optional[str] = None
        self._available = False
        self._detail = "stream cache not started"
        self._state: dict[str, Any] = {}
        self._primary_descriptors: set[str] = set()

    # -- consumption --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """A copy of the cached picture (``available`` + fields or reason)."""
        with self._lock:
            return {
                "available": self._available,
                "detail": self._detail,
                **dict(self._state),
            }

    # -- stream plumbing ----------------------------------------------------

    def ensure_started(self, doc_addr: Optional[str], info_addr: Optional[str]) -> None:
        """Start both consumer threads once (idempotent, never raises).

        Once-only BY DESIGN: the first call's addresses latch for the
        process lifetime (the #653 rule leaves no way to retire a zmq
        consumer thread).  A later call with *different* addresses —
        possible only after ``runtime.clear_runtime_cache()`` rebuilt the
        queue client against an edited config — is ignored with one
        warning; a config change needs a server restart, like runtime's
        other singletons.
        """
        with self._lock:
            if self._started:
                if doc_addr and doc_addr != self._doc_addr:
                    logger.warning(
                        "progress stream already consuming %r — new address %r "
                        "ignored (restart the server to re-point streams)",
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
            name="geecs-mcp-doc-stream",
            daemon=True,
        ).start()
        if info_addr:
            threading.Thread(
                target=self._run_console_text,
                args=(info_addr,),
                name="geecs-mcp-console-stream",
                daemon=True,
            ).start()

    def _run_documents(self, doc_addr: str) -> None:
        try:
            from bluesky.callbacks.zmq import RemoteDispatcher

            dispatcher = RemoteDispatcher(doc_addr)
            dispatcher.subscribe(self._on_document)
            with self._lock:
                self._available = True
                self._detail = ""
            dispatcher.start()  # blocks for the thread's lifetime
        except Exception as exc:
            logger.warning("document stream unavailable: %s", exc)
            with self._lock:
                self._available = False
                self._detail = f"document stream unavailable: {exc}"

    def _run_console_text(self, info_addr: str) -> None:
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
                    msg = monitor.next_msg(timeout=5.0)
                except Exception:  # timeout — nothing to read
                    continue
                # One message can carry several lines (the console's own
                # parsing rule) — match per line so the reason never
                # drags trailing unrelated output along.
                for line in str(msg.get("msg", "")).splitlines():
                    if FAILED_MOVE_LOG_PREFIX not in line:
                        continue
                    reason = line.split(FAILED_MOVE_LOG_PREFIX, 1)[1]
                    reason = reason.lstrip(" :-").strip()
                    with self._lock:
                        self._state["paused_reason"] = reason or FAILED_MOVE_LOG_PREFIX
        except Exception as exc:
            logger.warning("console-text stream unavailable: %s", exc)

    # -- document handling --------------------------------------------------

    def _on_document(self, name: str, doc: dict) -> None:
        try:
            with self._lock:
                if name == "start":
                    shots_per_step = int(doc.get("shots_per_step") or 1)
                    num_points = doc.get("num_points")
                    max_iterations = doc.get("max_iterations")
                    if num_points:
                        total: Optional[int] = int(num_points) * shots_per_step
                    elif max_iterations:
                        # Optimize runs record max_iterations, not
                        # num_points (the Scan013 live lesson).
                        total = int(max_iterations) * shots_per_step
                    else:
                        total = None
                    self._state = {
                        "run_uid": doc.get("uid"),
                        "scan_number": doc.get("scan_number"),
                        "shots_total": total,
                        "shots_done": 0,
                        "exit_status": None,
                        "paused_reason": None,
                    }
                    self._primary_descriptors = set()
                elif name == "descriptor":
                    if doc.get("name") == "primary":
                        self._primary_descriptors.add(str(doc.get("uid")))
                elif name == "event":
                    if str(doc.get("descriptor")) in self._primary_descriptors:
                        self._state["shots_done"] = int(doc.get("seq_num") or 0)
                        # Progress proves the resume: without this, a scan
                        # paused a second time (manually) would report the
                        # FIRST pause's failed-move text as the current why
                        # (review finding #683-1).
                        self._state["paused_reason"] = None
                elif name == "stop":
                    if doc.get("run_start") == self._state.get("run_uid"):
                        self._state["exit_status"] = doc.get("exit_status")
        except Exception:  # a malformed document must never kill the stream
            logger.debug("progress document ignored", exc_info=True)


#: The process-wide cache (the server is one process; tests build their own).
_cache = ProgressCache()


def get_progress_cache() -> ProgressCache:
    """The process-wide cache instance (module attribute = the patch seam)."""
    return _cache


def start_for_client(client: Any) -> ProgressCache:
    """Start the process-wide cache from a queue client's stream addresses.

    THE one address resolution — ``doc_addr`` / ``info_addr`` read off the
    client (``None`` on a stub or an unconfigured ``[qserver]``, which the
    cache reports honestly).  Two callers: ``scan_progress`` on every poll
    (the lazy start, stdio's posture) and the HTTP entry point once at
    startup (#685), so a long-lived service is already consuming when its
    first run's start document passes.  Returns the cache for the caller's
    snapshot.
    """
    cache = get_progress_cache()
    cache.ensure_started(
        getattr(client, "doc_addr", None), getattr(client, "info_addr", None)
    )
    return cache
