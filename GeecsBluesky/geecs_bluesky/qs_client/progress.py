"""The live-run progress picture every queue client reduces from the streams.

:class:`ProgressStream` consumes the worker's document stream and the
manager's console-output stream on daemon threads and reduces them to one
:class:`RunProgress`: the start document seeds the scan number and the
planned total, row events advance ``shots_done``, the stop document
records the exit, and a failed-move console line is the paused reason.
Clients render that snapshot in their own wire shape.

Best-effort by design.  A zmq socket must never be touched from another
thread (a cross-thread close can abort the process), so the threads live
for the process lifetime and there is no ``stop``; state is guarded by
one lock.  A stream that cannot be set up marks the picture unavailable
with a reason, and the manager poll stays authoritative.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: The streams whose events are shots: a run's ``primary`` rows.  Mirrors
#: ``geecs_bluesky.callbacks.ROW_STREAMS``
#: (that package imports the engine); ``tests/test_qs_client_progress.py``
#: pins the two equal.
ROW_STREAMS: tuple[str, ...] = ("primary",)


#: How many of the run's slowest devices :class:`RunProgress` keeps.
SLOWEST_KEPT = 5


class SlowDevice(BaseModel):
    """One required device as the run's rows saw it hold their shots."""

    device: str
    held: int = Field(description="Rows whose shot it was the last to deliver")
    mean_s: float = Field(description="Its mean delay after the fire on those rows")
    max_s: float = Field(description="Its longest delay after the fire")


class RunProgress(BaseModel):
    """The latest run as the streams describe it.

    The rates and the slowest devices come from the rows' ``fire_stamp``,
    ``shot_period_s``, ``slowest_device`` and ``slowest_s`` (a strict run's
    shot table; ``None`` / empty for a run without them).
    """

    available: bool = False
    detail: str = ""
    run_uid: Optional[str] = None
    scan_number: Optional[int] = None
    scan_folder: Optional[str] = None
    day: Optional[str] = None
    plan_name: Optional[str] = None
    planned_total: Optional[int] = None
    shots_done: int = 0
    state: Optional[str] = None
    exit_status: Optional[str] = None
    paused_reason: Optional[str] = None
    updated_at: Optional[float] = None
    current_rate_hz: Optional[float] = Field(
        default=None, description="1 / the latest row's shot_period_s"
    )
    average_rate_hz: Optional[float] = Field(
        default=None,
        description="(fires − 1) / (last fire − first fire) over the run",
    )
    waiting_on: Optional[str] = Field(
        default=None, description="The latest row's slowest_device"
    )
    waiting_s: Optional[float] = Field(
        default=None, description="The latest row's slowest_s"
    )
    slowest: list[SlowDevice] = Field(
        default_factory=list,
        description="The devices that held the most rows, most first",
    )


class ProgressStream:
    """Lock-guarded :class:`RunProgress` fed by the document and console streams.

    Feed it from the real streams with :meth:`ensure_started`, or directly
    with :meth:`on_document` / :meth:`on_console_line`.  :meth:`version`
    changes with every update, so a poller can skip unchanged rounds.
    Subclasses extend the two ``on_*`` hooks under the same (reentrant)
    lock.
    """

    def __init__(
        self, clock: Callable[[], float] = time.time, thread_prefix: str = "geecs"
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._thread_prefix = thread_prefix
        self._started = False
        self._doc_addr: Optional[str] = None
        self._progress = RunProgress(detail="stream cache not started")
        self._rows: set[str] = set()
        self._version = 0
        self._timing = _RowTiming()

    # ---------------------------------------------------------------- read

    def snapshot(self) -> RunProgress:
        """A copy of the current picture."""
        with self._lock:
            return self._progress.model_copy()

    def version(self) -> int:
        """A counter that changes with every update."""
        with self._lock:
            return self._version

    # ---------------------------------------------------------------- feed

    def mark_available(self, available: bool, detail: str = "") -> None:
        """Record whether the document stream is being consumed."""
        with self._lock:
            self._progress.available = available
            self._progress.detail = detail
            self._version += 1

    def on_document(self, name: str, doc: dict) -> None:
        """Reduce one bluesky document; a malformed one is logged and skipped."""
        with self._lock:
            try:
                self._reduce(name, doc, self._clock())
            except Exception:  # noqa: BLE001 — a bad document must not kill the stream
                logger.debug("progress document ignored", exc_info=True)
            self._version += 1

    def on_console_line(self, text: str, failed_move_prefix: Optional[str]) -> None:
        """Take one console line; a failed-move line pauses the picture."""
        with self._lock:
            if failed_move_prefix and failed_move_prefix in text:
                reason = text.split(failed_move_prefix, 1)[1].lstrip(" :-").strip()
                self._progress.paused_reason = reason or text
                self._progress.state = "paused"
            self._version += 1

    def _reduce(self, name: str, doc: dict, now: float) -> None:
        if name == "start":
            self._start(doc, now)
        elif name == "descriptor":
            if doc.get("name") in ROW_STREAMS and doc.get("uid"):
                self._rows.add(str(doc["uid"]))
        elif name in ("event", "event_page"):
            if str(doc.get("descriptor")) in self._rows:
                seq = doc.get("seq_num")
                seqs = seq if name == "event_page" else [seq]
                counted = [_as_int(v) or 0 for v in seqs or []]
                if counted:
                    self._count_rows(max(counted), now)
                self._timing.take(doc.get("data") or {}, page=name == "event_page")
                self._timing.write(self._progress)
        elif name == "stop":
            self._stop(doc, now)

    def _start(self, doc: dict, now: float) -> None:
        self._rows = set()
        self._timing = _RowTiming()
        self._progress = RunProgress(
            available=self._progress.available,
            detail=self._progress.detail,
            run_uid=_as_str(doc.get("uid")),
            scan_number=_as_int(doc.get("scan_number")),
            scan_folder=_as_str(doc.get("scan_folder")),
            day=_scan_day(doc.get("scan_tag")),
            plan_name=_as_str(doc.get("plan_name")),
            planned_total=_planned_total(doc),
            state="running",
            updated_at=now,
        )

    def _stop(self, doc: dict, now: float) -> None:
        p = self._progress
        run_start = doc.get("run_start")
        if (
            run_start is not None
            and p.run_uid is not None
            and str(run_start) != p.run_uid
        ):
            return  # another run's stop
        status = str(doc.get("exit_status") or "")
        p.exit_status = status
        p.state = "done" if status == "success" else "aborted"
        p.updated_at = now

    def _count_rows(self, seq: int, now: float) -> None:
        p = self._progress
        p.shots_done = max(seq, p.shots_done)
        # A row after a failed-move pause proves the operator resumed.
        if p.state == "paused":
            p.state = "running"
            p.paused_reason = None
        p.updated_at = now

    # ----------------------------------------------------------- real streams

    def ensure_started(self, doc_addr: Optional[str], info_addr: Optional[str]) -> None:
        """Start both consumer threads once (idempotent, never raises).

        The first call's addresses latch for the process lifetime; a later
        call with a different address warns and is ignored.
        """
        with self._lock:
            if self._started:
                if doc_addr and doc_addr != self._doc_addr:
                    logger.warning(
                        "progress stream already consuming %r — new address %r "
                        "ignored (restart the service to re-point streams)",
                        self._doc_addr,
                        doc_addr,
                    )
                return
            self._started = True
            self._doc_addr = doc_addr
            if not doc_addr:
                self._progress.detail = "no document-stream address configured"
                logger.warning(
                    "progress stream not started: no document-stream address "
                    "([qserver] unconfigured?)"
                )
                return
        logger.info("progress stream consuming document stream %s", doc_addr)
        threading.Thread(
            target=self._run_documents,
            args=(doc_addr,),
            name=f"{self._thread_prefix}-doc-stream",
            daemon=True,
        ).start()
        if info_addr:
            threading.Thread(
                target=self._run_console,
                args=(info_addr,),
                name=f"{self._thread_prefix}-console-stream",
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

            from geecs_bluesky.log_markers import FAILED_MOVE_LOG_PREFIX

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
                # One message can carry several lines: match per line.
                for raw in str(msg.get("msg", "")).splitlines():
                    line = raw.rstrip()
                    if line:
                        self.on_console_line(line, FAILED_MOVE_LOG_PREFIX)
        except Exception as exc:  # noqa: BLE001 — best-effort by design
            logger.warning("console stream at %s unavailable: %s", info_addr, exc)


def _planned_total(doc: dict) -> Optional[int]:
    """``num_points`` (or ``max_iterations``) × ``shots_per_step`` (default 1)."""
    points = doc.get("num_points") or doc.get("max_iterations")
    if not points:
        return None
    try:
        return int(points) * int(doc.get("shots_per_step") or 1)
    except (TypeError, ValueError):
        return None


def _scan_day(tag: Any) -> Optional[str]:
    """The ISO day from a start document's ``scan_tag``."""
    if not isinstance(tag, dict):
        return None
    try:
        if tag.get("year") and tag.get("month") and tag.get("day"):
            return (
                f"{int(tag['year']):04d}-{int(tag['month']):02d}-{int(tag['day']):02d}"
            )
    except (TypeError, ValueError):
        pass
    return None


class _RowTiming:
    """The run's rates and the devices that held its shots, from the rows' timing columns."""

    def __init__(self) -> None:
        self.first: float | None = None
        self.last: float | None = None
        self.fires = 0
        self.period: float | None = None
        self.waiting: tuple[str, float] | None = None
        self.held: dict[str, list[float]] = {}

    def take(self, data: dict, *, page: bool) -> None:
        """One event's ``data`` (or an event page's, columns of lists)."""
        rows = (
            [dict(zip(data, vals, strict=False)) for vals in zip(*data.values())]
            if page
            else [data]
        )
        for row in rows:
            fired = _finite(row.get("fire_stamp"))
            if fired is not None:
                self.first = fired if self.first is None else min(self.first, fired)
                self.last = fired if self.last is None else max(self.last, fired)
                self.fires += 1
            period = _finite(row.get("shot_period_s"))
            if period is not None and period > 0:
                self.period = period
            device = row.get("slowest_device")
            delay = _finite(row.get("slowest_s"))
            if device and delay is not None:
                self.waiting = (str(device), delay)
                self.held.setdefault(str(device), []).append(delay)

    def write(self, progress: RunProgress) -> None:
        """Set the picture's timing fields."""
        progress.current_rate_hz = None if self.period is None else 1.0 / self.period
        span = (
            None if self.first is None or self.last is None else self.last - self.first
        )
        progress.average_rate_hz = (
            (self.fires - 1) / span if span and self.fires > 1 else None
        )
        progress.waiting_on, progress.waiting_s = self.waiting or (None, None)
        ranked = sorted(self.held.items(), key=lambda kv: (-len(kv[1]), -max(kv[1])))
        progress.slowest = [
            SlowDevice(
                device=device,
                held=len(delays),
                mean_s=sum(delays) / len(delays),
                max_s=max(delays),
            )
            for device, delays in ranked[:SLOWEST_KEPT]
        ]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> Optional[str]:
    return str(value) if value else None
