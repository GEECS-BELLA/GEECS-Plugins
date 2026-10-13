"""The scanner's stream cache: the shared progress picture plus what the page adds.

The worker's document stream is pickled Python and the manager's console
stream is 0MQ, so this process consumes both and the web layer hands the
result to the page over Server-Sent Events as JSON; the pickled wire format
never leaves the process.  The run's progress is reduced by
:class:`geecs_bluesky.qs_client.progress.ProgressStream`, shared with every
queue client; this subclass keeps the ring of console lines a late-joining page
replays and the latest optimization record.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Optional

from geecs_bluesky.optimization.events import OptimizationRole, optimization_name
from geecs_bluesky.qs_client.progress import ProgressStream

from geecs_scanner.service.models import ConsoleLine, OptimizationOut

#: Console lines kept for late-joining pages.
_CONSOLE_KEEP = 400

#: An explicit best move is offered for fifteen minutes after successful completion.
SET_BEST_MAX_AGE_S = 15 * 60


class StreamCache(ProgressStream):
    """The shared progress picture plus a ring of console lines and the optimization record.

    Feed it from the real streams with :meth:`ensure_started`, or directly
    with :meth:`on_document` / :meth:`push_console_line` (tests, the demo
    backend).
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        super().__init__(clock=clock, thread_prefix="geecs-scanner")
        self._optimization = OptimizationOut()
        self._optimization_streams: set[str] = set()
        self._console: deque[ConsoleLine] = deque(maxlen=_CONSOLE_KEEP)
        self._seq = 0
        #: Distinguishes this process's sequence numbers from a restarted
        #: one's — a reconnecting page must not take a low seq for "seen".
        self.epoch = f"{int(clock() * 1000):x}"

    # ---------------------------------------------------------------- read

    def optimization(self) -> OptimizationOut:
        """Copy the latest optimization record under the stream lock."""
        with self._lock:
            result = self._optimization.model_copy(deep=True)
            result.expired = (
                result.expires_at is not None and self._clock() >= result.expires_at
            )
            return result

    def invalidate_optimization(self, reason: str) -> None:
        """Disable best moves after a subsequent service-submitted hardware change."""
        with self._lock:
            if self._optimization.run_uid:
                self._optimization.invalidated_reason = reason
                self._version += 1

    def console_since(self, seq: int, limit: int = 200) -> list[ConsoleLine]:
        """Console lines with ``seq`` greater than *seq*, oldest first."""
        with self._lock:
            return [ln for ln in self._console if ln.seq > seq][:limit]

    # ---------------------------------------------------------------- feed

    def on_document(self, name: str, doc: dict) -> None:
        """Reduce one bluesky document into the picture and the optimization record."""
        with self._lock:
            self._reduce_optimization(name, doc, self._clock())
            super().on_document(name, doc)

    def on_console_line(self, text: str, failed_move_prefix: Optional[str]) -> None:
        """Append one console line to the ring, then reduce it into the picture."""
        with self._lock:
            self._seq += 1
            self._console.append(
                ConsoleLine(seq=self._seq, text=text, at=self._clock())
            )
            super().on_console_line(text, failed_move_prefix)

    def push_console_line(
        self, text: str, failed_move_prefix: Optional[str] = None
    ) -> None:
        """Append one console line; a failed-move line sets the paused reason."""
        self.on_console_line(text, failed_move_prefix)

    def _reduce_optimization(self, name: str, doc: dict, now: float) -> None:
        """The optimization record (caller holds the lock)."""
        if name == "start":
            self._optimization_streams = set()
            self._optimization = OptimizationOut()
            if doc.get("plan_name") == "optimize":
                self._optimization = OptimizationOut(
                    run_uid=doc.get("uid"),
                    config=doc.get("optimizer_config"),
                    scan_number=_as_int(doc.get("scan_number")),
                    started_at=doc.get("time", now),
                    max_iterations=doc.get("max_iterations"),
                    objectives=doc.get("optimization_objectives", []),
                    best_moves=dict.fromkeys(doc.get("optimization_move_targets", [])),
                )
        elif name == "descriptor":
            if doc.get("name") == "optimization" and doc.get("uid"):
                self._optimization_streams.add(str(doc["uid"]))
        elif name == "event":
            if str(doc.get("descriptor")) in self._optimization_streams:
                data = doc.get("data", {})
                self._optimization.iteration = _as_int(data.get("iteration")) or 0
                for field, prefix in (
                    ("proposal", OptimizationRole.PROPOSAL + ":"),
                    ("measured", OptimizationRole.MEASURED + ":"),
                    ("outputs", OptimizationRole.OUTPUT + ":"),
                    ("valid_shots", OptimizationRole.VALID_SHOTS + ":"),
                    ("best", OptimizationRole.BEST + ":"),
                    ("best_moves", OptimizationRole.BEST_MOVE + ":"),
                ):
                    setattr(
                        self._optimization,
                        field,
                        {
                            optimization_name(key[len(prefix) :]): _finite(value)
                            for key, value in data.items()
                            if key.startswith(prefix)
                        },
                    )
        elif name == "stop":
            status = str(doc.get("exit_status") or "")
            if (
                self._optimization.run_uid
                and doc.get("run_start", self._optimization.run_uid)
                == self._optimization.run_uid
            ):
                self._optimization.finished = True
                self._optimization.exit_status = status
                self._optimization.completed_at = min(now, doc.get("time", now))
                self._optimization.expires_at = (
                    self._optimization.completed_at + SET_BEST_MAX_AGE_S
                )


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value: object) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None
