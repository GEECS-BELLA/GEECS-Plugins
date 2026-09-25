"""The Tiled writer's heartbeat as one kit word — a warning, never a gate.

The engine spools every run's documents to a local file and the
``geecs-tiled-writer`` service registers them in Tiled off the engine
thread (GeecsBluesky 0.103.0).  The writer says how it is doing in
``<GEECS_TILED_WRITER_STATE>/heartbeat.json``; this module reads that file
(through ``geecs_bluesky.tiled_spool``, the shared side — never the
service loop) and reduces it to the word the page's chip shows and the
``tiled_writer`` field of ``/health`` carries.

**Nothing here gates a submit.**  With the spool a dead writer loses
nothing — runs keep spooling and the first writer to run catches up — so
the verdict is shown, never enforced (owner's ruling, 2026-09-25).

The thresholds come from the measurement (25–28 s per run on the SQLite
catalog, 2026-09-25): between runs the writer is registering at most the
run that just ended, so ``pending`` ≤ 1 with a fresh heartbeat is the
writer keeping up; ≥ 3 is a backlog (it is not keeping up, or every
registration fails and is backing off — ``last_error`` says which).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from geecs_scanner.service.models import TiledWriterOut

#: ``pending`` at or below this with a fresh heartbeat reads ``ok``: the
#: writer registers one run in ~25–28 s, so the run that just ended is the
#: one complete file it may still hold.
PENDING_OK_MAX = 1
#: ``pending`` at or above this reads ``failed``: a backlog.
PENDING_FAILED_MIN = 3


def _age(now: float, then: Optional[float]) -> str:
    if then is None:
        return "never"
    seconds = max(0.0, now - then)
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago"


def writer_verdict(
    heartbeat: Any, path: Path, now: Optional[float] = None
) -> TiledWriterOut:
    """Reduce a heartbeat (``WriterHeartbeat`` or ``None``) to the chip's word.

    ``failed`` when a file was set aside for an operator or a backlog has
    formed; ``degraded`` when the writer is silent (no heartbeat, or a
    stale one — down or wedged), cannot reach Tiled, or holds two complete
    files; ``ok`` otherwise.  ``paused`` and ``running`` are never used:
    the writer has no run of its own.
    """
    now = time.time() if now is None else now
    if heartbeat is None:
        return TiledWriterOut(
            state="degraded",
            detail=f"no writer heartbeat at {path} — is geecs-tiled-writer running? "
            "(runs keep spooling; nothing reaches Tiled until it is)",
            stale=True,
        )
    counts = TiledWriterOut(
        state="unknown",
        pending=heartbeat.pending,
        in_progress=heartbeat.in_progress,
        failed=heartbeat.failed,
        last_ok=heartbeat.last_ok,
        last_error=heartbeat.last_error,
        stale=heartbeat.is_stale(now),
    )
    last_ok = _age(now, heartbeat.last_ok)
    if counts.stale:
        state, detail = (
            "degraded",
            f"writer heartbeat stale ({_age(now, heartbeat.last_sweep)}, pid "
            f"{heartbeat.pid}) — down or wedged; runs keep spooling",
        )
    elif heartbeat.failed > 0:
        state, detail = (
            "failed",
            f"{heartbeat.failed} run(s) set aside as .failed — an operator's "
            f"call{': ' + heartbeat.last_error if heartbeat.last_error else ''}",
        )
    elif heartbeat.pending >= PENDING_FAILED_MIN:
        state, detail = (
            "failed",
            f"{heartbeat.pending} runs waiting to register — the writer is not "
            f"keeping up{': ' + heartbeat.last_error if heartbeat.last_error else ''}",
        )
    elif not heartbeat.tiled_reachable:
        state, detail = (
            "degraded",
            f"Tiled at {heartbeat.tiled_uri} unreachable — {heartbeat.pending} "
            f"waiting; last registered {last_ok}",
        )
    elif heartbeat.pending > PENDING_OK_MAX:
        state, detail = (
            "degraded",
            f"{heartbeat.pending} runs waiting to register (one is usual "
            f"just after a run); last registered {last_ok}",
        )
    else:
        state, detail = (
            "ok",
            f"writer alive (pid {heartbeat.pid}); {heartbeat.pending} waiting, "
            f"{heartbeat.in_progress} in progress; last registered {last_ok}",
        )
    return counts.model_copy(update={"state": state, "detail": detail})


def default_heartbeat_path() -> Path:
    """Where the writer's heartbeat is on this host (``GEECS_TILED_WRITER_STATE``)."""
    from geecs_bluesky.tiled_spool import SpoolLayout, default_state_dir

    return SpoolLayout(default_state_dir()).heartbeat_path


def read_writer_status(path: Path, now: Optional[float] = None) -> TiledWriterOut:
    """The verdict over the heartbeat file at *path* (missing → ``degraded``)."""
    from geecs_bluesky.tiled_spool import read_heartbeat

    return writer_verdict(read_heartbeat(path), path, now)


__all__ = [
    "PENDING_FAILED_MIN",
    "PENDING_OK_MAX",
    "default_heartbeat_path",
    "read_writer_status",
    "writer_verdict",
]
