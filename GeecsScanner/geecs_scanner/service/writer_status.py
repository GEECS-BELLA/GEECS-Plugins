"""The Tiled writer's heartbeat as one kit word — a warning, never a gate.

The engine spools every run's documents to a local file and the
``geecs-tiled-writer`` service registers them in Tiled off the engine
thread (GeecsBluesky 0.103.0).  The writer says how it is doing in
``<GEECS_TILED_WRITER_STATE>/heartbeat.json``.  **The rule is the
writer's**, not this module's: ``geecs_bluesky.tiled_spool.heartbeat_verdict``
(the shared side of the spool, beside the heartbeat model — never the
service loop) reduces the file to ``ok`` / ``degraded`` / ``failed`` and a
reason, and this module only projects that onto the API model the page's
chip and ``/health`` carry.

**Nothing here gates a submit.**  With the spool a dead writer loses
nothing — runs keep spooling and the first writer to run catches up — so
the verdict is shown, never enforced (owner's ruling, 2026-09-25).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from geecs_scanner.service.models import TiledWriterOut


def writer_verdict(
    heartbeat: Any, path: Path, now: Optional[float] = None
) -> TiledWriterOut:
    """Project the shared verdict over *heartbeat* (or ``None``) onto the API model.

    The level is a kit word already (``ok`` / ``degraded`` / ``failed``);
    the counts are the heartbeat's own, zero when there is none.
    """
    from geecs_bluesky.tiled_spool import heartbeat_verdict

    verdict = heartbeat_verdict(heartbeat, now, path=path)
    if heartbeat is None:
        return TiledWriterOut(state=verdict.level, detail=verdict.reason, stale=True)
    return TiledWriterOut(
        state=verdict.level,
        detail=verdict.reason,
        pending=heartbeat.pending,
        in_progress=heartbeat.in_progress,
        failed=heartbeat.failed,
        last_ok=heartbeat.last_ok,
        last_error=heartbeat.last_error,
        stale=verdict.stale,
    )


def default_heartbeat_path() -> Path:
    """Where the writer's heartbeat is on this host (``GEECS_TILED_WRITER_STATE``)."""
    from geecs_bluesky.tiled_spool import SpoolLayout, default_state_dir

    return SpoolLayout(default_state_dir()).heartbeat_path


def read_writer_status(path: Path, now: Optional[float] = None) -> TiledWriterOut:
    """The verdict over the heartbeat file at *path* (missing or unreadable → ``degraded``)."""
    from geecs_bluesky.tiled_spool import read_heartbeat

    return writer_verdict(read_heartbeat(path), path, now)


__all__ = ["default_heartbeat_path", "read_writer_status", "writer_verdict"]
