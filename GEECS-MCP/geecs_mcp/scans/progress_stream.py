"""The ``scan_progress`` stream picture: the shared progress stream, in MCP's keys.

The reduction lives in :mod:`geecs_bluesky.qs_client.progress` (the same
one the web scanner renders); this module keeps the process-wide
instance, the one address resolution that starts it, and
:func:`stream_view`, the tool contract's shape of its snapshot.

Best-effort BY DESIGN: ``available=true`` means the dispatcher is
consuming, not that the address is *right* — zmq connects lazily, so a
wrong ``doc_addr`` reads as an available-but-forever-empty picture, and a
one-time setup failure stays down until process restart.  The manager
poll stays the authoritative answer either way.
"""

from __future__ import annotations

from typing import Any

from geecs_bluesky.qs_client.progress import ProgressStream, RunProgress

#: The process-wide stream (the server is one process; tests build their own).
_cache = ProgressStream(thread_prefix="geecs-mcp")


def get_progress_cache() -> ProgressStream:
    """The process-wide stream instance (module attribute = the patch seam)."""
    return _cache


def stream_view(progress: RunProgress) -> dict[str, Any]:
    """The tool contract's ``stream`` dict for one snapshot.

    Only ``available`` + ``detail`` until a run is seen; then the run's
    uid, scan number, ``shots_total`` / ``shots_done``, exit status and
    paused reason.
    """
    view: dict[str, Any] = {"available": progress.available, "detail": progress.detail}
    if progress.state is not None:
        view.update(
            run_uid=progress.run_uid,
            scan_number=progress.scan_number,
            shots_total=progress.planned_total,
            shots_done=progress.shots_done,
            exit_status=progress.exit_status,
            paused_reason=progress.paused_reason,
        )
    return view


def start_for_client(client: Any) -> ProgressStream:
    """Start the process-wide stream from a queue client's stream addresses.

    THE one address resolution — ``doc_addr`` / ``info_addr`` read off the
    client (``None`` on a stub or an unconfigured ``[qserver]``, which the
    stream reports honestly).  Two callers: ``scan_progress`` on every poll
    (the lazy start, stdio's posture) and the HTTP entry point once at
    startup (#685), so a long-lived service is already consuming when its
    first run's start document passes.  Returns the stream for the caller's
    snapshot.
    """
    cache = get_progress_cache()
    cache.ensure_started(
        getattr(client, "doc_addr", None), getattr(client, "info_addr", None)
    )
    return cache
