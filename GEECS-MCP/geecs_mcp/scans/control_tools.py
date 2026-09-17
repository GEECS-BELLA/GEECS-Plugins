"""The control tools: stop, pause/resume, clear-queue, progress.

Same conventions as :mod:`.read_tools` (sync ``_*_impl`` = the tested
surface, async wrappers via the guard, JSON envelopes, engine text
verbatim).  The safety doctrine (owner decisions 2026-08-22):

- ``stop_scan`` refuses another client's scan by name unless
  ``force=true`` (approval-gated osprey-side, and always logged in the
  result); it is approval-only and must NEVER sit behind the kill
  switch — a halt must always be possible.
- ``clear_queue`` is the one verb that removes items — explicit,
  approval-gated recovery from the failed-item-at-front state; nothing
  clears implicitly.
- ``pause_scan`` joins the halt family (never behind the headless gate
  — see ``tool_names.STOP_TOOLS``), ``resume_scan`` restarts motion so
  it gates like a submission, and ``scan_progress`` is the best-effort
  document-stream picture (:mod:`.progress_stream`).

**There is no write path left here.**  ``submit_scan``, ``run_action``,
``describe_action`` and ``move_scan_variable`` were deleted when the
native-Bluesky rebuild retired the client verbs they called
(``submit_scan``, ``submit_action``, ``move_variable``,
``describe_action`` are all gone from ``geecs_bluesky.qs_client``, whose
submission surface is now ``submit_plan``/``submit_preset`` over the
``count``/``sweep``/``optimize`` plans).  This server is an experiment,
not an operator surface, so the verbs were removed rather than rewired;
see #727 if a write path is ever wanted back.  The reference
implementation to copy is then
``GeecsScanner/geecs_scanner/service/scanner.py``.
"""

from __future__ import annotations

import logging

from geecs_mcp import errors, runtime, tool_names
from geecs_mcp.server import mcp
from geecs_mcp.scans.read_tools import _run_guarded

logger = logging.getLogger("geecs_mcp.scans.control")


# ---------------------------------------------------------------------------
# stop_scan
# ---------------------------------------------------------------------------


def _stop_scan_impl(force: bool) -> str:
    """Ownership-gated graceful stop (partial data preserved)."""
    client = runtime.get_queue_client()
    status = client.status()
    if not status.connected:
        return errors.make_error("manager_unreachable", status.detail)
    running = None
    try:
        running = client.running_item()
    except Exception:  # fail-open: an unreadable item never blocks a halt
        logger.debug("running-item read failed before stop", exc_info=True)
    owner = (running or {}).get("user")
    foreign = bool(owner and owner != runtime.client_identity())
    if foreign and not force:
        return errors.make_error(
            "policy_refusal",
            f"the running scan was submitted by {owner!r} — pass "
            "force=true only if the operator explicitly asks for the stop",
        )
    ok, message = client.stop_scan()
    if not ok:
        return errors.make_error("worker_refused", message)
    # forced marks "an operator authorized stopping ANOTHER client's scan"
    # — a habitual force=true on the MCP's own scan must not pollute the
    # audit marker (review finding).
    return errors.make_ok(message=message, forced=bool(force and foreign))


@mcp.tool(name=tool_names.STOP_SCAN)
async def stop_scan(force: bool = False) -> str:
    """Gracefully stop the current scan (partial data preserved).

    May take up to ~120 s from a running scan (a deferred pause waits out
    the in-flight move before stopping). Refuses a scan submitted by
    another client (the submitting identity is named) unless
    ``force=true`` — pass force only when the operator explicitly asks.
    """
    return await _run_guarded(_stop_scan_impl, force)


# ---------------------------------------------------------------------------
# clear_queue
# ---------------------------------------------------------------------------


def _clear_queue_impl() -> str:
    """List what is queued, then remove it — the explicit recovery verb."""
    client = runtime.get_queue_client()
    try:
        pending = client.queue_items()
    except Exception as exc:
        return errors.make_error("manager_unreachable", str(exc))
    if not pending:
        return errors.make_ok(cleared=[], message="queue already empty")
    ok, message = client.clear_queue()
    if not ok:
        return errors.make_error("worker_refused", message)
    return errors.make_ok(
        cleared=[
            {
                "item_uid": item.get("item_uid"),
                "plan": item.get("name"),
                "user": item.get("user"),
            }
            for item in pending
        ],
        message=message,
    )


@mcp.tool(name=tool_names.CLEAR_QUEUE)
async def clear_queue() -> str:
    """Remove every queued item — the ONLY verb that clears the queue.

    Usually recovers from one failed item returned to the queue front.
    The result lists exactly what was removed, with each item's
    submitting client.
    """
    return await _run_guarded(_clear_queue_impl)


# ---------------------------------------------------------------------------
# pause_scan / resume_scan
# ---------------------------------------------------------------------------


def _running_scan_owner(client) -> tuple[str | None, bool]:
    """The running item's submitted-as identity, as ``(owner, readable)``.

    The two consumers treat an unreadable item OPPOSITELY by doctrine:
    the halt family (pause, like stop) fails open — a flaky read must
    never block making the machine quieter — while resume (a go verb)
    fails closed, so a transient read failure cannot let this client
    restart another client's scan unforced (review finding #683-2).
    """
    try:
        running = client.running_item()
    except Exception:
        logger.debug("running-item read failed before pause/resume", exc_info=True)
        return None, False
    return (running or {}).get("user") or None, True


def _pause_scan_impl(force: bool) -> str:
    """Ownership-gated deferred pause (the halt family, like stop)."""
    client = runtime.get_queue_client()
    status = client.status()
    if not status.connected:
        return errors.make_error("manager_unreachable", status.detail)
    if status.re_state != "running":
        return errors.make_error(
            "invalid_request", f"nothing to pause (RE state: {status.re_state})"
        )
    owner, _readable = _running_scan_owner(client)  # unreadable = fail open
    foreign = bool(owner and owner != runtime.client_identity())
    if foreign and not force:
        return errors.make_error(
            "policy_refusal",
            f"the running scan was submitted by {owner!r} — pass "
            "force=true only if the operator explicitly asks for the pause",
        )
    ok, message = client.request_pause()
    if not ok:
        return errors.make_error("worker_refused", message)
    return errors.make_ok(message=message, forced=bool(force and foreign))


@mcp.tool(name=tool_names.PAUSE_SCAN)
async def pause_scan(force: bool = False) -> str:
    """Pause the running scan at the next checkpoint (deferred pause).

    The in-flight shot and any in-flight move always finish first (1–2
    shots of latency — the architectural floor). Nothing is lost: resume
    continues exactly where the scan paused; stop ends it gracefully with
    partial data. Refuses a scan submitted by another client (named in
    the refusal) unless ``force=true`` — pass force only when the
    operator explicitly asks.
    """
    return await _run_guarded(_pause_scan_impl, force)


def _resume_scan_impl(force: bool) -> str:
    """Ownership-gated resume from the paused state."""
    client = runtime.get_queue_client()
    status = client.status()
    if not status.connected:
        return errors.make_error("manager_unreachable", status.detail)
    if status.re_state != "paused":
        return errors.make_error(
            "invalid_request", f"nothing to resume (RE state: {status.re_state})"
        )
    owner, readable = _running_scan_owner(client)
    if not readable and not force:
        # Fail CLOSED: resume restarts motion, so unknown ownership
        # refuses (unlike the halt family's fail-open).
        return errors.make_error(
            "policy_refusal",
            "the paused scan's owner could not be read — retry, or pass "
            "force=true only if the operator explicitly asks for the resume",
        )
    foreign = bool(owner and owner != runtime.client_identity())
    if foreign and not force:
        return errors.make_error(
            "policy_refusal",
            f"the paused scan was submitted by {owner!r} — pass "
            "force=true only if the operator explicitly asks for the resume",
        )
    ok, message = client.request_resume()
    if not ok:
        return errors.make_error("worker_refused", message)
    # forced also covers force past UNKNOWN ownership — the audit marker
    # means "an operator authorized resuming a scan not known to be ours".
    return errors.make_ok(
        message=message, forced=bool(force and (foreign or not readable))
    )


@mcp.tool(name=tool_names.RESUME_SCAN)
async def resume_scan(force: bool = False) -> str:
    """Resume a paused scan (nothing replays; a failed move is retried).

    Check scan_progress first — a scan paused on a failed axis move
    reports the reason, and resuming retries that exact move. This
    restarts motion and acquisition, so it is gated like a submission.
    Refuses a scan submitted by another client unless ``force=true`` —
    pass force only when the operator explicitly asks.
    """
    return await _run_guarded(_resume_scan_impl, force)


# ---------------------------------------------------------------------------
# scan_progress (read-only: manager poll + best-effort document stream)
# ---------------------------------------------------------------------------


def _stream_snapshot(client, re_state: str | None) -> dict:
    """The document-stream picture, started lazily from the client's addrs.

    The HTTP service warms the same cache at startup
    (``__main__.warm_progress_stream``, #685) and this call is then the
    idempotent no-op; stdio relies on this lazy start alone, so a run
    submitted before the first poll streams no counts there.  Best-effort
    BY DESIGN: ``available=false`` (with the reason) when the
    stream cannot be consumed, and the poll fields stand alone.  The
    sticky ``paused_reason`` (the console-text stream's failed-move line)
    is only surfaced while the RE is actually paused — after a resume the
    stale reason would read as current.
    """
    from geecs_mcp.scans import progress_stream

    snapshot = progress_stream.start_for_client(client).snapshot()
    if re_state != "paused":
        snapshot.pop("paused_reason", None)
    return snapshot


def _scan_progress_impl() -> str:
    """Manager poll (authoritative) + the per-shot stream picture (best-effort)."""
    client = runtime.get_queue_client()
    status = client.status()
    if not status.connected:
        return errors.make_ok(state="unknown", detail=status.detail, running_item=None)
    running = None
    try:
        raw = client.running_item()
        if raw:
            running = {
                "item_uid": raw.get("item_uid"),
                "plan": raw.get("name"),
                "user": raw.get("user"),
            }
    except Exception:
        logger.debug("running-item read failed in progress poll", exc_info=True)
    last = None
    try:
        items = client.history_items()
        if items:
            tail = items[-1]
            result = tail.get("result") or {}
            last = {
                "plan": tail.get("name"),
                "user": tail.get("user"),
                "exit_status": result.get("exit_status"),
                "scan_ids": result.get("scan_ids"),
            }
    except Exception:
        logger.debug("history read failed in progress poll", exc_info=True)
    try:
        stream = _stream_snapshot(client, status.re_state)
    except Exception:  # the poll answer must never die on the stream extra
        logger.debug("stream snapshot failed in progress poll", exc_info=True)
        stream = {"available": False, "detail": "stream snapshot failed"}
    return errors.make_ok(
        state=status.re_state or "idle",
        running_item=running,
        items_in_queue=status.items_in_queue,
        last_completed=last,
        stream=stream,
    )


@mcp.tool(name=tool_names.SCAN_PROGRESS)
async def scan_progress() -> str:
    """Progress for the current scan: manager state + per-shot counts.

    The manager poll is authoritative: RE state (idle/running/paused/…),
    the running item (with its submitting client), queue depth, and the
    last completed item's outcome. ``stream`` adds the document-stream
    picture when available — scan number, shots done / planned total,
    and (while paused) the failed-move reason; ``stream.available=false``
    means only the poll fields apply (one manager runs one scan at a
    time, so the stream's latest run IS the running scan).
    """
    return await _run_guarded(_scan_progress_impl)
