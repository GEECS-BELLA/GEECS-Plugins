"""The v1 control tools: submit, stop, clear-queue, and poll progress.

Same conventions as :mod:`.read_tools` (sync ``_*_impl`` = the tested
surface, async wrappers via the guard, JSON envelopes, engine text
verbatim).  The safety doctrine (owner decisions 2026-08-22 + the
planning doc §2):

- ``submit_scan`` takes a preset name OR a composed request dict (both
  from day one — presets are barely used in practice), enforces the
  agent scan-size cap (default 1,000 shots), refuses while anything is
  queued or running, runs the full preflight, and **never continues
  silently past a warning** — unacknowledged questions come back as
  ``needs_acknowledgement`` and each acknowledgement is stamped
  ``continued`` into the run's ``SubmissionRecord``.  ``clear_pending``
  is never used.
- ``stop_scan`` refuses another client's scan by name unless
  ``force=true`` (approval-gated osprey-side, and always logged in the
  result); it is approval-only and must NEVER sit behind the kill
  switch — a halt must always be possible.
- ``clear_queue`` is the one verb that removes items — explicit,
  approval-gated recovery from the failed-item-at-front state; nothing
  clears implicitly.
"""

from __future__ import annotations

import logging

from geecs_scan_mcp import errors, runtime, tool_names
from geecs_scan_mcp.server import mcp
from geecs_scan_mcp.tools.read_tools import _run_guarded

logger = logging.getLogger("geecs_scan_mcp.tools.control")


# ---------------------------------------------------------------------------
# submit_scan
# ---------------------------------------------------------------------------


def _planned_shots(request) -> int | None:
    """Total planned shots, or ``None`` when the request cannot say.

    step/noscan: product of axis position counts × ``shots_per_step``
    (noscan = one no-move bin).  optimize: ``max_iterations ×
    shots_per_step`` — an optimize request WITHOUT an explicit
    ``max_iterations`` returns ``None`` (the engine's auto budget is
    open-ended; the agent cap needs a number, so submission requires one).
    """
    mode = getattr(request.mode, "value", request.mode)
    if mode == "optimize":
        spec = request.optimization
        iterations = getattr(spec, "max_iterations", None) if spec else None
        if not iterations:
            return None
        return int(iterations) * int(request.shots_per_step)
    steps = 1
    for axis in request.axes or []:
        positions = axis.positions
        values = getattr(positions, "values", None)
        steps *= len(values) if values is not None else len(positions.to_values())
    return steps * int(request.shots_per_step)


def _submit_scan_impl(
    request: dict | None,
    preset: str | None,
    description: str | None,
    acknowledge_warnings: list[str] | None,
) -> str:
    """Validate → cap → etiquette → preflight → acknowledge → stamp → queue."""
    from geecs_schemas import ScanRequest

    if (request is None) == (preset is None):
        return errors.make_error(
            "invalid_request", "pass exactly one of request (a dict) or preset (a name)"
        )
    experiment = runtime.get_experiment()
    if not experiment:
        return errors.make_error(
            "invalid_request",
            "no experiment configured ([Experiment] expt in config.ini)",
        )
    # -- resolve + schema-validate -----------------------------------------
    try:
        if preset is not None:
            resolver = runtime.get_resolver()
            validated = resolver.resolve_preset(preset)
        else:
            validated = ScanRequest.model_validate(request)
    except Exception as exc:
        return errors.make_error("invalid_request", str(exc))
    if description:
        validated = validated.model_copy(update={"description": description})

    # -- agent scan-size cap -------------------------------------------------
    cap = runtime.max_shots()
    shots = _planned_shots(validated)
    if shots is None:
        return errors.make_error(
            "policy_refusal",
            "optimize submissions need an explicit max_iterations (the "
            f"engine's auto budget is open-ended; the agent cap is {cap} "
            "shots)",
        )
    if shots > cap:
        return errors.make_error(
            "policy_refusal",
            f"{shots} planned shots exceeds the agent cap of {cap} "
            "([scan_mcp] max_shots) — shrink the scan or have an operator "
            "run it from the console",
        )

    # -- queue etiquette: one scan in flight, never clear implicitly --------
    client = runtime.get_queue_client()
    status = client.status()
    if not status.connected:
        return errors.make_error("manager_unreachable", status.detail)
    if status.re_state not in (None, "idle"):
        return errors.make_error(
            "policy_refusal",
            f"a scan is active (RE state: {status.re_state}) — wait for it "
            "or stop it first",
        )
    try:
        pending = client.queue_items()
    except Exception as exc:
        return errors.make_error("manager_unreachable", str(exc))
    if pending:
        return errors.make_error(
            "policy_refusal",
            f"{len(pending)} item(s) already queued (usually a failed item "
            "returned to the queue front) — inspect with scan_status and "
            "clear explicitly with clear_queue",
            pending_items=[
                {
                    "item_uid": item.get("item_uid"),
                    "plan": item.get("name"),
                    "user": item.get("user"),
                }
                for item in pending
            ],
        )

    # -- preflight + acknowledge-warnings loop -------------------------------
    from geecs_bluesky.qs_client import run_submit_preflight, stamp_submission

    report = run_submit_preflight(validated, experiment)
    if report.refusal is not None:
        return errors.make_error("invalid_request", report.refusal)
    acknowledged = set(acknowledge_warnings or [])
    unacknowledged = [q for q in report.questions if q.check not in acknowledged]
    if unacknowledged:
        return errors.make_error(
            "policy_refusal",
            "preflight raised warnings that need explicit acknowledgement — "
            "surface them to the human, then resubmit with "
            "acknowledge_warnings listing each check name",
            needs_acknowledgement=[
                {"check": q.check, "title": q.title, "message": q.message}
                for q in unacknowledged
            ],
        )
    outcomes = list(report.outcomes) + [
        (q.check, "continued", q.message[:200]) for q in report.questions
    ]
    stamped = stamp_submission(validated, outcomes, client=runtime.client_identity())

    # -- queue it (never clear_pending) --------------------------------------
    result = client.submit_scan(stamped.model_dump(mode="json"), clear_pending=False)
    if not result.ok:
        return errors.make_error(
            "worker_refused",
            result.message or "submission refused",
            pending_items=result.pending_items,
        )
    return errors.make_ok(
        item_uid=result.item_uid,
        message=result.message,
        planned_shots=shots,
        submitted_as=runtime.client_identity(),
    )


@mcp.tool(name=tool_names.SUBMIT_SCAN)
async def submit_scan(
    request: dict | None = None,
    preset: str | None = None,
    description: str | None = None,
    acknowledge_warnings: list[str] | None = None,
) -> str:
    """Queue one GEECS scan and start it (submit-and-poll).

    Returns the queue item_uid immediately — track it with
    scan_progress / scan_history. Pass exactly one of ``request`` (a composed ScanRequest dict — validate
    with validate_scan_request first) or ``preset`` (a name from
    list_scan_configs presets). Refuses while anything is queued or
    running (one scan in flight; clear_queue is the only remover).
    Preflight warnings return as ``needs_acknowledgement`` — show them to
    the human, then resubmit naming each acknowledged check; the answers
    are recorded in the run's provenance. Planned shots are capped
    (default 1,000); optimize submissions need an explicit max_iterations.
    """
    return await _run_guarded(
        _submit_scan_impl, request, preset, description, acknowledge_warnings
    )


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
    if running and not force:
        owner = running.get("user")
        if owner and owner != runtime.client_identity():
            return errors.make_error(
                "policy_refusal",
                f"the running scan was submitted by {owner!r} — pass "
                "force=true only if the operator explicitly asks for the stop",
            )
    ok, message = client.stop_scan()
    if not ok:
        return errors.make_error("worker_refused", message)
    return errors.make_ok(message=message, forced=bool(force and running))


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
# scan_progress (poll-shaped, read-only)
# ---------------------------------------------------------------------------


def _scan_progress_impl() -> str:
    """Coarse poll: RE state, the running item, and the last outcome."""
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
    return errors.make_ok(
        state=status.re_state or "idle",
        running_item=running,
        items_in_queue=status.items_in_queue,
        last_completed=last,
    )


@mcp.tool(name=tool_names.SCAN_PROGRESS)
async def scan_progress() -> str:
    """Poll-shaped progress for the current scan.

    RE state (idle/running/paused/…), the running item (with its
    submitting client), queue depth, and the last completed item's
    outcome. Coarse by design in v1 — per-shot counts arrive with the
    document-stream upgrade (v2).
    """
    return await _run_guarded(_scan_progress_impl)
