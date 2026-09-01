"""The control tools: submit, stop, clear-queue, progress, and the v2 verbs.

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

The v2 verbs (#676) extend the same doctrine: ``run_action`` /
``move_scan_variable`` are idle-only writes gated like submission
(``describe_action`` is their read-only preview), ``pause_scan`` joins
the halt family (never behind the headless gate — see
``tool_names.STOP_TOOLS``), ``resume_scan`` restarts motion so it gates
like a submission, and ``scan_progress`` grows the best-effort
document-stream picture (:mod:`.progress_stream`).
"""

from __future__ import annotations

import logging

from geecs_mcp import errors, runtime, tool_names
from geecs_mcp.server import mcp
from geecs_mcp.scans.read_tools import _run_guarded

logger = logging.getLogger("geecs_mcp.scans.control")


# ---------------------------------------------------------------------------
# submit_scan
# ---------------------------------------------------------------------------


#: The preflight checks whose warnings can be acknowledged.  Hand-kept in
#: sync with the ``PreflightOutcome.check`` vocabulary of
#: ``geecs_bluesky.qs_client.submit_preflight`` (fail-closed on drift: a
#: new check's question could not be acknowledged until this tuple is
#: updated — update BOTH when adding a check).  Names outside
#: this vocabulary in ``acknowledge_warnings`` are refused (typo guard).
#: HONEST RESIDUAL (review finding): the server is stateless, so an agent
#: that pre-acknowledges these known names on its FIRST call skips the
#: warning round trip — the backstop is OSPREY's approval prompt, which
#: shows the tool arguments (a human sees the pre-acknowledgement), and
#: the provenance record, which stamps ``continued`` only for questions
#: actually raised.
_ACKNOWLEDGEABLE_CHECKS = (
    "unserved_variables",
    "gateway_liveness",
    "free_run_staleness",
)


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
    # planned_shots() is THE schema derivation — arithmetic, never
    # materializing positions, so an agent-composed pathological range
    # ({start: 0, end: 1e15, step: 1e-9}) is counted, not expanded
    # (review finding: the old expanding count made the guard the crash).
    cap = runtime.max_shots()
    shots = validated.planned_shots()
    if shots is None:
        return errors.make_error(
            "policy_refusal",
            "optimize submissions need an explicit max_iterations (without "
            "one the engine applies its own default budget; the agent cap "
            f"of {cap} shots needs the number stated up front)",
        )
    if shots > cap:
        return errors.make_error(
            "policy_refusal",
            f"{shots} planned shots exceeds the agent cap of {cap} "
            "([mcp] max_shots) — shrink the scan or have an operator "
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
    from geecs_bluesky.qs_client import build_submission_record, run_submit_preflight

    report = run_submit_preflight(validated, experiment)
    if report.refusal is not None:
        return errors.make_error("invalid_request", report.refusal)
    acknowledged = set(acknowledge_warnings or [])
    unknown = acknowledged - set(_ACKNOWLEDGEABLE_CHECKS)
    if unknown:
        return errors.make_error(
            "invalid_request",
            f"unknown acknowledge_warnings name(s): {', '.join(sorted(unknown))} "
            f"— acknowledgeable checks are {', '.join(_ACKNOWLEDGEABLE_CHECKS)}",
        )
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
    record = build_submission_record(outcomes, client=runtime.client_identity())

    # -- queue it (never clear_pending) --------------------------------------
    # The record travels beside the request (request/record split,
    # geecs-schemas 0.14.0).
    result = client.submit_scan(
        validated.model_dump(mode="json"),
        submission=record.model_dump(mode="json"),
        clear_pending=False,
    )
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
# run_action / describe_action
# ---------------------------------------------------------------------------


def _run_action_impl(name: str) -> str:
    """Queue the named ActionPlan (idle-only, same etiquette as submit)."""
    name = (name or "").strip()  # validated stripped ⇒ submitted stripped
    if not name:
        return errors.make_error("invalid_request", "pass the action plan's name")
    client = runtime.get_queue_client()
    status = client.status()
    if not status.connected:
        return errors.make_error("manager_unreachable", status.detail)
    # Actions ride the queue: submitting while a scan runs would silently
    # queue the action to auto-run the moment the scan finishes (the queue
    # is already started) — refuse instead, exactly like submit_scan.
    if status.re_state not in (None, "idle"):
        return errors.make_error(
            "policy_refusal",
            f"a scan is active (RE state: {status.re_state}) — actions are "
            "idle-only; wait for it or stop it first",
        )
    result = client.submit_action(name)
    if not result.ok:
        if result.pending_items:
            return errors.make_error(
                "policy_refusal",
                result.message,
                pending_items=[
                    {
                        "item_uid": item.get("item_uid"),
                        "plan": item.get("name"),
                        "user": item.get("user"),
                    }
                    for item in result.pending_items
                ],
            )
        return errors.make_error(
            "worker_refused", result.message or "action submission refused"
        )
    return errors.make_ok(
        item_uid=result.item_uid,
        message=result.message,
        submitted_as=runtime.client_identity(),
    )


@mcp.tool(name=tool_names.RUN_ACTION)
async def run_action(name: str) -> str:
    """Run a named action plan on demand (idle-only).

    Actions are the experiment's configured rituals (insert a screen,
    block the laser, …) — list them with list_scan_configs and preview
    the exact steps with describe_action first. Refuses while a scan is
    active or anything is queued. Returns the queue item_uid; observe
    completion with scan_progress / scan_history.
    """
    return await _run_guarded(_run_action_impl, name)


def _describe_action_impl(name: str) -> str:
    """Dry-run the named action against the worker's configs (idle-only)."""
    name = (name or "").strip()
    if not name:
        return errors.make_error("invalid_request", "pass the action plan's name")
    client = runtime.get_queue_client()
    status = client.status()
    if not status.connected:
        return errors.make_error("manager_unreachable", status.detail)
    try:
        steps = client.describe_action(name)
    except Exception as exc:
        return errors.make_error(_task_error_kind(exc), str(exc))
    steps = list(steps or [])
    return errors.make_ok(action=name, step_count=len(steps), steps=steps)


@mcp.tool(name=tool_names.DESCRIBE_ACTION)
async def describe_action(name: str) -> str:
    """Preview a named action plan's resolved steps without running it.

    A worker-side dry-run against the live configs — the flattened
    device writes/waits the action would perform, in order. Read-only,
    but needs an idle manager to answer (refused mid-scan).
    """
    return await _run_guarded(_describe_action_impl, name)


# ---------------------------------------------------------------------------
# move_scan_variable
# ---------------------------------------------------------------------------


def _task_error_kind(exc: Exception) -> str:
    """``task_timeout`` for the client's task-poll timeout, else worker_refused.

    The string match is the only seam: ``_wait_for_task`` raises a plain
    ``RuntimeError("worker task did not finish within N s")`` on timeout
    (same type as every other task failure).
    """
    return "task_timeout" if "did not finish within" in str(exc) else "worker_refused"


def _move_scan_variable_impl(name: str, value: float) -> str:
    """One manual scan-variable move on the worker (idle-only, blocking)."""
    import math

    name = (name or "").strip()
    if not name:
        return errors.make_error("invalid_request", "pass the scan variable's name")
    try:
        target = float(value)
    except (TypeError, ValueError):
        return errors.make_error("invalid_request", f"value {value!r} is not a number")
    if not math.isfinite(target):
        return errors.make_error("invalid_request", f"value {value!r} is not finite")
    client = runtime.get_queue_client()
    status = client.status()
    if not status.connected:
        return errors.make_error("manager_unreachable", status.detail)
    try:
        result = client.move_variable(name, target)
    except Exception as exc:
        return errors.make_error(_task_error_kind(exc), str(exc))
    return errors.make_ok(variable=name, requested=target, result=result)


@mcp.tool(name=tool_names.MOVE_SCAN_VARIABLE)
async def move_scan_variable(name: str, value: float) -> str:
    """Move one scan variable to a value, outside any scan.

    Prefer a catalog scan-variable name from list_scan_configs (plain,
    confirm, or pseudo — the worker resolves it exactly as a scan axis
    would, confirmation included). A raw ``Device:Variable`` string is
    ALSO accepted (the worker's manual-move surface): a direct setpoint
    write with no catalog semantics — setpoint limits are whatever the
    gateway enforces on that variable. Idle-only (the manager refuses
    while a scan runs) and BLOCKING: the call returns when the move
    completes or fails, up to ~120 s. The result carries the worker's
    move report verbatim.
    """
    return await _run_guarded(_move_scan_variable_impl, name, value)


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

    Lazy is the stdio posture; the HTTP service warms the same cache at
    startup (``__main__.warm_progress_stream``, #685) and this call is
    then the idempotent no-op.  Best-effort BY DESIGN: ``available=false`` (with the reason) when the
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
