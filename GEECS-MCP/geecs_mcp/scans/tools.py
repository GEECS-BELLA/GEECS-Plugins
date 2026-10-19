"""The scans domain over the web scanner's HTTP API.

Every verb here is one of the scanner's routes (``GeecsScanner``,
:8300) seen through the envelope — what the agent does is what an
operator does on the scanner page, with the scanner's own preflight,
limits and words.  Nothing is re-derived: no preset expansion, no
ownership, no shot cap (DESIGN.md §4).

Tools are ``async def`` over :func:`geecs_mcp.clients.call`; each tool's
``_*`` sibling is the tested surface and the wrapper is transport glue
through :func:`geecs_mcp.guard.run_async`.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import anyio

from geecs_mcp import clients, errors, guard, tool_names
from geecs_mcp.clients import ServiceRefusal, ServiceTimeout, ServiceUnavailable
from geecs_mcp.identity import InvalidArgument, identity_for
from geecs_mcp.server import mcp
from geecs_mcp.waits import (
    BUDGET_SPENT,
    DEFAULT_WAIT_S,
    CallBudget,
    heartbeat,
    keepalive,
)
from geecs_mcp.waits import bounded_wait as _bounded_wait

#: The scanner's ``GET /api/configs/{kind}`` kinds (the page's dropdowns).
CONFIG_KINDS = (
    "presets",
    "trigger_profiles",
    "scan_variables",
    "actions",
    "optimizer_configs",
)

#: Polling cadence (s): the scanner page itself polls at 1 Hz.
SCAN_POLL_S = 2.0
MOVE_POLL_S = 1.0
#: A move's own wait cap: a move waits less than a scan, leaving its
#: readback room inside waits.CALL_BUDGET_S.
MOVE_MAX_WAIT_S = 60.0
#: The scanner holds a stop up to 120 s for the pause to land; the MCP
#: waits at most this long (under the client's 2-minute auto-background,
#: waits.CALL_BUDGET_S) and then answers that the stop has not landed yet.
STOP_READ_TIMEOUT_S = 100.0
#: How deep into the manager's history the wait verbs look.
HISTORY_DEPTH = 50
#: The scanner reads the running item, the waiting items and the history
#: in separate manager round trips, so an item moving waiting → running
#: can be in none of the three for one read.  An
#: item seen nowhere is re-read this many times before it is "not found".
VANISHED_REREADS = 3


async def _sleep(seconds: float) -> None:
    """The wait verbs' pause — a module attribute so tests can step a fake manager."""
    await anyio.sleep(seconds)


def _clock() -> float:
    return time.monotonic()


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


async def _scan_status() -> str:
    status = await clients.call("scanner", "GET", "/api/status")
    # /api (not /health): the version without a second manager round trip
    about = await clients.call("scanner", "GET", "/api")
    return errors.make_ok(**status, scanner_version=(about or {}).get("version"))


@mcp.tool(name=tool_names.SCAN_STATUS)
async def scan_status() -> str:
    """The scanner's status snapshot: manager connection, RE state, queue depth, readiness.

    ``re_state`` is the manager's word (``idle``/``running``/``paused``);
    ``items_in_queue`` counts what waits; ``readiness`` + ``readiness_detail``
    say whether a submit would be accepted now; ``identity`` is the name
    the scanner submits under.  Read-only.
    """
    return await guard.run_async(_scan_status)


async def _scan_progress() -> str:
    progress = await clients.call("scanner", "GET", "/api/progress")
    return errors.make_ok(**progress)


@mcp.tool(name=tool_names.SCAN_PROGRESS)
async def scan_progress() -> str:
    """The running (or last) scan's progress: scan number, shots done of planned, state.

    ``state`` is ``running`` / ``paused`` / ``done`` / ``aborted``;
    ``paused_reason`` carries the failed-move line while paused;
    ``scan_folder`` and ``day`` name where the data lands.  Read-only.
    """
    return await guard.run_async(_scan_progress)


async def _scan_history(limit: int) -> str:
    limit = max(1, min(int(limit), 200))
    queue = await clients.call(
        "scanner", "GET", "/api/queue", params={"history": limit}
    )
    return errors.make_ok(**queue)


@mcp.tool(name=tool_names.SCAN_HISTORY)
async def scan_history(limit: int = 10) -> str:
    """The queue as the scanner shows it: the running item, what waits, the last finished.

    Each row carries ``item_uid``, ``plan``, ``summary``, ``user``,
    ``state``/``word`` and, for finished scans, ``scan_numbers`` and
    ``run_uids`` — the keys the analysis and logbook verbs take.
    """
    return await guard.run_async(_scan_history, limit)


async def _list_scan_configs(kind: str, name: Optional[str]) -> str:
    kind = str(kind).strip()
    if kind == "devices":
        if name:
            return errors.make_error(
                "invalid_request", "only a preset can be fetched by name"
            )
        # The manager lists a device and its children (U_S1H, U_S1H.current
        # …); a preset names devices, so only the bare names are offered —
        # the scanner page's add-device list does the same.
        references = await clients.call("scanner", "GET", "/api/devices")
        names = [n for n in references or [] if "." not in str(n)]
        return errors.make_ok(kind=kind, names=names)
    if kind not in CONFIG_KINDS:
        return errors.make_error(
            "not_found",
            f"unknown config kind {kind!r}",
            kinds=[*CONFIG_KINDS, "devices"],
        )
    if name:
        if kind != "presets":
            return errors.make_error(
                "invalid_request", "only a preset can be fetched by name"
            )
        document = await clients.call("scanner", "GET", f"/api/configs/presets/{name}")
        return errors.make_ok(kind=kind, name=name, preset=document)
    listing = await clients.call("scanner", "GET", f"/api/configs/{kind}")
    return errors.make_ok(**listing)


@mcp.tool(name=tool_names.LIST_SCAN_CONFIGS)
async def list_scan_configs(kind: str, name: Optional[str] = None) -> str:
    """The experiment's config catalog, as the scanner lists it.

    ``kind`` is one of ``presets``, ``trigger_profiles``, ``scan_variables``,
    ``actions``, ``optimizer_configs`` → the names (+ ``unavailable`` with
    reasons).  With ``name`` on ``presets``, the preset document itself.
    ``devices`` → the device names a preset's ``devices[].device`` takes
    (the scanner page's add-device list).  Names must come from here —
    never invent one.
    """
    return await guard.run_async(_list_scan_configs, kind, name)


# ---------------------------------------------------------------------------
# halts and the queue
# ---------------------------------------------------------------------------


def _verb_body(force: bool, operator: Optional[str]) -> dict[str, Any]:
    """``{force, operator?}`` — the operator as the identity string when named."""
    body: dict[str, Any] = {"force": bool(force)}
    if operator and operator.strip():
        body["operator"] = identity_for(operator)
    return body


def _halt_body(force: bool, operator: Optional[str]) -> tuple[dict[str, Any], str]:
    """The halt family's body, and a note when the name could not be recorded.

    An invalid optional name (over the identity cap) must never stop a
    halt: the request goes without it and the note says so.
    """
    try:
        return _verb_body(force, operator), ""
    except InvalidArgument as exc:
        return {"force": bool(force)}, f"operator not recorded: {exc}"


async def _verb(
    path: str,
    body: Optional[dict[str, Any]],
    *,
    read_timeout_s=None,
    note: str = "",
) -> str:
    out = await keepalive(
        lambda: clients.call(
            "scanner", "POST", path, json=body, read_timeout_s=read_timeout_s
        ),
        message=f"waiting for the scanner's {path}",
    )
    if not out.get("ok", False):
        # The scanner answers 200 {ok:false} for a verb that does not apply
        # (nothing running, not paused, the stop budget expired) — a verdict
        # on the ask, so policy_refusal with the scanner's words.
        return errors.make_error("policy_refusal", str(out.get("message") or "refused"))
    payload: dict[str, Any] = {"message": str(out.get("message") or "")}
    if note:
        payload["note"] = note
    return errors.make_ok(**payload)


async def _halt(path: str, force: bool, operator: str, *, read_timeout_s=None) -> str:
    body, note = _halt_body(force, operator)  # inside the guard; never raises past it
    return await _verb(path, body, read_timeout_s=read_timeout_s, note=note)


@mcp.tool(name=tool_names.STOP_SCAN)
async def stop_scan(force: bool = False, operator: str = "") -> str:
    """Stop the running scan gracefully (after the current shot) — the halt family.

    Waits for the scanner to report the stop landed, up to 100 s; if it
    has not by then the answer is ``ok`` with ``landed: false`` — the
    request is in and the scanner keeps landing it; check
    ``scan_status``.  ``operator`` is recorded when given; a halt is never
    blocked on a missing or malformed name.  ``force`` is passed through
    for the scanner's ownership registry.
    """
    return await guard.run_async(_stop, force, operator)


async def _stop(force: bool, operator: str) -> str:
    try:
        return await _halt(
            "/api/stop", force, operator, read_timeout_s=STOP_READ_TIMEOUT_S
        )
    except ServiceTimeout as exc:
        # Sent, not yet answered: the scanner is landing it (its own budget
        # is 120 s) — say so rather than report an outage.
        return errors.make_ok(
            landed=False,
            message=(
                f"stop requested; the scanner had not confirmed it after "
                f"{STOP_READ_TIMEOUT_S:.0f} s and keeps landing it — check "
                f"scan_status before acting on it ({exc.message})"
            ),
        )


@mcp.tool(name=tool_names.PAUSE_SCAN)
async def pause_scan(force: bool = False, operator: str = "") -> str:
    """Pause the running scan at the next step boundary — the halt family.

    Deferred on the manager: ``scan_status.re_state`` says ``paused`` once
    it lands.  A pause makes the machine strictly quieter; ``operator`` is
    recorded when given and never required.
    """
    return await guard.run_async(_halt, "/api/pause", force, operator)


async def _resume(operator: str, force: bool) -> str:
    identity_for(
        operator
    )  # required on a go verb — inside the guard, so → invalid_request
    return await _verb("/api/resume", _verb_body(force, operator))


@mcp.tool(name=tool_names.RESUME_SCAN)
async def resume_scan(operator: str, force: bool = False) -> str:
    """Resume a paused scan — restarts motion and acquisition, so it is a WRITE.

    ``operator`` names the person supervising; the scanner records it.
    Refused (``policy_refusal``) when nothing is paused.
    """
    return await guard.run_async(_resume, operator, force)


@mcp.tool(name=tool_names.CLEAR_QUEUE)
async def clear_queue() -> str:
    """Remove every WAITING item from the queue — the running one is untouched."""
    return await guard.run_async(_verb, "/api/clear", None)


# ---------------------------------------------------------------------------
# submit_preset
# ---------------------------------------------------------------------------


async def _preset_document(
    name: Optional[str], preset: Optional[dict[str, Any]]
) -> tuple[Optional[dict[str, Any]], str]:
    """The preset document to send, and an error envelope when there is none.

    Exactly one of a saved preset's ``name`` or a whole ``preset``
    document.  A document goes to the scanner as given: its schema, its
    names and its limits are the scanner's to check (preflight).
    """
    name = str(name or "").strip()
    if bool(name) == (preset is not None):
        raise InvalidArgument(
            "give exactly one of `name` (a saved preset, list_scan_configs"
            "('presets')) or `preset` (a whole preset document)"
        )
    if preset is not None:
        if not isinstance(preset, dict):
            raise InvalidArgument("`preset` must be a preset document (an object)")
        return preset, ""
    # the scanner has no submit-by-name: fetch the document first
    try:
        document = await clients.call("scanner", "GET", f"/api/configs/presets/{name}")
    except ServiceRefusal as exc:
        if exc.kind != "not_found":
            raise
        listing = await clients.call("scanner", "GET", "/api/configs/presets")
        return None, errors.make_error(
            "not_found", exc.message, names=list(listing.get("names") or [])
        )
    return document, ""


async def _preflight_preset(
    name: Optional[str], preset: Optional[dict[str, Any]] = None
) -> str:
    document, refused = await _preset_document(name, preset)
    if document is None:
        return refused
    report = await clients.call("scanner", "POST", "/api/preflight", json=document)
    return errors.make_ok(
        refusal=report.get("refusal"),
        questions=list(report.get("questions") or []),
        outcomes=report.get("outcomes") or [],
        summary=report.get("summary", ""),
        planned_shots=report.get("planned_shots"),
        plan=report.get("plan"),
    )


@mcp.tool(name=tool_names.PREFLIGHT_PRESET)
async def preflight_preset(
    name: Optional[str] = None, preset: Optional[dict[str, Any]] = None
) -> str:
    """The scanner's preflight of a scan, with nothing submitted — a read.

    Give a saved preset's ``name`` or a whole ``preset`` document (the
    shape ``list_scan_configs('presets', name=...)`` returns:
    ``devices`` + ``plan``).  The answer is the scanner's verdict:
    ``refusal`` (why it cannot run, else null), ``questions`` it would
    ask before queueing, the per-check ``outcomes``, and for a runnable
    scan the expanded ``plan``, ``summary`` and ``planned_shots``.  Show
    it to the operator before ``submit_preset``.
    """
    return await guard.run_async(_preflight_preset, name, preset)


async def _submit_preset(
    name: Optional[str],
    acknowledged: list[str],
    operator: str,
    preset: Optional[dict[str, Any]] = None,
) -> str:
    identity = identity_for(operator)
    acknowledged = [str(a).strip() for a in (acknowledged or []) if str(a).strip()]

    # 1. the document — a saved preset's, or the one given
    document, refused = await _preset_document(name, preset)
    if document is None:
        return refused
    preset = document

    # 2. preflight — the scanner's checks, its questions verbatim
    report = await clients.call("scanner", "POST", "/api/preflight", json=preset)
    if report.get("refusal"):
        return errors.make_error(
            "invalid_request",
            str(report["refusal"]),
            outcomes=report.get("outcomes") or [],
        )
    questions = list(report.get("questions") or [])
    pending = [q for q in questions if q.get("check") not in acknowledged]
    if pending:
        # Never acknowledged here: the agent passes the check names back
        # once the supervising person has seen the questions.
        return errors.make_error(
            "policy_refusal",
            "preflight asks before this scan runs — pass the check names back in "
            "`acknowledged` once the operator has confirmed",
            needs_acknowledgement=pending,
            summary=report.get("summary", ""),
            planned_shots=report.get("planned_shots"),
        )

    # 3. submit — the scanner reruns preflight; a new question lands as a 409
    #    we return exactly like step 2 (the guard maps the kind and extras)
    out = await clients.call(
        "scanner",
        "POST",
        "/api/submit",
        json={"preset": preset, "acknowledged": acknowledged, "operator": identity},
    )
    return errors.make_ok(
        item_uid=out.get("item_uid"),
        submitted_as=out.get("submitted_as"),
        operator=identity,
        message=out.get("message", ""),
        planned_shots=out.get("planned_shots", report.get("planned_shots")),
        summary=out.get("summary") or report.get("summary", ""),
        plan=report.get("plan"),
        preflight=report.get("outcomes") or [],
        acknowledged=acknowledged,
    )


@mcp.tool(name=tool_names.SUBMIT_PRESET)
async def submit_preset(
    operator: str,
    name: Optional[str] = None,
    preset: Optional[dict[str, Any]] = None,
    acknowledged: Optional[list[str]] = None,
) -> str:
    """Queue one scan, through the scanner's own preflight.

    Give exactly one of ``name`` (a saved preset, from
    ``list_scan_configs('presets')``) or ``preset`` (a whole preset
    document, composed for this scan: ``devices`` from
    ``list_scan_configs('devices')``, a ``plan`` whose scan variable comes
    from ``list_scan_configs('scan_variables')``).  A document is never
    saved; the scanner validates it exactly as it validates the page's.
    ``operator`` is the supervising person (recorded on the run as
    ``agent, admin supervision (<operator>)``).  If preflight asks a
    question, the answer is ``policy_refusal`` with the questions under
    ``needs_acknowledgement``: show them to the operator and call again
    with their ``check`` names in ``acknowledged`` — this server never
    acknowledges on its own.  Success returns ``item_uid``, the handle
    ``wait_for_scan`` takes; the scan number exists only once the worker
    claims it.  Refused while anything is already queued (``pending_items``).
    """
    return await guard.run_async(
        _submit_preset, name, list(acknowledged or []), operator, preset
    )


# ---------------------------------------------------------------------------
# wait_for_scan
# ---------------------------------------------------------------------------


def _find_item(queue: dict, item_uid: str) -> tuple[str, Optional[dict]]:
    running = queue.get("running")
    if isinstance(running, dict) and running.get("item_uid") == item_uid:
        return "running", running
    for row in queue.get("waiting") or []:
        if row.get("item_uid") == item_uid:
            return "waiting", row
    for row in queue.get("finished") or []:
        if row.get("item_uid") == item_uid:
            return "finished", row
    return "", None


def _correlated_progress(
    where: str, row: Optional[dict], progress: Any
) -> tuple[Optional[dict], Optional[dict]]:
    """Split the snapshot into ``(the item's own progress or None, another scan's live picture)``.

    ``/api/progress`` is the scanner's latest-run picture, not a lookup by
    item: once the next queued scan starts it describes that one.  A
    finished item's progress is attached only when the snapshot names one
    of its scan numbers and has ended; a running item's only while the
    snapshot is live (a ``done``/``aborted`` snapshot is the previous
    run's, before this one's start document); a waiting item's never.
    Anything else comes back as ``live_scan`` — number and state — so the
    caller sees the machine's state without mistaking it for this item's.
    """
    if not isinstance(progress, dict):
        return None, None
    state = progress.get("state")
    if where == "finished" and row is not None:
        mine = progress.get("scan_number") in (row.get("scan_numbers") or [])
        mine = mine and state in ("done", "aborted")
    elif where == "running":
        mine = state in ("running", "paused")
    else:
        mine = False
    if mine:
        return progress, None
    return None, {"scan_number": progress.get("scan_number"), "state": state}


async def _wait_for_item(
    item_uid: str, timeout_s: float, *, poll_s: float, budget: CallBudget
) -> tuple[str, Optional[dict], Optional[dict]]:
    """Poll the queue until *item_uid* is finished or the budget is spent.

    Returns ``(where, row, progress)``: ``where`` is ``finished``,
    ``running``, ``waiting`` or ``""`` (nowhere — the manager's bounded
    history no longer holds it, or it never existed).
    """
    start = _clock()
    deadline = start + timeout_s
    vanished = 0
    while True:
        await heartbeat(_clock() - start, timeout_s, f"waiting for {item_uid}")
        queue = await clients.call(
            "scanner",
            "GET",
            "/api/queue",
            params={"history": HISTORY_DEPTH},
            read_timeout_s=budget.read_timeout(),
        )
        where, row = _find_item(queue, item_uid)
        if where == "finished":
            return where, row, None
        if not where:
            vanished += 1
            if vanished > VANISHED_REREADS:
                return where, row, None
            await _sleep(poll_s)
            continue
        vanished = 0
        if _clock() >= deadline:
            progress = await clients.call(
                "scanner", "GET", "/api/progress", read_timeout_s=budget.read_timeout()
            )
            return where, row, progress
        await _sleep(poll_s)


async def _wait_for_scan(item_uid: str, timeout_s: float) -> str:
    item_uid = str(item_uid).strip()
    if not item_uid:
        return errors.make_error(
            "invalid_request", "item_uid is the handle submit_preset returned"
        )
    budget = CallBudget()
    with budget.scope():
        return await _wait_within(item_uid, timeout_s, budget)
    # Only on expiry: the scope ended the wait mid-request.
    return errors.make_ok(
        finished=False,
        item_uid=item_uid,
        where="unknown",
        progress=None,
        live_scan=None,
        note=BUDGET_SPENT,
    )


async def _wait_within(item_uid: str, timeout_s: float, budget: CallBudget) -> str:
    where, row, progress = await _wait_for_item(
        item_uid, _bounded_wait(timeout_s), poll_s=SCAN_POLL_S, budget=budget
    )
    if where == "finished":
        snapshot = await clients.call(
            "scanner", "GET", "/api/progress", read_timeout_s=budget.read_timeout()
        )
        progress, live_scan = _correlated_progress(where, row, snapshot)
        return errors.make_ok(
            finished=True,
            item_uid=item_uid,
            state=row.get("state"),
            word=row.get("word"),
            plan=row.get("plan"),
            summary=row.get("summary"),
            scan_numbers=row.get("scan_numbers") or [],
            run_uids=row.get("run_uids") or [],
            detail=row.get("detail", ""),
            progress=progress,
            live_scan=live_scan,
        )
    if not where:
        return errors.make_error(
            "not_found",
            f"no queue item {item_uid!r} is running, waiting or in the last "
            f"{HISTORY_DEPTH} finished items — the manager's history is bounded "
            "and clearable; scan_history shows what it still holds",
        )
    progress, live_scan = _correlated_progress(where, row, progress)
    return errors.make_ok(
        finished=False,
        item_uid=item_uid,
        where=where,
        position=row.get("position"),
        plan=row.get("plan"),
        summary=row.get("summary"),
        progress=progress,
        live_scan=live_scan,
    )


@mcp.tool(name=tool_names.WAIT_FOR_SCAN)
async def wait_for_scan(item_uid: str, timeout_s: float = DEFAULT_WAIT_S) -> str:
    """Block until the queue item finishes, up to ``timeout_s`` (at most 90 s).

    Short by design: a scan outlasts one call, so call again until
    ``finished`` — each call is bounded under the client's own timers.

    Finished: ``{finished: true, state, word, scan_numbers, run_uids,
    detail, progress}`` — ``scan_numbers``/``run_uids`` are the keys for
    ``run_analysis`` and ``post_logbook_entry``.  Still going when the
    budget ends: ``{finished: false, where: running|waiting, position,
    progress}`` — a result, not an error; call again.  ``progress`` is
    THIS item's picture or null; when the scanner's live snapshot belongs
    to another scan it comes as ``live_scan`` instead.  ``not_found``
    when the item is in none of the queue's three lists.
    """
    return await guard.run_async(_wait_for_scan, item_uid, timeout_s)


# ---------------------------------------------------------------------------
# move
# ---------------------------------------------------------------------------


async def _readback_target(
    variable: str, budget: CallBudget
) -> tuple[Optional[str], str]:
    """The canonical ``Device:Variable`` to read back, or why there is none.

    The scanner's readback route takes only the canonical form (never
    the dotted ``reference`` the move reply carries): a ``Device:Variable``
    is read as given; a catalog name through the scan-variable listing's
    ``target``; a bare device or a pseudo (no single target) has none.
    """
    if ":" in variable:
        return variable, ""
    listing = await clients.call(
        "scanner", "GET", "/api/scan-variables", read_timeout_s=budget.read_timeout()
    )
    for row in listing or []:
        if row.get("name") == variable:
            target = row.get("target")
            if isinstance(target, str) and ":" in target:
                return target, ""
            return (
                None,
                f"{variable!r} is a {row.get('kind', 'pseudo')} entry with no single target to read back",
            )
    return (
        None,
        f"{variable!r} is not a Device:Variable name and not in the scan-variable catalog",
    )


async def _move(
    variable: str, value: Any, operator: str, wait: bool, timeout_s: float
) -> str:
    identity = identity_for(operator)
    variable = str(variable).strip()
    if not variable:
        return errors.make_error("invalid_request", "name the variable to move")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return errors.make_error("invalid_request", f"value {value!r} is not a number")
    if not math.isfinite(number):
        return errors.make_error("invalid_request", f"value {value!r} is not finite")

    budget = CallBudget()
    out = await clients.call(
        "scanner",
        "POST",
        "/api/move",
        json={"variable": variable, "value": number, "operator": identity},
        read_timeout_s=budget.read_timeout(),
    )
    item_uid = out.get("item_uid")
    payload: dict[str, Any] = {
        "item_uid": item_uid,
        "reference": out.get("reference"),
        "message": out.get("message", ""),
        "submitted_as": out.get("submitted_as"),
        "operator": identity,
        "finished": False,
        "readback": None,
        "readback_detail": "",
    }
    if not wait or not item_uid:
        payload["readback_detail"] = "not waited for"
        return errors.make_ok(**payload)
    # The move is ACCEPTED (item_uid): an outage or a spent budget while it
    # is followed is not a failed move, and the handle must reach the agent.
    follow = f"wait_for_scan({payload['item_uid']!r}) follows it"
    with budget.scope():
        try:
            return await _follow_move(payload, variable, timeout_s, budget)
        except ServiceUnavailable as exc:
            payload["where"] = "unknown"
            payload["readback_detail"] = (
                f"the scanner accepted the move but stopped answering while it "
                f"was followed ({exc.message}); {follow}"
            )
            return errors.make_ok(**payload)
    payload["where"] = "unknown"
    payload["readback_detail"] = f"{BUDGET_SPENT}; {follow}"
    return errors.make_ok(**payload)


async def _follow_move(
    payload: dict[str, Any], variable: str, timeout_s: float, budget: CallBudget
) -> str:
    """Wait for an accepted move, then read back — the part that may outlive the scanner."""
    where, row, _ = await _wait_for_item(
        str(payload["item_uid"]),
        min(_bounded_wait(timeout_s), MOVE_MAX_WAIT_S),
        poll_s=MOVE_POLL_S,
        budget=budget,
    )
    if where != "finished":
        payload["where"] = where or "unknown"
        payload["readback_detail"] = "the move had not finished within the budget"
        return errors.make_ok(**payload)
    payload.update(
        finished=True,
        state=row.get("state"),
        word=row.get("word"),
        detail=row.get("detail", ""),
    )
    target, why_not = await _readback_target(variable, budget)
    if target is None:
        payload["readback_detail"] = why_not
        return errors.make_ok(**payload)
    try:
        readback = await clients.call(
            "scanner",
            "GET",
            "/api/readback",
            params={"variable": target},
            read_timeout_s=budget.read_timeout(),
        )
    except ServiceRefusal as exc:
        payload["readback_detail"] = exc.message
        return errors.make_ok(**payload)
    payload["readback"] = {
        k: readback.get(k)
        for k in ("variable", "value", "units", "timestamp", "age_s", "ok", "detail")
    }
    return errors.make_ok(**payload)


@mcp.tool(name=tool_names.MOVE)
async def move(
    variable: str,
    value: float,
    operator: str,
    wait: bool = True,
    timeout_s: float = 60.0,
) -> str:
    """Move one settable through the scanner (the Bluesky ``mv`` plan), idle-only.

    ``variable`` is a catalog name, a ``Device:Variable`` or a bare
    device, resolved by the scanner exactly as a scan axis is; the range
    limits are the DB's, enforced on the setpoint by the gateway.  Refused
    (``policy_refusal``) while a plan runs or anything waits.  With
    ``wait`` the call returns once the item finished, with the readback
    (``null`` + ``readback_detail`` for a pseudo or bare device).
    """
    return await guard.run_async(_move, variable, value, operator, wait, timeout_s)
