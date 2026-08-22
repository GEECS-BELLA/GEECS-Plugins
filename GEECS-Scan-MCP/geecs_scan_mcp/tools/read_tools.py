"""The v0 read-only tools: status, history, results, configs, validation.

Conventions (the osprey bluesky-server pattern): ``async def`` tool
wrappers whose blocking body runs via ``anyio.to_thread.run_sync``; every
return is a JSON envelope from :mod:`geecs_scan_mcp.errors` — tools never
raise to the agent, and engine message text is preserved verbatim.  Each
tool's synchronous ``_*_impl`` sibling is the tested surface; the
wrappers are transport glue.  Singletons come from
:mod:`geecs_scan_mcp.runtime` via module-attribute calls (the patch seam).
"""

from __future__ import annotations

import logging
from datetime import date

import anyio

from geecs_scan_mcp import errors, runtime, tool_names
from geecs_scan_mcp.server import mcp

logger = logging.getLogger("geecs_scan_mcp.tools.read")


async def _run_guarded(impl, *args) -> str:
    """Dispatch one impl to a worker thread; a raise becomes an envelope.

    The tools-never-raise backstop: every async wrapper routes through
    here, so a bug anywhere in an impl reaches the agent as
    ``internal_error`` instead of a protocol exception.
    """
    try:
        return await anyio.to_thread.run_sync(impl, *args)
    except Exception as exc:
        logger.exception("tool %s failed", getattr(impl, "__name__", impl))
        return errors.make_error("internal_error", str(exc))


#: Cap on per-column summary statistics in get_scan_result — metadata and
#: shapes always return; a wide run's full stats table would bloat agent
#: context for little value.
_MAX_STAT_COLUMNS = 40

#: The list_scan_configs kinds and the resolver capability each maps to.
_CONFIG_KINDS = (
    "save_sets",
    "trigger_profiles",
    "presets",
    "optimizer_configs",
    "scan_variables",
    "actions",
)


# ---------------------------------------------------------------------------
# scan_status
# ---------------------------------------------------------------------------


def _scan_status_impl() -> str:
    """One manager snapshot plus the queue's pending items."""
    client = runtime.get_queue_client()
    status = client.status()
    pending: list[dict] = []
    if status.connected:
        try:
            pending = [
                {
                    "item_uid": item.get("item_uid"),
                    "plan": (item.get("name") or item.get("plan")),
                    "user": item.get("user"),
                }
                for item in client.queue_items()
            ]
        except Exception as exc:  # queue read is best-effort beside status
            logger.debug("queue read failed: %s", exc)
    return errors.make_ok(
        connected=status.connected,
        re_state=status.re_state,
        manager_state=status.manager_state,
        worker_exists=status.worker_exists,
        items_in_queue=status.items_in_queue,
        running_item_uid=status.running_item_uid,
        pending_items=pending,
        detail=status.detail,
    )


@mcp.tool(name=tool_names.SCAN_STATUS)
async def scan_status() -> str:
    """One snapshot of the GEECS RE Manager: engine state, queue, worker.

    Always answers — ``connected: false`` with a ``detail`` string when
    the manager is unreachable. ``pending_items`` non-empty usually means
    a failed item returned to the queue front (nothing new can be
    submitted until an operator clears it).
    """
    return await _run_guarded(_scan_status_impl)


# ---------------------------------------------------------------------------
# scan_history
# ---------------------------------------------------------------------------


def _scan_history_impl(limit: int) -> str:
    """The manager's most recent history items, newest last, field-tolerant."""
    client = runtime.get_queue_client()
    try:
        items = client.history_items()
    except Exception as exc:
        return errors.make_error("manager_unreachable", str(exc))
    tail = items[-max(1, int(limit)) :]
    rows = []
    for item in tail:
        result = item.get("result") or {}
        traceback_text = (result.get("traceback") or "").strip()
        rows.append(
            {
                "plan": item.get("name"),
                "user": item.get("user"),
                "exit_status": result.get("exit_status"),
                "msg": result.get("msg") or None,
                "scan_ids": result.get("scan_ids"),
                "run_uids": result.get("run_uids"),
                # The last traceback line carries the operator-facing error.
                "error": traceback_text.splitlines()[-1] if traceback_text else None,
            }
        )
    return errors.make_ok(items=rows, total_in_history=len(items))


@mcp.tool(name=tool_names.SCAN_HISTORY)
async def scan_history(limit: int = 10) -> str:
    """Recent queue items and how they ended (newest last).

    ``error`` carries the final line of a failed item's traceback — the
    engine's operator-facing message (e.g. which device was down).
    """
    return await _run_guarded(_scan_history_impl, limit)


# ---------------------------------------------------------------------------
# get_scan_result
# ---------------------------------------------------------------------------


def _finite(value) -> float | None:
    """A JSON-safe float: non-finite (NaN/inf) reads as ``None``.

    ``json.dumps`` serializes NaN as a bare ``NaN`` token — invalid JSON
    that strict consumers reject — and NaN stats are routine here: a
    one-row run's ddof=1 std, or an all-NaN column from a device that
    was dead for the whole scan (the event schema's designed null cell).
    """
    import math

    number = float(value)
    return number if math.isfinite(number) else None


def _summarize_dataframe(data) -> dict:
    """Column names plus capped per-column mean/std for numeric columns."""
    if data is None:
        return {"columns": [], "rows": 0, "stats": {}}
    columns = [str(c) for c in data.columns]
    stats: dict[str, dict] = {}
    numeric = data.select_dtypes(include="number")
    for column in list(numeric.columns)[:_MAX_STAT_COLUMNS]:
        series = numeric[column]
        stats[str(column)] = {
            "mean": _finite(series.mean()),
            "std": _finite(series.std()),
        }
    return {"columns": columns, "rows": int(len(data)), "stats": stats}


def _get_scan_result_impl(
    scan_number: int | None, day: str | None, uid: str | None
) -> str:
    """Completed-run lookup: by uid, or by day-scoped GEECS scan number."""
    try:
        when = date.fromisoformat(day) if day else date.today()
    except ValueError as exc:  # bad day string — decided before any catalog I/O
        return errors.make_error("invalid_request", str(exc))
    catalog = runtime.get_catalog()
    try:
        if uid:
            detail = catalog.load_run(uid)
        else:
            if scan_number is None:
                return errors.make_error(
                    "invalid_request", "pass scan_number (with optional day) or uid"
                )
            experiment = runtime.get_experiment()
            if not experiment:
                return errors.make_error(
                    "invalid_request",
                    "no experiment configured ([Experiment] expt in config.ini)",
                )
            runs = catalog.list_runs(experiment, when)
            match = next((r for r in runs if r.scan_number == int(scan_number)), None)
            if match is None:
                return errors.make_error(
                    "not_found",
                    f"no Scan{int(scan_number):03d} in the archive for "
                    f"{experiment} on {when.isoformat()}",
                )
            detail = catalog.load_run(match.uid)
    except KeyError as exc:  # the catalog's unknown-uid contract
        return errors.make_error("not_found", f"no run with uid {exc}")
    except Exception as exc:  # network / unconfigured catalog
        return errors.make_error("tiled_unreachable", str(exc))
    summary = detail.summary
    start = detail.start_doc or {}
    return errors.make_ok(
        uid=summary.uid,
        scan_number=summary.scan_number,
        mode=summary.mode,
        shots=summary.shots,
        exit_status=summary.exit_status,
        experiment=summary.experiment,
        description=summary.description,
        save_sets=list(summary.save_sets),
        start_time=summary.start_time,
        scan_folder=start.get("scan_folder"),
        submission=start.get("submission"),
        data=_summarize_dataframe(detail.data),
    )


@mcp.tool(name=tool_names.GET_SCAN_RESULT)
async def get_scan_result(
    scan_number: int | None = None,
    day: str | None = None,
    uid: str | None = None,
) -> str:
    """Look up one completed run in the Tiled archive.

    Pass ``scan_number`` (day-scoped; ``day`` as YYYY-MM-DD, default
    today in the server host's local timezone) or a run ``uid``. Returns run metadata — including the
    ``submission`` provenance record — plus column names and capped
    per-column mean/std. Never the full event table.
    """
    return await _run_guarded(_get_scan_result_impl, scan_number, day, uid)


# ---------------------------------------------------------------------------
# list_scan_configs
# ---------------------------------------------------------------------------


def _scan_variable_row(name: str, spec) -> dict:
    """A compact catalog row from the real schema shape.

    ``ScanVariable`` carries ``target``/``kind``/``confirm``;
    ``PseudoScanVariable`` carries ``kind``/``targets``/``mode``.  Limits
    and units deliberately do NOT live in this schema (device limits are
    hardware truth — the schema module's own rule), so rows never carry
    bounds; an agent that needs limits must not infer "unbounded" from
    their absence here.
    """
    row: dict = {"name": name, "kind": getattr(spec, "kind", None)}
    target = getattr(spec, "target", None)
    if target is not None:
        row["target"] = target
    confirm = getattr(spec, "confirm", None)
    if confirm is not None:
        row["confirm"] = confirm
    targets = getattr(spec, "targets", None)
    if targets is not None:
        row["targets"] = [getattr(t, "target", str(t)) for t in targets]
        mode = getattr(spec, "mode", None)
        if mode is not None:
            row["mode"] = getattr(mode, "value", mode)
    return row


def _list_scan_configs_impl(kind: str) -> str:
    """The experiment's name catalog for one config kind."""
    if kind not in _CONFIG_KINDS:
        return errors.make_error(
            "invalid_request", f"kind must be one of {', '.join(_CONFIG_KINDS)}"
        )
    resolver = runtime.get_resolver()
    if resolver is None:
        return errors.make_error(
            "invalid_request",
            "no experiment configured ([Experiment] expt in config.ini)",
        )
    try:
        if kind == "scan_variables":
            catalog = resolver.scan_variable_catalog()
            names = [
                _scan_variable_row(name, spec)
                for name, spec in sorted(catalog.variables.items())
            ]
        elif kind == "actions":
            names = sorted(resolver.action_plan_registry())
        else:
            names = getattr(resolver, f"list_{kind}")()
    except Exception as exc:  # resolver failures read as empty-with-message
        return errors.make_error("not_found", f"listing {kind} failed: {exc}")
    return errors.make_ok(kind=kind, names=names, experiment=runtime.get_experiment())


@mcp.tool(name=tool_names.LIST_SCAN_CONFIGS)
async def list_scan_configs(kind: str) -> str:
    """The experiment's config catalogs — the names a ScanRequest may use.

    ``kind``: save_sets | trigger_profiles | presets | optimizer_configs |
    scan_variables | actions. NEVER invent catalog names — resolve them
    here. scan_variables rows carry kind/target(s) — never limits (device
    limits are hardware truth, not catalog data; absence here does NOT
    mean unbounded).
    """
    return await _run_guarded(_list_scan_configs_impl, kind)


# ---------------------------------------------------------------------------
# validate_scan_request
# ---------------------------------------------------------------------------


def _validate_scan_request_impl(request: dict) -> str:
    """Schema validation plus the full client-side preflight — no submission."""
    from geecs_schemas import ScanRequest

    try:
        validated = ScanRequest.model_validate(request)
    except Exception as exc:
        return errors.make_ok(valid=False, refusal=str(exc), warnings=[])
    experiment = runtime.get_experiment()
    if not experiment:
        return errors.make_error(
            "invalid_request",
            "no experiment configured ([Experiment] expt in config.ini)",
        )
    from geecs_bluesky.qs_client import run_submit_preflight

    report = run_submit_preflight(validated, experiment)
    if report.refusal is not None:
        return errors.make_ok(valid=False, refusal=report.refusal, warnings=[])
    warnings = [
        {"check": q.check, "title": q.title, "message": q.message}
        for q in report.questions
    ]
    outcomes = [
        {"check": check, "result": result, "detail": detail}
        for check, result, detail in report.outcomes
    ]
    return errors.make_ok(
        valid=True, refusal=None, warnings=warnings, outcomes=outcomes
    )


@mcp.tool(name=tool_names.VALIDATE_SCAN_REQUEST)
async def validate_scan_request(request: dict) -> str:
    """Full dry-run of a ScanRequest dict; nothing is submitted.

    Runs schema validation, the engine's own validation, and the
    client-side preflight (unserved variables, device liveness, trigger
    staleness).

    ``valid: false`` with ``refusal`` means fix the
    request; ``warnings`` are the questions an operator would be asked
    (each will require explicit acknowledgement at submission, once the
    v1 submit verb exists). Costs a DB query and a few CA reads.
    """
    return await _run_guarded(_validate_scan_request_impl, request)
