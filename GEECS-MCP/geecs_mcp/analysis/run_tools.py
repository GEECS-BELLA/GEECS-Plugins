"""The analysis-domain execution tools: run ScanAnalyzers on demand (#686).

``run_scan_analysis`` restores a currently-dormant capability through the
new architecture (the LiveTaskRunner fleet has not run post-migration):
it enqueues and executes the ScanAnalysis pipeline for one scan, on this
serving host.  **ScanAnalysis as-is is the backend by owner decision
(2026-08-24)** — the Tiled-based analysis stack is the recorded long-term
direction, so the verb surface stays backend-neutral (names a diagnostic
or group, a scan, a day) and the ``analysis_status/`` YAMLs are the
progress contract: a future backend reports into the same files and the
existing ``get_scan_analysis`` poll never changes.

Execution shape — submit-and-poll, never a blocking tool call:

1. The tool validates everything refusable *before* side effects: exactly
   one of ``analyzer``/``group``, the configs root resolves, the scan
   folder EXISTS (analysis never creates ``scans/ScanNNN/`` — the
   cross-package invariant; a missing folder is a ``not_found`` refusal,
   pinned by a nothing-created test), and the analyzer(s) actually
   construct on this host — a diagnostic whose image-analyzer class needs
   a Windows-only SDK (HASO WaveKit, Grenouille FROG.dll) fails right
   here with a clear refusal instead of a half-run (those diagnostics
   belong to the future Windows satellite server, per the domain
   roadmap).
2. It initializes the ``analysis_status/`` files server-side
   (``init_status_for_scan``, idempotent) — so a worker that dies before
   claiming leaves *visible* queued rows, never a silent nothing.
3. It spawns a detached worker subprocess (``run_worker``) that builds
   the worklist and runs it — the ScanAnalysis task queue's own claim /
   heartbeat / stale-reclaim machinery narrates progress into the status
   files the read tools already parse.

Google-Doc upload stays hard-off (``run_worklist``'s ``gdoc_enabled``
default): publishing to the experiment log is an outward-facing action
that would need its own explicitly-gated verb.

ScanAnalysis imports are lazy and guarded — the ``analysis-run`` extra is
optional, and without it every tool here refuses with a message naming
the extra (the server always starts).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from geecs_mcp import errors, tool_names
from geecs_mcp.analysis import read_tools
from geecs_mcp.scans.read_tools import _run_guarded
from geecs_mcp.server import mcp

logger = logging.getLogger("geecs_mcp.analysis.run")

#: Cap on names returned by the listing tools — a map, not a dump.
_MAX_LISTED_NAMES = 200

_EXTRA_HINT = (
    "ScanAnalysis is not installed on the serving host — install the "
    "geecs-mcp 'analysis-run' extra (poetry install -E analysis-run)"
)

_ROOT_HINT = (
    "scan-analysis configs root not configured on the serving host — set "
    "SCAN_ANALYSIS_CONFIG_DIR or config.ini [Paths] "
    "scan_analysis_configs_path (the GEECS-Plugins-Configs checkout)"
)


def _config_root() -> Optional[Path]:
    """The resolved scan-analysis configs root, or ``None`` unconfigured.

    A module-level seam (tests patch it); production defers to the shared
    ``geecs_data_utils.config_roots`` manager — the same resolution every
    ScanAnalysis consumer uses.
    """
    from geecs_data_utils.config_roots import scan_analysis_config

    return scan_analysis_config.base_dir


def _spawn_worker(payload: dict) -> int:
    """Launch the detached run_worker subprocess; return its pid.

    A module-level seam (tests capture the payload instead of spawning).
    The child is detached (own session on POSIX) with all stdio dropped —
    its observable output is the status files; analyzer failures land in
    each task's ``error`` field via ``run_worklist``'s own capture.  The
    unwaited child is reaped by the stdlib's internal bookkeeping on a
    later spawn — acceptable for the verb's low call rate.
    """
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, our own module
        [sys.executable, "-m", "geecs_mcp.analysis.run_worker", json.dumps(payload)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name == "posix"),
    )
    return proc.pid


# ---------------------------------------------------------------------------
# listings
# ---------------------------------------------------------------------------


def _listing_impl(kind: str) -> str:
    """Shared body for the two discovery listings (``kind``: analyzers|groups)."""
    try:
        from scan_analysis.config import discover_analyzers, discover_groups
    except ImportError as exc:
        return errors.make_error("invalid_request", f"{_EXTRA_HINT}: {exc}")
    root = _config_root()
    if root is None:
        return errors.make_error("invalid_request", _ROOT_HINT)
    discover = discover_analyzers if kind == "analyzers" else discover_groups
    try:
        index = discover(root)
    except FileNotFoundError as exc:
        return errors.make_error("not_found", str(exc))
    except ValueError as exc:  # duplicate stems — a configs-repo defect
        return errors.make_error("invalid_request", str(exc))
    names = sorted(index)
    # count = distinct config FILES: discover_groups indexes each group
    # under two accepted names (bare stem + namespace/stem), and an agent
    # asking "how many groups exist" must not get double.
    payload: dict[str, Any] = {
        kind: names[:_MAX_LISTED_NAMES],
        "count": len(set(index.values())),
        "truncated": len(names) > _MAX_LISTED_NAMES,
        "config_root": str(root),
    }
    if kind == "groups":
        payload["note"] = (
            "names include both bare stems and namespace/stem forms — "
            "every listed name is accepted; count is distinct groups"
        )
    return errors.make_ok(**payload)


def _list_analyzers_impl() -> str:
    """Diagnostic IDs ``run_scan_analysis(analyzer=...)`` accepts."""
    return _listing_impl("analyzers")


def _list_analysis_groups_impl() -> str:
    """Group names ``run_scan_analysis(group=...)`` accepts."""
    return _listing_impl("groups")


@mcp.tool(name=tool_names.LIST_ANALYZERS)
async def list_analyzers() -> str:
    """List the diagnostic IDs available to run_scan_analysis.

    Returns
    -------
    str
        JSON envelope with ``analyzers`` (sorted diagnostic IDs from the
        configs repo's ``analyzers/`` tree), ``count``, and ``truncated``.
    """
    return await _run_guarded(_list_analyzers_impl)


@mcp.tool(name=tool_names.LIST_ANALYSIS_GROUPS)
async def list_analysis_groups() -> str:
    """List the analysis-group names available to run_scan_analysis.

    Returns
    -------
    str
        JSON envelope with ``groups`` (sorted names from the configs
        repo's ``groups/`` tree — bare stems and namespace-qualified
        forms both resolve), ``count``, and ``truncated``.
    """
    return await _run_guarded(_list_analysis_groups_impl)


# ---------------------------------------------------------------------------
# run_scan_analysis
# ---------------------------------------------------------------------------


def _run_scan_analysis_impl(
    scan_number: int,
    day: Optional[str],
    analyzer: Optional[str],
    group: Optional[str],
    rerun_failed: bool,
    rerun_completed: bool,
) -> str:
    """Validate, enqueue statuses, and start the detached analysis worker."""
    if bool(analyzer) == bool(group):
        return errors.make_error(
            "invalid_request",
            "pass exactly one of 'analyzer' (a diagnostic ID from "
            "list_analyzers) or 'group' (a name from list_analysis_groups)",
        )
    try:
        from scan_analysis import task_queue
    except ImportError as exc:
        return errors.make_error("invalid_request", f"{_EXTRA_HINT}: {exc}")
    root = _config_root()
    if root is None:
        return errors.make_error("invalid_request", _ROOT_HINT)
    try:
        tag, scan_folder, _ = read_tools._resolve_folders(scan_number, day)
    except (ValueError, RuntimeError) as exc:
        return errors.make_error("invalid_request", str(exc))
    if not scan_folder.is_dir():
        # The cross-package invariant: analysis is a consumer of scan
        # folders, never a producer — a missing folder is refused, never
        # created (an SMB blip healed by mkdir orphans the real data).
        return errors.make_error(
            "not_found",
            f"scan folder does not exist: {scan_folder} — analysis never "
            "creates scan folders; check the scan number/day",
        )
    # Build the analyzer(s) now, in-server, through the SAME builder the
    # worker runs (run_worker.build_analyzers — one implementation, so
    # this validation can never drift from the execution): validates the
    # name, the YAML, the image-analyzer class path — and this HOST.  A
    # diagnostic needing a Windows-only SDK fails its import here and is
    # refused before anything is enqueued.
    from geecs_mcp.analysis.run_worker import build_analyzers

    try:
        analyzers = build_analyzers(analyzer, group, root)
    except (FileNotFoundError, KeyError) as exc:
        # An unknown diagnostic stem raises KeyError (with the known-names
        # list in the message), an unknown group FileNotFoundError.
        return errors.make_error("not_found", str(exc))
    except ImportError as exc:
        return errors.make_error(
            "invalid_request",
            f"diagnostic cannot run on this host (import failed: {exc}) — "
            "Windows-only SDK analyzers need the future Windows satellite "
            "server, not this one",
        )
    except Exception as exc:
        return errors.make_error(
            "invalid_request", f"could not build the analyzer(s): {exc}"
        )
    if not analyzers:
        return errors.make_error(
            "invalid_request", f"group {group!r} resolved to zero analyzers"
        )
    task_ids = [task_queue.analyzer_task_id(a) for a in analyzers]
    base = read_tools._base_directory()
    # The active-claim refusal comes FIRST, before any status write, so a
    # refused call is side-effect-free: a task another runner is actively
    # working (fresh heartbeat) refuses the whole call — two workers
    # claiming the same task would run the same analysis twice into the
    # same output files.  (Resetting first would re-queue the scan's other
    # rows and then refuse, leaving rows that read exactly like the
    # "worker died before claiming" signature.)
    pre = {s.analyzer_id: s for s in task_queue.read_statuses(scan_folder)}
    active = [
        tid for tid in task_ids if tid in pre and task_queue.claim_is_active(pre[tid])
    ]
    if active:
        return errors.make_error(
            "policy_refusal",
            f"analysis already running for task(s) {active} on scan "
            f"{tag.number} (claimed, heartbeat fresh) — poll "
            "get_scan_analysis and retry after it finishes",
        )
    # Server-side status bookkeeping BEFORE the spawn, so every requested
    # task is visibly queued even if the worker dies pre-claim: init the
    # missing rows, and — the rerun paths — reset done/failed (and
    # stale-claimed) rows to queued; reset refuses to touch a live claim.
    task_queue.init_status_for_scan(tag, analyzers, base_directory=base)
    reset_states = tuple(
        state
        for state, wanted in (("failed", rerun_failed), ("done", rerun_completed))
        if wanted
    )
    if reset_states:
        task_queue.reset_status_for_scan(
            tag,
            analyzers,
            base_directory=base,
            states_to_reset=reset_states + ("claimed",),
        )
    # Classify what this call will actually run from the post-init/reset
    # states: done/failed rows without their rerun flag are reported as
    # skipped, not silently re-listed as work.
    statuses = {s.analyzer_id: s for s in task_queue.read_statuses(scan_folder)}
    # Any remaining "claimed" is stale (active ones refused above) —
    # build_worklist reclaims those, so they count as runnable.
    runnable = [
        tid
        for tid in task_ids
        if tid not in statuses or statuses[tid].state in ("queued", "claimed")
    ]
    skipped = {tid: statuses[tid].state for tid in task_ids if tid not in runnable}
    common = {
        "scan_number": tag.number,
        "day": f"{tag.year:04d}-{tag.month:02d}-{tag.day:02d}",
        "scan_folder": str(scan_folder),
    }
    if not runnable:
        return errors.make_ok(
            started=False,
            tasks=[],
            skipped=skipped,
            note=(
                "nothing to run — every requested task is already "
                "done/failed; pass rerun_completed/rerun_failed to re-run"
            ),
            **common,
        )
    payload = {
        "year": tag.year,
        "month": tag.month,
        "day": tag.day,
        "number": tag.number,
        "experiment": tag.experiment,
        "analyzer": analyzer,
        "group": group,
        "rerun_failed": bool(rerun_failed),
        "rerun_completed": bool(rerun_completed),
        "config_root": str(root),
        "base_directory": str(base) if base is not None else None,
    }
    pid = _spawn_worker(payload)
    logger.info(
        "run_scan_analysis: scan %s (%s) -> %d task(s), worker pid %s",
        tag.number,
        analyzer or group,
        len(runnable),
        pid,
    )
    return errors.make_ok(
        started=True,
        worker_pid=pid,
        tasks=runnable,
        skipped=skipped,
        note=(
            "analysis runs detached — poll get_scan_analysis for task "
            "states (queued → claimed → done/failed/no_data).  A task "
            "stuck 'queued' means the worker died before claiming; "
            "'claimed' with a growing heartbeat_age_s means it died "
            "mid-run (the claim goes stale after 180 s and a repeat "
            "call re-runs it)"
        ),
        **common,
    )


@mcp.tool(name=tool_names.RUN_SCAN_ANALYSIS)
async def run_scan_analysis(
    scan_number: int,
    day: str | None = None,
    analyzer: str | None = None,
    group: str | None = None,
    rerun_failed: bool = False,
    rerun_completed: bool = False,
) -> str:
    """Run a ScanAnalysis diagnostic or group for one scan, detached.

    Submit-and-poll: the call returns as soon as the tasks are enqueued
    and the worker is started; poll ``get_scan_analysis`` for progress
    and ``get_scan_figure`` for the produced figures.  The scan folder
    must already exist (analysis never creates scan folders).

    Parameters
    ----------
    scan_number : int
        The scan to analyze.
    day : str, optional
        ISO date (``YYYY-MM-DD``); today when omitted.
    analyzer : str, optional
        One diagnostic ID (from ``list_analyzers``).  Exactly one of
        ``analyzer``/``group`` is required.
    group : str, optional
        An analysis-group name (from ``list_analysis_groups``).
    rerun_failed, rerun_completed : bool
        Re-run tasks already recorded ``failed``/``done`` (default:
        only never-run tasks execute — a repeat call is a cheap no-op).

    Returns
    -------
    str
        JSON envelope: the enqueued task ids, worker pid, and how to
        poll — or a refusal (bad names, missing scan folder, a
        diagnostic this host cannot run).
    """
    return await _run_guarded(
        _run_scan_analysis_impl,
        scan_number,
        day,
        analyzer,
        group,
        rerun_failed,
        rerun_completed,
    )
