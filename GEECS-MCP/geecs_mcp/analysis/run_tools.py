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


def _analyzer_id(analyzer: Any) -> str:
    """The status-file id for *analyzer* — task_queue's own derivation.

    Mirrors ``init_status_for_scan`` exactly so the envelope's task list
    names the same files the worker will claim.
    """
    analyzer_id = getattr(analyzer, "id", getattr(analyzer, "device_name", "unknown"))
    return analyzer_id or (
        f"{analyzer.__class__.__name__}_{getattr(analyzer, 'device_name', '')}"
    )


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
    payload: dict[str, Any] = {
        kind: names[:_MAX_LISTED_NAMES],
        "count": len(names),
        "truncated": len(names) > _MAX_LISTED_NAMES,
        "config_root": str(root),
    }
    if kind == "groups":
        # discover_groups indexes each group under its bare stem AND its
        # namespace-qualified form — every listed name is accepted.
        payload["note"] = "names include both bare stems and namespace/stem forms"
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
        from scan_analysis.config import create_scan_analyzer

        from image_analysis.config import load_diagnostic
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
    # Build the analyzer(s) now, in-server: this validates the name, the
    # YAML, the image-analyzer class path — and this HOST.  A diagnostic
    # needing a Windows-only SDK fails its import here and is refused
    # before anything is enqueued.
    try:
        if analyzer:
            analyzers = [
                create_scan_analyzer(
                    load_diagnostic(analyzer, config_dir=root), id=analyzer
                )
            ]
        else:
            analyzers = task_queue.load_analyzers_from_config(group, config_dir=root)
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
    base = read_tools._base_directory()
    # Server-side init (idempotent): a worker that dies pre-claim leaves
    # visible queued rows for get_scan_analysis instead of silence.
    task_queue.init_status_for_scan(tag, analyzers, base_directory=base)
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
    task_ids = [_analyzer_id(a) for a in analyzers]
    logger.info(
        "run_scan_analysis: scan %s (%s) -> %d task(s), worker pid %s",
        tag.number,
        analyzer or group,
        len(task_ids),
        pid,
    )
    return errors.make_ok(
        started=True,
        worker_pid=pid,
        scan_number=tag.number,
        day=f"{tag.year:04d}-{tag.month:02d}-{tag.day:02d}",
        scan_folder=str(scan_folder),
        tasks=task_ids,
        note=(
            "analysis runs detached — poll get_scan_analysis for task "
            "states (queued → claimed → done/failed/no_data; a task "
            "stuck 'queued' means the worker died before claiming)"
        ),
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
