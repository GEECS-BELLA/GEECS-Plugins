"""The analysis-domain read tools: per-scan analysis results and figures.

Closes the "scan → analyze → present" loop (#675): ``get_scan_analysis``
reports what the ScanAnalysis pipeline produced for a scan (task
statuses from ``analysis_status/`` plus the output tree), and
``get_scan_figure`` returns a rendered summary figure as actual MCP
image content (downscaled — context-sized, never the raw file).

Everything is **strictly read-only over the data share**:

- Paths come from :class:`geecs_data_utils.ScanPaths`'s PURE static
  builders (``get_scan_folder_path`` / ``get_scan_analysis_folder_path``)
  — never the instance accessors, whose ``get_analysis_folder()``
  silently ``os.makedirs`` a missing folder.  A missing folder is a
  ``not_found`` envelope naming the path; nothing on the data share is
  ever created (the repo's scan-folder invariant, applied to the whole
  share).
- The task-status YAMLs are parsed tolerantly (``.get`` everywhere) —
  their schema is ScanAnalysis-owned (``task_queue.TaskStatus``); this
  module reads the documented fields and survives extras/absences.

Same conventions as every domain: sync ``_*_impl`` = the tested surface,
async wrappers via the shared guard, JSON envelopes (except a successful
figure fetch, which returns MCP image content).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Optional

from geecs_mcp import errors, runtime, tool_names
from geecs_mcp.scans.read_tools import _run_guarded
from geecs_mcp.server import mcp

logger = logging.getLogger("geecs_mcp.analysis.read")

#: Cap on file names listed per output directory — the tree is a map for
#: the agent, not a dump.
_MAX_FILES_PER_DIR = 30

#: Cap on figure candidates gathered/listed.
_MAX_FIGURE_CANDIDATES = 60

#: Longest image edge after downscaling (agent context, not archival).
_MAX_FIGURE_EDGE_PX = 1024

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _base_directory() -> Optional[Path]:
    """The data-share base directory override (``None`` = production config).

    A module-level seam so hermetic tests point the pure path builders at
    a tmp tree; production always returns ``None`` (the builders then use
    ``GeecsPathsConfig``).
    """
    return None


def _resolve_folders(scan_number: int, day: str | None) -> tuple[Any, Path, Path]:
    """Resolve (tag, scan_folder, analysis_folder) — pure paths, no I/O.

    Raises
    ------
    ValueError
        Bad ``day`` string (caller maps to ``invalid_request``).
    RuntimeError
        No experiment configured (caller maps to ``invalid_request``).
    """
    from geecs_data_utils import ScanPaths, ScanTag

    experiment = runtime.get_experiment()
    if not experiment:
        raise RuntimeError("no experiment configured ([Experiment] expt in config.ini)")
    when = date.fromisoformat(day) if day else date.today()
    tag = ScanTag(
        year=when.year,
        month=when.month,
        day=when.day,
        number=int(scan_number),
        experiment=experiment,
    )
    base = _base_directory()
    if base is None:
        # ScanPaths runs reload_paths_config() at module import, but it
        # swallows ConfigurationError and leaves paths_config = None on an
        # unconfigured host (live-run finding: the bare attribute then
        # raised AttributeError downstream instead of degrading).  The
        # retry here is belt; the honest refusal below is ours.
        if ScanPaths.paths_config is None:
            ScanPaths.reload_paths_config()
        if ScanPaths.paths_config is None:
            raise RuntimeError(
                "data share not configured — set [Paths] geecs_data in "
                "config.ini (must point at the dir containing "
                "Configurations.INI) on the serving host"
            )
    scan_folder = ScanPaths.get_scan_folder_path(tag=tag, base_directory=base)
    analysis_folder = ScanPaths.get_scan_analysis_folder_path(
        tag=tag, base_directory=base
    )
    return tag, scan_folder, analysis_folder


def _heartbeat_age_s(value: Any, now: "Any") -> Optional[float]:
    """Age of an ISO-8601 heartbeat, tolerant; ``None`` when unparseable.

    The writer stamps ``datetime.now(timezone.utc).isoformat()`` — parse
    per task_queue's own ``_parse_ts`` semantics (assume UTC when naive).
    """
    from datetime import datetime, timezone

    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except Exception:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return round((now - stamp).total_seconds(), 1)


def _read_task_statuses(scan_folder: Path) -> dict[str, dict]:
    """Parse ``analysis_status/*.yaml`` tolerantly; missing folder → empty.

    THE AUTHORITATIVE SHAPE IS ``TaskStatus.to_dict()`` in
    ``ScanAnalysis/scan_analysis/task_queue.py`` — keys ``state``
    (queued/claimed/done/failed/no_data), ``error``, ``claimed_by``,
    ``claimed_at``, ``last_heartbeat`` (ISO-8601 string), and
    ``display_files``.  (Review finding on the #675 PR: the package's
    CLAUDE.md prose was stale — read the writer, not the doc.)  Every
    field coercion sits inside the per-file guard: one odd YAML on a
    writable share degrades that one entry to ``unreadable``, never the
    whole tool.
    """
    import yaml
    from datetime import datetime, timezone

    status_dir = scan_folder / "analysis_status"
    statuses: dict[str, dict] = {}
    if not status_dir.is_dir():
        return statuses
    now = datetime.now(timezone.utc)
    for entry in sorted(status_dir.iterdir()):
        if entry.suffix not in (".yaml", ".yml"):
            continue
        try:
            document = yaml.safe_load(entry.read_text()) or {}
            if not isinstance(document, dict):
                raise ValueError("not a mapping")
            display_files = document.get("display_files") or []
            if not isinstance(display_files, list):
                display_files = []
            statuses[entry.stem] = {
                "state": document.get("state"),
                "error": document.get("error"),
                "claimed_by": document.get("claimed_by"),
                "heartbeat_age_s": _heartbeat_age_s(
                    document.get("last_heartbeat"), now
                ),
                "display_files": [
                    name for name in display_files if isinstance(name, str)
                ],
            }
        except Exception as exc:  # a torn write mid-heartbeat is not our error
            statuses[entry.stem] = {"state": "unreadable", "detail": str(exc)}
    return statuses


def _get_scan_analysis_impl(scan_number: int, day: str | None) -> str:
    """Task statuses + the analysis output tree for one scan."""
    try:
        _tag, scan_folder, analysis_folder = _resolve_folders(scan_number, day)
    except ValueError as exc:
        return errors.make_error("invalid_request", str(exc))
    except RuntimeError as exc:
        return errors.make_error("invalid_request", str(exc))
    if not scan_folder.is_dir():
        return errors.make_error(
            "not_found", f"no scan folder at {scan_folder} (share mounted?)"
        )
    tasks = _read_task_statuses(scan_folder)
    outputs: dict[str, dict] = {}
    if analysis_folder.is_dir():
        for entry in sorted(analysis_folder.iterdir()):
            if entry.is_dir():
                # The production layout nests analyzer subdirs
                # (Scan<NNN>/<device>/<Analyzer>/files) — a one-level
                # listing read every device as empty (n_files: 0; live
                # deployment finding 2026-08-24), so walk the whole
                # device tree, names relative to it.
                names = sorted(
                    p.relative_to(entry).as_posix()
                    for p in entry.rglob("*")
                    if p.is_file()
                )
                outputs[entry.name] = {
                    "n_files": len(names),
                    "files": names[:_MAX_FILES_PER_DIR],
                }
            elif entry.is_file():
                outputs.setdefault("(top level)", {"n_files": 0, "files": []})
                top = outputs["(top level)"]
                top["n_files"] += 1
                if len(top["files"]) < _MAX_FILES_PER_DIR:
                    top["files"].append(entry.name)
    return errors.make_ok(
        scan_number=int(scan_number),
        scan_folder=str(scan_folder),
        analysis_folder=str(analysis_folder),
        analysis_present=analysis_folder.is_dir(),
        tasks=tasks,
        outputs=outputs,
    )


@mcp.tool(name=tool_names.GET_SCAN_ANALYSIS)
async def get_scan_analysis(scan_number: int, day: str | None = None) -> str:
    """What the analysis pipeline produced for one scan.

    Returns per-analyzer task statuses (queued/claimed/done/failed, with
    ``display_files`` — the rendered summary figures) and the analysis
    output tree (per-analyzer directories, file names capped). ``day`` as
    YYYY-MM-DD, default today in the server host's local timezone.
    Requires the data share mounted on the serving host.
    """
    return await _run_guarded(_get_scan_analysis_impl, scan_number, day)


def _localize_display_entry(name: str, analysis_folder: Path) -> Optional[Path]:
    r"""A ``display_files`` entry as a path on THIS host, or ``None``.

    Production entries are written by the WINDOWS analysis machines
    (``Z:\\data\\...\\analysis\\Scan<NNN>\\...`` — live deployment
    finding 2026-08-24: on the Linux service host such a string is not
    absolute, so it was joined onto the analysis folder as one giant
    backslash component and the stat blew up the whole tool).  A
    Windows-style entry is re-rooted by its tail after the scan's own
    ``analysis\\Scan<NNN>\\`` onto the local *analysis_folder*; an entry
    whose parts never match that pattern returns ``None`` (skipped with
    a warning — never a crash).  Anything else passes through unchanged
    for the normal absolute/relative handling.
    """
    from pathlib import PureWindowsPath

    looks_windows = "\\" in name or (len(name) >= 2 and name[1] == ":")
    if not looks_windows:
        return Path(name)
    parts = PureWindowsPath(name).parts
    lowered = [part.lower() for part in parts]
    scan_name = analysis_folder.name.lower()
    for i in range(len(parts) - 1):
        if lowered[i] == "analysis" and lowered[i + 1] == scan_name:
            tail = parts[i + 2 :]
            if tail:
                return analysis_folder.joinpath(*tail)
            break
    logger.warning("display_files entry not under this scan's analysis tree: %s", name)
    return None


def _gather_figure_candidates(scan_folder: Path, analysis_folder: Path) -> list[Path]:
    """Figure candidates: display_files first, then images in the tree.

    Every candidate must resolve **inside this scan's analysis folder**
    — the ``display_files`` entries come from YAML on a writable share,
    and an absolute or ``../`` entry must not let the tool serve
    anything else (review finding 3 bounded to the share root; the #675
    codex review tightened it to the scan's own analysis folder, which
    is where the writer actually puts every legitimate entry —
    ScanAnalysis analyzers write to ``<date>/analysis/Scan<NNN>/...`` —
    so a poisoned entry cannot reach even another scan's outputs).
    Windows-written entries are localized first (see
    :func:`_localize_display_entry`), and every per-candidate
    filesystem touch is guarded — one hostile or malformed entry must
    never take down the tree-scan fallback (the live crash did exactly
    that).
    """
    candidates: list[Path] = []
    seen: set[Path] = set()
    try:
        root = analysis_folder.resolve()
    except OSError:
        return []

    def _add(path: Path) -> None:
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            return
        if not path.is_absolute():
            path = analysis_folder / path
        try:
            path = path.resolve()
            if not path.is_relative_to(root):
                logger.warning(
                    "figure candidate outside the scan's analysis folder: %s", path
                )
                return
            if path not in seen and path.is_file():
                seen.add(path)
                candidates.append(path)
        except OSError:
            # e.g. a stat on a pathologically-shaped name (EINVAL on the
            # NFS mount) — skip the entry, never the tool.
            logger.warning("figure candidate unreadable, skipped: %s", path)

    for status in _read_task_statuses(scan_folder).values():
        for name in status.get("display_files") or []:
            localized = _localize_display_entry(str(name), analysis_folder)
            if localized is not None:
                _add(localized)
    if analysis_folder.is_dir():
        for path in sorted(analysis_folder.rglob("*")):
            if len(candidates) >= _MAX_FIGURE_CANDIDATES:
                break
            if path.is_file():
                _add(path)
    return candidates[:_MAX_FIGURE_CANDIDATES]


def _relative_label(path: Path, analysis_folder: Path) -> str:
    """A candidate's agent-facing name (relative to the analysis folder)."""
    try:
        return str(path.relative_to(analysis_folder))
    except ValueError:
        return path.name


#: Pixel cap for figure decode — matplotlib summaries are a few MP; far
#: below Pillow's ~178 MP bomb ceiling so a giant share-resident image
#: refuses instead of decoding hundreds of MB in the server process.
_MAX_FIGURE_PIXELS = 64_000_000


def _load_downscaled_png(path: Path) -> bytes:
    """The figure as PNG bytes, longest edge capped (context, not archive)."""
    import io

    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        if image.width * image.height > _MAX_FIGURE_PIXELS:
            raise ValueError(
                f"figure {path.name} is {image.width}x{image.height} — "
                f"over the {_MAX_FIGURE_PIXELS / 1e6:.0f} MP decode cap"
            )
        image.load()
        if max(image.size) > _MAX_FIGURE_EDGE_PX:
            image.thumbnail((_MAX_FIGURE_EDGE_PX, _MAX_FIGURE_EDGE_PX))
        if image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _get_scan_figure_impl(scan_number: int, name: str | None, day: str | None):
    """One figure as image content, or the candidate list to choose from."""
    try:
        _tag, scan_folder, analysis_folder = _resolve_folders(scan_number, day)
    except ValueError as exc:
        return errors.make_error("invalid_request", str(exc))
    except RuntimeError as exc:
        return errors.make_error("invalid_request", str(exc))
    if not scan_folder.is_dir():
        return errors.make_error(
            "not_found", f"no scan folder at {scan_folder} (share mounted?)"
        )
    candidates = _gather_figure_candidates(scan_folder, analysis_folder)
    if not candidates:
        return errors.make_error(
            "not_found",
            f"no figures found for Scan{int(scan_number):03d} "
            f"(analysis folder: {analysis_folder})",
        )
    labels = [_relative_label(path, analysis_folder) for path in candidates]
    if name is None:
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            return errors.make_ok(
                message="multiple figures — call again with name=<one of these>",
                figures=labels,
            )
    else:
        matches = [
            path
            for path, label in zip(candidates, labels)
            if name.lower() in label.lower()
        ]
        if not matches:
            return errors.make_error(
                "not_found", f"no figure matching {name!r}; available: {labels}"
            )
        if len(matches) > 1:
            return errors.make_ok(
                message=f"{name!r} is ambiguous — pick one",
                figures=[_relative_label(p, analysis_folder) for p in matches],
            )
        chosen = matches[0]
    from fastmcp.utilities.types import Image

    return Image(data=_load_downscaled_png(chosen), format="png")


@mcp.tool(name=tool_names.GET_SCAN_FIGURE)
async def get_scan_figure(
    scan_number: int, name: str | None = None, day: str | None = None
):
    """Fetch one rendered analysis figure for a scan, as an image.

    With no ``name`` and exactly one figure, returns it; otherwise
    returns the candidate list to pick from (``name`` matches by
    substring of the listed path). Images are downscaled to ≤1024 px on
    the longest edge — ask for the analysis folder path via
    get_scan_analysis when the full-resolution file is needed.
    """
    return await _run_guarded(_get_scan_figure_impl, scan_number, name, day)
