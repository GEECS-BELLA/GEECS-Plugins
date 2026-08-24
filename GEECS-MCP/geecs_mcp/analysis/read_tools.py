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
import time
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
        # ScanPaths.paths_config starts as None and nothing initializes it
        # at import — consumers must call reload_paths_config() once
        # (live-run finding: an unconfigured class attribute raised
        # AttributeError instead of degrading).  reload swallows its own
        # ConfigurationError and leaves None, so the honest refusal is ours.
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


def _read_task_statuses(scan_folder: Path) -> dict[str, dict]:
    """Parse ``analysis_status/*.yaml`` tolerantly; missing folder → empty."""
    import yaml

    status_dir = scan_folder / "analysis_status"
    statuses: dict[str, dict] = {}
    if not status_dir.is_dir():
        return statuses
    now = time.time()
    for entry in sorted(status_dir.iterdir()):
        if entry.suffix not in (".yaml", ".yml"):
            continue
        try:
            document = yaml.safe_load(entry.read_text()) or {}
        except Exception as exc:  # a torn write mid-heartbeat is not our error
            statuses[entry.stem] = {"status": "unreadable", "detail": str(exc)}
            continue
        if not isinstance(document, dict):
            statuses[entry.stem] = {"status": "unreadable", "detail": "not a mapping"}
            continue
        heartbeat = document.get("heartbeat")
        statuses[entry.stem] = {
            "status": document.get("status"),
            "claimed_by": document.get("claimed_by"),
            "heartbeat_age_s": (
                round(now - float(heartbeat), 1) if heartbeat else None
            ),
            "display_files": document.get("display_files") or [],
        }
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
                names = sorted(p.name for p in entry.iterdir() if p.is_file())
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


def _gather_figure_candidates(scan_folder: Path, analysis_folder: Path) -> list[Path]:
    """Figure candidates: display_files first, then images in the tree."""
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            return
        if not path.is_absolute():
            path = analysis_folder / path
        if path not in seen and path.is_file():
            seen.add(path)
            candidates.append(path)

    for status in _read_task_statuses(scan_folder).values():
        for name in status.get("display_files") or []:
            _add(Path(name))
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


def _load_downscaled_png(path: Path) -> bytes:
    """The figure as PNG bytes, longest edge capped (context, not archive)."""
    import io

    from PIL import Image as PILImage

    with PILImage.open(path) as image:
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
