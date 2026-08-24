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

#: Payload budgets for get_scan_analysis (osprey-side audit ask): one
#: task's display_files list and the number of device dirs listed —
#: truncation is flagged, never silent.
_MAX_DISPLAY_FILES_PER_TASK = 20
_MAX_OUTPUT_DIRS = 40

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
            display_files = [name for name in display_files if isinstance(name, str)]
            record = {
                "state": document.get("state"),
                "error": document.get("error"),
                "claimed_by": document.get("claimed_by"),
                "heartbeat_age_s": _heartbeat_age_s(
                    document.get("last_heartbeat"), now
                ),
                # Payload budget (osprey-side audit ask, 2026-08-24): the
                # writer owns this list's length; cap what one task can
                # put into a tool answer.
                "display_files": display_files[:_MAX_DISPLAY_FILES_PER_TASK],
            }
            if len(display_files) > _MAX_DISPLAY_FILES_PER_TASK:
                record["display_files_truncated"] = True
            statuses[entry.stem] = record
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
    outputs_truncated = False
    if analysis_folder.is_dir():
        for entry in sorted(analysis_folder.iterdir()):
            if len(outputs) >= _MAX_OUTPUT_DIRS:
                # Payload budget: a pathological analysis tree must not
                # balloon the answer (osprey-side audit ask).
                outputs_truncated = True
                break
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
    extra = {"outputs_truncated": True} if outputs_truncated else {}
    return errors.make_ok(
        scan_number=int(scan_number),
        scan_folder=str(scan_folder),
        analysis_folder=str(analysis_folder),
        analysis_present=analysis_folder.is_dir(),
        tasks=tasks,
        outputs=outputs,
        **extra,
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
        except (OSError, ValueError):
            # OSError: a stat on a pathologically-shaped name (EINVAL on
            # the NFS mount, ENAMETOOLONG on any fs).  ValueError:
            # resolve() on an embedded null byte — legal YAML, so
            # reachable from the writable share (review finding on this
            # PR).  Skip the entry, never the tool.
            logger.warning("figure candidate unreadable, skipped: %s", path)

    # NOTE: the statuses' display_files lists arrive payload-capped (20
    # per task) — the cap deliberately bounds discovery here too; a
    # figure past the cap is still found by the tree scan below.
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
    """A candidate's agent-facing name (relative, POSIX separators).

    Candidates arrive RESOLVED (``_add`` resolves before bounding), so
    the folder must be resolved too before ``relative_to`` — with a
    symlinked base path, an unresolved folder would fail every
    ``relative_to`` and collapse all labels to colliding basenames
    (#687 review finding 3: the route's exact match would then serve
    the first collision — wrong bytes under the right name).
    """
    try:
        root = analysis_folder.resolve()
    except OSError:
        root = analysis_folder
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


#: Pixel cap for figure decode — matplotlib summaries are a few MP; far
#: below Pillow's ~178 MP bomb ceiling so a giant share-resident image
#: refuses instead of decoding hundreds of MB in the server process.
_MAX_FIGURE_PIXELS = 64_000_000

#: Thumbnail bounds (the opt-in ``thumbnail=true`` return — the only
#: path that puts image bytes through model context, so it is bounded
#: hard: ≤768 px longest edge, JPEG q80 ≈ 30–60 KB for a matplotlib
#: summary).  The first web-UI integration blew a haiku-tier context on
#: a 247 KB inline PNG (osprey-side finding, 2026-08-24) — inline image
#: content is never the default again.
_THUMBNAIL_EDGE_PX = 768
_THUMBNAIL_JPEG_QUALITY = 80

#: Byte cap for the raw ``/figures/`` route (streams the original file,
#: never through model context — the cap is a DoS guard, not a context
#: budget).
_MAX_ROUTE_FILE_BYTES = 50_000_000


def _load_thumbnail_jpeg(path: Path) -> bytes:
    """The figure as bounded JPEG bytes (model context, not archive)."""
    import io

    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        if image.width * image.height > _MAX_FIGURE_PIXELS:
            raise ValueError(
                f"figure {path.name} is {image.width}x{image.height} — "
                f"over the {_MAX_FIGURE_PIXELS / 1e6:.0f} MP decode cap"
            )
        image.load()
        if max(image.size) > _THUMBNAIL_EDGE_PX:
            image.thumbnail((_THUMBNAIL_EDGE_PX, _THUMBNAIL_EDGE_PX))
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=_THUMBNAIL_JPEG_QUALITY)
        return buffer.getvalue()


def _day_string(tag) -> str:
    """The resolved day as YYYY-MM-DD (the route's day segment)."""
    return f"{int(tag.year):04d}-{int(tag.month):02d}-{int(tag.day):02d}"


def _figure_url(day: str, scan_number: int, label: str) -> str:
    """The SERVER-RELATIVE fetch URL for one figure.

    Relative by design: the server binds 0.0.0.0 and cannot know its
    advertised host, and baking an address into results would break the
    planned service re-homing.  Clients resolve it against the MCP base
    URL they already hold (the profile's ``url:`` minus the ``/mcp``
    path).
    """
    from urllib.parse import quote

    return f"/figures/{day}/{int(scan_number)}/{quote(label, safe='/')}"


def _share_relative(path: Path) -> str:
    r"""The path relative to the GEECS data root (POSIX), else absolute.

    The *primary* file handle in tool results — never a ``Z:\\`` or
    ``/mnt/`` form (osprey-side ask): clients resolve it against their
    own mount of the share.
    """
    from geecs_data_utils import ScanPaths

    base = _base_directory()
    if base is None and ScanPaths.paths_config is not None:
        base = ScanPaths.paths_config.base_path
    if base is not None:
        try:
            return path.relative_to(Path(base).resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _figure_metadata(path: Path, label: str, day: str, scan_number: int) -> dict:
    """One figure's reference record (no image bytes)."""
    from PIL import Image as PILImage

    width = height = None
    size = None
    try:
        size = path.stat().st_size
        with PILImage.open(path) as image:  # header read only — no decode
            width, height = image.width, image.height
    except Exception:  # a vanished/unreadable file degrades the fields
        logger.debug("figure header unreadable: %s", path, exc_info=True)
    return {
        "figure": label,
        "day": day,
        "scan_number": int(scan_number),
        "share_relative_path": _share_relative(path),
        "bytes": size,
        "width": width,
        "height": height,
        "figure_url": _figure_url(day, scan_number, label),
    }


def _candidate_listing(paths, analysis_folder: Path, day: str, scan_number: int):
    """Compact reference entries for a candidate list (cheap stats only)."""
    entries = []
    for path in paths:
        label = _relative_label(path, analysis_folder)
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        entries.append(
            {
                "figure": label,
                "bytes": size,
                "figure_url": _figure_url(day, scan_number, label),
            }
        )
    return entries


def _get_scan_figure_impl(
    scan_number: int, name: str | None, day: str | None, thumbnail: bool
):
    """Figure reference (default) / bounded thumbnail / candidate list."""
    try:
        tag, scan_folder, analysis_folder = _resolve_folders(scan_number, day)
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
    resolved_day = _day_string(tag)
    labels = [_relative_label(path, analysis_folder) for path in candidates]
    if name is None:
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            return errors.make_ok(
                message="multiple figures — call again with name=<one of these>",
                figures=_candidate_listing(
                    candidates, analysis_folder, resolved_day, scan_number
                ),
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
                figures=_candidate_listing(
                    matches, analysis_folder, resolved_day, scan_number
                ),
            )
        chosen = matches[0]
    if thumbnail:
        from fastmcp.utilities.types import Image

        return Image(data=_load_thumbnail_jpeg(chosen), format="jpeg")
    meta = _figure_metadata(
        chosen, _relative_label(chosen, analysis_folder), resolved_day, scan_number
    )
    return errors.make_ok(
        message=(
            "figure reference — fetch the full-resolution bytes from "
            "figure_url (relative to the MCP server's base URL) or the "
            "share_relative_path; pass thumbnail=true only when the "
            "model itself needs to see it"
        ),
        **meta,
    )


@mcp.tool(name=tool_names.GET_SCAN_FIGURE)
async def get_scan_figure(
    scan_number: int,
    name: str | None = None,
    day: str | None = None,
    thumbnail: bool = False,
):
    """Locate one rendered analysis figure for a scan.

    Default return is a REFERENCE, not the image: the figure's name,
    dimensions, byte size, its path relative to the GEECS data root,
    and ``figure_url`` — a server-relative HTTP route serving the
    original file bytes (resolve against the MCP base URL). Fetch the
    URL and save it as a local artifact instead of pulling pixels
    through the model. ``thumbnail=true`` returns a bounded preview
    (≤768 px JPEG) as image content — use it only when the figure must
    actually be looked at. With no ``name`` and several figures, the
    candidate list (with URLs) comes back to pick from; ``name``
    matches by substring.
    """
    return await _run_guarded(_get_scan_figure_impl, scan_number, name, day, thumbnail)


def _locate_figure(day: str, scan_number: int, label: str):
    """The route's SYNC lookup: ``(status_code, detail)`` or a file Path.

    Runs in a worker thread (see :func:`_serve_figure_route`) — every
    filesystem touch here can hang on the share's mount timeout, and
    must never do so on the event loop.
    """
    try:
        _tag, scan_folder, analysis_folder = _resolve_folders(scan_number, day)
    except (ValueError, RuntimeError) as exc:
        return (400, str(exc))
    if not scan_folder.is_dir():
        return (404, "no such scan")
    for path in _gather_figure_candidates(scan_folder, analysis_folder):
        if _relative_label(path, analysis_folder) == label:
            if path.stat().st_size > _MAX_ROUTE_FILE_BYTES:
                return (413, "figure too large")
            return path
    return (404, "no such figure")


async def _serve_figure_route(request):
    """GET /figures/{day}/{scan_number}/{label:path} — the raw file bytes.

    The fetch counterpart of ``get_scan_figure``'s ``figure_url``:
    serves the ORIGINAL figure (it bypasses model context, so no
    downscale), bounded by exactly the same candidate set as the tool —
    only files the scan's own analysis folder legitimately offers, by
    exact label match — plus a byte cap.  The label is only ever
    COMPARED against candidate labels, never used to build a path, so
    traversal sequences can only fail to match.  Registered on the
    FastMCP app, so it is live under ``--transport http`` and simply
    never reachable over stdio.  Never raises: every failure is a plain
    status response.

    The share lookup runs via ``anyio.to_thread`` — fastmcp runs async
    endpoints directly on the event loop, and one hung NFS stat here
    would otherwise stall EVERY request on the server, including the
    halt verbs (#687 review finding 1; the tools already thread via
    ``_run_guarded``).
    """
    import anyio.to_thread
    from starlette.responses import FileResponse, PlainTextResponse

    try:
        day = request.path_params["day"]
        try:
            scan_number = int(request.path_params["scan_number"])
        except (TypeError, ValueError):
            return PlainTextResponse("scan_number must be an integer", status_code=400)
        label = request.path_params["label"]
        located = await anyio.to_thread.run_sync(
            _locate_figure, day, scan_number, label
        )
        if isinstance(located, tuple):
            status, detail = located
            return PlainTextResponse(detail, status_code=status)
        media = "image/png" if located.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(located, media_type=media)
    except Exception:
        logger.exception("figure route failed")
        return PlainTextResponse("internal error", status_code=500)


mcp.custom_route("/figures/{day}/{scan_number}/{label:path}", methods=["GET"])(
    _serve_figure_route
)
