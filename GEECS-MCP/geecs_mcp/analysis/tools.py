"""The analysis domain over the Data Portal's HTTP API.

Every verb here is one of the portal's routes seen through the envelope —
the same analysis runs, listings and artifacts an operator sees on the
portal's Analysis tab (DESIGN.md §4.4–§4.9).  A scan is named the way the
loop knows it, by scan number and day; the portal's ``jump`` route turns
that into the run uid every other route takes, and a scan number the day
does not hold is ``not_found`` (the jump route otherwise returns the day's
newest run, which must never be analysed by mistake).

Nothing here reads the data share, runs an analyzer or interprets a
column: the portal does, and this module forwards its words.
"""

from __future__ import annotations

import io
import json
import threading
from typing import Any, Optional

import anyio

from geecs_mcp import clients, errors, guard, portal, tool_names
from geecs_mcp.clients import ServiceRefusal
from geecs_mcp.identity import InvalidArgument
from geecs_mcp.server import mcp
from geecs_mcp.waits import DEFAULT_WAIT_S, CallBudget, bounded_wait, heartbeat

#: Job states the portal's analysis runner reports.
ACTIVE_STATES = ("queued", "running")
#: Polling cadence for wait_for_analysis (the portal's tab polls at 1.5 s).
ANALYSIS_POLL_S = 2.0
#: The portal keeps the last 400 log lines per job; the agent gets a tail.
LOG_TAIL_LINES = 40
#: get_analysis_artifact's default and ceiling (bytes) — a sidecar JSON is
#: a few KiB; a per-shot table can be MBs, and never belongs in context whole.
ARTIFACT_DEFAULT_BYTES = 64 * 1024
ARTIFACT_MAX_BYTES = 256 * 1024
#: Figure bytes fetched for the opt-in thumbnail (a DoS guard, not a
#: context budget — only the ≤768 px JPEG reaches the agent).
FIGURE_FETCH_MAX_BYTES = 50_000_000
THUMBNAIL_EDGE_PX = 768
THUMBNAIL_JPEG_QUALITY = 80
MAX_FIGURE_PIXELS = 80_000_000


async def _sleep(seconds: float) -> None:
    """The wait verb's pause — a module attribute so tests can replace it."""
    await anyio.sleep(seconds)


def _bounded_job(job: Any) -> Any:
    """The portal's job record with its log cut to a tail."""
    if not isinstance(job, dict):
        return job
    out = dict(job)
    log = out.get("log")
    if isinstance(log, list) and len(log) > LOG_TAIL_LINES:
        out["log"] = log[-LOG_TAIL_LINES:]
        out["log_truncated"] = True
        out["log_lines_total"] = len(log)
    return out


# ---------------------------------------------------------------------------
# listings
# ---------------------------------------------------------------------------


async def _catalog() -> dict:
    return await clients.call("portal", "GET", "/api/analysis/catalog") or {}


async def _list_analyzers() -> str:
    catalog = await _catalog()
    analyzers = list(catalog.get("analyzers") or [])
    return errors.make_ok(analyzers=analyzers, count=len(analyzers))


@mcp.tool(name=tool_names.LIST_ANALYZERS)
async def list_analyzers() -> str:
    """Every analysis document the portal can run: ``id``, ``device``, ``output_dir``, ``destructive``, ``kind``.

    The ids are what ``run_analysis`` takes — never invent one.  A
    ``destructive`` kind (the ``.himg`` compaction) needs ``confirm`` set
    to the scan number.
    """
    return await guard.run_async(_list_analyzers)


async def _list_analysis_groups() -> str:
    catalog = await _catalog()
    groups = list(catalog.get("groups") or [])
    return errors.make_ok(
        groups=groups,
        count=len(groups),
        note=(
            "the portal runs one diagnostic per job: a group is a plan to "
            "expand — call run_analysis for each member in order"
        ),
    )


@mcp.tool(name=tool_names.LIST_ANALYSIS_GROUPS)
async def list_analysis_groups() -> str:
    """The analysis groups as written: ``id``, ``name``, ``members`` in run order.

    Not runnable as one unit here — the portal runs one diagnostic per
    job — so a group is the order to call ``run_analysis`` in.
    """
    return await guard.run_async(_list_analysis_groups)


# ---------------------------------------------------------------------------
# the per-scan picture
# ---------------------------------------------------------------------------


async def _get_scan_analysis(scan_number: Any, day: Optional[str]) -> str:
    uid, iso, number = await portal.resolve_uid(scan_number, day)
    listing = await portal.analysis_listing(uid, iso)
    analyzers = []
    for entry in listing.get("analyzers") or []:
        row = dict(entry)
        row["job"] = _bounded_job(row.get("job"))
        analyzers.append(row)
    return errors.make_ok(
        uid=uid,
        day=iso,
        scan_number=listing.get("scan_number", number),
        running=listing.get("running"),
        analyzers=analyzers,
    )


@mcp.tool(name=tool_names.GET_SCAN_ANALYSIS)
async def get_scan_analysis(scan_number: int, day: Optional[str] = None) -> str:
    """The portal's analysis picture of one scan: every analyzer with its job, files and artifacts.

    ``day`` is ISO (``2026-10-16``), today when omitted.  Per analyzer:
    ``applicable`` (its device has data in this scan), ``job`` (the
    portal's record — ``queued``/``running``/``done``/``failed``/``no_data``,
    ``error``, a bounded ``log`` tail; null when nothing ran since the
    portal started), ``files`` on disk and ``artifacts`` (``path``,
    ``kind`` summary|other|bin, ``inline``).  Paths feed
    ``get_analysis_artifact`` and ``get_scan_figure``.
    """
    return await guard.run_async(_get_scan_analysis, scan_number, day)


# ---------------------------------------------------------------------------
# run_analysis / wait_for_analysis
# ---------------------------------------------------------------------------


async def _run_analysis(
    scan_number: Any, analyzer: str, day: Optional[str], confirm: Any
) -> str:
    analyzer = str(analyzer or "").strip()
    if not analyzer:
        return errors.make_error("invalid_request", "name an analyzer (list_analyzers)")
    uid, iso, number = await portal.resolve_uid(scan_number, day)
    params: dict[str, Any] = {"analyzer": analyzer, "day": iso}
    if confirm is not None:
        try:
            params["confirm"] = int(confirm)
        except (TypeError, ValueError):
            raise InvalidArgument(
                f"confirm {confirm!r} is not an integer (the scan number)"
            ) from None
    try:
        job = await clients.call(
            "portal", "POST", f"/api/run/{uid}/analysis", params=params
        )
    except ServiceRefusal as exc:
        if exc.kind == "not_found" and "no diagnostic" in exc.message:
            catalog = await _catalog()
            ids = [a.get("id") for a in catalog.get("analyzers") or []]
            raise ServiceRefusal(exc.kind, exc.message, analyzers=ids) from None
        raise
    return errors.make_ok(
        uid=uid, day=iso, scan_number=number, analyzer=analyzer, job=_bounded_job(job)
    )


@mcp.tool(name=tool_names.RUN_ANALYSIS)
async def run_analysis(
    scan_number: int,
    analyzer: str,
    day: Optional[str] = None,
    confirm: Optional[int] = None,
) -> str:
    """Start one analysis document on one scan through the portal (its Analysis tab's run).

    Returns at once with the portal's job record (usually ``queued``);
    follow it with ``wait_for_analysis``.  One job per scan at a time
    (``policy_refusal`` with the active job otherwise).  A ``destructive``
    kind needs ``confirm=<scan_number>`` — the same typed confirmation the
    tab asks a person for; this server never fills it in.
    """
    return await guard.run_async(_run_analysis, scan_number, analyzer, day, confirm)


def _job_for(listing: dict, analyzer: str) -> tuple[Optional[dict], Optional[dict]]:
    for entry in listing.get("analyzers") or []:
        if entry.get("id") == analyzer:
            return entry, entry.get("job")
    return None, None


async def _wait_for_analysis(
    scan_number: Any, analyzer: str, day: Optional[str], timeout_s: Any
) -> str:
    analyzer = str(analyzer or "").strip()
    if not analyzer:
        return errors.make_error("invalid_request", "name the analyzer you started")
    call_budget = CallBudget()
    uid, iso, number = await portal.resolve_uid(scan_number, day)
    budget = bounded_wait(timeout_s)
    start = anyio.current_time()
    deadline = start + budget
    while True:
        await heartbeat(
            anyio.current_time() - start,
            budget,
            f"waiting for {analyzer} on scan {number}",
        )
        listing = await portal.analysis_listing(
            uid, iso, read_timeout_s=call_budget.read_timeout()
        )
        entry, job = _job_for(listing, analyzer)
        if entry is None:
            return errors.make_error(
                "not_found",
                f"no analyzer {analyzer!r} in the portal's listing for scan {number}",
                analyzers=[e.get("id") for e in listing.get("analyzers") or []],
            )
        state = (job or {}).get("state")
        if (
            job is None
            or state not in ACTIVE_STATES
            or anyio.current_time() >= deadline
        ):
            break
        await _sleep(ANALYSIS_POLL_S)
    finished = job is not None and state not in ACTIVE_STATES
    return errors.make_ok(
        finished=finished,
        uid=uid,
        day=iso,
        scan_number=number,
        analyzer=analyzer,
        # "unknown": the portal restarted since the run — the files on disk
        # are what is known, never a guess at done
        state=state if job is not None else "unknown",
        error=(job or {}).get("error"),
        progress=(job or {}).get("progress"),
        artifacts=entry.get("artifacts") or [],
        files=entry.get("files") or [],
        job=_bounded_job(job),
    )


@mcp.tool(name=tool_names.WAIT_FOR_ANALYSIS)
async def wait_for_analysis(
    scan_number: int,
    analyzer: str,
    day: Optional[str] = None,
    timeout_s: float = DEFAULT_WAIT_S,
) -> str:
    """Block until the analyzer's job on this scan leaves ``queued``/``running`` (at most 90 s).

    ``{finished, state, error, artifacts, files, job}`` — ``state`` is
    ``done``/``failed``/``no_data``, or ``unknown`` when the portal has no
    job record (it restarted; the files on disk are listed).  A timeout is
    a result with ``finished: false``; call again.
    """
    return await guard.run_async(
        _wait_for_analysis, scan_number, analyzer, day, timeout_s
    )


# ---------------------------------------------------------------------------
# reading numbers: artifacts and scalars
# ---------------------------------------------------------------------------


async def _get_analysis_artifact(
    scan_number: Any, path: str, day: Optional[str], max_bytes: Any
) -> str:
    path = str(path or "").strip()
    if not path:
        return errors.make_error(
            "invalid_request", "name an artifact path from get_scan_analysis"
        )
    if path.lower().endswith(portal.IMAGE_SUFFIXES):
        return errors.make_error(
            "invalid_request", f"{path!r} is an image — use get_scan_figure for figures"
        )
    try:
        cap = int(max_bytes)
    except (TypeError, ValueError):
        cap = ARTIFACT_DEFAULT_BYTES
    cap = max(1, min(cap, ARTIFACT_MAX_BYTES))
    uid, iso, number = await portal.resolve_uid(scan_number, day)
    body, content_type, truncated, total = await clients.get_bytes(
        "portal",
        f"/run/{uid}/artifact",
        params={"path": path, "day": iso},
        max_bytes=cap,
    )
    payload: dict[str, Any] = {
        "uid": uid,
        "day": iso,
        "scan_number": number,
        "path": path,
        "content_type": content_type,
        "size_bytes": total,
        "truncated": truncated,
    }
    if b"\x00" in body:
        # HDF5, NPY, pickles: the listing shows them and the portal serves
        # them as text/plain, but replacement characters are not numbers
        return errors.make_error(
            "invalid_request",
            f"{path!r} is a binary file — read it with the tools that know its "
            "format, not through model context",
        )
    text = body.decode("utf-8", errors="replace")
    if not truncated:
        try:
            payload["data"] = json.loads(text)
            return errors.make_ok(**payload)
        except ValueError:
            pass
    payload["text"] = text
    return errors.make_ok(**payload)


@mcp.tool(name=tool_names.GET_ANALYSIS_ARTIFACT)
async def get_analysis_artifact(
    scan_number: int,
    path: str,
    day: Optional[str] = None,
    max_bytes: int = ARTIFACT_DEFAULT_BYTES,
) -> str:
    """Read one non-image artifact a run produced — the numbers the loop decides on.

    ``path`` as ``get_scan_analysis`` lists it (relative to the scan's
    analysis folder): a ``*_summary_*.json`` sidecar comes back parsed as
    ``data``; a text table as ``text``.  Capped at ``max_bytes`` (ceiling
    256 KiB) with ``truncated`` and ``size_bytes`` when cut.  Images are
    refused — ``get_scan_figure`` is for those.
    """
    return await guard.run_async(
        _get_analysis_artifact, scan_number, path, day, max_bytes
    )


async def _get_scan_scalars(
    scan_number: Any,
    columns: list[str],
    day: Optional[str],
    x: str,
    bincfg: str,
    filters: str,
) -> str:
    cols = [str(c).strip() for c in (columns or []) if str(c).strip()]
    if not cols:
        return errors.make_error("invalid_request", "name at least one scalar column")
    # the column cap is the portal's (400 above its four): forwarded, not restated
    uid, iso, number = await portal.resolve_uid(scan_number, day)
    params: dict[str, Any] = {"cols": cols, "day": iso}
    params.update(
        {k: v for k, v in (("x", x), ("bincfg", bincfg), ("filters", filters)) if v}
    )
    binned = (
        await clients.call("portal", "GET", f"/api/run/{uid}/binned", params=params)
        or {}
    )
    # the plot and the notebook snippet are for browsers; the numbers stay
    numbers = {k: v for k, v in binned.items() if k not in ("figure", "code")}
    return errors.make_ok(uid=uid, day=iso, scan_number=number, columns=cols, **numbers)


@mcp.tool(name=tool_names.GET_SCAN_SCALARS)
async def get_scan_scalars(
    scan_number: int,
    columns: list[str],
    day: Optional[str] = None,
    x: str = "",
    bincfg: str = "",
    filters: str = "",
) -> str:
    """Per-bin centres, error bands and counts of scalar columns (the portal's Plot tab numbers; it caps them at four).

    ``bincfg`` and ``filters`` are the portal's own JSON strings, passed
    through unparsed, so a binning set up in the Plot tab is reproducible
    from its URL; with none, the portal's default binning applies.  The
    reply is the portal's numeric payload (``bins``, ``counts``,
    ``series``, ``x_centers`` when ``x`` is given) with the figure and the
    notebook snippet dropped.
    """
    return await guard.run_async(
        _get_scan_scalars, scan_number, columns, day, x, bincfg, filters
    )


# ---------------------------------------------------------------------------
# get_scan_figure
# ---------------------------------------------------------------------------


#: One figure decode at a time: an 80 Mpx RGBA decode peaks near 0.8 GB
#: resident, so two at once would outgrow the unit's MemoryMax and the
#: OOM kill would take every in-flight call with it.
_DECODE_LOCK = threading.Lock()


def _thumbnail_jpeg(data: bytes) -> tuple[bytes, int, int]:
    with _DECODE_LOCK:
        return _decode_thumbnail(data)


def _decode_thumbnail(data: bytes) -> tuple[bytes, int, int]:
    from PIL import Image as PILImage

    with PILImage.open(io.BytesIO(data)) as image:
        if image.width * image.height > MAX_FIGURE_PIXELS:
            raise InvalidArgument(
                f"figure is {image.width}x{image.height} — over the "
                f"{MAX_FIGURE_PIXELS / 1e6:.0f} MP decode cap"
            )
        image.load()
        width, height = image.size
        if max(image.size) > THUMBNAIL_EDGE_PX:
            image.thumbnail((THUMBNAIL_EDGE_PX, THUMBNAIL_EDGE_PX))
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY)
        return buffer.getvalue(), width, height


async def _get_scan_figure(
    scan_number: Any, label: Optional[str], day: Optional[str], thumbnail: bool
) -> str:
    uid, iso, number = await portal.resolve_uid(scan_number, day)
    listing = await portal.analysis_listing(uid, iso)
    figures = portal.figure_entries(listing, uid, iso)
    wanted = str(label or "").strip()
    if not wanted:
        summaries = [f for f in figures if f.get("kind") == "summary"]
        if len(summaries) == 1 and not thumbnail:
            wanted = summaries[0]["figure"]
        else:
            return errors.make_ok(
                uid=uid,
                day=iso,
                scan_number=number,
                figures=figures,
                note="name one of these as `label` for its reference (or thumbnail=true)",
            )
    match = next((f for f in figures if f["figure"] == wanted), None)
    if match is None:
        return errors.make_error(
            "not_found",
            f"no figure {wanted!r} among scan {number}'s artifacts",
            figures=[f["figure"] for f in figures],
        )
    payload: dict[str, Any] = {"uid": uid, "day": iso, "scan_number": number, **match}
    if not thumbnail:
        return errors.make_ok(**payload)
    data, content_type, truncated, total = await clients.get_bytes(
        "portal",
        f"/run/{uid}/artifact",
        params={"path": wanted, "day": iso},
        max_bytes=FIGURE_FETCH_MAX_BYTES,
    )
    if truncated:
        return errors.make_error(
            "invalid_request",
            f"figure {wanted!r} is over the {FIGURE_FETCH_MAX_BYTES // 1_000_000} MB fetch cap",
        )
    jpeg, width, height = await anyio.to_thread.run_sync(_thumbnail_jpeg, data)
    from fastmcp.utilities.types import Image

    payload.update(
        bytes=total, width=width, height=height, thumbnail="jpeg image block follows"
    )
    # Two content blocks: the reference as text, the bounded JPEG as a real
    # MCP image block (a base64 string inside the JSON would reach the model
    # as text — thousands of characters it cannot look at).
    return [errors.make_ok(**payload), Image(data=jpeg, format="jpeg")]


@mcp.tool(name=tool_names.GET_SCAN_FIGURE)
async def get_scan_figure(
    scan_number: int,
    label: Optional[str] = None,
    day: Optional[str] = None,
    thumbnail: bool = False,
):
    """A figure REFERENCE from the scan's analysis outputs — never the image through context by default.

    Without ``label``: the list of figures (``figure`` path, ``analyzer``,
    ``kind``, the portal ``figure_url`` a client fetches directly), or the
    one summary figure's reference when there is exactly one.  With
    ``label`` (a listed ``figure`` path): that reference.  ``thumbnail=true``
    adds a ≤768 px JPEG as an MCP **image** content block beside the
    reference — the only way image bytes ride model context.  (No return
    annotation on purpose: the image block is not a string.)
    """
    return await guard.run_async(_get_scan_figure, scan_number, label, day, thumbnail)
