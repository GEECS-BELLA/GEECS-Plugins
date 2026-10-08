"""The HTML pages: the index redirects, the day list and the run page."""

from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.tiled_catalog import fmt_time_of_day, metadata_rows

from geecs_portal import figures, resources
from geecs_portal.routes.common import (
    _acq_timestamp,
    _jump_target,
    _parse_day,
    _parse_iso_day,
    _portal_version,
    _resolved_folder,
    _root,
    _scan_label,
    _sticky_query,
)
from geecs_portal.state import PortalState, get_state

router = APIRouter()


@router.get("/", response_class=RedirectResponse)
def index(request: Request) -> str:
    """Redirect to today's day view."""
    return f"{_root(request)}/day/{date.today().isoformat()}"


@router.get("/go", response_class=RedirectResponse)
def go(request: Request, day: str = "", experiment: str = "", filter: str = "") -> str:
    """The day/experiment picker form's target: redirect to the day view."""
    selected = _parse_day(day)
    query = _sticky_query({"experiment": experiment, "filter": filter})
    return f"{_root(request)}/day/{selected.isoformat()}{'?' + query if query else ''}"


@router.get("/day/{day}", response_class=HTMLResponse)
def day_view(
    request: Request,
    day: str,
    experiment: str = "",
    filter: str = "",
    portal: PortalState = Depends(get_state),
) -> HTMLResponse:
    """The run list for one day (newest first, as the catalog lists)."""
    selected = _parse_iso_day(day)
    exp = experiment or portal.default_experiment
    runs, error = portal.list_day(exp, selected, filter)
    if error:
        error = f"catalog error: {error}"
    day_state = {"experiment": exp, "filter": filter}
    return portal.templates.TemplateResponse(
        request,
        "day.html",
        {
            "day": selected,
            "prev_day": (selected - timedelta(days=1)).isoformat(),
            "next_day": (selected + timedelta(days=1)).isoformat(),
            "experiment": exp,
            "filter": filter,
            "rows": [(run, fmt_time_of_day(run.start_time)) for run in runs],
            "scan_label": _scan_label,
            "error": error,
            "qs": lambda **kw: _sticky_query(day_state, **kw),
        },
    )


@router.get("/run/jump/{day}", response_class=RedirectResponse)
def run_jump(
    request: Request,
    day: str,
    prefer: int = 0,
    portal: PortalState = Depends(get_state),
) -> str:
    """Day-step from the scan page without losing the analysis.

    Redirects to the target day's run with scan number *prefer*
    (else its newest run), carrying every other query param through
    verbatim — the rail's day steppers point here so filters,
    columns, and tab survive the hop.  A day with no runs falls
    back to the day page.
    """
    selected = _parse_iso_day(day)
    carried = [
        (key, value)
        for key, value in request.query_params.multi_items()
        if key != "prefer"
    ]
    experiment = request.query_params.get("experiment", "") or portal.default_experiment
    runs, _ = portal.list_day(experiment, selected, "")  # failure → the day page
    carried = [(k, v) for (k, v) in carried if k != "day"]
    carried.append(("day", selected.isoformat()))
    query = urlencode(carried, doseq=True)
    if not runs:
        day_query = _sticky_query(
            {
                "experiment": experiment,
                "filter": request.query_params.get("filter", ""),
            }
        )
        return (
            f"{_root(request)}/day/{selected.isoformat()}"
            f"{'?' + day_query if day_query else ''}"
        )
    target = _jump_target(runs, prefer)
    assert target is not None  # runs is non-empty here
    return f"{_root(request)}/run/{target.uid}?{query}"


@router.get("/run/{uid}", response_class=HTMLResponse)
def run_view(
    request: Request,
    uid: str,
    day: str = "",
    experiment: str = "",
    y: list[str] = Query(default=[]),
    x: str = "",
    device: str = "",
    shot: int = 1,
    filter: str = "",
    tab: str = "",
    filters: str = "",
    bincfg: str = "",
    view: str = "",
    display: str = "",
    processing: str = "",
    gridcfg: str = "",
    gridbin: str = "",
    imagebin: str = "",
    portal: PortalState = Depends(get_state),
) -> HTMLResponse:
    """One run: the rail + tabs (Overview / Plot / Images).

    ``tab``/``filters``/``bincfg``/``view``/``y``/``x`` are the
    analysis-tab state, carried in the URL (statelessness doctrine:
    a link IS the analysis) and consumed by the page's JS — the
    server only threads them through the sticky query so steppers
    keep the whole setup.
    """
    detail = portal.load_run(uid)
    run_day, folder = _resolved_folder(detail, day)
    devices = resources.image_devices(folder) if folder else []
    sel_device = device if device in devices else ""
    if sel_device:
        # Reuse the listing just computed — no second directory scan.
        probe = resources.device_kind(folder, sel_device, devices=devices)
        kind, kind_path = probe.kind, probe.path
        # Pixels or x-vs-y: the stack says which, and the gallery
        # renders a line for the array kinds (an (n, 2) lineout
        # drawn as pixels is a two-pixel-wide strip).
        content_kind = resources.stack_content(probe)
    else:
        kind, kind_path = "", None
        content_kind = "image"
    n_rows = None if detail.data is None else len(detail.data)
    shot = max(1, min(shot, n_rows) if n_rows else shot)
    analysis_enabled = portal.analysis_enabled_for(folder)
    if (
        kind == "native"
        and folder is not None
        and n_rows
        and detail.summary.exit_status
        and detail.data is not None
        and schema_map.device_acq_timestamp_column(
            [str(c) for c in detail.data.columns], sel_device
        )
        is not None
    ):
        # Background-warm the whole diagnostic (timestamp-joined shots
        # only — ordinal resolutions are never cached), so stepping
        # through shots serves from memory.
        warm_key = (uid, sel_device)
        warm_folder, warm_device, warm_detail = folder, sel_device, detail

        def _warm_one(s: int) -> None:
            acq_s, present = _acq_timestamp(warm_detail, warm_device, s)
            if acq_s is None:
                return  # device missed the shot (or no column)
            resources.load_shot_image(
                warm_folder,
                warm_device,
                s,
                acq_timestamp=acq_s,
                data_cache=portal.data_cache,
                cache_key=warm_key,
            )

        portal.data_cache.warm_native(
            warm_key, _warm_one, list(range(1, min(n_rows, 2000) + 1))
        )
    prev_uid, next_uid, day_runs = portal.neighbours(
        uid, experiment or portal.default_experiment, run_day
    )
    state = {
        "day": day,
        "experiment": experiment or portal.default_experiment,
        # The analysis-tab state (URL-carried; the page JS owns it):
        "tab": tab,
        "y": [c for c in y if c],
        "x": x,
        "view": view,
        "filters": filters,
        "bincfg": bincfg,
        "display": display,
        "processing": processing,
        "gridcfg": gridcfg,
        "gridbin": gridbin,
        "imagebin": imagebin,
        "device": sel_device,
        "shot": shot if sel_device else "",
        "filter": filter,  # the day list's filter, carried for the back link
    }
    return portal.templates.TemplateResponse(
        request,
        "run.html",
        {
            "uid": uid,
            "day": day,
            "run_day": run_day.isoformat() if run_day else "",
            "experiment": experiment or portal.default_experiment,
            "summary": detail.summary,
            "rows": metadata_rows(detail),
            "start_time_of_day": fmt_time_of_day(detail.summary.start_time),
            "prev_uid": prev_uid,
            "next_uid": next_uid,
            "day_runs": day_runs,
            "scan_number": detail.summary.scan_number or 0,
            "logbook_url": portal.logbook_url(request, detail, run_day),
            "logbook_send": portal.logbook_sendable(detail, run_day),
            "prev_day": ((run_day - timedelta(days=1)).isoformat() if run_day else ""),
            "next_day": ((run_day + timedelta(days=1)).isoformat() if run_day else ""),
            "tab": (
                tab
                if tab in ("overview", "plot", "grid", "images")
                or (tab == "analysis" and analysis_enabled)
                else "plot"
            ),
            "analysis_enabled": analysis_enabled,
            "config_editor": portal.config_editor_enabled and analysis_enabled,
            "devices": devices,
            "sel_device": sel_device,
            "kind": kind,
            "kind_path": str(kind_path) if kind_path else "",
            "content_kind": content_kind,
            "is_trace": content_kind != "image",
            "shot": shot,
            "has_next_shot": n_rows is None or shot < n_rows,
            "total_shots": detail.summary.shots,
            "processing": processing,
            "processing_options": (
                portal.processing_names()
                if sel_device and content_kind == "image"
                else []
            ),
            "display": display,
            "portal_version": _portal_version(),
            # The rail's chips and the display popup must stay in
            # step with the server-authored figures — one palette,
            # one marker default, both injected.
            "trace_colors": list(figures.TRACE_COLORS),
            "msize_default": figures.MARKER_SIZE_DEFAULT,
            "qs": lambda **kw: _sticky_query(state, **kw),
        },
    )
