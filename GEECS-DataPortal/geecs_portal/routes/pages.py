"""The HTML pages: the index redirects, the day list and the run page."""

from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import urlencode

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.requests import Request

from geecs_web_theme.web import root_of

from geecs_data_utils.io import shot_images
from geecs_data_utils.tiled_catalog import fmt_time_of_day, metadata_rows

from geecs_portal.routes.common import (
    _jump_target,
    _parse_day,
    _parse_iso_day,
    _portal_version,
    _resolved_folder,
    _scan_label,
    _sticky_query,
)
from geecs_portal.state import PortalState, get_state

router = APIRouter()


@router.get("/", response_class=RedirectResponse)
def index(request: Request) -> str:
    """Redirect to today's day view."""
    return f"{root_of(request)}/day/{date.today().isoformat()}"


@router.get("/go", response_class=RedirectResponse)
def go(request: Request, day: str = "", experiment: str = "", filter: str = "") -> str:
    """The day/experiment picker form's target: redirect to the day view."""
    selected = _parse_day(day)
    query = _sticky_query({"experiment": experiment, "filter": filter})
    return (
        f"{root_of(request)}/day/{selected.isoformat()}{'?' + query if query else ''}"
    )


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
            # The plotter opens the portal's own experiment only.
            "plot_page": portal.plot_page and exp == portal.default_experiment,
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
            f"{root_of(request)}/day/{selected.isoformat()}"
            f"{'?' + day_query if day_query else ''}"
        )
    target = _jump_target(runs, prefer)
    assert target is not None  # runs is non-empty here
    return f"{root_of(request)}/run/{target.uid}?{query}"


@router.get("/run/{uid}", response_class=HTMLResponse)
def run_view(
    request: Request,
    uid: str,
    day: str = "",
    experiment: str = "",
    filter: str = "",
    tab: str = "",
    portal: PortalState = Depends(get_state),
) -> Response:
    """One run: the rail + tabs (Overview / Plot / Images / Analysis).

    Plot and Images are the plotter's two pages (``/plot/``,
    ``/plot/images``), reading the viewer's settings set; this page
    keeps the Overview and the Analysis tab, and a ``tab=images`` link
    from before goes to the Images page. ``tab`` rides the URL.
    """
    detail = portal.load_run(uid)
    run_day, folder = _resolved_folder(detail, day)
    devices = shot_images.image_devices(folder) if folder else []
    analysis_enabled = portal.analysis_enabled_for(folder)
    prev_uid, next_uid, day_runs = portal.neighbours(
        uid, experiment or portal.default_experiment, run_day
    )
    # The Plot pages open a scan by day and number in the portal's own
    # experiment; a run they cannot open has neither tab.
    plot_page = (
        portal.plot_page
        and run_day is not None
        and detail.summary.scan_number is not None
        and experiment in ("", portal.default_experiment)
    )
    if tab == "images" and plot_page:
        query = urlencode(
            {"day": run_day.isoformat(), "scan": detail.summary.scan_number}
        )
        return RedirectResponse(f"{root_of(request)}/plot/images?{query}", 303)
    state = {
        "day": day,
        "experiment": experiment or portal.default_experiment,
        "tab": tab,
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
            "prev_day": ((run_day - timedelta(days=1)).isoformat() if run_day else ""),
            "next_day": ((run_day + timedelta(days=1)).isoformat() if run_day else ""),
            "tab": "analysis" if tab == "analysis" and analysis_enabled else "overview",
            "plot_page": plot_page,
            "analysis_enabled": analysis_enabled,
            "config_editor": portal.config_editor_enabled and analysis_enabled,
            "devices": devices,
            # The config editor previews this shot of the recipe's device.
            "shot": 1,
            "total_shots": detail.summary.shots,
            "portal_version": _portal_version(),
            "qs": lambda **kw: _sticky_query(state, **kw),
        },
    )
