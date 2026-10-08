"""Health and the browsing JSON API: the pages' own data, as JSON."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from starlette.requests import Request

from geecs_data_utils.tiled_catalog import metadata_rows

from geecs_portal import analysis, resources
from geecs_portal.routes.common import (
    _LISTING_HEADERS,
    _jump_target,
    _parse_iso_day,
    _portal_version,
    _resolved_folder,
    _root,
    _summary_json,
)
from geecs_portal.state import PortalState, get_state

router = APIRouter()


@router.get("/health")
def health(portal: PortalState = Depends(get_state)) -> dict:
    """Liveness + catalog probe (the fleet-map health check) + version."""
    status = portal.catalog.probe()
    return {
        "ok": status.ok,
        "catalog": status.label,
        "version": _portal_version(),
    }


# ------------------------- browsing JSON API -------------------------
# The page-shaped reads (day list, run overview, device probe, the
# day jump) as JSON — for scripts and agents, so everything a
# browser shows is readable without scraping HTML.  Same helpers as
# the templates, so the two surfaces cannot drift.
@router.get("/api/day/{day}")
def api_day(
    request: Request,
    day: str,
    experiment: str = "",
    filter: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """The day page as JSON: its runs (newest first), filtered."""
    selected = _parse_iso_day(day)
    exp = experiment or portal.default_experiment
    runs, error = portal.list_day(exp, selected, filter)
    if error:
        raise HTTPException(status_code=503, detail=f"catalog unavailable: {error}")
    payload = {
        "day": selected.isoformat(),
        "prev_day": (selected - timedelta(days=1)).isoformat(),
        "next_day": (selected + timedelta(days=1)).isoformat(),
        "experiment": exp,
        "filter": filter,
        "runs": [_summary_json(run) for run in runs],
        "page": f"{_root(request)}/day/{selected.isoformat()}",
    }
    return JSONResponse(payload, headers=_LISTING_HEADERS)


@router.get("/api/run/jump/{day}")
def api_run_jump(
    request: Request,
    day: str,
    prefer: int = 0,
    experiment: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """The day steppers' target as JSON (see :func:`run_jump`)."""
    selected = _parse_iso_day(day)
    exp = experiment or portal.default_experiment
    runs, error = portal.list_day(exp, selected, "")
    if error:
        raise HTTPException(status_code=503, detail=f"catalog unavailable: {error}")
    target = _jump_target(runs, prefer)
    page = (
        f"{_root(request)}/run/{target.uid}"
        if target
        else f"{_root(request)}/day/{selected.isoformat()}"
    )
    payload = {
        "day": selected.isoformat(),
        "experiment": exp,
        "prefer": prefer,
        "uid": target.uid if target else None,
        "scan_number": target.scan_number if target else None,
        "matched": bool(target and prefer and target.scan_number == prefer),
        "runs": len(runs),
        "page": page,
    }
    return JSONResponse(payload, headers=_LISTING_HEADERS)


@router.get("/api/run/{uid}")
def api_run(
    request: Request,
    uid: str,
    day: str = "",
    experiment: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """One run as JSON: the rail, the Overview tab and the device list.

    ``metadata`` is the Overview table verbatim (``metadata_rows``);
    ``devices`` are the image device folders (their tier is the
    separate ``/device`` probe — one listing per device); the
    neighbours and ``day_runs`` are the scan steppers and dropdown;
    ``analysis_enabled`` says whether the page offers the Analysis
    tab (same gate as the template).
    """
    detail = portal.load_run(uid)
    run_day, folder = _resolved_folder(detail, day)
    exp = experiment or portal.default_experiment
    prev_uid, next_uid, day_runs = portal.neighbours(uid, exp, run_day)
    payload = {
        "uid": uid,
        "run_day": run_day.isoformat() if run_day else None,
        "experiment": exp,
        "summary": _summary_json(detail.summary),
        "metadata": [[field, value] for field, value in metadata_rows(detail)],
        "start_doc": analysis.jsonable_document(detail.start_doc or {}),
        "stop_doc": analysis.jsonable_document(detail.stop_doc or {}),
        "event_rows": None if detail.data is None else len(detail.data),
        "scan_folder": str(folder) if folder else None,
        "devices": resources.image_devices(folder) if folder else [],
        "neighbours": {"prev_uid": prev_uid, "next_uid": next_uid},
        "day_runs": [_summary_json(run) for run in day_runs],
        "prev_day": (run_day - timedelta(days=1)).isoformat() if run_day else None,
        "next_day": (run_day + timedelta(days=1)).isoformat() if run_day else None,
        "processing_options": portal.processing_names(),
        "analysis_enabled": portal.analysis_enabled_for(folder),
        "config_editor": portal.config_editor_enabled,
        "logbook": portal.logbook_url(request, detail, run_day) or None,
        "logbook_send": portal.logbook_sendable(detail, run_day),
        "page": f"{_root(request)}/run/{uid}",
        "portal_version": _portal_version(),
    }
    return JSONResponse(payload, headers=_LISTING_HEADERS)


@router.get("/api/run/{uid}/device")
def api_device(
    uid: str, device: str = "", day: str = "", portal: PortalState = Depends(get_state)
) -> JSONResponse:
    """One device's gallery tier — what clicking its name decides.

    Same ``device_kind`` probe as the page (one directory listing,
    no pixel reads), so the JSON and the Images tab can never
    disagree about whether a device renders.
    """
    detail = portal.load_run(uid)
    if not device:
        raise HTTPException(status_code=400, detail="device is required")
    run_day, folder = _resolved_folder(detail, day)
    if folder is None:
        raise HTTPException(status_code=404, detail="scan folder not resolvable")
    probe = resources.device_kind(folder, device)
    if probe.kind == "missing":
        raise HTTPException(status_code=404, detail=f"unknown device {device!r}")
    renderable = probe.kind in ("stack", "native")
    payload = {
        "uid": uid,
        "run_day": run_day.isoformat() if run_day else None,
        "device": device,
        "kind": probe.kind,
        "renderable": renderable,
        "path": str(probe.path) if probe.path else None,
        "ext": probe.ext,
        "event_rows": None if detail.data is None else len(detail.data),
        "planned_shots": detail.summary.shots,
        "processing_options": portal.processing_names() if renderable else [],
    }
    return JSONResponse(payload, headers=_LISTING_HEADERS)
