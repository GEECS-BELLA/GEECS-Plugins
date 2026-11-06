"""The day page's batch analysis: the analysis worker's API, through the portal.

Every route forwards to the worker (:mod:`geecs_portal.batch`) and hands
its answer back as it came; the portal adds only the experiment (its own,
when the request names none) and two errors of its own: no worker
configured (404) and a worker that does not answer (503).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse

from geecs_portal.batch import WorkerUnavailable
from geecs_portal.state import PortalState, get_state

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}


def _forward(portal: PortalState, method: str, path: str, body: Any = None):
    if portal.worker is None:
        raise HTTPException(status_code=404, detail="no analysis worker configured")
    try:
        status, answer = portal.worker.call(method, path, json=body)
    except WorkerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(answer, status_code=status, headers=_NO_STORE)


def _with_experiment(portal: PortalState, body: dict) -> dict:
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="the body must be a JSON object")
    return {"experiment": portal.default_experiment or None, **body}


@router.get("/api/batch/status")
def batch_status(portal: PortalState = Depends(get_state)) -> JSONResponse:
    """Whether a worker is configured, answers now, and may be started here."""
    worker = portal.worker
    return JSONResponse(
        {
            "configured": worker is not None,
            "reachable": bool(worker and worker.reachable()),
            "autostart": bool(worker and worker.can_autostart),
        },
        headers=_NO_STORE,
    )


@router.get("/api/batch/groups")
def batch_groups(portal: PortalState = Depends(get_state)):
    """The worker's analysis groups (starts it when allowed)."""
    return _forward(portal, "GET", "/api/groups")


@router.post("/api/batch/plan")
def batch_plan(body: dict = Body(...), portal: PortalState = Depends(get_state)):
    """What a job would run now."""
    return _forward(portal, "POST", "/api/plan", _with_experiment(portal, body))


@router.post("/api/batch/jobs")
def batch_submit(body: dict = Body(...), portal: PortalState = Depends(get_state)):
    """Queue a job on the worker."""
    return _forward(portal, "POST", "/api/jobs", _with_experiment(portal, body))


@router.get("/api/batch/jobs")
def batch_jobs(portal: PortalState = Depends(get_state)):
    """The worker's jobs, newest first."""
    return _forward(portal, "GET", "/api/jobs")


@router.get("/api/batch/jobs/{job_id}")
def batch_job(job_id: str, portal: PortalState = Depends(get_state)):
    """One job with its tasks."""
    return _forward(portal, "GET", f"/api/jobs/{job_id}")


@router.post("/api/batch/jobs/{job_id}/cancel")
def batch_cancel(job_id: str, portal: PortalState = Depends(get_state)):
    """Stop a job before its next task."""
    return _forward(portal, "POST", f"/api/jobs/{job_id}/cancel")


@router.get("/api/batch/watch")
def batch_watch(portal: PortalState = Depends(get_state)):
    """The worker's running watch, or ``{watch: null}``."""
    return _forward(portal, "GET", "/api/watch")


@router.post("/api/batch/watch")
def batch_watch_start(body: dict = Body(...), portal: PortalState = Depends(get_state)):
    """Start watching a day: the group's ticked analyzers on each scan as it closes."""
    return _forward(portal, "POST", "/api/watch", _with_experiment(portal, body))


@router.post("/api/batch/watch/stop")
def batch_watch_stop(portal: PortalState = Depends(get_state)):
    """Stop watching (jobs it already queued still run)."""
    return _forward(portal, "POST", "/api/watch/stop")
