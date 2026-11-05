"""The analysis worker's HTTP API (FastAPI, the ``worker`` extra).

::

    GET  /health                 {ok, service, experiment, config_dir, active}
    GET  /api/groups             the configs tree's loadable groups
    POST /api/plan               {day, group, scans?, redo?, experiment?} -> what a job would run
    POST /api/jobs               same body -> 202 + the queued job
    GET  /api/jobs               every job, newest first (no task lists)
    GET  /api/jobs/{id}          one job with its tasks
    POST /api/jobs/{id}/cancel   stop it before its next task

Unauthenticated, like the portal's run verb: bind it to the host or the
lab network only.  Errors: an unknown group or a day with no scans is a
404, a malformed body a 422.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from scan_analysis.worker.jobs import JobSpec, Worker

__all__ = ["JobRequest", "create_worker_app"]


class JobRequest(BaseModel):
    """A job request: one group over one day."""

    day: date = Field(description="The scans' day, YYYY-MM-DD.")
    group: str = Field(
        min_length=1, description="An analysis group: PW/frog_only or its stem."
    )
    scans: Optional[list[int]] = Field(
        None, description="Only these scan numbers; all when omitted."
    )
    redo: bool = Field(
        False, description="Run again tasks that already finished or failed."
    )
    experiment: Optional[str] = Field(
        None, description="The experiment folder; the worker's own when omitted."
    )


def create_worker_app(worker: Worker) -> FastAPI:
    """The API over *worker*, which it starts and stops with the app."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        try:
            yield
        finally:
            worker.stop(timeout=1.0)

    app = FastAPI(
        title="GEECS analysis worker", lifespan=lifespan, docs_url=None, redoc_url=None
    )
    _info_routes(app, worker)
    _job_routes(app, worker)
    return app


def _spec_of(worker: Worker, body: JobRequest) -> JobSpec:
    """The request as a job spec; an unloadable group is a 404."""
    from scan_analysis.config import load_analysis_group

    try:
        load_analysis_group(body.group, config_dir=worker.config_dir)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(
            status_code=404, detail=f"group {body.group!r}: {exc}"
        ) from exc
    return JobSpec(
        day=body.day,
        group=body.group,
        experiment=body.experiment or worker.default_experiment,
        scans=tuple(body.scans) if body.scans is not None else None,
        redo=body.redo,
    )


def _info_routes(app: FastAPI, worker: Worker) -> None:
    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "service": "scan-analysis-worker",
            "experiment": worker.default_experiment,
            "config_dir": str(worker.config_dir),
            "active": sum(1 for j in worker.jobs() if j.state in ("queued", "running")),
        }

    @app.get("/api/groups")
    def groups() -> dict:
        return {"groups": worker.groups()}

    @app.post("/api/plan")
    def plan(body: JobRequest) -> dict:
        spec = _spec_of(worker, body)
        try:
            return worker.plan(spec).to_json()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


def _job_routes(app: FastAPI, worker: Worker) -> None:
    @app.post("/api/jobs", status_code=202)
    def submit(body: JobRequest) -> dict:
        return worker.submit(_spec_of(worker, body)).to_json()

    @app.get("/api/jobs")
    def jobs() -> JSONResponse:
        return JSONResponse(
            {"jobs": [j.to_json(with_tasks=False) for j in worker.jobs()]},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> JSONResponse:
        found = worker.get(job_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return JSONResponse(found.to_json(), headers={"Cache-Control": "no-store"})

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        found = worker.cancel(job_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return found.to_json(with_tasks=False)
