"""The analysis worker's HTTP API (FastAPI, the ``worker`` extra).

::

    GET  /health                 {ok, service, experiment, config_dir, active}
    GET  /api/groups             the configs tree's loadable groups
    POST /api/plan               {day, group, scans?, redo?, analyzers?, from_scan?, experiment?}
                                 -> what a job would run
    POST /api/jobs               same body -> 202 + the queued job
    GET  /api/jobs               every job, newest first (no task lists)
    GET  /api/jobs/{id}          one job with its tasks
    POST /api/jobs/{id}/cancel   stop it before its next task
    GET  /api/watch              {watch}: the running watch, or null
    POST /api/watch              {day, group, analyzers?, from_scan?, experiment?} -> start
                                 watching (replaces any watch)
    POST /api/watch/stop         stop watching (its queued jobs still run)

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

from scan_analysis.worker.jobs import JobSpec, WatchSpec, Worker

__all__ = ["JobRequest", "WatchRequest", "create_worker_app"]


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
    analyzers: Optional[list[str]] = Field(
        None,
        description="Only these of the group's analyzers (by id); all when omitted.",
    )
    from_scan: int = Field(0, ge=0, description="Only scans numbered this or higher.")


class WatchRequest(BaseModel):
    """A watch request: the group's ticked analyzers on a day's scans as they close."""

    day: date = Field(description="The day to watch, YYYY-MM-DD.")
    group: str = Field(min_length=1, description="An analysis group.")
    analyzers: Optional[list[str]] = Field(
        None,
        description="Only these of the group's analyzers (by id); all when omitted.",
    )
    from_scan: int = Field(0, ge=0, description="Only scans numbered this or higher.")
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
    _watch_routes(app, worker)
    return app


def _check_group(worker: Worker, group: str) -> None:
    """An unloadable group is a 404."""
    from scan_analysis.config import load_analysis_group

    try:
        load_analysis_group(group, config_dir=worker.config_dir)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=f"group {group!r}: {exc}") from exc


def _spec_of(worker: Worker, body: JobRequest) -> JobSpec:
    """The request as a job spec; an unloadable group is a 404."""
    _check_group(worker, body.group)
    return JobSpec(
        day=body.day,
        group=body.group,
        experiment=body.experiment or worker.default_experiment,
        scans=tuple(body.scans) if body.scans is not None else None,
        redo=body.redo,
        analyzers=tuple(body.analyzers) if body.analyzers is not None else None,
        from_scan=body.from_scan,
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


def _watch_routes(app: FastAPI, worker: Worker) -> None:
    def state() -> JSONResponse:
        watch = worker.watching()
        return JSONResponse(
            {"watch": watch.to_json() if watch else None},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/watch")
    def get_watch() -> JSONResponse:
        return state()

    @app.post("/api/watch")
    def start_watch(body: WatchRequest) -> JSONResponse:
        _check_group(worker, body.group)
        worker.watch(
            WatchSpec(
                day=body.day,
                group=body.group,
                experiment=body.experiment or worker.default_experiment,
                analyzers=tuple(body.analyzers) if body.analyzers is not None else None,
                from_scan=body.from_scan,
            )
        )
        return state()

    @app.post("/api/watch/stop")
    def stop_watch() -> JSONResponse:
        worker.unwatch()
        return state()
