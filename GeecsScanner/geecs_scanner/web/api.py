"""The JSON API — one route per service verb.

Sync handlers on purpose: every service call blocks on the manager, and
FastAPI runs a ``def`` route on its threadpool.  Listing answers are
``no-cache`` — a queue changes while it is looked at.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from geecs_scanner.service.models import (
    ConfigListOut,
    HealthOut,
    PreflightOut,
    ProgressOut,
    QueueOut,
    ScanVariableOut,
    StatusOut,
    SubmitIn,
    SubmitOut,
    VerbIn,
    VerbOut,
)
from geecs_scanner.service.scanner import ScannerService

_NO_CACHE = {"Cache-Control": "no-cache"}


def register(router: APIRouter, service: ScannerService) -> None:
    """Attach the API routes to *router*."""

    @router.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        """Liveness + the manager probe + version (the fleet-map check)."""
        return service.health()

    @router.get("/api/status", response_model=StatusOut)
    def status(response: Response) -> StatusOut:
        """One manager poll plus the readiness verdict."""
        response.headers.update(_NO_CACHE)
        return service.status()

    @router.get("/api/queue", response_model=QueueOut)
    def queue(response: Response, history: int = 10) -> QueueOut:
        """Running, waiting and recently finished items."""
        response.headers.update(_NO_CACHE)
        return service.queue(history_limit=history)

    @router.get("/api/progress", response_model=ProgressOut)
    def progress(response: Response) -> ProgressOut:
        """The latest-run picture from the streams (also carried by /api/events)."""
        response.headers.update(_NO_CACHE)
        return service.progress()

    @router.get("/api/configs/{kind}", response_model=ConfigListOut)
    def configs(kind: str, response: Response) -> ConfigListOut:
        """Names of one config kind: presets, trigger_profiles, scan_variables, actions, optimizer_configs."""
        response.headers.update(_NO_CACHE)
        return service.list_configs(kind)

    @router.get("/api/configs/presets/{name}")
    def preset(name: str, response: Response) -> dict[str, Any]:
        """One preset document."""
        response.headers.update(_NO_CACHE)
        return service.preset(name)

    @router.get("/api/scan-variables", response_model=list[ScanVariableOut])
    def scan_variables(response: Response) -> list[ScanVariableOut]:
        """The catalog as the picker lists it; pseudo entries present but not scannable."""
        response.headers.update(_NO_CACHE)
        return service.scan_variables()

    @router.get("/api/devices", response_model=list[str])
    def devices(response: Response) -> list[str]:
        """Every device reference the manager resolves."""
        response.headers.update(_NO_CACHE)
        return service.devices()

    @router.post("/api/preflight", response_model=PreflightOut)
    def preflight(preset_doc: dict[str, Any]) -> PreflightOut:
        """Validate and pre-check a preset; submit nothing."""
        return service.preflight(preset_doc)

    @router.post("/api/submit", response_model=SubmitOut)
    def submit(body: SubmitIn) -> SubmitOut:
        """Preflight, require every question acknowledged, stamp, queue."""
        return service.submit(body)

    @router.post("/api/pause", response_model=VerbOut)
    def pause(body: VerbIn | None = None) -> VerbOut:
        """Deferred pause at the next step boundary."""
        return service.pause(body or VerbIn())

    @router.post("/api/resume", response_model=VerbOut)
    def resume(body: VerbIn | None = None) -> VerbOut:
        """Resume a paused plan."""
        return service.resume(body or VerbIn())

    @router.post("/api/stop", response_model=VerbOut)
    def stop(body: VerbIn | None = None) -> VerbOut:
        """Graceful stop."""
        return service.stop(body or VerbIn())

    @router.post("/api/clear", response_model=VerbOut)
    def clear() -> VerbOut:
        """Remove every waiting item."""
        return service.clear()
