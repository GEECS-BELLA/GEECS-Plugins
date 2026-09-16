"""The JSON API — one route per service verb.

Sync handlers on purpose: every service call blocks on the manager, and
FastAPI runs a ``def`` route on its threadpool.  Listing answers are
``no-cache`` — a queue changes while it is looked at.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from geecs_scanner.service.models import (
    ActionDetailOut,
    ActionOut,
    CalibrationIn,
    CalibrationOut,
    ConfigListOut,
    HealthOut,
    ItemOut,
    MoveIn,
    OptimizationOut,
    OptimizerConfigOut,
    SetBestIn,
    PreflightOut,
    ProgressOut,
    QueueOut,
    ReadbackOut,
    SavePresetIn,
    SavePresetOut,
    ScanLogOut,
    ScanVariableOut,
    SettablesOut,
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

    @router.get("/api/scanlog", response_model=ScanLogOut)
    def scanlog(response: Response, offset: int = 0) -> ScanLogOut:
        """The latest run's scan.log from *offset* (also carried by /api/events as `log`)."""
        response.headers.update(_NO_CACHE)
        return service.scan_log(offset)

    @router.get("/api/configs/{kind}", response_model=ConfigListOut)
    def configs(kind: str, response: Response) -> ConfigListOut:
        """Names of one config kind: presets, trigger_profiles, scan_variables, actions, optimizer_configs."""
        response.headers.update(_NO_CACHE)
        return service.list_configs(kind)

    @router.get(
        "/api/configs/optimizer_configs/{name}", response_model=OptimizerConfigOut
    )
    def optimizer_config(name: str) -> OptimizerConfigOut:
        """The selected optimizer document and required devices."""
        return service.optimizer_config(name)

    @router.get("/api/optimization", response_model=OptimizationOut)
    def optimization() -> OptimizationOut:
        """Latest optimization iteration, also carried over SSE."""
        return service.optimization()

    @router.post("/api/optimization/best", response_model=ItemOut)
    def set_best(body: SetBestIn) -> ItemOut:
        """Move to the completed run's recorded best physical positions."""
        return service.set_optimization_best(body)

    @router.get("/api/configs/presets/{name}")
    def preset(name: str, response: Response) -> dict[str, Any]:
        """One preset document."""
        response.headers.update(_NO_CACHE)
        return service.preset(name)

    @router.post("/api/configs/presets/{name}", response_model=SavePresetOut)
    def save_preset(name: str, body: SavePresetIn) -> SavePresetOut:
        """Write a preset document to the configs tree (the URL names the file)."""
        return service.save_preset(name, body)

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

    # ---------------------------------------------------- idle-only items

    @router.get("/api/settables", response_model=SettablesOut)
    def settables(response: Response) -> SettablesOut:
        """Every numeric settable of the experiment, alias-first, from the GEECS DB."""
        response.headers.update(_NO_CACHE)
        return service.settables()

    @router.get("/api/readback", response_model=ReadbackOut)
    async def readback(
        variable: str, response: Response, units: str = ""
    ) -> ReadbackOut:
        """One live reading of ``Device:Variable`` over the gateway — the readback, not ``:SP``."""
        response.headers.update(_NO_CACHE)
        return await service.readback(variable, units=units)

    @router.post("/api/move", response_model=ItemOut)
    def move(body: MoveIn) -> ItemOut:
        """One manual move as an ``mv`` queue item; refused unless idle."""
        return service.move(body)

    @router.get("/api/actions", response_model=list[ActionOut])
    def actions(response: Response) -> list[ActionOut]:
        """The action library: names, step counts, nested plans."""
        response.headers.update(_NO_CACHE)
        return service.actions()

    @router.get("/api/actions/{name}", response_model=ActionDetailOut)
    def action(name: str, response: Response) -> ActionDetailOut:
        """The preview: every step the action would run, nested plans inlined."""
        response.headers.update(_NO_CACHE)
        return service.action(name)

    @router.post("/api/actions/{name}/run", response_model=ItemOut)
    def run_action(name: str, body: VerbIn | None = None) -> ItemOut:
        """Queue ``run_action(name)``; refused unless idle."""
        return service.run_action(name, body or VerbIn())

    @router.get("/api/calibration", response_model=CalibrationOut)
    def calibration(response: Response) -> CalibrationOut:
        """The stored shot offsets, summarized."""
        response.headers.update(_NO_CACHE)
        return service.calibration()

    @router.post("/api/calibration/check", response_model=ItemOut)
    def calibration_check(body: CalibrationIn) -> ItemOut:
        """Queue ``check_shot_sync`` (box OFF, costs no shot); refused unless idle."""
        return service.calibration_check(body)

    @router.post("/api/calibration/measure", response_model=ItemOut)
    def calibration_measure(body: CalibrationIn) -> ItemOut:
        """Queue ``measure_shot_offsets``; ``write`` stores the result. Refused unless idle."""
        return service.calibration_measure(body)
