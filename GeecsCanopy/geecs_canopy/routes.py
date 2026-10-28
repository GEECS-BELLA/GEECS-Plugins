"""The module's routes, mounted at ``/geecs``; no Canopy import (condition (b)).

Every route, the put included, needs Canopy read access only (condition
(a)): an operator's key never carries Canopy ``write``; the module
authorizes its own writes by the operator policy. Errors are Canopy's
``{"code", "detail"}`` envelope.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Query, Request, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from geecs_canopy.freshness import view
from geecs_canopy.gates import PutRequest, put_role
from geecs_canopy.hare import WriteValue
from geecs_canopy.module import DevicesUnavailable, Module


class PutBody(BaseModel):
    """A write to the device in the path: role (and axis), value, operator, hazard confirm."""

    model_config = ConfigDict(extra="forbid")

    role: str
    axis: str | None = None
    value: WriteValue
    operator: str = Field(min_length=1)
    confirm: bool = False


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "detail": detail})


def build_app(module: Module, *, read_access: Callable[..., Any]) -> FastAPI:
    """Build the sub-app; the wrapper mounts and tags it and supplies Canopy's read check."""
    app = FastAPI(title="Canopy / GEECS", description="GEECS devices over Hare.")
    stale, read = module.settings.stale_after_s, [Security(read_access)]

    @app.exception_handler(DevicesUnavailable)
    async def unavailable(_: Request, exc: DevicesUnavailable) -> JSONResponse:
        return _error(503, "devices_unavailable", str(exc))

    @app.get("/v1/devices", dependencies=read)
    async def roster() -> Any:
        listed = [
            {"name": d.name, "family": d.family.template, "axes": list(d.axes)}
            | {"state": view(d, module.tracker, stale)["state"]}
            for d in (await module.devices()).values()
        ]
        return {"experiment": module.settings.experiment, "devices": listed}

    @app.get("/v1/devices/{name}", dependencies=read)
    async def device(name: str) -> Any:
        if (d := (await module.devices()).get(name)) is None:
            return _error(404, "not_found", f"no device {name!r}")
        body = {"name": d.name, "family": d.family.template, "axes": list(d.axes)}
        body |= {"actions": list(d.actions), "staleAfterS": stale}
        return body | view(d, module.tracker, stale)

    @app.post("/v1/devices/{name}/put")
    async def put(name: str, body: PutBody, key: Any = Security(read_access)) -> Any:
        key_name = getattr(key, "name", None)
        request = PutRequest(device=name, key_name=key_name, **body.model_dump())
        outcome = await put_role(module, request)
        return JSONResponse(status_code=outcome.status, content=outcome.body())

    @app.get("/v1/journal", dependencies=read)
    async def journal(entries: int = Query(50, ge=1, le=1000)) -> Any:
        return [row.model_dump(mode="json") for row in module.journal.last(entries)]

    return app
