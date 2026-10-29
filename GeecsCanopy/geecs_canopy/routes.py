"""The module's routes, mounted at ``/geecs``; no Canopy import (condition (b)).

Every route, the put included, needs Canopy read access only (condition
(a)): an operator's key never carries Canopy ``write``; the module
authorizes its own writes by the operator policy. Errors are Canopy's
``{"code", "detail"}`` envelope.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import yaml
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
            {"name": d.name, "family": d.family, "source": d.source}
            | {"axes": list(d.axes), "state": view(d, module.tracker, stale)["state"]}
            for d in (await module.devices()).values()
        ]
        body = {"experiment": module.settings.experiment, "devices": listed}
        return body | {"sources": module.sources}

    @app.get("/v1/devices/{name}", dependencies=read)
    async def device(name: str) -> Any:
        if (d := (await module.devices()).get(name)) is None:
            return _error(404, "not_found", f"no device {name!r}")
        body = {"name": d.name, "family": d.family, "source": d.source}
        body |= {"axes": list(d.axes), "actions": list(d.actions), "staleAfterS": stale}
        body |= view(d, module.tracker, stale)
        try:  # the policy's hazards, so the page asks before it writes
            policy = await module.policy()
        except (OSError, ValueError, TimeoutError, yaml.YAMLError):
            return body  # unreadable: every put is denied, and journaled
        for entry in body["roles"]:
            if policy.hazardous(d.name, d.family, entry["role"]):
                entry["confirm"] = "hazard"
        return body

    @app.post("/v1/devices/{name}/put")
    async def put(name: str, body: PutBody, key: Any = Security(read_access)) -> Any:
        key_name = getattr(key, "name", None)
        request = PutRequest(device=name, key_name=key_name, **body.model_dump())
        outcome = await put_role(module, request)
        return JSONResponse(status_code=outcome.status, content=outcome.body())

    @app.get("/v1/journal", dependencies=read)
    async def journal(entries: int = Query(50, ge=1, le=1000)) -> Any:
        rows = await asyncio.to_thread(module.journal.last, entries)  # off the loop
        return [row.model_dump(mode="json") for row in rows]

    return app
