"""The module's routes, mounted at ``/geecs``; no Canopy import (condition (b)).

Every route, the put and an action run included, needs Canopy read access
only (condition (a)): an operator's key never carries Canopy ``write``; the
module authorizes its own writes by the operator policy, step by step. Errors are Canopy's
``{"code", "detail"}`` envelope.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated, Any

import yaml
import httpx
from fastapi import FastAPI, Query, Request, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from geecs_canopy.actions import Executor
from geecs_canopy.freshness import view
from geecs_canopy.gates import PutRequest, Refusal, put_role
from geecs_canopy.hare import WriteValue
from geecs_canopy.module import DevicesUnavailable, Module
from geecs_canopy.presets import Presets
from geecs_canopy.squirrel import SquirrelClient, SquirrelError


class PutBody(BaseModel):
    """A write to the device in the path: role (and axis), value, operator, hazard confirm."""

    model_config = ConfigDict(extra="forbid")

    role: str
    axis: str | None = None
    value: WriteValue
    operator: str = Field(min_length=1)
    confirm: bool = False


class RunBody(BaseModel):
    """A run of a plan: who asks, and the one hazard confirm for the whole plan."""

    model_config = ConfigDict(extra="forbid")

    operator: str = Field(min_length=1)
    confirm: bool = False


#: A name in a description line: no newline, no comma (the grammar's separators).
_Name = Annotated[str, Field(min_length=1, pattern=r"^[^,\r\n]+$")]


class Scope(BaseModel):
    """What a preset covers: named devices, whole families, screens (their files' bindings)."""

    model_config = ConfigDict(extra="forbid")

    devices: list[_Name] = []
    families: list[_Name] = []
    screen: list[_Name] = []


class SaveBody(BaseModel):
    """A save: the snapshot's name, who saves, the scope its description records."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    operator: _Name
    scope: Scope = Scope()


class GoldenBody(BaseModel):
    """The golden mark: who sets it, for this scope (default: the snapshot's own)."""

    model_config = ConfigDict(extra="forbid")

    operator: _Name
    scope: Scope = Scope()


class RestoreBody(BaseModel):
    """A restore: who asks, the one hazard confirm, and a device subset of the scope."""

    model_config = ConfigDict(extra="forbid")

    operator: str = Field(min_length=1)
    confirm: bool = False
    devices: list[str] | None = None


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "detail": detail})


def _error_handlers(app: FastAPI) -> None:
    """Canopy's ``{code, detail}`` envelope for what the module refuses before it writes."""

    @app.exception_handler(DevicesUnavailable)
    async def unavailable(_: Request, exc: DevicesUnavailable) -> JSONResponse:
        return _error(503, "devices_unavailable", str(exc))

    @app.exception_handler(Refusal)
    async def refused(_: Request, exc: Refusal) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=exc.body())


def build_app(module: Module, *, read_access: Callable[..., Any], squirrel_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:  # fmt: skip
    """Build the sub-app; the wrapper mounts and tags it and supplies Canopy's read check (``squirrel_transport``: a test's)."""
    app = FastAPI(title="Canopy / GEECS", description="GEECS devices over Hare.")
    stale, read = module.settings.stale_after_s, [Security(read_access)]

    _error_handlers(app)

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

    actions = Executor(module)
    _action_routes(app, actions, read, read_access)
    s = module.settings
    squirrel = SquirrelClient(s.squirrel_url, s.squirrel_key, transport=squirrel_transport) if s.squirrel_url and s.squirrel_key else None  # fmt: skip
    _preset_routes(app, Presets(module, actions, squirrel), read, read_access)

    @app.get("/v1/journal", dependencies=read)
    async def journal(entries: int = Query(50, ge=1, le=1000)) -> Any:
        rows = await asyncio.to_thread(module.journal.last, entries)  # off the loop
        return [row.model_dump(mode="json") for row in rows]

    return app


def _action_routes(app: FastAPI, actions: Executor, read: list[Any], read_access: Callable[..., Any]) -> None:  # fmt: skip
    """The action plans: the list, a run's start, its state and its stop; read access, as every route."""
    app.state.actions = actions

    @app.get("/v1/actions", dependencies=read)
    async def action_plans() -> Any:
        return {"plans": await actions.listing()}

    @app.post("/v1/actions/{name}/run")
    async def run(name: str, body: RunBody, key: Any = Security(read_access)) -> Any:
        started = await actions.start(name, body.operator, body.confirm, getattr(key, "name", None))  # fmt: skip
        return JSONResponse(status_code=202, content={"runId": started.id} | started.body())  # fmt: skip

    @app.get("/v1/actions/runs/{run_id}", dependencies=read)
    async def action_run(run_id: str) -> Any:
        if (found := actions.runs.get(run_id)) is None:
            return _error(404, "not_found", f"no run {run_id!r} (the last {len(actions.runs)} are kept)")  # fmt: skip
        return found.body()

    @app.post("/v1/actions/runs/{run_id}/stop", dependencies=read)
    async def stop(run_id: str) -> Any:
        if (found := actions.stop(run_id)) is None:
            return _error(404, "not_found", f"no run {run_id!r}")
        return JSONResponse(status_code=202, content=found.body())


def _preset_routes(app: FastAPI, presets: Presets, read: list[Any], read_access: Callable[..., Any]) -> None:  # fmt: skip
    """The presets: Squirrel's snapshots listed, saved, compared with live, marked golden, restored as a run."""
    app.state.presets = presets

    @app.exception_handler(SquirrelError)
    async def squirrel_down(_: Request, exc: SquirrelError) -> JSONResponse:
        status = 502 if exc.status in (0, 401, 403) else exc.status
        return _error(status, "squirrel_unavailable", exc.detail)

    @app.get("/v1/presets", dependencies=read)
    async def listing() -> Any:
        return {"presets": await presets.listing()}

    @app.post("/v1/presets")
    async def save(body: SaveBody, key: Any = Security(read_access)) -> Any:
        saved = await presets.save(body.name, body.operator, body.scope.model_dump(), getattr(key, "name", None))  # fmt: skip
        return JSONResponse(status_code=201, content=saved)

    @app.get("/v1/presets/{snapshot_id}/diff", dependencies=read)
    async def diff(snapshot_id: str, devices: list[str] = Query([])) -> Any:
        return await presets.diff(snapshot_id, devices or None)

    @app.post("/v1/presets/{snapshot_id}/golden")
    async def golden(snapshot_id: str, body: GoldenBody, key: Any = Security(read_access)) -> Any:  # fmt: skip
        return await presets.golden(snapshot_id, body.operator, body.scope.model_dump(), getattr(key, "name", None))  # fmt: skip

    @app.post("/v1/presets/{snapshot_id}/restore")
    async def restore(snapshot_id: str, body: RestoreBody, key: Any = Security(read_access)) -> Any:  # fmt: skip
        run = await presets.restore(snapshot_id, body.operator, body.confirm, body.devices, getattr(key, "name", None))  # fmt: skip
        return JSONResponse(status_code=202, content={"runId": run.id} | run.body())
