"""Screen files: read one, bind it against the experiment's devices, and the routes that draw and write it.

``<experiment>/screens/<name>.yaml`` is validated by GEECS-Schemas'
``Screen`` and bound by ``resolve_screen`` against the installed families,
``panel_devices.yaml`` and ``native_devices.yaml``. The page reads live
values from Hare itself; the routes here only pass the module's answers
through: a device's verdicts, and a put, sent once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from geecs_schemas import (
    NATIVE_DEVICES_FILENAME,
    PANEL_DEVICES_FILENAME,
    SCREENS_DIRNAME,
    Bound,
    NativeDevices,
    PanelDevices,
    Screen,
    ScreenError,
    load_families,
    resolve_screen,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from geecs_control_room.module_client import ModuleUnavailable
from geecs_control_room.settings import Settings

_NAME = re.compile(r"^[\w-]+$")
#: Raster images only: an SVG opened directly would run its script on this origin.
_IMAGE = re.compile(r"^[\w-]+\.(png|jpe?g|gif|webp)$", re.IGNORECASE)

#: §3's words, each worn as a kit state: crit for an alarm, warn for any other doubt.
KIT_STATE = {
    "live": "ok",
    "stale": "degraded",
    "not answering": "degraded",
    "unreachable": "degraded",
    "off": "unknown",
    "error": "failed",
    "unknown": "unknown",
}


@dataclass
class Loaded:
    """A screen bound to its roles, or why it could not be: the HTTP status and every problem."""

    screen: Optional[Screen] = None
    rows: list[list[Bound]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    status: int = 200

    @property
    def columns(self) -> int:
        """The grid's width: the right edge of the widest group."""
        groups = self.screen.groups if self.screen else []
        return max((g.at[1] + g.size[1] - 1 for g in groups), default=1)


def load(settings: Settings, name: str) -> Loaded:
    """Read and bind ``<name>.yaml``; a missing file is 404, any misfit 422."""
    folder = settings.experiment_dir
    path = folder / SCREENS_DIRNAME / f"{name}.yaml"
    if not _NAME.fullmatch(name) or not path.is_file():
        return Loaded(problems=[f"no screen {name!r} in {settings.experiment}"], status=404)  # fmt: skip
    try:
        screen = Screen.from_path(path)
        panel, native = PanelDevices(), NativeDevices.model_validate({})
        if (folder / PANEL_DEVICES_FILENAME).is_file():
            panel = PanelDevices.from_path(folder / PANEL_DEVICES_FILENAME)
        if (folder / NATIVE_DEVICES_FILENAME).is_file():
            native = NativeDevices.from_path(folder / NATIVE_DEVICES_FILENAME)
        return Loaded(screen, resolve_screen(screen, load_families(), panel, native))
    except ScreenError as exc:
        return Loaded(problems=exc.problems, status=422)
    except (ValidationError, yaml.YAMLError, OSError) as exc:
        return Loaded(problems=[str(exc)], status=422)


class PutBody(BaseModel):
    """A write as the module takes it: role (and axis), value, operator, hazard confirm."""

    model_config = ConfigDict(extra="forbid")

    role: str
    axis: Optional[str] = None
    value: Union[bool, float, int, str]
    operator: str = Field(min_length=1)
    confirm: bool = False


def router(settings: Settings) -> APIRouter:
    """The screen page, the screens' images, and the two module pass-throughs."""
    r = APIRouter()

    @r.get("/screens/{name}")
    async def screen_page(request: Request, name: str) -> Any:
        loaded = load(settings, name)
        context = {"loaded": loaded, "name": name, "kit_state": KIT_STATE}
        templates = request.app.state.templates
        return templates.TemplateResponse(
            request, "screen.html", context, status_code=loaded.status
        )

    @r.get("/screens/files/{file}")
    async def screen_image(file: str) -> Any:
        path: Path = settings.experiment_dir / SCREENS_DIRNAME / file
        if not _IMAGE.fullmatch(file) or not path.is_file():
            return JSONResponse({"code": "not_found", "detail": file}, status_code=404)
        return FileResponse(path)

    def unreachable(exc: ModuleUnavailable, what: str) -> JSONResponse:
        body = {"outcome": "unknown", "code": "module_unreachable", "detail": f"{what}: {exc}"}  # fmt: skip
        return JSONResponse(body, status_code=502)

    @r.get("/api/devices/{name}")
    async def device(request: Request, name: str) -> Any:
        try:
            status, body = await request.app.state.module.device(name)
        except ModuleUnavailable as exc:
            return unreachable(exc, "the module did not answer")
        return JSONResponse(body, status_code=status)

    @r.post("/api/devices/{name}/put")
    async def put(request: Request, name: str, body: PutBody) -> Any:
        try:
            sent = body.model_dump(exclude_none=True)
            status, answer = await request.app.state.module.put(name, sent)
        except ModuleUnavailable as exc:
            why = "no answer; read the device and the journal before repeating it"
            return unreachable(exc, why)
        return JSONResponse(answer, status_code=status)

    return r
