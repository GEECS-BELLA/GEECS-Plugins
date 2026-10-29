"""Screen files: read one, bind it against the experiment's devices, and the routes that draw and write it.

``<experiment>/screens/<name>.yaml`` is validated by GEECS-Schemas'
``Screen`` and bound by ``resolve_screen`` against the installed families,
``panel_devices.yaml`` and ``native_devices.yaml``, and the module's roster
(read at most every ``ROSTER_TTL_S``): a device only the module lists (a
DB device) is bound deferred, its roles judged on the page by the module's
device route. The page reads live values from Hare itself; the routes here
only pass the module's answers through: a device's verdicts, and a put,
sent once.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Collection
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

from geecs_control_room.module_client import ModuleClient, ModuleUnavailable
from geecs_control_room.settings import Settings

_NAME = re.compile(r"^[\w-]+$")
#: Raster images only: an SVG opened directly would run its script on this origin.
_IMAGE = re.compile(r"^[\w-]+\.(png|jpe?g|gif|webp)$", re.IGNORECASE)
#: How long the module's roster names are reused: the module's own native cadence.
ROSTER_TTL_S = 10.0

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

    @property
    def stacks(self) -> Optional[list[list[int]]]:
        """Each column's groups top to bottom, packed as a LabVIEW panel is; ``None`` (the grid) when one spans columns."""
        groups = self.screen.groups if self.screen else []
        if any(g.size[1] > 1 for g in groups):
            return None
        columns: list[list[int]] = [[] for _ in range(self.columns)]
        for i in sorted(range(len(groups)), key=lambda i: groups[i].at):
            columns[groups[i].at[1] - 1].append(i)
        return columns


def load(settings: Settings, name: str, deferred: Collection[str] = ()) -> Loaded:
    """Read and bind ``<name>.yaml`` (``deferred``: the module's device names); a missing file is 404, any misfit 422."""
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
        bound = resolve_screen(screen, load_families(), panel, native, deferred)
        return Loaded(screen, bound)
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


class _Roster:
    """The module's device names, re-read every ``ROSTER_TTL_S``; the last good set (and why) while it is unreadable."""

    def __init__(self) -> None:
        self._names: frozenset[str] = frozenset()
        self._at, self._error = -math.inf, None

    async def names(self, module: ModuleClient) -> tuple[frozenset[str], Optional[str]]:
        if time.monotonic() - self._at < ROSTER_TTL_S:
            return self._names, self._error
        self._at = time.monotonic()
        try:
            status, body = await module.devices()
        except ModuleUnavailable as exc:
            self._error = str(exc)
            return self._names, self._error
        devices = body.get("devices") if status == 200 else None
        if devices is None:
            self._error = f"the module answered {status}"
        else:
            self._names, self._error = frozenset(d["name"] for d in devices), None
        return self._names, self._error


def router(settings: Settings) -> APIRouter:
    """The screen page, the screens' images, and the two module pass-throughs."""
    r = APIRouter()
    roster = _Roster()

    @r.get("/screens/{name}")
    async def screen_page(request: Request, name: str) -> Any:
        names, error = await roster.names(request.app.state.module)
        loaded = load(settings, name, names)
        if (
            error and loaded.status == 422
        ):  # a DB device may be refused for want of the roster
            loaded.problems.append(f"the module's roster is unreadable ({error}): a DB device cannot be bound")  # fmt: skip
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
