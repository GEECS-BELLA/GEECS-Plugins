"""The panels process: its app factory and routes.

- ``/`` — the experiment's native devices from the registry, by kind, each
  linking to its panel.
- ``/camera/{device}`` — the camera panel: the live image (pulled one
  frame at a time) and the live readings (Server-Sent Events).
- ``/camera/{device}/frame.jpg`` — the latest frame, shared by every
  viewer (:class:`~geecs_panels.feed.FrameSource`).
- ``/camera/{device}/events`` — the camera's readings as SSE, one
  ``state`` event every :data:`PUSH_S`; ``?once=1`` sends one and closes.
- ``/health``.

Read-only in this build: no route writes a PV. The web glue (prefix
middleware, ``/theme``, ``root`` in every template) is GeecsWebTheme's.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Protocol, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from geecs_web_theme.web import (
    ForwardedPrefixMiddleware,
    RevalidatingStaticFiles,
    make_templates,
    mount_theme,
)

from geecs_panels.camera import (
    CROSSHAIRS,
    READINGS,
    TRANSFORMS,
    NotWritable,
    check_write,
)
from geecs_panels.coords import to_image
from geecs_panels.feed import CameraFeed, FrameSource, FrameUnavailable, WriteFailed
from geecs_panels.registry import Device

_PKG = Path(__file__).resolve().parent
TEMPLATES_DIR = _PKG / "templates"
STATIC_DIR = _PKG / "static"

#: Device kind → the panel route that serves it. A kind without a panel is
#: listed on the landing page, unlinked.
PANELS = {"aravis_camera": "camera"}

#: Seconds between SSE pushes: ages tick on the page at this rate.
PUSH_S = 0.5

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # nginx: never buffer this response
}


def _version() -> str:
    try:
        return version("geecs-panels")
    except PackageNotFoundError:  # pragma: no cover - a source tree without install
        return "0.0.0"


class Cameras(Protocol):
    """Where the app gets each camera's live state (real: :class:`geecs_panels.ca.LiveCameras`)."""

    def feed(self, device: Device) -> CameraFeed:
        """The camera's started feed."""

    def frames(self, device: Device) -> FrameSource:
        """The camera's shared frame source."""

    async def put(self, device: Device, suffix: str, value: object) -> None:
        """Write one PV under the camera's prefix; :class:`WriteFailed` if the IOC refuses."""


class Colour(BaseModel):
    """A crosshair colour, 0..255 a channel (the IOC stores it per overlay)."""

    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)


class PutRequest(BaseModel):
    """One allow-listed write: a PV suffix under the camera's prefix and its value."""

    pv: str
    value: Union[bool, int, float, str]


def create_app(
    experiment: str,
    devices: list[Device],
    cameras: Cameras,
    *,
    registry_path: Optional[Path] = None,
    registry_error: Optional[str] = None,
    stale_after: float = 3.0,
    max_fps: float = 10.0,
    root_path: str = "",
    writable: bool = True,
) -> FastAPI:
    """Build the panels application.

    Parameters
    ----------
    experiment : str
        The experiment whose registry was read (shown on every page).
    devices : list of Device
        The registry's devices, in its order.
    cameras : Cameras
        The live state of each camera.
    registry_path : Path, optional
        Where the registry was looked for (said on the landing page).
    registry_error : str, optional
        Why the registry could not be read; the landing page says so.
    stale_after : float
        Seconds without a frame-counter advance before a value is stale.
    max_fps : float
        The frame rate the page asks for at most.
    root_path : str
        URL prefix behind a prefix-preserving reverse proxy.
    writable : bool
        Whether the write routes (crosshairs, analysis ROI) are open; off,
        every write is a 403 and the page draws its controls disabled.
    """
    app = FastAPI(
        title="GEECS Panels",
        version=_version(),
        root_path=root_path,
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(ForwardedPrefixMiddleware)
    mount_theme(app)
    app.mount(
        "/static",
        RevalidatingStaticFiles(directory=str(STATIC_DIR)),
        name="panels_static",
    )
    templates = make_templates(TEMPLATES_DIR)
    by_name = {d.name: d for d in devices}

    def camera(name: str) -> Device:
        device = by_name.get(name)
        if device is None or PANELS.get(device.kind) != "camera":
            raise HTTPException(404, f"no camera named {name!r} in {experiment}")
        return device

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        kinds: dict[str, list[Device]] = {}
        for device in devices:
            kinds.setdefault(device.kind, []).append(device)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "experiment": experiment,
                "kinds": kinds,
                "panels": PANELS,
                "registry_path": registry_path,
                "registry_error": registry_error,
            },
        )

    @app.get("/camera/{name}", response_class=HTMLResponse)
    def camera_page(request: Request, name: str) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "camera.html",
            {
                "experiment": experiment,
                "device": camera(name),
                "readings": READINGS,
                "transforms": TRANSFORMS,
                "stale_after": stale_after,
                "max_fps": max_fps,
                "writable": writable,
            },
        )

    _add_live_routes(app, cameras, camera, stale_after)
    _add_write_routes(app, cameras, camera, stale_after, writable)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "experiment": experiment, "devices": len(devices)}

    return app


def _add_live_routes(
    app: FastAPI,
    cameras: Cameras,
    camera: Callable[[str], Device],
    stale_after: float,
) -> None:
    """The camera's frame and its readings: the routes the page polls and streams."""

    @app.get("/camera/{name}/frame.jpg")
    async def frame(name: str) -> Response:
        source = cameras.frames(camera(name))
        try:
            frame = await source.latest()
        except FrameUnavailable as exc:
            return Response(str(exc), status_code=503, media_type="text/plain")
        # The page skips decoding a frame it already shows.
        headers = {"Cache-Control": "no-store", "X-Frame-Tag": frame.tag}
        return Response(frame.jpeg, media_type="image/jpeg", headers=headers)

    @app.get("/camera/{name}/events")
    async def events(request: Request, name: str, once: int = 0) -> StreamingResponse:
        feed = cameras.feed(camera(name))

        async def gen() -> AsyncIterator[str]:
            while True:
                payload = json.dumps(feed.snapshot(stale_after), default=str)
                yield f"event: state\ndata: {payload}\n\n"
                if once or await request.is_disconnected():
                    return
                await asyncio.sleep(PUSH_S)

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers=_SSE_HEADERS
        )


def centroid_in_image(feed: CameraFeed, stale_after: float) -> tuple[int, int]:
    """The centroid (STAT's, in ROI pixels) in image pixels; ``LookupError`` if it is not fresh."""
    states = {r["key"]: r for r in feed.snapshot(stale_after)["readings"]}
    if states["cx"]["state"] != "fresh" or states["cy"]["state"] != "fresh":
        raise LookupError("the centroid is not fresh: nothing to move to")
    ax, ay = feed.roi_axes()
    # Half up, not Python's half-to-even: 120.5 and 121.5 both go up.
    return (
        math.floor(to_image(ax, float(states["cx"]["value"])) + 0.5),
        math.floor(to_image(ay, float(states["cy"]["value"])) + 0.5),
    )


class _Writer:
    """What every write route shares: the read-only refusal and the put."""

    def __init__(
        self, cameras: Cameras, camera: Callable[[str], Device], writable: bool
    ) -> None:
        self.cameras, self.camera, self.writable = cameras, camera, writable

    def device(self, name: str, crosshair: Optional[int] = None) -> Device:
        """The camera to write to, refusing a read-only panel or an unknown crosshair."""
        device = self.camera(name)
        if not self.writable:
            raise HTTPException(403, "this panel is read-only (--read-only)")
        if crosshair is not None and crosshair not in CROSSHAIRS:
            raise HTTPException(404, f"no crosshair {crosshair}")
        return device

    async def put(self, device: Device, suffix: str, value: object) -> None:
        """One allow-listed write; the IOC refusing it is a 502."""
        try:
            value = check_write(suffix, value)
        except NotWritable as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            await self.cameras.put(device, suffix, value)
        except WriteFailed as exc:
            raise HTTPException(502, str(exc)) from exc


def _add_write_routes(
    app: FastAPI,
    cameras: Cameras,
    camera: Callable[[str], Device],
    stale_after: float,
    writable: bool,
) -> None:
    """The panel's writes: one allow-listed PV at a time (design note §4)."""
    w = _Writer(cameras, camera, writable)

    @app.post("/camera/{name}/put")
    async def put_one(name: str, request: PutRequest) -> dict[str, object]:
        device = w.device(name)
        if request.pv == "TRANS1:Type":
            # The background is grabbed AFTER the transform (cam1 → TRANS1 →
            # PROC1) and NDProcess only checks its size: rotated frames minus
            # an un-rotated background is silently wrong. Stop subtracting
            # first; the page says to grab again.
            try:
                check_write(
                    request.pv, request.value
                )  # refuse before changing anything
            except NotWritable as exc:
                raise HTTPException(422, str(exc)) from exc
            await w.put(device, "PROC1:EnableBackground", 0)
        await w.put(device, request.pv, request.value)
        return {
            "ok": True,
            "pv": request.pv,
            "value": check_write(request.pv, request.value),
        }

    @app.post("/camera/{name}/crosshair/{n}/colour")
    async def colour(name: str, n: int, request: Colour) -> dict[str, object]:
        """Set crosshair *n*'s colour: its three channel PVs, each allow-listed."""
        device = w.device(name, n)
        for field, value in (
            ("Red", request.r),
            ("Green", request.g),
            ("Blue", request.b),
        ):
            await w.put(device, f"OVER1:{n}:{field}", value)
        return {"ok": True}

    @app.post("/camera/{name}/crosshair/{n}/to-centroid")
    async def to_centroid(name: str, n: int) -> dict[str, object]:
        """Move crosshair *n* to the centroid, converted from ROI to image pixels."""
        device = w.device(name, n)
        try:
            x, y = centroid_in_image(cameras.feed(device), stale_after)
        except LookupError as exc:
            raise HTTPException(409, str(exc)) from exc
        await w.put(device, f"OVER1:{n}:CenterX", x)
        await w.put(device, f"OVER1:{n}:CenterY", y)
        return {"ok": True, "x": x, "y": y}
