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
from collections.abc import AsyncIterator, Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from geecs_web_theme.web import (
    ForwardedPrefixMiddleware,
    RevalidatingStaticFiles,
    make_templates,
    mount_theme,
)

from geecs_panels.camera import READINGS
from geecs_panels.feed import CameraFeed, FrameSource, FrameUnavailable
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
                "stale_after": stale_after,
                "max_fps": max_fps,
            },
        )

    _add_live_routes(app, cameras, camera, stale_after)

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
