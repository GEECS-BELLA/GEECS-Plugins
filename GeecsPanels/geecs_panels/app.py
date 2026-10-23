"""The panels process: its app factory and routes.

- ``/`` — the experiment's native devices from the registry, by kind, each
  linking to its panel.
- ``/camera/{device}`` — the camera panel: the live image (pulled one
  frame at a time) and the live readings (Server-Sent Events).
- ``/camera/{device}/frame.jpg`` — the latest frame, shared by every
  viewer (:class:`~geecs_panels.feed.FrameSource`).
- ``/camera/{device}/events`` — the camera's readings as SSE, one
  ``state`` event every :data:`PUSH_S`; ``?once=1`` sends one and closes.
- ``/scope/{device}`` and its stream and writes — the oscilloscope panel
  (:mod:`geecs_panels.scope_app`).
- ``/health``.

Read-only in this build: no route writes a PV. The web glue (prefix
middleware, ``/theme``, ``root`` in every template) is GeecsWebTheme's.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    LOCK_FIELDS,
    LOCKED_PREFIXES,
    READINGS,
    TRANSFORMS,
    NotWritable,
    check_write,
    lock_state,
)
from geecs_panels.coords import to_image, to_roi
from geecs_panels.feed import (
    ArraysSource,
    ArraysUnavailable,
    CameraFeed,
    FrameSource,
    FrameUnavailable,
    WriteFailed,
    plot_payload,
)
from geecs_panels.registry import Device
from geecs_panels.scope import SCOPE_KIND
from geecs_panels.scope_app import Scopes, add_scope_routes

logger = logging.getLogger(__name__)

_PKG = Path(__file__).resolve().parent
TEMPLATES_DIR = _PKG / "templates"
STATIC_DIR = _PKG / "static"

#: Device kind → the panel route that serves it. A kind without a panel is
#: listed on the landing page, unlinked.
PANELS = {"aravis_camera": "camera", SCOPE_KIND: "scope"}

#: Seconds between SSE pushes: ages tick on the page at this rate.
PUSH_S = 0.5
#: How much history the plots are sent.
HISTORY_S = 120.0

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # nginx: never buffer this response
}


def finite(value: object) -> object:
    """*value* with every non-finite float (NaN, ±inf) replaced by ``None``, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


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

    def arrays(self, device: Device) -> ArraysSource:
        """The camera's STAT arrays, read on request and shared."""


class Colour(BaseModel):
    """A crosshair colour, 0..255 a channel (the IOC stores it per overlay)."""

    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)


class Point(BaseModel):
    """A point in image pixels."""

    x: float
    y: float


class Override(BaseModel):
    """The stuck-lock override: who is releasing it (the surfaces carry no user identity)."""

    operator: str = Field(min_length=1, max_length=60)


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
    scopes: Optional[Scopes] = None,
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
    scopes : Scopes, optional
        The live state of each oscilloscope; without it no scope panel is served.
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

    def scope(name: str) -> Device:
        device = by_name.get(name)
        if device is None or PANELS.get(device.kind) != "scope":
            raise HTTPException(404, f"no oscilloscope named {name!r} in {experiment}")
        return device

    if scopes is not None:
        add_scope_routes(app, templates, experiment, scopes, scope, writable, finite)

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
    async def events(
        request: Request, name: str, once: int = 0, plots: int = 0
    ) -> StreamingResponse:
        """The readings as SSE; ``?plots=1`` adds the plots' arrays and history."""
        device = camera(name)
        feed = cameras.feed(device)

        def snapshot() -> dict[str, object]:
            snap = feed.snapshot(stale_after)
            if plots:
                # Never awaited: the arrays are read in the background, and
                # the readings go out on time whatever the read is doing.
                source = cameras.arrays(device)
                source.kick()
                try:
                    snap["plots"] = plot_payload(
                        feed, source.current(), HISTORY_S, stale_after
                    )
                except ArraysUnavailable as exc:  # said on the page, never fatal
                    snap["plots"] = {"error": str(exc)}
            return snap

        async def gen() -> AsyncIterator[str]:
            while True:
                # Any NaN or inf anywhere (a reading, a plot) becomes null: the
                # page's JSON.parse rejects them, and one must not kill the
                # stream. allow_nan=False is the backstop.
                payload = json.dumps(finite(snapshot()), default=str, allow_nan=False)
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
    """What every write route shares: the read-only refusal, the run lock and the put."""

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
        """One allow-listed write; a locked camera is a 423, the IOC refusing it a 502."""
        try:
            value = check_write(suffix, value)
        except NotWritable as exc:
            raise HTTPException(422, str(exc)) from exc
        if suffix.startswith(LOCKED_PREFIXES):
            # The IOC enforces this too (its access security); refusing here
            # first says why, and covers an IOC whose lock is not served.
            feed = self.cameras.feed(device)
            state = lock_state(feed.values, feed.connected)
            if state != "free":
                owner = feed.values.get(LOCK_FIELDS["owner"]) or "unknown"
                raise HTTPException(
                    423,
                    f"{suffix}: the camera belongs to a run ({owner})"
                    if state == "locked"
                    else f"{suffix}: the camera's run lock cannot be read",
                )
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
    _add_plot_routes(app, cameras, w)
    _add_lock_routes(app, cameras, w)

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
        if (
            request.pv == "PROC1:EnableBackground"
            and request.value
            and cameras.feed(device).background_stale
        ):
            # For every viewer: the held background predates a rotation.
            raise HTTPException(
                409, "the background predates a transform change: grab a new one first"
            )
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


def _add_plot_routes(app: FastAPI, cameras: Cameras, w: "_Writer") -> None:
    """The lineout cursor: a clicked image point, written in the ROI's pixels."""

    @app.post("/camera/{name}/cursor")
    async def cursor(name: str, point: Point) -> dict[str, object]:
        device = w.device(name)
        try:
            ax, ay = cameras.feed(device).roi_axes()
        except LookupError as exc:
            raise HTTPException(409, str(exc)) from exc
        out = []
        for axis, value in ((ax, point.x), (ay, point.y)):
            # The page sends a position; pixel i covers [i, i+1), as the
            # cursor readout names it. Its index, then the ROI pixel nearest.
            index = math.floor(value)
            last = max(axis.size // max(axis.bin, 1) - 1, 0)
            out.append(min(max(math.floor(to_roi(axis, index) + 0.5), 0), last))
        await w.put(device, "STAT:CursorX", out[0])
        await w.put(device, "STAT:CursorY", out[1])
        return {"ok": True, "x": out[0], "y": out[1]}


def _add_lock_routes(app: FastAPI, cameras: Cameras, w: "_Writer") -> None:
    """The stuck-lock override: release ``RUN:Lock``, never anything else."""

    @app.post("/camera/{name}/unlock")
    async def unlock(name: str, request: Override) -> dict[str, object]:
        """Release a stuck run lock (a killed worker), naming who did, in the log.

        Writes ``RUN:Lock`` 0 only — never the writer or the driver — so
        misjudging a live run as stuck cannot end its stack file; at worst
        it reopens, for that run, the race the lock closes.
        """
        device = w.device(name)
        feed = cameras.feed(device)
        values = feed.values
        # Directly, not through the allow-list: the lock is not a plain write
        # (it is never in WRITABLE), only this named, logged release.
        try:
            await cameras.put(device, LOCK_FIELDS["lock"], 0)
        except WriteFailed as exc:
            raise HTTPException(502, str(exc)) from exc
        logger.warning(
            "run lock of %s released by %s (owner %r, HDF capture %s, image age %s)",
            device.name,
            request.operator,
            values.get(LOCK_FIELDS["owner"]),
            values.get(LOCK_FIELDS["capture"]),
            feed.counter_ages().get("MJPG1"),
        )
        return {"ok": True}
