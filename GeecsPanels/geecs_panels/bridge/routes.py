"""The bridge's routes under ``/api`` (``Planning/epics_estate.md`` §2.3).

- ``GET /api/devices`` — the experiment's devices with a stream: the
  registry joined with its panel data (:mod:`~geecs_panels.bridge.catalog`).
- ``GET /api/families`` — the family files the bridge loaded.
- ``GET /api/devices/{name}/events`` — the device's stream (SSE, ``event:
  device``, ``id:`` its ``seq``): a full ``DeviceFrame``, then deltas
  (:mod:`~geecs_panels.bridge.stream`); ``Last-Event-ID`` resumes.
- ``GET /api/devices/{name}/arrays`` — its arrays, read on request and
  decimated (:mod:`~geecs_panels.bridge.arrays`).

``GET /api/frame/{name}`` (a camera's JPEG, one upstream fetch in flight)
is the per-kind camera route's, mounted beside it in :mod:`geecs_panels.app`.
The frames' schema is the events document (``api/events.schema.json``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from geecs_schemas.epics.family import FamilyFile, load_families

from geecs_panels.bridge.arrays import ArraysReply
from geecs_panels.bridge.catalog import BridgeDevice, DeviceCatalog, device_info
from geecs_panels.bridge.stream import DeviceStreams, UnknownDevice, sse

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _no_subscriptions(*_args: object) -> object:
    raise RuntimeError("this bridge has no CA transport")


@dataclass
class Bridge:
    """What the routes serve: the experiment's devices, why any are missing, and their streams."""

    experiment: str
    devices: list[BridgeDevice] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    streams: Optional[DeviceStreams] = None
    families: Mapping[str, FamilyFile] = field(default_factory=load_families)

    def __post_init__(self) -> None:
        """Give a bridge without a transport streams that refuse to subscribe (listings only)."""
        if self.streams is None:
            self.streams = DeviceStreams(
                {d.name: d for d in self.devices}, _no_subscriptions
            )

    def catalog(self) -> DeviceCatalog:
        """The ``GET /api/devices`` answer."""
        return DeviceCatalog(
            experiment=self.experiment,
            devices=[device_info(d) for d in self.devices],
            errors=list(self.errors),
        )


def _last_event_id(header: Optional[str]) -> Optional[int]:
    """The ``Last-Event-ID`` a reconnect sent, if it is one of ours (a number)."""
    if header is None or not header.strip().isdigit():
        return None
    return int(header.strip())


def add_bridge_routes(app: FastAPI, bridge: Bridge) -> None:
    """Mount the bridge's ``/api`` routes on *app*."""
    streams = bridge.streams
    assert streams is not None

    def known(name: str) -> None:
        if name not in streams.devices:
            raise HTTPException(404, f"no device named {name!r} in {bridge.experiment}")

    @app.get("/api/devices", response_model=DeviceCatalog)
    def devices() -> DeviceCatalog:
        """The experiment's devices with a stream, and the documents or devices that could not be read."""
        return bridge.catalog()

    @app.get("/api/families", response_model=dict[str, FamilyFile])
    def families() -> Mapping[str, FamilyFile]:
        """Every family file the bridge loaded, by template name."""
        return bridge.families

    _add_stream_routes(app, streams, known)


def _add_stream_routes(
    app: FastAPI, streams: DeviceStreams, known: Callable[[str], None]
) -> None:
    """A device's stream and its arrays."""

    @app.get(
        "/api/devices/{name}/events",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": (
                    "Server-Sent Events, `event: device`, `id:` the frame's seq; each "
                    "`data:` is a DeviceFrame (api/events.schema.json)."
                ),
                "content": {"text/event-stream": {}},
            },
            404: {"description": "No such device."},
        },
    )
    async def device_events(
        name: str,
        limit: int = Query(
            0, ge=0, description="Close after this many frames (0: never)."
        ),
        last_event_id: Optional[str] = Header(
            None,
            alias="Last-Event-ID",
            description="The seq a reconnecting EventSource last saw.",
        ),
    ) -> StreamingResponse:
        """The device's live state: a full frame, then the changed roles each push period."""
        known(name)
        resume = _last_event_id(last_event_id)

        async def gen() -> AsyncIterator[str]:
            frames = streams.listen(name, resume)
            sent = 0
            try:
                async for frame in frames:
                    if frame is None:
                        yield ": keepalive\n\n"
                        continue
                    yield sse(frame)
                    sent += 1
                    if limit and sent >= limit:
                        return
            finally:
                await frames.aclose()

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    @app.get("/api/devices/{name}/arrays", response_model=ArraysReply)
    async def device_arrays(
        name: str,
        columns: int = Query(
            1500, ge=1, description="The screen columns a trace is decimated to."
        ),
    ) -> ArraysReply:
        """The device's arrays (profiles, histogram, traces), read now or within the last half second."""
        known(name)
        try:
            hub = streams.hub(name)
        except UnknownDevice as exc:  # pragma: no cover - known() said otherwise
            raise HTTPException(404, str(exc)) from exc
        return await hub.arrays(columns)
