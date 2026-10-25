"""The bridge's routes under ``/api`` (``Planning/epics_estate.md`` §2.3).

- ``GET /api/devices`` — the experiment's devices with a stream: the
  registry joined with its panel data (:mod:`~geecs_panels.bridge.catalog`).
- ``GET /api/families`` — the family files the bridge loaded.
- ``GET /api/devices/{name}/events`` — the device's stream (SSE, ``event:
  device``, ``id:`` its ``seq``): a full ``DeviceFrame``, then deltas
  (:mod:`~geecs_panels.bridge.stream`); ``Last-Event-ID`` resumes.
- ``GET /api/devices/{name}/arrays`` — its arrays, read on request and
  decimated (:mod:`~geecs_panels.bridge.arrays`).
- ``POST /api/subscriptions`` / ``GET /api/stream/{handle}`` — the raw PV
  stream (:mod:`~geecs_panels.bridge.pvstream`); ``GET /api/pv/{name}`` —
  the inspector's read; ``GET /api/panels`` — the experiment's composed
  panels (:mod:`~geecs_panels.bridge.panels`).
- ``POST /api/put`` — one write through the three gates, journaled
  (:mod:`~geecs_panels.bridge.puts`); ``POST /api/devices/{name}/unlock``
  — a stuck run lock released, journaled; ``GET /api/journal`` — the
  last puts (:mod:`~geecs_panels.bridge.journal`).

``GET /api/frame/{name}`` (a camera's JPEG, one upstream fetch in flight)
is the per-kind camera route's, mounted beside it in :mod:`geecs_panels.app`.
The frames' schema is the events document (``api/events.schema.json``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from geecs_schemas.epics.family import FamilyFile, load_families

from geecs_panels.bridge.arrays import ArraysReply
from geecs_panels.bridge.catalog import BridgeDevice, DeviceCatalog, device_info
from geecs_panels.bridge.journal import MAX_ENTRIES, Journal
from geecs_panels.bridge.panels import PanelsReply, load_panels
from geecs_panels.bridge.pvstream import (
    PvInfo,
    PvStreams,
    ReadInfo,
    TooManyHandles,
    UnknownHandle,
    pv_info,
    pv_sse,
)
from geecs_panels.bridge.puts import Puts
from geecs_panels.bridge.stream import DeviceStreams, UnknownDevice, sse
from geecs_panels.contracts import (
    JournalEntry,
    LockRelease,
    PutRequest,
    PutResult,
    SubscriptionAck,
    SubscriptionRequest,
)

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _no_subscriptions(*_args: object) -> object:
    raise RuntimeError("this bridge has no CA transport")


class _NoWrites:
    """The put transport of a bridge built without one: never reached (that bridge is read-only)."""

    async def put(self, pv: str, value: object, wait: bool) -> None:
        raise RuntimeError("this bridge has no CA transport")

    async def get(self, pv: str, as_string: bool = False) -> object:
        raise RuntimeError("this bridge has no CA transport")

    async def choices(self, pv: str) -> list[str]:
        raise RuntimeError("this bridge has no CA transport")


@dataclass
class Bridge:
    """What the routes serve: the experiment's devices, why any are missing, and their streams."""

    experiment: str
    devices: list[BridgeDevice] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    streams: Optional[DeviceStreams] = None
    families: Mapping[str, FamilyFile] = field(default_factory=load_families)
    puts: Optional[Puts] = None
    #: The raw PV stream; ``None``: the bridge has no CA transport (503).
    pvs: Optional[PvStreams] = None
    #: The inspector's one-shot read; ``None``: no CA transport (503).
    read_info: Optional[ReadInfo] = None
    #: ``<experiment>/panels`` in the configs repository; ``None``: no panels.
    panels_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        """Give a bridge without a transport streams that refuse to subscribe and puts that deny (listings only)."""
        if self.streams is None:
            self.streams = DeviceStreams(
                {d.name: d for d in self.devices}, _no_subscriptions
            )
        if self.puts is None:
            # Read-only: every put is denied, so the journal holds only denials.
            self.puts = Puts(
                self.streams, _NoWrites(), Journal(warn=False), writable=False
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
    assert bridge.puts is not None
    _add_put_routes(app, bridge.puts)
    _add_pv_routes(app, bridge)

    @app.get("/api/panels", response_model=PanelsReply)
    def panels() -> PanelsReply:
        """The experiment's composed panels and family overrides, validated against the devices served; the others named."""
        return load_panels(
            bridge.panels_dir,
            {d.name: d.family.template for d in bridge.devices},
            bridge.families,
        )


#: Every status a put may answer, each with the PutResult body.
_PUT_STATUSES: dict[int | str, dict[str, object]] = {
    status: {"model": PutResult, "description": why}
    for status, why in {
        403: "Denied: not permitted on this device, a hazard unconfirmed, a read-only bridge.",
        404: "No such device (no body: nothing was journaled).",
        409: "Denied: the device's stream is closed, stale or disconnected.",
        422: "Refused by the bridge: the value is outside the role's or the record's bounds.",
        423: "Denied: the run lock is held, or cannot be read.",
        502: "Refused by the IOC: a CA refusal, or the record left INVALID.",
    }.items()
}


def _answer(status: int, result: PutResult) -> JSONResponse:
    return JSONResponse(result.model_dump(mode="json"), status_code=status)


def _add_put_routes(app: FastAPI, puts: Puts) -> None:
    """Puts with identity, the run lock's release, the journal."""

    @app.post("/api/put", response_model=PutResult, responses=_PUT_STATUSES)
    async def put(request: PutRequest) -> JSONResponse:
        """One write to one role, by a named operator, through the three gates; journaled whatever the outcome."""
        try:
            status, result = await puts.put(request)
        except UnknownDevice as exc:
            raise HTTPException(404, f"no device named {request.device!r}") from exc
        return _answer(status, result)

    @app.post(
        "/api/devices/{name}/unlock", response_model=PutResult, responses=_PUT_STATUSES
    )
    async def unlock(name: str, request: LockRelease) -> JSONResponse:
        """Release a stuck run lock (a killed worker): the lock written free and nothing else, journaled."""
        try:
            status, result = await puts.release_lock(name, request.operator)
        except UnknownDevice as exc:
            raise HTTPException(404, f"no device named {name!r}") from exc
        return _answer(status, result)

    @app.get("/api/journal", response_model=list[JournalEntry])
    def journal(
        entries: int = Query(
            50, ge=1, le=MAX_ENTRIES, description="How many, newest first."
        ),
    ) -> list[JournalEntry]:
        """The last puts the bridge answered (who, what, when, how it ended), newest first."""
        return puts.journal.last(entries)


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
            1500,
            ge=1,
            description=(
                "The screen columns a trace is decimated to (a scope's capture is "
                "decimated once for every viewer, to 1500)."
            ),
        ),
    ) -> ArraysReply:
        """The device's arrays (profiles, histogram, traces), read now or within the last half second."""
        known(name)
        try:
            hub = streams.hub(name)
        except UnknownDevice as exc:  # pragma: no cover - known() said otherwise
            raise HTTPException(404, str(exc)) from exc
        return await hub.arrays(columns)


def _add_pv_routes(app: FastAPI, bridge: Bridge) -> None:
    """The raw PV stream and the inspector's read."""

    def live() -> PvStreams:
        if bridge.pvs is None:
            raise HTTPException(503, "this bridge has no CA transport")
        return bridge.pvs

    @app.post("/api/subscriptions", response_model=SubscriptionAck)
    async def subscribe(request: SubscriptionRequest) -> SubscriptionAck:
        """A handle on bare PVs (monitors shared across handles), released after its grace period unread.

        ``async``: aioca subscribes on the running event loop, which a
        threadpool route has not (and the sweeper starts on it).
        """
        try:
            return live().open(request.pvs)
        except TooManyHandles as exc:
            raise HTTPException(503, str(exc)) from exc

    _add_raw_stream(app, live)

    @app.get("/api/pv/{name}", response_model=PvInfo)
    async def inspect(name: str) -> PvInfo:
        """One PV read now: value, alarm, timestamp, and its record's units, limits and choices."""
        if bridge.read_info is None:
            raise HTTPException(503, "this bridge has no CA transport")
        if not name or any(c.isspace() for c in name):
            raise HTTPException(422, f"not a PV name: {name!r}")
        return await pv_info(name, bridge.read_info)


def _add_raw_stream(app: FastAPI, live: Callable[[], PvStreams]) -> None:
    """``GET /api/stream/{handle}``: a handle's PVs as Server-Sent Events."""

    @app.get(
        "/api/stream/{handle}",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": (
                    "Server-Sent Events, `event: pv`, `id:` the frame's seq; each "
                    "`data:` is a PvFrame (api/events.schema.json)."
                ),
                "content": {"text/event-stream": {}},
            },
            404: {"description": "No such handle (never made, or released)."},
        },
    )
    async def stream(
        handle: str,
        limit: int = Query(
            0, ge=0, description="Close after this many frames (0: never)."
        ),
    ) -> StreamingResponse:
        """The handle's PVs: all of them on connect, then once per push period those that changed."""
        streams = live()
        streams.sweep()
        if handle not in streams.handles:
            raise HTTPException(404, f"no handle {handle!r}")

        async def gen() -> AsyncIterator[str]:
            frames = streams.listen(handle)
            sent = 0
            try:
                async for frame in frames:
                    if frame is None:
                        yield ": keepalive\n\n"
                        continue
                    yield pv_sse(frame)
                    sent += 1
                    if limit and sent >= limit:
                        return
            except UnknownHandle:  # released between the check and the stream
                return
            finally:
                await frames.aclose()

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers=_SSE_HEADERS
        )
