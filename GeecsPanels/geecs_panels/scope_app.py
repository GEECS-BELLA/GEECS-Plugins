"""The oscilloscope panel's routes.

- ``/scope/{device}`` — the panel: the traces on one time axis, the
  channel, timebase and trigger settings, the capture state.
- ``/scope/{device}/events`` — the settings and status as SSE, one
  ``state`` event every :data:`~geecs_panels.app.PUSH_S`; ``?traces=1``
  adds the decimated traces; ``?once=1`` sends one and closes.
- ``/scope/{device}/put`` — one allow-listed setting (:data:`geecs_panels.scope.WRITABLE`).

The page speaks geecs-epics' scope contract only, so it fits any scope
vendor: enum choices arrive in the stream (the PVs' own), the channels are
the ones the model reports. Every setting of the scope is writable: the
panel exists to drive it.
The IOC applies or refuses each one; a refused setting snaps back (its
setpoint follows the value in force) and the reason is ``LastError``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Protocol, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from geecs_panels.feed import ArraysSource, ArraysUnavailable, WriteFailed
from geecs_panels.registry import Device
from geecs_panels.scope import CHANNELS, NotWritable, check_write
from geecs_panels.scope_feed import ScopeFeed, traces_payload

#: Seconds between SSE pushes.
PUSH_S = 0.25

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class Scopes(Protocol):
    """Where the app gets each scope's live state (real: :class:`geecs_panels.ca.LiveScopes`)."""

    def feed(self, device: Device) -> ScopeFeed:
        """The scope's started feed."""

    def traces(self, device: Device) -> ArraysSource:
        """The scope's traces, read on request and shared."""

    async def put(self, device: Device, suffix: str, value: object) -> None:
        """Write one setting; :class:`WriteFailed` if the IOC refuses."""


class ScopePut(BaseModel):
    """One allow-listed setting: a PV suffix under the scope's prefix and its value."""

    pv: str
    value: Union[int, float, str]


def add_scope_routes(
    app: FastAPI,
    templates: Jinja2Templates,
    experiment: str,
    scopes: Scopes,
    scope: Callable[[str], Device],
    writable: bool,
    finite: Callable[[object], object],
) -> None:
    """Mount the scope panel's page, its stream and its write route."""

    @app.get("/scope/{name}", response_class=HTMLResponse)
    def scope_page(request: Request, name: str) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "scope.html",
            {
                "experiment": experiment,
                "device": scope(name),
                "writable": writable,
                "channels": CHANNELS,
            },
        )

    _add_stream(app, scopes, scope, finite)
    _add_write(app, scopes, scope, writable)


def _add_stream(
    app: FastAPI,
    scopes: Scopes,
    scope: Callable[[str], Device],
    finite: Callable[[object], object],
) -> None:
    """The settings, status and (on request) traces, as SSE."""

    @app.get("/scope/{name}/events")
    async def scope_events(
        request: Request, name: str, once: int = 0, traces: int = 0
    ) -> StreamingResponse:
        device = scope(name)
        feed = scopes.feed(device)

        def snapshot() -> dict[str, object]:
            snap = feed.snapshot()
            if traces:
                # Never awaited: the traces are read in the background and
                # the settings go out on time whatever the read is doing.
                source = scopes.traces(device)
                source.kick()
                try:
                    snap["traces"] = traces_payload(
                        source.current(), feed.present_channels()
                    )
                except ArraysUnavailable as exc:
                    snap["traces"] = {"error": str(exc)}
            return snap

        async def gen() -> AsyncIterator[str]:
            while True:
                payload = json.dumps(finite(snapshot()), default=str, allow_nan=False)
                yield f"event: state\ndata: {payload}\n\n"
                if once or await request.is_disconnected():
                    return
                await asyncio.sleep(PUSH_S)

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers=_SSE_HEADERS
        )


def _add_write(
    app: FastAPI, scopes: Scopes, scope: Callable[[str], Device], writable: bool
) -> None:
    """One allow-listed setting per request."""

    @app.post("/scope/{name}/put")
    async def scope_put(name: str, request: ScopePut) -> dict[str, object]:
        device = scope(name)
        if not writable:
            raise HTTPException(403, "this panel is read-only (--read-only)")
        try:
            value = check_write(request.pv, request.value, scopes.feed(device).enums)
        except NotWritable as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            await scopes.put(device, request.pv, value)
        except WriteFailed as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"ok": True, "pv": request.pv, "value": value}
