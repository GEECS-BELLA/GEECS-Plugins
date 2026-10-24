"""The positioner panel's routes.

- ``/motor/{device}`` — the panel: one card per axis (position and
  setpoint, moves, stop, limits, homing, velocity).
- ``/motor/{device}/events`` — every axis's fields as SSE, one ``state``
  event every :data:`PUSH_S`; ``?once=1`` sends one and closes.
- ``/motor/{device}/put`` — one allow-listed write to one axis
  (:func:`geecs_panels.motor.check_write`): a move is checked against the
  soft limits the feed holds before it reaches the record; a field outside
  the device's permitted actions (``Device.allow``) is a 403, and the page
  does not draw its control.
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

from geecs_panels.feed import WriteFailed
from geecs_panels.motor import (
    ACTIONS,
    NotAllowed,
    NotWritable,
    allowed_actions,
    allowed_fields,
    check_write,
)
from geecs_panels.motor_feed import MotorFeed
from geecs_panels.registry import Device

#: Seconds between SSE pushes.
PUSH_S = 0.25

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class Motors(Protocol):
    """Where the app gets each positioner's live state (real: :class:`geecs_panels.ca.LiveMotors`)."""

    def feed(self, device: Device) -> MotorFeed:
        """The positioner's started feed."""

    async def put(self, device: Device, axis: str, field: str, value: object) -> None:
        """Write one field of one axis; :class:`WriteFailed` if the IOC refuses."""


class MotorPut(BaseModel):
    """One allow-listed write: an axis of the device, a record field and its value."""

    axis: str
    field: str
    value: Union[int, float]


def add_motor_routes(
    app: FastAPI,
    templates: Jinja2Templates,
    experiment: str,
    motors: Motors,
    motor: Callable[[str], Device],
    writable: bool,
    finite: Callable[[object], object],
) -> None:
    """Mount the positioner panel's page, its stream and its write route."""

    @app.get("/motor/{name}", response_class=HTMLResponse)
    def motor_page(request: Request, name: str) -> HTMLResponse:
        device = motor(name)
        return templates.TemplateResponse(
            request,
            "motor.html",
            {
                "experiment": experiment,
                "device": device,
                "writable": writable,
                "actions": allowed_actions(device.allow),
                "held": [a for a in ACTIONS if a not in allowed_actions(device.allow)],
                "restricted": device.allow is not None,
            },
        )

    @app.get("/motor/{name}/events")
    async def motor_events(
        request: Request, name: str, once: int = 0
    ) -> StreamingResponse:
        feed = motors.feed(motor(name))

        async def gen() -> AsyncIterator[str]:
            while True:
                payload = json.dumps(
                    finite(feed.snapshot()), default=str, allow_nan=False
                )
                yield f"event: state\ndata: {payload}\n\n"
                if once or await request.is_disconnected():
                    return
                await asyncio.sleep(PUSH_S)

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    _add_write(app, motors, motor, writable)


def _add_write(
    app: FastAPI, motors: Motors, motor: Callable[[str], Device], writable: bool
) -> None:
    """One allow-listed write to one axis per request."""

    @app.post("/motor/{name}/put")
    async def motor_put(name: str, request: MotorPut) -> dict[str, object]:
        device = motor(name)
        if not writable:
            raise HTTPException(403, "this panel is read-only (--read-only)")
        if request.axis not in device.axes:
            raise HTTPException(404, f"{device.name} has no axis {request.axis!r}")
        try:
            value = check_write(
                request.field,
                request.value,
                motors.feed(device).axis(request.axis),
                allowed_fields(device.allow),
            )
        except NotAllowed as exc:
            raise HTTPException(403, f"{device.name}: {exc}") from exc
        except NotWritable as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            await motors.put(device, request.axis, request.field, value)
        except WriteFailed as exc:
            raise HTTPException(502, str(exc)) from exc
        return {
            "ok": True,
            "axis": request.axis,
            "field": request.field,
            "value": value,
        }
