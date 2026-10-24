"""The power-supply panel's routes.

- ``/supply/{device}`` — the panel: readbacks, the setpoint, output on/off,
  the fault reset, the loop mode (read-only) and the status bits.
- ``/supply/{device}/events`` — the supply's state as SSE, one ``state``
  event every :data:`PUSH_S`; ``?once=1`` sends one and closes.
- ``/supply/{device}/put`` — one allowed action
  (:func:`geecs_panels.power.check_write`): a setpoint is checked against its
  record's drive limits before it reaches the IOC; an action the device does
  not permit (``Device.allow``) is a 403, and the page does not draw it.
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
from geecs_panels.motor import NotAllowed, NotWritable
from geecs_panels.power import ACTIONS, allowed_actions, check_write
from geecs_panels.registry import Device
from geecs_panels.supply_feed import SupplyFeed

#: Seconds between SSE pushes.
PUSH_S = 0.5

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class Supplies(Protocol):
    """Where the app gets each supply's live state (real: :class:`geecs_panels.ca.LiveSupplies`)."""

    def feed(self, device: Device) -> SupplyFeed:
        """The supply's started feed."""

    async def put(self, device: Device, role: str, value: object) -> None:
        """Write one role's PV; :class:`WriteFailed` if the IOC or the supply refuses."""


class SupplyPut(BaseModel):
    """One action: its name (``current``, ``output``, ``reset`` …) and its value."""

    action: str
    value: Union[int, float]


def add_supply_routes(
    app: FastAPI,
    templates: Jinja2Templates,
    experiment: str,
    supplies: Supplies,
    supply: Callable[[str], Device],
    writable: bool,
    finite: Callable[[object], object],
) -> None:
    """Mount the power-supply panel's page, its stream and its write route."""

    @app.get("/supply/{name}", response_class=HTMLResponse)
    def supply_page(request: Request, name: str) -> HTMLResponse:
        device = supply(name)
        roles = dict(device.roles)
        actions = allowed_actions(roles, device.allow)
        return templates.TemplateResponse(
            request,
            "supply.html",
            {
                "experiment": experiment,
                "device": device,
                "roles": roles,
                "writable": writable,
                "actions": actions,
                "held": [
                    a for a in ACTIONS if ACTIONS[a] in roles and a not in actions
                ],
                "restricted": device.allow is not None,
            },
        )

    @app.get("/supply/{name}/events")
    async def supply_events(
        request: Request, name: str, once: int = 0
    ) -> StreamingResponse:
        feed = supplies.feed(supply(name))

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

    _add_write(app, supplies, supply, writable)


def _add_write(
    app: FastAPI, supplies: Supplies, supply: Callable[[str], Device], writable: bool
) -> None:
    """One allowed action per request."""

    @app.post("/supply/{name}/put")
    async def supply_put(name: str, request: SupplyPut) -> dict[str, object]:
        device = supply(name)
        if not writable:
            raise HTTPException(403, "this panel is read-only (--read-only)")
        roles = dict(device.roles)
        try:
            role, value = check_write(
                request.action,
                request.value,
                supplies.feed(device).current(),
                roles,
                device.allow,
            )
        except NotAllowed as exc:
            raise HTTPException(403, f"{device.name}: {exc}") from exc
        except NotWritable as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            await supplies.put(device, role, value)
        except WriteFailed as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"ok": True, "action": request.action, "value": value}
