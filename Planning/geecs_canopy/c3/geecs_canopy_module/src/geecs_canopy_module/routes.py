"""The GEECS device module: two routes over a Hare port, no Canopy import.

``GET /v1/devices/{name}`` reads every role of a device from Hare and judges
each one's freshness by Appendix A rule 1 of ``Planning/epics_estate.md``: a
value changing is never evidence of freshness, the producing plugin's counter
advancing is. ``POST /v1/devices/{name}/put`` denies a role outside the
device's allowed actions before Hare is touched, otherwise puts through Hare
with ``wait: true``; every attempt leaves one journal row. Errors use
Canopy's ``{"code", "detail"}`` envelope.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

from fastapi import FastAPI, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from geecs_canopy_module.family import Device, Role
from geecs_canopy_module.journal import Journal

# Canopy's PVWriteValue (canopy.contracts.pv), restated so this file needs no Canopy.
WriteValue = float | int | bool | str | list[float | int | bool | str]
Wire = dict[str, object]  # a PVValue as Hare sends it (camelCase keys)


class HareRefused(Exception):
    """Hare's answer to a failed put: its HTTP status, ``code`` and ``detail``."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


class Hare(Protocol):
    """What the module needs from Hare: a value read and a put."""

    async def get(self, pv: str) -> Wire:
        """Return Hare's PVValue for ``pv`` as its wire dict (camelCase)."""
        ...

    async def put(
        self, pv: str, value: WriteValue, *, wait: bool, timeout: float | None
    ) -> None:
        """Write ``value``; raise ``HareRefused`` on failure."""
        ...


class PutBody(BaseModel):
    """Body of the put route; camelCase or snake_case on input, as Canopy's DTOs."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    role: str
    axis: str | None = None
    value: WriteValue
    operator: str = Field(min_length=1)


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "detail": detail})


def _advanced(first: Wire, second: Wire) -> bool:
    a, b = first.get("value"), second.get("value")
    numbers = all(isinstance(v, int | float) for v in (a, b))
    return bool(
        first.get("connected") and second.get("connected") and numbers and b > a
    )


def verdict(role: Role, value: Wire, flag: Wire | None, advanced: bool | None) -> str:
    """Judge one role: ``disconnected``, ``off``, ``fresh``/``stale``, else ``connected``."""
    off = flag is not None and flag.get("connected") and flag.get("value") in (0, False)
    if not value.get("connected"):
        return "disconnected"
    if role.computed_by and off:
        return "off"
    if role.fresh_by:
        return "fresh" if advanced else "stale"
    return "connected"


async def _read(hare: Hare, pvs: list[str]) -> list[Wire]:
    return list(await asyncio.gather(*(hare.get(pv) for pv in pvs)))


def _answer(
    journal: Journal, status: int, row: Wire, code: str | None, detail: str | None
) -> JSONResponse:
    """Journal one put attempt and answer with its outcome (plus Canopy's envelope if failed)."""
    journal_id = journal.record(**row, canopy_code=code, detail=detail)
    content = {k: row[k] for k in ("outcome", "device", "role", "axis")} | {
        "journalId": journal_id
    }
    error = {"code": code, "detail": detail} if code else {}
    return JSONResponse(status_code=status, content=content | error)


def build_app(
    hare: Hare,
    journal: Journal,
    devices: dict[str, Device],
    *,
    read_access: Callable[..., object],
    write_access: Callable[..., object],
    window_s: float = 1.0,
    put_timeout_s: float | None = 30.0,
) -> FastAPI:
    """Build the module's sub-app; the caller mounts it and supplies Canopy's key checks."""
    app = FastAPI(title="Canopy / GEECS", description="GEECS devices over Hare.")

    @app.get("/v1/devices/{name}", dependencies=[Security(read_access)])
    async def get_device(name: str) -> JSONResponse:
        if (device := devices.get(name)) is None:
            return _error(404, "not_found", f"no device {name!r}")
        roles = device.family.roles
        entries = [(r, a) for a in device.axes or (None,) for r in roles.values()]
        counters = sorted({(r.fresh_by, a) for r, a in entries if r.fresh_by}, key=str)
        counter_pvs = [device.pv(roles[c], a) for c, a in counters]
        pvs = [device.pv(r, a) for r, a in entries]
        first, values = await asyncio.gather(_read(hare, counter_pvs), _read(hare, pvs))
        if counters:
            await asyncio.sleep(window_s)
        second = await _read(hare, counter_pvs)
        advanced = {k: _advanced(f, s) for k, f, s in zip(counters, first, second)}
        by_role = {(r.name, a): v for (r, a), v in zip(entries, values)}

        def judged(role: Role, axis: str | None, value: Wire) -> Wire:
            flag = by_role.get((role.computed_by, axis))
            judge = verdict(role, value, flag, advanced.get((role.fresh_by, axis)))
            return {**value, "role": role.name, "axis": axis, "verdict": judge}

        body = {"device": name, "family": device.family.template, "windowS": window_s}
        body["roles"] = [judged(r, a, v) for (r, a), v in zip(entries, values)]
        return JSONResponse(body)

    @app.post("/v1/devices/{name}/put")
    async def put(name: str, body: PutBody, key=Security(write_access)) -> JSONResponse:
        if (device := devices.get(name)) is None:
            return _error(404, "not_found", f"no device {name!r}")
        if not device.axis_ok(body.axis):
            return _error(
                400, "bad_request", f"axis must be one of {list(device.axes)}"
            )
        row = {
            "device": name,
            "role": body.role,
            "axis": body.axis,
            "value": body.value,
        }
        row |= {"operator": body.operator, "key_name": getattr(key, "name", None)}
        role = device.family.roles.get(body.role)
        if role is None or body.role not in device.allowed_roles():
            detail = f"{body.role!r} is in none of {name}'s allowed actions {list(device.allow)}"
            return _answer(journal, 403, row | {"outcome": "denied"}, "denied", detail)
        try:
            pv = device.pv(role, body.axis, write=True)
            await hare.put(pv, body.value, wait=True, timeout=put_timeout_s)
        except HareRefused as exc:
            return _answer(
                journal, exc.status, row | {"outcome": "refused"}, exc.code, exc.detail
            )
        return _answer(journal, 200, row | {"outcome": "accepted"}, None, None)

    return app
