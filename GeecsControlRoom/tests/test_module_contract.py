"""The Control Room against GeecsCanopy's own module, in process: the route shapes the pages read.

The module is built the way its own suite builds it (``geecs_canopy.testing``:
a ``FakeHare``, the device documents, the journal in a temp file), its app
mounted at ``/geecs`` on an ``httpx.ASGITransport`` that ``ModuleClient`` is
pointed at. A route or field rename in GeecsCanopy fails this suite in the
same PR. geecs-canopy is a dev-only dependency: ``geecs_control_room`` never
imports it (the ``control-room-edges`` contract).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from geecs_canopy.testing import EXPERIMENT, live, make_env

from geecs_control_room.app import create_app
from geecs_control_room.module_client import ModuleClient
from geecs_control_room.settings import Settings

#: Every field screen.js reads from a role of /api/devices/{name}.
ROLE_FIELDS = {"role", "axis", "pv", "read", "writable", "verdict"}
#: Every field home.js reads from a journal row (and home.py's ``device``).
JOURNAL_FIELDS = {"device", "time", "outcome", "role", "axis", "value", "operator", "canopy_code"}  # fmt: skip


class _Failing(httpx.AsyncBaseTransport):
    """The module's transport, failing every request whose path ends in one of ``broken``."""

    def __init__(self, inner: httpx.AsyncBaseTransport, broken: set[str]) -> None:
        self.inner, self.broken = inner, broken

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if any(request.url.path.endswith(b) for b in self.broken):
            raise httpx.ConnectError("refused", request=request)
        return await self.inner.handle_async_request(request)


@pytest.fixture
async def room(tmp_path: Path):
    env = make_env(tmp_path)
    canopy = FastAPI()
    canopy.mount("/geecs", env.app)
    broken: set[str] = set()
    module = ModuleClient(
        "http://canopy.test",
        "read-key",
        transport=_Failing(httpx.ASGITransport(app=canopy), broken),
    )
    settings = Settings(experiment=EXPERIMENT, configs_root=tmp_path, canopy_url="http://canopy.test", read_key="read-key")  # fmt: skip
    app = create_app(settings, module)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://room.test") as client:  # fmt: skip
        yield SimpleNamespace(env=env, client=client, broken=broken)


async def put(room, device: str, **body):
    r = await room.client.post(f"/api/devices/{device}/put", json={"operator": "ann", **body})  # fmt: skip
    return r.status_code, r.json()


async def test_a_device_carries_what_the_screen_reads(room) -> None:
    first = await room.client.get("/api/devices/ps1")
    assert first.status_code == 200 and {"state", "heartbeat"} <= set(first.json())  # fmt: skip
    roles = {r["role"]: r for r in first.json()["roles"]}
    assert all(ROLE_FIELDS <= set(r) for r in roles.values())
    await asyncio.sleep(0)  # the module subscribes as a task
    room.env.hare.on_frame({roles["current"]["pv"]: live(1.0)})
    again = {r["role"]: r for r in (await room.client.get("/api/devices/ps1")).json()["roles"]}  # fmt: skip
    assert again["current"]["verdict"] == "live"
    assert again["current_sp"]["writable"] is True and again["current"]["read"] == "monitor"  # fmt: skip


async def test_an_accepted_put_names_its_journal_row(room) -> None:
    status, body = await put(room, "Mot1", role="homf", axis="m1", value=1, confirm=True)  # fmt: skip
    assert (status, body["outcome"]) == (200, "accepted")
    assert isinstance(body["journalId"], int) and "code" not in body


@pytest.mark.parametrize(
    "device, body, status, code",
    [
        ("ps1", {"role": "reset", "value": 1, "operator": "bob"}, 403, "not_permitted"),
        ("Mot1", {"role": "homf", "axis": "m1", "value": 1}, 428, "confirm_required"),
    ],
)  # fmt: skip
async def test_a_refusal_carries_outcome_code_and_journal_row(room, device, body, status, code) -> None:  # fmt: skip
    got, answer = await put(room, device, **body)
    assert (got, answer["code"], answer["outcome"]) == (status, code, "denied")
    assert isinstance(answer["journalId"], int)


async def test_out_of_bounds_carries_its_detail(room) -> None:
    roles = (await room.client.get("/api/devices/ps1")).json()["roles"]
    sp = next(r["pv"] for r in roles if r["role"] == "current_sp")
    room.env.hare.meta[sp] = {"controlLow": 0.0, "controlHigh": 10.0}
    status, body = await put(room, "ps1", role="current_sp", value=11.0)
    assert (status, body["code"]) == (422, "out_of_bounds")
    assert body["outcome"] and body["detail"]


async def test_the_home_lists_come_from_the_roster_and_the_journal(room) -> None:
    await put(room, "Mot1", role="homf", axis="m1", value=1, confirm=True)
    roles = (await room.client.get("/api/devices/ps1")).json()["roles"]
    await asyncio.sleep(0)
    current = next(r["pv"] for r in roles if r["role"] == "current")
    room.env.hare.on_frame({current: live(1.0, severity="MAJOR", alarmStatus=3)})
    body = (await room.client.get("/api/home/devices")).json()
    assert body["ok"] and body["journal"] and body["unread"] == []
    assert {d["name"] for d in body["devices"]} == {"Cam1", "Mot1", "ps1"}
    assert {d["name"] for d in body["not_live"]} == {
        "Cam1",
        "Mot1",
        "ps1",
    }  # no heartbeat yet
    assert [d["name"] for d in body["in_alarm"]] == ["ps1"]
    last = next(d["last"] for d in body["not_live"] if d["name"] == "Mot1")
    assert JOURNAL_FIELDS <= set(last) and last["outcome"] == "accepted"


async def test_a_live_device_leaves_not_live_and_its_family_reaches_the_pickers(room) -> None:  # fmt: skip
    roles = (await room.client.get("/api/devices/ps1")).json()["roles"]
    await asyncio.sleep(0)
    room.env.hare.on_frame({r["pv"]: live(0) for r in roles if r["read"] == "monitor"})
    body = (await room.client.get("/api/home/devices")).json()
    families = {d["name"]: d["family"] for d in body["devices"]}
    assert families == {"Cam1": "ad_camera_chain", "Mot1": "motor_record", "ps1": "easydriver"}  # fmt: skip
    assert {d["name"] for d in body["not_live"]} == {"Cam1", "Mot1"}


async def test_a_device_route_that_fails_is_unread(room) -> None:
    room.broken.add("/devices/Cam1")
    body = (await room.client.get("/api/home/devices")).json()
    assert body["ok"] and body["unread"] == [{"name": "Cam1"}]
    assert (await room.client.get("/api/devices/Cam1")).json()["code"] == "module_unreachable"  # fmt: skip
