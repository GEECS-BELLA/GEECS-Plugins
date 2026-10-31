"""The frame proxy and the arrays route over fake transports: no CA library, no IOC, the module stubbed."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from geecs_control_room.images import ARRAYS_EVERY_S, FRAME_EVERY_S, ROUTE_TTL_S, Images, router, with_ca_host  # fmt: skip
from geecs_control_room.module_client import ModuleClient

IMAGE_PV = "bench:cam:MJPG1:JPG_URL_RBV"
ROLES = [
    {"role": "image", "pv": IMAGE_PV, "read": "once"},
    {
        "role": "profile_x",
        "pv": "bench:cam:STAT:ProfileAverageX_RBV",
        "read": "on_request",
    },
    {"role": "histogram", "pv": "bench:cam:STAT:Histogram_RBV", "read": "on_request"},
    {"role": "exposure", "pv": "bench:cam:cam1:AcquireTime_RBV", "read": "monitor"},
]


class FakeCa:
    """The three reads, recorded; ``fail`` makes the next read raise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.fail = False
        self.gate: asyncio.Event | None = None

    async def text(self, pv: str) -> str:
        self.calls.append(("text", pv))
        return "http://iochost.lab:8081/CAM1.MJPG.jpg"

    async def host(self, pv: str) -> str:
        self.calls.append(("host", pv))
        return "10.0.0.5:5064"

    async def arrays(self, pvs: list[str]) -> list[list[float]]:
        self.calls.append(("arrays", tuple(pvs)))
        if self.gate:
            await self.gate.wait()
        if self.fail:
            raise TimeoutError("no answer from the IOC")
        return [[1.0, float("nan")], [3.0, float("inf"), 4.0]]


class Env:
    def __init__(self, roles: list[dict[str, Any]] | None = ROLES, status: int = 200) -> None:  # fmt: skip
        self.ca, self.urls, self.reads, self.clock = FakeCa(), [], 0, [0.0]
        self.fetch_fails, self.gate = False, None
        self.status, self.roles = status, roles

        def answer(request: httpx.Request) -> httpx.Response:
            self.reads += 1
            if self.status == 0:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(
                self.status, json={"name": "Cam", "roles": self.roles}
            )

        self.module = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(answer))  # fmt: skip
        self.images = Images(self.ca, self.fetch, clock=lambda: self.clock[0])

    async def fetch(self, url: str) -> bytes:
        self.urls.append(url)
        if self.gate:
            await self.gate.wait()
        if self.fetch_fails:
            raise httpx.ConnectError("refused")
        return b"\xff\xd8jpeg" + str(len(self.urls)).encode()

    def client(self) -> httpx.AsyncClient:
        from fastapi import FastAPI

        app = FastAPI()
        app.state.module = self.module
        app.include_router(router(self.images))
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://room.test")  # fmt: skip


def test_the_url_is_rehosted_to_the_ca_address_keeping_its_port() -> None:
    assert with_ca_host("http://iochost.lab:8081/CAM1.MJPG.jpg", "10.0.0.5:5064") == "http://10.0.0.5:8081/CAM1.MJPG.jpg"  # fmt: skip
    assert with_ca_host("http://iochost.lab/x.jpg", "localhost:5066") == "http://localhost/x.jpg"  # fmt: skip


async def test_the_frame_is_the_jpeg_at_the_url_the_module_names() -> None:
    env = Env()
    async with env.client() as c:
        r = await c.get("/api/devices/Cam/frame.jpg")
    assert r.status_code == 200 and r.content.startswith(b"\xff\xd8")
    assert r.headers["content-type"] == "image/jpeg" and r.headers["cache-control"] == "no-store"  # fmt: skip
    assert r.headers["x-frame-tag"]
    assert env.ca.calls == [("text", IMAGE_PV), ("host", IMAGE_PV)]
    assert env.urls == ["http://10.0.0.5:8081/CAM1.MJPG.jpg"]


async def test_viewers_share_one_fetch_in_flight_and_one_frame_within_the_floor() -> None:  # fmt: skip
    env = Env()
    env.gate = asyncio.Event()
    async with env.client() as c:
        pending = [asyncio.create_task(c.get("/api/devices/Cam/frame.jpg")) for _ in range(5)]  # fmt: skip
        await asyncio.sleep(0.05)
        env.gate.set()
        tags = {(await t).headers["x-frame-tag"] for t in pending}
        assert len(env.urls) == 1 and len(tags) == 1
        env.clock[0] += FRAME_EVERY_S / 2
        await c.get("/api/devices/Cam/frame.jpg")
        assert len(env.urls) == 1  # inside the floor: the held frame
        env.clock[0] += FRAME_EVERY_S
        again = await c.get("/api/devices/Cam/frame.jpg")
    assert len(env.urls) == 2 and again.headers["x-frame-tag"] not in tags
    assert env.reads == 1  # the module's route, read once for all of it


async def test_a_failed_fetch_is_a_503_with_words_and_the_url_is_read_again() -> None:
    env = Env()
    env.fetch_fails = True
    async with env.client() as c:
        r = await c.get("/api/devices/Cam/frame.jpg")
        assert r.status_code == 503 and "no frame from the IOC" in r.json()["detail"]
        env.fetch_fails = False
        assert (await c.get("/api/devices/Cam/frame.jpg")).status_code == 200
    assert [k for k, _ in env.ca.calls] == ["text", "host", "text", "host"]


@pytest.mark.parametrize(
    ("roles", "status", "code", "words"),
    [
        ([r for r in ROLES if r["role"] != "image"], 200, 404, "names no image role"),
        (ROLES, 404, 404, "not on the module's roster"),
        (ROLES, 503, 503, "answered 503"),
        (ROLES, 0, 503, "did not answer"),
    ],
)
async def test_a_frame_the_module_does_not_name_is_never_derived(roles, status, code, words) -> None:  # fmt: skip
    env = Env(roles, status)
    async with env.client() as c:
        r = await c.get("/api/devices/Cam/frame.jpg")
    assert r.status_code == code and words in r.json()["detail"]
    assert env.ca.calls == [] and env.urls == []


async def test_the_arrays_are_the_on_request_roles_by_name_finite_only() -> None:
    env = Env()
    async with env.client() as c:
        body = (await c.get("/api/devices/Cam/arrays")).json()
    assert body["arrays"] == {"profile_x": [1.0, None], "histogram": [3.0, None, 4.0]}
    assert body["pvs"]["profile_x"] == "bench:cam:STAT:ProfileAverageX_RBV"
    assert env.ca.calls == [("arrays", ("bench:cam:STAT:ProfileAverageX_RBV", "bench:cam:STAT:Histogram_RBV"))]  # fmt: skip


async def test_the_arrays_are_shared_and_read_at_most_every_half_second() -> None:
    env = Env()
    env.ca.gate = asyncio.Event()
    async with env.client() as c:
        pending = [
            asyncio.create_task(c.get("/api/devices/Cam/arrays")) for _ in range(4)
        ]
        await asyncio.sleep(0.05)
        env.ca.gate.set()
        assert [(await t).status_code for t in pending] == [200] * 4
        env.clock[0] += ARRAYS_EVERY_S / 2
        await c.get("/api/devices/Cam/arrays")
        assert len(env.ca.calls) == 1
        env.clock[0] += ARRAYS_EVERY_S
        await c.get("/api/devices/Cam/arrays")
    assert len(env.ca.calls) == 2


async def test_unreadable_arrays_are_a_503_and_none_named_a_404() -> None:
    env = Env()
    env.ca.fail = True
    async with env.client() as c:
        r = await c.get("/api/devices/Cam/arrays")
    assert r.status_code == 503 and "could not be read" in r.json()["detail"]
    bare = Env([ROLES[0], ROLES[3]])
    async with bare.client() as c:
        r = await c.get("/api/devices/Cam/arrays")
    assert r.status_code == 404 and "no on-request array" in r.json()["detail"]
    assert bare.ca.calls == []


async def test_the_module_route_is_reread_after_its_ttl() -> None:
    env = Env()
    async with env.client() as c:
        await c.get("/api/devices/Cam/arrays")
        env.clock[0] += ROUTE_TTL_S + 1
        await c.get("/api/devices/Cam/arrays")
    assert env.reads == 2


async def test_a_silent_module_serves_the_last_good_route_and_is_asked_once_per_ttl() -> None:  # fmt: skip
    env = Env()
    async with env.client() as c:
        assert (await c.get("/api/devices/Cam/frame.jpg")).status_code == 200
        env.status = 0  # the module stops answering
        env.clock[0] += ROUTE_TTL_S + 1
        assert (await c.get("/api/devices/Cam/frame.jpg")).status_code == 200
        assert (await c.get("/api/devices/Cam/arrays")).status_code == 200
        assert env.reads == 2  # the failed refresh, then the held roles for another TTL
        env.status = 503
        env.clock[0] += ROUTE_TTL_S + 1
        assert (await c.get("/api/devices/Cam/frame.jpg")).status_code == 200
        env.status = 404  # off the roster: forgotten, never served from memory
        env.clock[0] += ROUTE_TTL_S + 1
        r = await c.get("/api/devices/Cam/frame.jpg")
    assert r.status_code == 404 and "not on the module's roster" in r.json()["detail"]


async def test_waiters_share_one_route_read_even_when_it_fails() -> None:
    env = Env(status=0)
    gate = asyncio.Event()
    device = env.module.device

    async def slow(name: str) -> Any:
        await gate.wait()
        return await device(name)

    env.module.device = slow  # type: ignore[method-assign]
    async with env.client() as c:
        pending = [asyncio.create_task(c.get("/api/devices/Cam/frame.jpg")) for _ in range(6)]  # fmt: skip
        await asyncio.sleep(0.05)
        gate.set()
        codes = [(await t).status_code for t in pending]
    assert codes == [503] * 6 and env.reads == 1
