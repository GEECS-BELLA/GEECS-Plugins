"""The frame proxy and the arrays route over fake transports: no CA library, no IOC, the module stubbed."""

from __future__ import annotations

import asyncio
import gzip
from typing import Any

import httpx
import numpy as np
import pytest

from geecs_control_room.images import ARRAYS_EVERY_S, FRAME_EVERY_S, JPEG_REUSE_S, RAW_HOLD_S, ROUTE_TTL_S, AiocaCa, Images, router, with_ca_host  # fmt: skip
from geecs_control_room.module_client import ModuleClient

IMAGE_PV = "bench:cam:MJPG1:JPG_URL_RBV"
IMAGE_COUNTER = "bench:cam:MJPG1:ArrayCounter_RBV"
ROLES = [
    {"role": "image", "pv": IMAGE_PV, "read": "once", "type": "image"},
    {
        "role": "profile_x",
        "pv": "bench:cam:STAT:ProfileAverageX_RBV",
        "read": "on_request",
        "type": "array",
    },
    {
        "role": "histogram",
        "pv": "bench:cam:STAT:Histogram_RBV",
        "read": "on_request",
        "type": "array",
    },  # fmt: skip
    {
        "role": "exposure",
        "pv": "bench:cam:cam1:AcquireTime_RBV",
        "read": "monitor",
        "type": "setpoint",
    },  # fmt: skip
    # the raw frame's source: never JSON, and its on-request scalars are not arrays of the page's
    {
        "role": "raw_frame",
        "pv": "bench:cam:ARR:ArrayData",
        "read": "never",
        "type": "array",
    },  # fmt: skip
    {
        "role": "raw_width",
        "pv": "bench:cam:ARR:ArraySize0_RBV",
        "read": "on_request",
        "type": "readback",
    },  # fmt: skip
    {
        "role": "raw_height",
        "pv": "bench:cam:ARR:ArraySize1_RBV",
        "read": "on_request",
        "type": "readback",
    },  # fmt: skip
    {
        "role": "raw_counter",
        "pv": "bench:cam:ARR:ArrayCounter_RBV",
        "read": "on_request",
        "type": "counter",
    },  # fmt: skip
    {
        "role": "full_scale",
        "pv": "bench:cam:FullScale",
        "read": "once",
        "type": "readback",
    },  # fmt: skip
    {
        "role": "image_counter",
        "pv": IMAGE_COUNTER,
        "read": "monitor",
        "type": "counter",
    },  # fmt: skip
]
RAW = ("bench:cam:ARR:ArrayData", "bench:cam:ARR:ArraySize0_RBV", "bench:cam:ARR:ArraySize1_RBV")  # fmt: skip
COUNTER, FULL_SCALE = "bench:cam:ARR:ArrayCounter_RBV", "bench:cam:FullScale"


class FakeCa:
    """The reads, recorded; ``fail`` makes the arrays read raise, ``frame_fails`` / ``counter_fails`` theirs.

    The raw frame: ``raw`` is what CA delivers asked as ``DBR_SHORT`` (int16
    values, a count above 32767 negative; width, height),
    ``counters`` the successive ``ArrayCounter_RBV`` reads (the last repeats),
    ``full_scale`` the ``FullScale`` read (``None``: unreadable).
    The JPEG: ``image_counters`` the successive ``MJPG1:ArrayCounter_RBV``
    reads (the last repeats), ``image_counter_fails`` makes them raise.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.fail = self.frame_fails = self.counter_fails = False
        self.gate: asyncio.Event | None = None
        self.raw: tuple[np.ndarray, int, int] = (np.arange(12, dtype=np.int16), 4, 3)
        self.counters: list[float] = [7.0]
        self.image_counters: list[float] = [3.0]
        self.image_counter_fails = False
        self.full_scale: float | None = 4095.0
        self.scale_gate: asyncio.Event | None = None

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

    async def scalar(self, pv: str) -> float:
        self.calls.append(("scalar", pv))
        if "MJPG1:" in pv:  # the JPEG's counter, the module's or this file's
            if self.image_counter_fails:
                raise TimeoutError("no answer from the IOC")
            return self.image_counters.pop(0) if len(self.image_counters) > 1 else self.image_counters[0]  # fmt: skip
        if pv.endswith("ArrayCounter_RBV"):
            if self.counter_fails:
                raise TimeoutError("no answer from the IOC")
            return self.counters.pop(0) if len(self.counters) > 1 else self.counters[0]
        if self.scale_gate:
            await self.scale_gate.wait()
        if self.full_scale is None:
            raise TimeoutError("no answer from the IOC")
        return self.full_scale

    async def frame(self, frame: str, width: str, height: str) -> tuple[np.ndarray, int, int]:  # fmt: skip
        self.calls.append(("frame", (frame, width, height)))
        if self.gate:
            await self.gate.wait()
        if self.frame_fails:
            raise TimeoutError("no answer from the IOC")
        return self.raw

    def kinds(self) -> list[str]:
        """The reads by kind, the full scale's apart: ``counter`` / ``frame`` / ``scale`` / ``image_counter``."""
        names = {FULL_SCALE: "scale", COUNTER: "counter", IMAGE_COUNTER: "image_counter"}  # fmt: skip
        return [names.get(a, k) if isinstance(a, str) else k for k, a in self.calls]


class Env:
    def __init__(self, roles: list[dict[str, Any]] | None = ROLES, status: int = 200) -> None:  # fmt: skip
        self.ca, self.urls, self.reads, self.clock = FakeCa(), [], 0, [0.0]
        self.fetch_fails, self.gate = False, None
        self.body: bytes | None = (
            None  # set: every fetch returns these bytes (else each fetch's own)
        )
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
        return self.body if self.body is not None else b"\xff\xd8jpeg" + str(len(self.urls)).encode()  # fmt: skip

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
    tag = r.headers["x-frame-tag"]  # the bytes' checksum
    assert (
        tag and r.headers["etag"] == f'"3;{tag}"'
    )  # the counter read before the fetch, and the bytes
    assert env.ca.calls == [("scalar", IMAGE_COUNTER), ("text", IMAGE_PV), ("host", IMAGE_PV)]  # fmt: skip
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
        env.ca.image_counters = [4.0]  # the plugin moved
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
    assert [k for k, _ in env.ca.calls if k != "scalar"] == ["text", "host", "text", "host"]  # fmt: skip


JPG_URL = "/api/devices/Cam/frame.jpg"
#: Past the floor (a float sum of 0.1 steps can land a hair inside it).
STEP = FRAME_EVERY_S * 1.01


async def test_an_unmoved_image_counter_serves_the_held_jpeg_without_a_fetch() -> None:
    env = Env()
    async with env.client() as c:
        first = await c.get(JPG_URL)
        for _ in range(
            9
        ):  # a 10 Hz poller on a camera that has not moved, inside JPEG_REUSE_S
            env.clock[0] += STEP
            r = await c.get(JPG_URL)
            assert (r.status_code, r.content) == (200, first.content)
            assert (r.headers["etag"], r.headers["x-frame-tag"]) == (first.headers["etag"], first.headers["x-frame-tag"])  # fmt: skip
        assert len(env.urls) == 1  # one upstream fetch for ten requests
        assert env.ca.kinds() == ["image_counter", "text", "host"] + ["image_counter"] * 9  # fmt: skip
        env.clock[0] += FRAME_EVERY_S / 2
        await c.get(JPG_URL)  # inside the floor: not even a probe
        assert len(env.ca.calls) == 12
        env.ca.image_counters = [4.0]
        env.clock[0] += STEP
        moved = await c.get(JPG_URL)
    assert len(env.urls) == 2 and moved.headers["etag"].startswith('"4;')
    assert moved.headers["x-frame-tag"] != first.headers["x-frame-tag"]


async def test_an_unmoved_counter_is_trusted_for_jpeg_reuse_s_then_fetched_again() -> None:  # fmt: skip
    env = Env()
    env.body = b"\xff\xd8the same frame"
    async with env.client() as c:
        first = await c.get(JPG_URL)
        env.clock[0] += JPEG_REUSE_S
        same = await c.get(JPG_URL, headers={"If-None-Match": first.headers["etag"]})
        assert (
            len(env.urls) == 2 and same.status_code == 304
        )  # fetched again, the same bytes: nothing new
        env.body = b"\xff\xd8the frame its counter announced"  # a counter posted before its JPEG
        env.clock[0] += JPEG_REUSE_S
        fixed = await c.get(JPG_URL, headers={"If-None-Match": first.headers["etag"]})
    # the same counter, other bytes: a 200 carrying them, never a 304 that hides them
    assert len(env.urls) == 3 and fixed.status_code == 200 and fixed.content == env.body
    assert fixed.headers["etag"].startswith('"3;') and fixed.headers["etag"] != first.headers["etag"]  # fmt: skip


async def test_a_matching_if_none_match_on_the_jpeg_is_a_304_with_its_tags() -> None:
    env = Env()
    async with env.client() as c:
        first = await c.get(JPG_URL)
        etag = first.headers["etag"]
        for asked in (etag, f"W/{etag}", f'"1", {etag}', "*"):
            env.clock[0] += STEP
            same = await c.get(JPG_URL, headers={"If-None-Match": asked})
            assert (same.status_code, same.content) == (304, b"")
            assert (same.headers["etag"], same.headers["x-frame-tag"]) == (etag, first.headers["x-frame-tag"])  # fmt: skip
        other = await c.get(
            JPG_URL, headers={"If-None-Match": '"3"'}
        )  # the counter alone is not the ETag
    assert other.status_code == 200 and other.content == first.content
    assert len(env.urls) == 1


async def test_the_jpeg_is_tagged_by_the_counter_read_before_its_fetch() -> None:
    env = Env()
    env.ca.image_counters = [3.0, 4.0]  # a frame lands while the first JPEG is fetched
    async with env.client() as c:
        first = await c.get(JPG_URL)
        env.clock[0] += STEP
        second = await c.get(JPG_URL)
    # at least as new as counter 3; counter 4 is not that JPEG's, so the next probe fetches again
    assert first.headers["etag"].startswith('"3;') and second.headers["etag"].startswith('"4;')  # fmt: skip
    assert len(env.urls) == 2 and env.ca.kinds().count("image_counter") == 2


@pytest.mark.parametrize("why", ["unreadable", "unnamed"])
async def test_without_a_counter_the_jpeg_is_fetched_as_before_and_untagged(why) -> None:  # fmt: skip
    env = Env([r for r in ROLES if r["role"] != "image_counter"] if why == "unnamed" else ROLES)  # fmt: skip
    env.ca.image_counter_fails = why == "unreadable"
    async with env.client() as c:
        answers = []
        for _ in range(3):
            answers.append(await c.get(JPG_URL, headers={"If-None-Match": "*"}))
            env.clock[0] += STEP
    assert [r.status_code for r in answers] == [
        200
    ] * 3  # the image never fails on its counter
    assert all("etag" not in r.headers and r.headers["x-frame-tag"] for r in answers)
    assert len(env.urls) == 3  # one fetch per floor, as before the hold
    # an unreadable counter is asked once per ROUTE_TTL_S, never on every frame (a silent one costs the CA timeout)
    assert env.ca.kinds().count("image_counter") == (1 if why == "unreadable" else 0)


async def test_an_unreadable_counter_is_asked_again_after_the_ttl_and_the_hold_resumes() -> None:  # fmt: skip
    env = Env()
    env.ca.image_counter_fails = True
    async with env.client() as c:
        assert "etag" not in (await c.get(JPG_URL)).headers
        env.ca.image_counter_fails = False
        env.clock[0] += STEP
        assert "etag" not in (await c.get(JPG_URL)).headers  # inside the TTL: not asked
        env.clock[0] += ROUTE_TTL_S
        back = await c.get(JPG_URL)
    assert back.headers["etag"].startswith('"3;') and env.ca.kinds().count("image_counter") == 2  # fmt: skip


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
    assert (
        "raw_frame" not in body["pvs"] and "raw_width" not in body["pvs"]
    )  # never JSON; a scalar is no array
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


@pytest.mark.parametrize("refusal", [401, 403])
async def test_a_refused_key_forgets_the_held_route(refusal) -> None:
    env = Env()
    async with env.client() as c:
        assert (await c.get("/api/devices/Cam/frame.jpg")).status_code == 200
        env.status = refusal
        env.clock[0] += ROUTE_TTL_S + 1
        r = await c.get("/api/devices/Cam/frame.jpg")
        assert r.status_code == 503 and f"answered {refusal}" in r.json()["detail"]
        env.status = 0  # then silent: nothing held to serve
        env.clock[0] += ROUTE_TTL_S + 1
        assert (await c.get("/api/devices/Cam/frame.jpg")).status_code == 503


RAW_URL = "/api/devices/Cam/frame.raw"


async def _raw(c: httpx.AsyncClient, **headers: str) -> tuple[httpx.Response, bytes]:
    """The response and its body as sent (httpx would decode the gzip): the bytes on the wire."""
    async with c.stream("GET", RAW_URL, headers=headers) as r:
        return r, b"".join([chunk async for chunk in r.aiter_raw()])


@pytest.mark.parametrize("width, height", [(2592, 2048), (600, 600)])
async def test_the_raw_frame_is_uint16_little_endian_row_major_gzip_framed(width, height) -> None:  # fmt: skip
    env = Env()
    frame = np.random.default_rng(1).integers(0, 4096, size=width * height, dtype=np.int16)  # fmt: skip
    env.ca.raw = (frame, width, height)
    async with env.client() as c:
        r, wire = await _raw(c)
    assert (
        r.status_code == 200 and wire[:2] == b"\x1f\x8b"
    )  # gzip framing, never bare zlib
    assert int(r.headers["content-length"]) == len(wire)
    assert gzip.decompress(wire) == frame.astype("<u2").tobytes()
    assert (r.headers["x-frame-width"], r.headers["x-frame-height"]) == (str(width), str(height))  # fmt: skip


async def test_the_raw_frame_headers() -> None:
    env = Env()
    async with env.client() as c:
        r, _ = await _raw(c)
    h = r.headers
    assert h["content-type"] == "application/octet-stream" and h["content-encoding"] == "gzip"  # fmt: skip
    assert h["cache-control"] == "no-store" and h["x-frame-tag"] == '"7"'
    assert h["etag"] == '"7;4095"'  # the representation: the counter and the full scale
    assert h["x-frame-full-scale"] == "4095" and "x-frame-transform" not in h
    assert env.ca.kinds() == ["counter", "frame", "counter", "scale"]
    assert ("frame", RAW) in env.ca.calls  # the frame and its geometry in one batch


async def test_an_unreadable_full_scale_is_omitted_and_asked_again_after_the_ttl() -> None:  # fmt: skip
    env = Env()
    env.ca.full_scale = None
    async with env.client() as c:
        r, _ = await _raw(c)
        assert r.status_code == 200 and "x-frame-full-scale" not in r.headers
        env.ca.counters, env.ca.full_scale = [8.0], 4095.0
        env.clock[0] += RAW_HOLD_S
        await _raw(c)  # inside the TTL: not asked again
        assert env.ca.kinds().count("scale") == 1
        env.ca.counters = [9.0]
        env.clock[0] += ROUTE_TTL_S
        r, _ = await _raw(c)
    assert r.headers["x-frame-full-scale"] == "4095" and env.ca.kinds().count("scale") == 2  # fmt: skip


async def test_a_slow_full_scale_never_holds_the_frame() -> None:
    env = Env()
    env.ca.scale_gate = asyncio.Event()  # FullScale silent: its read would time out
    async with env.client() as c:
        first, wire = await asyncio.wait_for(_raw(c), 1.0)
        assert first.status_code == 200 and "x-frame-full-scale" not in first.headers
        assert first.headers["etag"] == '"7"'  # the scale unknown: the counter alone
        env.ca.scale_gate.set()
        await asyncio.sleep(0.01)
        env.clock[0] += RAW_HOLD_S
        # the camera has not moved; the full scale has landed: the page's conditional ask is a 200 carrying it
        held, body = await _raw(c, **{"If-None-Match": first.headers["etag"]})
        assert held.status_code == 200 and held.headers["x-frame-full-scale"] == "4095"
        assert held.headers["x-frame-tag"] == first.headers["x-frame-tag"] and body == wire  # fmt: skip
        assert held.headers["etag"] == '"7;4095"'
        env.clock[0] += RAW_HOLD_S
        same, _ = await _raw(c, **{"If-None-Match": held.headers["etag"]})
    assert same.status_code == 304 and env.ca.kinds().count("frame") == 1


async def test_an_unmoved_counter_serves_the_held_frame_without_an_array_read() -> None:  # fmt: skip
    env = Env()
    async with env.client() as c:
        first, wire = await _raw(c)
        env.clock[0] += RAW_HOLD_S / 2
        await _raw(c)  # inside the hold: no read at all
        assert env.ca.kinds() == ["counter", "frame", "counter", "scale"]
        env.clock[0] += RAW_HOLD_S
        again, same = await _raw(c)  # the probe: one scalar read, the held frame
        assert env.ca.kinds()[4:] == ["counter"] and same == wire
        assert again.headers["etag"] == first.headers["etag"]
        env.ca.counters = [8.0]
        env.clock[0] += RAW_HOLD_S
        moved, _ = await _raw(c)  # the counter advanced: one array read
    assert env.ca.kinds()[5:] == ["counter", "frame", "counter"]
    assert moved.headers["etag"] == '"8;4095"'


async def test_a_counter_moving_mid_batch_reads_once_more_tagged_by_the_second_counter() -> None:  # fmt: skip
    env = Env()
    env.ca.counters = [7.0, 8.0]
    async with env.client() as c:
        r, _ = await _raw(c)
    assert env.ca.kinds() == ["counter", "frame", "counter", "frame", "scale"]
    assert r.status_code == 200 and r.headers["x-frame-tag"] == '"8"'


async def test_a_matching_if_none_match_is_a_304_with_the_same_tag() -> None:
    env = Env()
    async with env.client() as c:
        r, _ = await _raw(c)
        etag = r.headers["etag"]
        for asked in (etag, f"W/{etag}", f'"1", {etag}', "*"):
            same, body = await _raw(c, **{"If-None-Match": asked})
            assert same.status_code == 304 and body == b""
            assert (same.headers["etag"], same.headers["x-frame-tag"]) == (etag, '"7"')
        other, body = await _raw(c, **{"If-None-Match": '"6"'})
    assert other.status_code == 200 and gzip.decompress(body)


@pytest.mark.parametrize("gone", ["raw_frame", "raw_counter"])
async def test_no_raw_frame_role_is_a_404_and_nothing_is_read(gone) -> None:
    env = Env([r for r in ROLES if r["role"] != gone])
    async with env.client() as c:
        r = await c.get(RAW_URL)
    assert r.status_code == 404 and f"names no {gone} role" in r.json()["detail"]
    assert env.ca.calls == []  # never derived


@pytest.mark.parametrize(
    "fault, words",
    [
        ("counter", f"the frame counter ({COUNTER}) could not be read"),
        (
            "frame",
            f"the raw frame ({RAW[0]}) could not be read: no answer from the IOC",
        ),  # fmt: skip
        ("geometry", "geometry disagrees: 12 values for 4×4"),
        (
            "colour",
            "geometry disagrees: 36 values for 3×4",
        ),  # an RGB frame: ArraySize0 = 3
        ("empty", "geometry disagrees: 0 values for 0×0"),
        ("wide", "the raw frame arrived as int32, not 16-bit"),
    ],
)
async def test_a_silent_source_is_a_503_with_words(fault, words) -> None:
    env = Env()
    if fault == "counter":
        env.ca.counter_fails = True
    elif fault == "frame":
        env.ca.frame_fails = True
    elif fault == "geometry":  # a camera-ROI change between the array and its sizes
        env.ca.raw = (np.arange(12, dtype=np.int16), 4, 4)
    elif fault == "colour":
        env.ca.raw = (np.arange(36, dtype=np.int16), 3, 4)
    elif (
        fault == "wide"
    ):  # the native DBR_LONG read: twice the bytes, never viewed as 16-bit
        env.ca.raw = (np.arange(12, dtype=np.int32), 4, 3)
    else:
        env.ca.raw = (np.zeros(0, dtype=np.int16), 0, 0)
    async with env.client() as c:
        r = await c.get(RAW_URL)
    assert r.status_code == 503 and r.json()["detail"] == words


async def test_viewers_share_one_raw_read_in_flight() -> None:
    env = Env()
    env.ca.gate = asyncio.Event()
    async with env.client() as c:
        pending = [asyncio.create_task(_raw(c)) for _ in range(20)]
        await asyncio.sleep(0.05)
        env.ca.gate.set()
        answers = [await t for t in pending]
    assert {r.status_code for r, _ in answers} == {200} and len({w for _, w in answers}) == 1  # fmt: skip
    assert sorted(env.ca.kinds()) == [
        "counter",
        "counter",
        "frame",
        "scale",
    ]  # one read for all twenty


async def test_values_above_32767_survive_the_cast_exactly() -> None:
    env = Env()
    values = np.array([0, 1, 4095, 32767, 32768, 40000, 65534, 65535], dtype=np.uint16)
    sent = values.view(np.int16)  # as DBR_SHORT delivers them: 32768 arrives as -32768
    assert sent.tolist()[4:] == [-32768, -25536, -2, -1]
    env.ca.raw = (sent, 4, 2)
    async with env.client() as c:
        _, wire = await _raw(c)
    assert np.frombuffer(gzip.decompress(wire), dtype="<u2").tolist() == values.tolist()  # fmt: skip


async def test_the_real_ca_asks_for_the_frame_as_16_bit(monkeypatch) -> None:
    """``AiocaCa.frame`` asks ``DBR_SHORT`` for the frame alone (half the native ``DBR_LONG``'s bytes), its sizes as they are."""
    aioca = pytest.importorskip("aioca")
    asked: list[tuple[Any, Any]] = []

    async def caget(pv: Any, datatype: Any = None, timeout: Any = None) -> Any:
        asked.append((pv, datatype))
        if isinstance(pv, list):
            return [600, 2]
        return np.arange(1200, dtype=np.int32 if datatype is None else datatype)

    monkeypatch.setattr(aioca, "caget", caget)
    data, width, height = await AiocaCa().frame(*RAW)
    assert sorted(asked, key=str) == sorted([(RAW[0], np.int16), (list(RAW[1:]), None)], key=str)  # fmt: skip
    assert data.dtype == np.int16 and (width, height) == (600, 2)
