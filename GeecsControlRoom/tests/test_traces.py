"""The trace route over the fake CA: the shape, the hold on the counter, the axis re-read, the refusals."""

from __future__ import annotations

import gzip

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from geecs_control_room.traces import AXIS_EVERY_S, TRACE_HOLD_S, Traces, router
from tests.test_images import Env

P = "bench:scope:"
DATA, AXIS, COUNT, CHANNELS = P + "ARR:ArrayData", P + "TimeAxis_RBV", P + "ARR:ArrayCounter_RBV", P + "ARR:ArraySize0_RBV"  # fmt: skip
ROLES = [
    {"role": "data", "pv": DATA, "read": "on_request", "type": "array"},
    {"role": "time_axis", "pv": AXIS, "read": "on_request", "type": "array"},
    {"role": "arr_counter", "pv": COUNT, "read": "monitor", "type": "counter"},
    {"role": "arr_channels", "pv": CHANNELS, "read": "on_request", "type": "readback"},
]  # fmt: skip
URL = "/api/devices/Cam/trace"
#: Past the hold (a float sum can land a hair inside it).
STEP = TRACE_HOLD_S * 1.01


def capture(
    channels: int = 4, samples: int = 5, off: tuple[int, ...] = (3,)
) -> np.ndarray:
    """[channels, samples], channels fastest: channel c's sample s is ``10 c + s``; an ``off`` channel NaN."""
    y = np.array([[np.nan if c in off else 10.0 * c + s for c in range(channels)] for s in range(samples)], dtype=np.float32)  # fmt: skip
    return y.ravel()


def scope(roles: list | None = None, family: str = "scope_contract") -> Env:
    env = Env(roles=ROLES if roles is None else roles)
    env.family = family
    env.ca.floats_by = {DATA: capture(), AXIS: np.arange(5, dtype=np.float64) * 1e-6}
    env.ca.scalars = {CHANNELS: 4.0}
    env.ca.counters = [7.0]
    env.traces = Traces(env.images, clock=lambda: env.clock[0])
    return env


def client(env: Env) -> httpx.AsyncClient:
    app = FastAPI()
    app.state.module = env.module
    app.include_router(router(env.traces))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://room.test")  # fmt: skip


async def wire(c: httpx.AsyncClient, **headers: str) -> tuple[httpx.Response, bytes]:
    """The response and the bytes as sent (httpx would undo the gzip)."""
    async with c.stream("GET", URL, headers=headers) as r:
        return r, b"".join([chunk async for chunk in r.aiter_raw()])


def reads(env: Env) -> list:
    return [(k, a) for k, a in env.ca.calls]


async def test_the_trace_is_the_axis_then_the_capture_float32_gzip_framed() -> None:
    env = scope()
    async with client(env) as c:
        r, sent = await wire(c)
    assert r.status_code == 200 and r.headers["content-encoding"] == "gzip"
    assert (r.headers["x-trace-channels"], r.headers["x-trace-samples"]) == ("4", "5")
    assert r.headers["etag"] == r.headers["x-trace-tag"] == '"7"' and r.headers["cache-control"] == "no-store"  # fmt: skip
    body = np.frombuffer(gzip.decompress(sent), dtype="<f4")
    assert body.size == 5 + 20
    assert np.allclose(body[:5], np.arange(5) * 1e-6)  # the axis, float32
    np.testing.assert_array_equal(
        body[5:], capture()
    )  # the capture as the IOC lays it out, NaN kept
    # the counter, the batch (array + channel count), the counter again, then the axis
    assert reads(env) == [("scalar", COUNT), ("floats", (DATA,)), ("scalar", CHANNELS), ("scalar", COUNT), ("floats", (AXIS,))]  # fmt: skip


async def test_an_unmoved_counter_costs_one_scalar_read_and_a_304() -> None:
    env = scope()
    async with client(env) as c:
        first = await c.get(URL)
        env.ca.calls.clear()
        env.clock[0] += STEP
        same = await c.get(URL, headers={"If-None-Match": first.headers["etag"]})
        assert (same.status_code, same.content, same.headers["etag"]) == (
            304,
            b"",
            '"7"',
        )
        assert reads(env) == [("scalar", COUNT)]
        env.ca.counters, env.clock[0] = [8.0], env.clock[0] + STEP
        new = await c.get(URL, headers={"If-None-Match": first.headers["etag"]})
    assert new.status_code == 200 and new.headers["etag"] == '"8"'
    # a new capture of the same length and timebase: the axis' first two values probed, the axis not read again
    assert ("head", (AXIS,)) in reads(env) and ("floats", (AXIS,)) not in reads(env)


async def test_a_new_timebase_of_the_same_length_reads_the_axis_again() -> None:
    env = scope()
    async with client(env) as c:
        first = await c.get(URL)
        env.ca.floats_by[AXIS] = (
            np.arange(5, dtype=np.float32) * 2e-6
        )  # the interval doubled, five samples still
        env.ca.counters, env.clock[0] = [8.0], env.clock[0] + STEP
        r = await c.get(URL)
    assert ("floats", (AXIS,)) in reads(env)[5:]  # past the first trace's reads
    axis = np.frombuffer(r.content, dtype="<f4")[:5]
    assert np.allclose(axis, np.arange(5) * 2e-6) and r.content != first.content


async def test_the_axis_is_read_again_when_its_length_changes_or_it_is_old() -> None:
    env = scope()
    async with client(env) as c:
        await c.get(URL)
        env.ca.floats_by = {
            DATA: capture(samples=8),
            AXIS: np.arange(8, dtype=np.float32),
        }
        env.ca.counters, env.clock[0] = [8.0], env.clock[0] + STEP
        env.ca.calls.clear()
        r = await c.get(URL)
        assert r.headers["x-trace-samples"] == "8" and ("floats", (AXIS,)) in reads(env)
        env.ca.counters, env.clock[0] = [9.0], env.clock[0] + AXIS_EVERY_S
        env.ca.calls.clear()
        await c.get(URL)
    assert ("floats", (AXIS,)) in reads(env)


async def test_a_capture_landing_mid_batch_is_read_again_and_tagged_by_the_second_counter() -> None:  # fmt: skip
    env = scope()
    env.ca.counters = [7.0, 8.0]
    async with client(env) as c:
        r = await c.get(URL)
    assert r.headers["etag"] == '"8"' and [k for k, a in env.ca.calls if a == (DATA,)] == ["floats", "floats"]  # fmt: skip


async def test_one_read_serves_every_viewer_within_the_hold() -> None:
    env = scope()
    async with client(env) as c:
        await c.get(URL)
        await c.get(URL)
    assert [a for _, a in env.ca.calls].count((DATA,)) == 1 and env.reads == 1


@pytest.mark.parametrize(
    "roles, family, words",
    [
        (ROLES, "ad_camera_chain", "names no trace"),
        (ROLES[:2] + ROLES[3:], "scope_contract", "the module names no arr_counter role for Cam"),
        (ROLES[:1], "scope_contract", "no time_axis, arr_counter, arr_channels roles"),
    ],
)  # fmt: skip
async def test_a_trace_the_family_or_the_module_does_not_name_is_a_404(roles, family, words) -> None:  # fmt: skip
    env = scope(roles, family)
    async with client(env) as c:
        r = await c.get(URL)
    assert r.status_code == 404 and words in r.json()["detail"] and not [a for k, a in env.ca.calls if k == "floats"]  # fmt: skip


@pytest.mark.parametrize(
    "break_it, words",
    [
        (lambda ca: setattr(ca, "counter_fails", True), f"the capture counter ({COUNT}) could not be read"),
        (lambda ca: ca.floats_by.pop(DATA), f"the trace ({DATA}) could not be read"),
        (lambda ca: ca.scalars.update({CHANNELS: 3.0}), "shape disagrees: 20 values for 3 channels"),
        (lambda ca: ca.floats_by.update({AXIS: np.zeros(4)}), f"the axis ({AXIS}) has 4 values for 5 samples"),
        (lambda ca: ca.floats_by.pop(AXIS), f"the axis ({AXIS}) could not be read"),
    ],
)  # fmt: skip
async def test_a_silent_source_or_a_shape_disagreement_is_a_503_with_words(break_it, words) -> None:  # fmt: skip
    env = scope()
    break_it(env.ca)
    async with client(env) as c:
        r = await c.get(URL)
    assert r.status_code == 503 and r.json()["detail"].startswith(words)


async def test_a_one_channel_family_reads_no_channel_count() -> None:
    """The spectrometer: ``[pixels]`` over its wavelengths, no ``channels_by``; one channel, no scalar read for the count."""
    sp = "bench:cct:"
    y, axis, count = (
        sp + "ARR:ArrayData",
        sp + "Wavelength_RBV",
        sp + "ARR:ArrayCounter_RBV",
    )
    roles = [
        {"role": "spectrum", "pv": y, "read": "on_request", "type": "array"},
        {"role": "wavelength", "pv": axis, "read": "on_request", "type": "array"},
        {"role": "arr_counter", "pv": count, "read": "monitor", "type": "counter"},
    ]
    env = scope(roles, "spectrometer_contract")
    env.ca.floats_by = {y: np.arange(6, dtype=np.float32) * 100, axis: 180.0 + np.arange(6, dtype=np.float64) * 0.36}  # fmt: skip
    async with client(env) as c:
        r, raw = await wire(c)
    assert r.status_code == 200 and (r.headers["x-trace-channels"], r.headers["x-trace-samples"]) == ("1", "6")  # fmt: skip
    values = np.frombuffer(gzip.decompress(raw), dtype="<f4")
    assert values[:6] == pytest.approx(180.0 + np.arange(6) * 0.36, abs=1e-4) and list(values[6:]) == [0, 100, 200, 300, 400, 500]  # fmt: skip
    # the counter, the spectrum alone (no channel count asked), the counter again, the axis
    assert [a for k, a in env.ca.calls if k in ("floats", "scalar")] == [
        count,
        (y,),
        count,
        (axis,),
    ]
