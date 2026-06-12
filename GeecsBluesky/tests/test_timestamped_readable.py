"""Tests for GeecsTimestampedReadable — the free-run sync contributor.

Each test runs a reference detector (pacemaker) and a timestamped readable
against two FakeGeecsServer instances, manipulating fake ``acq_timestamp``
values directly to simulate matched, missed, and late-arriving shots.
"""

from __future__ import annotations

import asyncio
import math
from contextlib import AsyncExitStack

import pytest
from bluesky.protocols import Triggerable

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.timestamped_readable import GeecsTimestampedReadable
from geecs_bluesky.plans.t0_sync import geecs_t0_sync
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

REF_T0 = 1000.0
CAM_T0 = 1000.05  # 50 ms NTP skew relative to the reference machine


async def _wait_for_ts(device, expected: float, timeout: float = 2.0) -> None:
    """Wait until the device's TCP cache reflects *expected* acq_timestamp."""
    deadline = asyncio.get_running_loop().time() + timeout
    while device.last_acq_timestamp != pytest.approx(expected):
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(
                f"cache never reached {expected}, last={device.last_acq_timestamp}"
            )
        await asyncio.sleep(0.05)


async def _setup(stack: AsyncExitStack):
    """Build, connect, configure, and seed a (reference, contributor) pair."""
    fake_ref = FakeGeecsDevice(
        name="U_Ref", variables={"Sig": 1.0, "acq_timestamp": REF_T0}
    )
    fake_cam = FakeGeecsDevice(
        name="U_Cam", variables={"Val": 2.0, "acq_timestamp": CAM_T0}
    )
    srv_ref = await stack.enter_async_context(FakeGeecsServer(fake_ref))
    srv_cam = await stack.enter_async_context(FakeGeecsServer(fake_cam))
    ref = GeecsGenericDetector("U_Ref", ["Sig"], srv_ref.host, srv_ref.port, name="ref")
    cam = GeecsTimestampedReadable(
        "U_Cam", ["Val"], srv_cam.host, srv_cam.port, name="cam"
    )
    await ref.connect()
    await cam.connect()
    ref.configure_shot_id(rep_rate_hz=1.0)
    cam.configure_shot_id(rep_rate_hz=1.0)
    cam.set_reference(ref)
    await _wait_for_ts(ref, REF_T0)
    await _wait_for_ts(cam, CAM_T0)
    ref.seed_shot_id(REF_T0)
    cam.seed_shot_id(CAM_T0)
    return fake_ref, fake_cam, ref, cam


async def test_not_triggerable() -> None:
    """Free-run contributors must never block trigger_and_read."""
    cam = GeecsTimestampedReadable("U_Cam", ["Val"], "127.0.0.1", 0, name="cam")
    assert not isinstance(cam, Triggerable)
    assert not hasattr(cam, "trigger")


async def test_matched_shot_is_valid() -> None:
    """Both devices caught the same physical trigger → offset 0, valid."""
    async with AsyncExitStack() as stack:
        fake_ref, fake_cam, ref, cam = await _setup(stack)
        fake_ref.variables["acq_timestamp"] = REF_T0 + 1.0
        fake_cam.variables["acq_timestamp"] = CAM_T0 + 1.0
        await _wait_for_ts(ref, REF_T0 + 1.0)
        await _wait_for_ts(cam, CAM_T0 + 1.0)

        reading = await cam.read()
        assert reading["cam-shot_id"]["value"] == 2
        assert reading["cam-shot_offset"]["value"] == 0
        assert reading["cam-valid"]["value"] is True
        assert reading["cam-acq_timestamp"]["value"] == pytest.approx(CAM_T0 + 1.0)
        assert reading["cam-val"]["value"] == pytest.approx(2.0)

        desc = await cam.describe()
        for key in desc:
            assert key in reading, f"described key {key} missing from reading"


async def test_missed_shots_are_invalid_with_real_offset() -> None:
    """Contributor stuck while the reference advances → negative offset, invalid."""
    async with AsyncExitStack() as stack:
        fake_ref, fake_cam, ref, cam = await _setup(stack)
        cam.set_reference(ref, grace_wait_s=0.05)
        # Reference catches two more shots; contributor sees neither
        fake_ref.variables["acq_timestamp"] = REF_T0 + 2.0
        await _wait_for_ts(ref, REF_T0 + 2.0)

        reading = await cam.read()
        assert reading["cam-shot_id"]["value"] == 1
        assert reading["cam-shot_offset"]["value"] == -2
        assert reading["cam-valid"]["value"] is False
        # Data values stay real — stale, truthfully labeled, never NaN-ed here
        assert reading["cam-val"]["value"] == pytest.approx(2.0)


async def test_grace_wait_recovers_late_frame() -> None:
    """A frame arriving within the grace window flips the row to valid."""
    async with AsyncExitStack() as stack:
        fake_ref, fake_cam, ref, cam = await _setup(stack)
        cam.set_reference(ref, grace_wait_s=2.0)
        fake_ref.variables["acq_timestamp"] = REF_T0 + 1.0
        await _wait_for_ts(ref, REF_T0 + 1.0)

        async def deliver_late_frame() -> None:
            await asyncio.sleep(0.15)  # longer than one push, well inside grace
            fake_cam.variables["acq_timestamp"] = CAM_T0 + 1.0

        task = asyncio.create_task(deliver_late_frame())
        reading = await cam.read()
        await task
        assert reading["cam-shot_offset"]["value"] == 0
        assert reading["cam-valid"]["value"] is True
        assert reading["cam-acq_timestamp"]["value"] == pytest.approx(CAM_T0 + 1.0)


async def test_no_reference_never_claims_validity() -> None:
    """Without a reference the offset is NaN and valid is False."""
    async with AsyncExitStack() as stack:
        fake_ref, fake_cam, ref, cam = await _setup(stack)
        cam.set_reference(ref, grace_wait_s=0.0)
        cam._reference = None  # simulate unanchored standalone use

        reading = await cam.read()
        assert reading["cam-shot_id"]["value"] == 1
        assert math.isnan(reading["cam-shot_offset"]["value"])
        assert reading["cam-valid"]["value"] is False


async def test_t0_sync_seeds_timestamped_readables() -> None:
    """The t0-sync stage treats contributors like any other sync device."""
    async with AsyncExitStack() as stack:
        fake_ref, fake_cam, ref, cam = await _setup(stack)
        # Re-seed both through the plan stage instead of the manual seeds
        plan = geecs_t0_sync([ref, cam], window_s=0.2)
        try:
            plan.send(None)
        except StopIteration as stop:
            t0s = stop.value
        assert t0s == {
            "U_Ref": pytest.approx(REF_T0),
            "U_Cam": pytest.approx(CAM_T0),
        }
        assert cam.shot_id_tracker is not None
        assert cam.shot_id_tracker.t0_acq_timestamp == pytest.approx(CAM_T0)
