"""Tests for CaTimestampedReadable — the free-run sync contributor.

Each test runs a CA-mock reference detector (pacemaker) and contributor,
driving ``acq_timestamp`` with ``set_mock_value`` to simulate matched, missed,
and late-arriving shots.  The labeling semantics under test live in the shared
``FreeRunContributorSupport`` mixin.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from bluesky.protocols import Triggerable

pytest.importorskip("aioca")

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import (  # noqa: E402
    CaGenericDetector,
    CaTimestampedReadable,
)
from geecs_bluesky.plans.t0_sync import geecs_t0_sync  # noqa: E402

REF_T0 = 1000.0
CAM_T0 = 1000.05  # 50 ms NTP skew relative to the reference machine


async def _advance(device, value: float) -> None:
    """Set a device's acq_timestamp and let the monitor callback deliver."""
    set_mock_value(device.acq_timestamp, value)
    await asyncio.sleep(0)


async def _setup():
    """Build, connect, configure, and seed a (reference, contributor) pair."""
    ref = CaGenericDetector("U_Ref", ["Sig"], experiment="Test", name="ref")
    cam = CaTimestampedReadable("U_Cam", ["Val"], experiment="Test", name="cam")
    await ref.connect(mock=True)
    await cam.connect(mock=True)
    ref.configure_shot_id(rep_rate_hz=1.0)
    cam.configure_shot_id(rep_rate_hz=1.0)
    cam.set_reference(ref)
    await _advance(ref, REF_T0)
    await _advance(cam, CAM_T0)
    set_mock_value(cam.val, 2.0)
    ref.seed_shot_id(REF_T0)
    cam.seed_shot_id(CAM_T0)
    return ref, cam


async def test_not_triggerable() -> None:
    """Free-run contributors must never block trigger_and_read."""
    cam = CaTimestampedReadable("U_Cam", ["Val"], name="cam")
    assert not isinstance(cam, Triggerable)
    assert not hasattr(cam, "trigger")


async def test_set_reference_does_not_adopt_the_reference() -> None:
    """The reference must stay a peer device.

    ophyd-async adopts Device-valued attributes as renamed children, after
    which bluesky's separate_devices silently drops the reference from
    scans as "redundant" — this pins the Reference-holder behavior.
    """
    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    cam = CaTimestampedReadable("U_Cam", ["Val"], name="cam")
    cam.set_reference(ref)
    assert ref.parent is None
    assert ref.name == "ref"


async def test_matched_shot_is_valid() -> None:
    """Both devices caught the same physical trigger → offset 0, valid."""
    ref, cam = await _setup()
    await _advance(ref, REF_T0 + 1.0)
    await _advance(cam, CAM_T0 + 1.0)

    reading = await asyncio.wait_for(cam.read(), timeout=3.0)
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
    ref, cam = await _setup()
    cam.set_reference(ref, grace_wait_s=0.05)
    # Reference catches two more shots; contributor sees neither
    await _advance(ref, REF_T0 + 2.0)

    reading = await asyncio.wait_for(cam.read(), timeout=3.0)
    assert reading["cam-shot_id"]["value"] == 1
    assert reading["cam-shot_offset"]["value"] == -2
    assert reading["cam-valid"]["value"] is False
    # Data values stay real — stale, truthfully labeled, never NaN-ed here
    assert reading["cam-val"]["value"] == pytest.approx(2.0)


async def test_grace_wait_recovers_late_frame() -> None:
    """A frame arriving within the grace window flips the row to valid."""
    ref, cam = await _setup()
    cam.set_reference(ref, grace_wait_s=2.0)
    await _advance(ref, REF_T0 + 1.0)

    async def deliver_late_frame() -> None:
        await asyncio.sleep(0.15)  # well inside the grace window
        await _advance(cam, CAM_T0 + 1.0)

    task = asyncio.create_task(deliver_late_frame())
    reading = await asyncio.wait_for(cam.read(), timeout=3.0)
    await task
    assert reading["cam-shot_offset"]["value"] == 0
    assert reading["cam-valid"]["value"] is True
    assert reading["cam-acq_timestamp"]["value"] == pytest.approx(CAM_T0 + 1.0)


async def test_no_reference_never_claims_validity() -> None:
    """Without a reference the offset is NaN and valid is False."""
    ref, cam = await _setup()
    cam.set_reference(ref, grace_wait_s=0.0)
    cam._reference = None  # simulate unanchored standalone use

    reading = await asyncio.wait_for(cam.read(), timeout=3.0)
    assert reading["cam-shot_id"]["value"] == 1
    assert math.isnan(reading["cam-shot_offset"]["value"])
    assert reading["cam-valid"]["value"] is False


async def test_save_nonscalar_emits_save_path_column() -> None:
    """A contributor with save_nonscalar_data saves files like the reference."""
    cam = CaTimestampedReadable(
        "U_Cam", ["Val"], experiment="Test", name="cam", save_nonscalar_data=True
    )
    await cam.connect(mock=True)
    cam.configure_shot_id(rep_rate_hz=1.0)
    # The writable save-control signals exist (used by the run wrapper to turn
    # native saving on/off through the gateway :SP PVs)
    assert cam.localsavingpath.source.endswith("Test:U_Cam:localsavingpath")
    assert cam.save.source.endswith("Test:U_Cam:save")
    cam.configure_nonscalar_file_logging("/data/Scan001/U_Cam")

    desc = await asyncio.wait_for(cam.describe(), timeout=3.0)
    assert desc["cam-nonscalar_save_path"]["dtype"] == "string"
    reading = await asyncio.wait_for(cam.read(), timeout=3.0)
    assert reading["cam-nonscalar_save_path"]["value"] == "/data/Scan001/U_Cam"
    for key in desc:
        assert key in reading, f"described key {key} missing from reading"


async def test_t0_sync_seeds_timestamped_readables() -> None:
    """The t0-sync stage treats contributors like any other sync device."""
    ref, cam = await _setup()
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
