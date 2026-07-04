"""Tests for ShotIdTracker and the coordinated t0-sync plan stage.

ShotIdTracker tests are pure unit tests; t0-sync tests run CA-mock
CaGenericDetector instances (set_mock_value drives acq_timestamp) and drive the
plan generator by hand (no RunEngine needed).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from geecs_bluesky.devices.shot_id import ShotIdTracker
from geecs_bluesky.exceptions import GeecsT0SyncError
from geecs_bluesky.plans.t0_sync import geecs_t0_sync

pytest.importorskip("aioca")

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaGenericDetector  # noqa: E402


# ---------------------------------------------------------------------------
# ShotIdTracker
# ---------------------------------------------------------------------------


class TestShotIdTracker:
    def test_unseeded_returns_none(self) -> None:
        tracker = ShotIdTracker(rep_rate_hz=1.0)
        assert not tracker.is_seeded
        assert tracker.update(1000.0) is None

    def test_seed_defines_shot_one(self) -> None:
        tracker = ShotIdTracker(rep_rate_hz=1.0)
        tracker.seed(1000.0)
        assert tracker.is_seeded
        assert tracker.t0_acq_timestamp == 1000.0
        assert tracker.current_shot_id == 1

    def test_consecutive_shots_increment_by_one(self) -> None:
        tracker = ShotIdTracker(rep_rate_hz=1.0)
        tracker.seed(1000.0)
        assert tracker.update(1001.0) == 2
        assert tracker.update(1002.0) == 3

    def test_dead_time_jump_advances_by_elapsed_periods(self) -> None:
        tracker = ShotIdTracker(rep_rate_hz=1.0)
        tracker.seed(1000.0)
        assert tracker.update(1005.0) == 6  # 5 trigger periods elapsed

    def test_repeated_timestamp_is_idempotent(self) -> None:
        """A device timeout repeats acq_timestamp — the ID must not advance."""
        tracker = ShotIdTracker(rep_rate_hz=1.0)
        tracker.seed(1000.0)
        assert tracker.update(1001.0) == 2
        assert tracker.update(1001.0) == 2

    def test_backwards_timestamp_is_ignored(self) -> None:
        tracker = ShotIdTracker(rep_rate_hz=1.0)
        tracker.seed(1000.0)
        tracker.update(1001.0)
        assert tracker.update(1000.5) == 2

    def test_incremental_derivation_immune_to_rep_rate_drift(self) -> None:
        """A 0.05% rate mismatch over 1800 shots must not misquantize.

        Absolute derivation (``round((ts - t0) * rep_rate) + 1``) accumulates
        the mismatch (~0.9 shots here) and lands on the wrong ID; incremental
        derivation re-zeros the error every shot.
        """
        true_period = 1.0005  # nominal rep rate 1 Hz, trigger 0.05% slow
        tracker = ShotIdTracker(rep_rate_hz=1.0)
        t0 = 1000.0
        tracker.seed(t0)
        shot_id = None
        for n in range(1, 1801):
            ts = t0 + n * true_period
            shot_id = tracker.update(ts)
        assert shot_id == 1801
        # The absolute method misquantizes by now — pin the contrast
        absolute = round((t0 + 1800 * true_period - t0) * 1.0) + 1
        assert absolute != 1801

    def test_rejects_invalid_rep_rate(self) -> None:
        with pytest.raises(ValueError):
            ShotIdTracker(rep_rate_hz=0.0)


# ---------------------------------------------------------------------------
# geecs_t0_sync (CA-mock detectors)
# ---------------------------------------------------------------------------


async def _drive(plan, on_sleep=None) -> Any:
    """Drive a plan generator by hand, optionally reacting to sleep messages."""
    result = None
    try:
        msg = plan.send(None)
        while True:
            if msg.command == "sleep" and on_sleep is not None:
                await on_sleep()
            msg = plan.send(None)
    except StopIteration as stop:
        result = stop.value
    return result


async def _sync_detector(
    device: str, name: str, acq_timestamp: float | None
) -> CaGenericDetector:
    """Connected CA-mock detector, optionally with a populated shot cache."""
    det = CaGenericDetector(device, ["Signal"], experiment="Test", name=name)
    await det.connect(mock=True)
    det.configure_shot_id(rep_rate_hz=1.0)
    if acq_timestamp is not None:
        set_mock_value(det.acq_timestamp, acq_timestamp)
        await asyncio.sleep(0)  # deliver the monitor update
    return det


class TestGeecsT0Sync:
    async def test_seeds_all_devices_from_common_shot(self) -> None:
        """Timestamps within the window seed every tracker with its own t0."""
        det_a = await _sync_detector("U_CamA", "cam_a", 1000.00)
        det_b = await _sync_detector("U_CamB", "cam_b", 1000.05)  # 50 ms skew

        t0s = await _drive(geecs_t0_sync([det_a, det_b], window_s=0.2))

        assert t0s == {
            "U_CamA": pytest.approx(1000.00),
            "U_CamB": pytest.approx(1000.05),
        }
        assert det_a.shot_id_tracker is not None
        assert det_b.shot_id_tracker is not None
        assert det_a.shot_id_tracker.t0_acq_timestamp == pytest.approx(1000.00)
        assert det_b.shot_id_tracker.t0_acq_timestamp == pytest.approx(1000.05)
        # Same physical trigger → equal shot IDs despite the skew
        assert det_a.shot_id_tracker.update(1003.00) == 4
        assert det_b.shot_id_tracker.update(1003.05) == 4

    async def test_raises_when_spread_exceeds_window(self) -> None:
        """Caches from different physical triggers must fail loudly."""
        det_a = await _sync_detector("U_CamA", "cam_a", 1000.0)
        det_b = await _sync_detector("U_CamB", "cam_b", 1005.0)  # 5 periods apart

        with pytest.raises(GeecsT0SyncError) as excinfo:
            await _drive(geecs_t0_sync([det_a, det_b], window_s=0.2, retries=0))
        assert "spread" in str(excinfo.value)
        assert det_a.shot_id_tracker is not None
        assert not det_a.shot_id_tracker.is_seeded

    async def test_retry_recovers_when_lagging_frame_arrives(self) -> None:
        """A frame still propagating at the first attempt succeeds on retry."""
        det_a = await _sync_detector("U_CamA", "cam_a", 1000.0)
        det_b = await _sync_detector("U_CamB", "cam_b", 995.0)  # stale frame

        async def deliver_lagging_frame() -> None:
            set_mock_value(det_b.acq_timestamp, 1000.1)
            await asyncio.sleep(0)

        t0s = await _drive(
            geecs_t0_sync([det_a, det_b], window_s=0.2, retries=2),
            on_sleep=deliver_lagging_frame,
        )
        assert t0s["U_CamB"] == pytest.approx(1000.1)

    async def test_missing_acq_timestamp_reports_device(self) -> None:
        """A device whose acq_timestamp never populated names itself in the error."""
        det = await _sync_detector("U_CamA", "cam_a", None)  # cache never fills

        with pytest.raises(GeecsT0SyncError) as excinfo:
            await _drive(geecs_t0_sync([det], window_s=0.2, retries=0))
        assert "U_CamA" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Stable keys / NaN policy on the generic detector
# ---------------------------------------------------------------------------


async def test_detector_emits_nan_companions_before_first_shot() -> None:
    """All described companion columns appear even with no shot data yet."""
    det = await _sync_detector("U_CamA", "cam_a", None)  # no acq_timestamp yet

    desc = await det.describe()
    reading = await det.read()
    for key in desc:
        assert key in reading, f"described key {key} missing from reading"
    assert math.isnan(reading["cam_a-shot_id"]["value"])
    assert math.isnan(reading["cam_a-shot_offset"]["value"])
    assert reading["cam_a-valid"]["value"] is False
