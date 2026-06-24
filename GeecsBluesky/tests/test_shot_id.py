"""Tests for ShotIdTracker and the coordinated t0-sync plan stage.

ShotIdTracker tests are pure unit tests; t0-sync tests run real
GeecsGenericDetector instances against FakeGeecsServer and drive the plan
generator by hand (no RunEngine needed).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.shot_id import ShotIdTracker
from geecs_bluesky.exceptions import GeecsT0SyncError
from geecs_bluesky.plans.t0_sync import geecs_t0_sync
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer


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
# geecs_t0_sync
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


async def _wait_for_cache(det: GeecsGenericDetector, timeout: float = 2.0) -> None:
    """Wait until the detector's TCP cache holds an acq_timestamp."""
    deadline = asyncio.get_running_loop().time() + timeout
    while det.last_acq_timestamp is None:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("TCP cache never populated")
        await asyncio.sleep(0.05)


def _sync_device(name: str, acq_timestamp: float) -> FakeGeecsDevice:
    return FakeGeecsDevice(
        name=name,
        variables={"Signal": 1.0, "acq_timestamp": acq_timestamp},
    )


@pytest.mark.fake_server
class TestGeecsT0Sync:
    async def test_seeds_all_devices_from_common_shot(self) -> None:
        """Timestamps within the window seed every tracker with its own t0."""
        fake_a = _sync_device("U_CamA", 1000.00)
        fake_b = _sync_device("U_CamB", 1000.05)  # 50 ms NTP skew
        async with FakeGeecsServer(fake_a) as srv_a, FakeGeecsServer(fake_b) as srv_b:
            det_a = GeecsGenericDetector(
                "U_CamA", ["Signal"], srv_a.host, srv_a.port, name="cam_a"
            )
            det_b = GeecsGenericDetector(
                "U_CamB", ["Signal"], srv_b.host, srv_b.port, name="cam_b"
            )
            await det_a.connect()
            await det_b.connect()
            det_a.configure_shot_id(rep_rate_hz=1.0)
            det_b.configure_shot_id(rep_rate_hz=1.0)
            await _wait_for_cache(det_a)
            await _wait_for_cache(det_b)

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
        fake_a = _sync_device("U_CamA", 1000.0)
        fake_b = _sync_device("U_CamB", 1005.0)  # 5 trigger periods apart
        async with FakeGeecsServer(fake_a) as srv_a, FakeGeecsServer(fake_b) as srv_b:
            det_a = GeecsGenericDetector(
                "U_CamA", ["Signal"], srv_a.host, srv_a.port, name="cam_a"
            )
            det_b = GeecsGenericDetector(
                "U_CamB", ["Signal"], srv_b.host, srv_b.port, name="cam_b"
            )
            await det_a.connect()
            await det_b.connect()
            det_a.configure_shot_id(rep_rate_hz=1.0)
            det_b.configure_shot_id(rep_rate_hz=1.0)
            await _wait_for_cache(det_a)
            await _wait_for_cache(det_b)

            with pytest.raises(GeecsT0SyncError) as excinfo:
                await _drive(geecs_t0_sync([det_a, det_b], window_s=0.2, retries=0))
            assert "spread" in str(excinfo.value)
            assert det_a.shot_id_tracker is not None
            assert not det_a.shot_id_tracker.is_seeded

    async def test_retry_recovers_when_lagging_frame_arrives(self) -> None:
        """A frame still propagating at the first attempt succeeds on retry."""
        fake_a = _sync_device("U_CamA", 1000.0)
        fake_b = _sync_device("U_CamB", 995.0)  # stale — last push pre-dates A's
        async with FakeGeecsServer(fake_a) as srv_a, FakeGeecsServer(fake_b) as srv_b:
            det_a = GeecsGenericDetector(
                "U_CamA", ["Signal"], srv_a.host, srv_a.port, name="cam_a"
            )
            det_b = GeecsGenericDetector(
                "U_CamB", ["Signal"], srv_b.host, srv_b.port, name="cam_b"
            )
            await det_a.connect()
            await det_b.connect()
            det_a.configure_shot_id(rep_rate_hz=1.0)
            det_b.configure_shot_id(rep_rate_hz=1.0)
            await _wait_for_cache(det_a)
            await _wait_for_cache(det_b)

            async def deliver_lagging_frame() -> None:
                fake_b.variables["acq_timestamp"] = 1000.1
                await asyncio.sleep(0.3)  # let the 5 Hz push propagate

            t0s = await _drive(
                geecs_t0_sync([det_a, det_b], window_s=0.2, retries=2),
                on_sleep=deliver_lagging_frame,
            )
            assert t0s["U_CamB"] == pytest.approx(1000.1)

    async def test_missing_acq_timestamp_reports_device(self) -> None:
        """A device whose acq_timestamp never populated names itself in the error."""
        # Empty value → push frame pair never parses → cache never gets the key
        fake = FakeGeecsDevice(
            name="U_CamA", variables={"Signal": 1.0, "acq_timestamp": ""}
        )
        async with FakeGeecsServer(fake) as srv:
            det = GeecsGenericDetector(
                "U_CamA", ["Signal"], srv.host, srv.port, name="cam_a"
            )
            await det.connect()
            det.configure_shot_id(rep_rate_hz=1.0)
            await asyncio.sleep(0.3)  # pushes arrive, but carry no acq_timestamp

            with pytest.raises(GeecsT0SyncError) as excinfo:
                await _drive(geecs_t0_sync([det], window_s=0.2, retries=0))
            assert "U_CamA" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Stable keys / NaN policy on GeecsGenericDetector
# ---------------------------------------------------------------------------


@pytest.mark.fake_server
async def test_detector_emits_nan_companions_before_first_shot() -> None:
    """All described companion columns appear even with no shot data yet."""
    # Empty value → push frame pair never parses → cache never gets the key
    fake = FakeGeecsDevice(
        name="U_CamA", variables={"Signal": 1.0, "acq_timestamp": ""}
    )
    async with FakeGeecsServer(fake) as srv:
        det = GeecsGenericDetector(
            "U_CamA", ["Signal"], srv.host, srv.port, name="cam_a"
        )
        await det.connect()
        det.configure_shot_id(rep_rate_hz=1.0)
        await asyncio.sleep(0.3)  # pushes arrive, but carry no acq_timestamp

        desc = await det.describe()
        reading = await det.read()
        for key in desc:
            assert key in reading, f"described key {key} missing from reading"
        assert math.isnan(reading["cam_a-shot_id"]["value"])
        assert math.isnan(reading["cam_a-shot_offset"]["value"])
        assert reading["cam_a-valid"]["value"] is False
