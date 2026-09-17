"""ShotSampler — one event per tick of the clock (phase 2b, ``08`` §4.7)."""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import set_mock_value, soft_signal_rw  # noqa: E402

from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable  # noqa: E402
from geecs_bluesky.devices.detector import GeecsDetector  # noqa: E402
from geecs_bluesky.devices.sampler import ShotSampler  # noqa: E402
from geecs_bluesky.exceptions import GeecsTriggerTimeoutError  # noqa: E402
from geecs_bluesky.plans.strict import BinCounter  # noqa: E402
from tests.ca_mock_helpers import connect_mock  # noqa: E402


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _run(RE: RunEngine, coro_factory, timeout: float = 10.0):
    async def call():
        return await coro_factory()

    return asyncio.run_coroutine_threadsafe(call(), RE._loop).result(timeout=timeout)


def _members(RE: RunEngine):
    ict = GeecsDetector("U_ICT", ["Charge"], experiment="TestExp", name="u_ict")
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, ict, gauge)
    set_mock_value(ict.acq_timestamp, 100.0)
    set_mock_value(ict.charge, 1.5)
    set_mock_value(gauge.pressure, 2e-7)
    bins = BinCounter()
    bins.value = 3
    return ict, gauge, bins


def test_one_row_per_tick_with_every_members_latest_value(RE: RunEngine) -> None:
    ict, gauge, bins = _members(RE)
    sampler = ShotSampler([ict, gauge, bins], ict.acq_timestamp, clock_name="U_ICT")

    async def scenario():
        await sampler.prepare(3)
        keys = set(await sampler.describe_collect())
        await sampler.kickoff()
        status = sampler.complete()
        for i, stamp in enumerate((101.0, 102.0, 103.0)):
            await asyncio.sleep(0.02)
            if i == 1:
                set_mock_value(gauge.pressure, 3e-7)  # the latest value rides
            set_mock_value(ict.acq_timestamp, stamp)
        await status
        rows = [row async for row in sampler.collect()]
        again = [row async for row in sampler.collect()]
        return keys, rows, again

    keys, rows, again = _run(RE, scenario)
    assert keys == {
        "u_ict-charge",
        "u_ict-acq_timestamp",
        "u_gauge-pressure",
        "bin_number",
    }
    assert len(rows) == 3 and again == []
    assert [r["data"]["u_ict-acq_timestamp"] for r in rows] == [101.0, 102.0, 103.0]
    assert [r["data"]["u_gauge-pressure"] for r in rows] == [2e-7, 3e-7, 3e-7]
    assert [r["data"]["bin_number"] for r in rows] == [3, 3, 3]
    assert all(set(r["timestamps"]) == keys for r in rows)
    assert all(abs(r["time"] - time.time()) < 5 for r in rows)
    assert sampler.sampled == 3


def test_baseline_echo_and_repeats_are_not_ticks_and_extra_ticks_are_ignored(
    RE: RunEngine,
) -> None:
    """The subscribe echo (the current stamp) and a re-published stamp count for nothing."""
    ict, gauge, bins = _members(RE)
    sampler = ShotSampler([gauge], ict.acq_timestamp, clock_name="U_ICT")

    async def scenario():
        await sampler.prepare(2)
        await sampler.kickoff()
        status = sampler.complete()
        await asyncio.sleep(0.02)
        set_mock_value(ict.acq_timestamp, 100.0)  # the baseline again
        set_mock_value(ict.acq_timestamp, 0.0)  # the gateway placeholder
        await asyncio.sleep(0.02)
        assert not status.done
        set_mock_value(ict.acq_timestamp, 101.0)
        set_mock_value(ict.acq_timestamp, 102.0)
        await status
        set_mock_value(ict.acq_timestamp, 103.0)  # after the quota: ignored
        await asyncio.sleep(0.02)
        return [row async for row in sampler.collect()]

    rows = _run(RE, scenario)
    assert [r["data"]["u_ict-acq_timestamp"] for r in rows] == [101.0, 102.0]


def test_silent_clock_fails_complete_with_the_geecs_timeout(RE: RunEngine) -> None:
    ict, gauge, _ = _members(RE)
    sampler = ShotSampler(
        [gauge], ict.acq_timestamp, clock_name="U_ICT", shot_timeout=0.15
    )

    async def scenario():
        await sampler.prepare(2)
        await sampler.kickoff()
        with pytest.raises(GeecsTriggerTimeoutError, match="U_ICT.*shot clock"):
            await sampler.complete()

    _run(RE, scenario)


def test_cancel_step_settles_a_pending_complete_and_drops_the_rows(
    RE: RunEngine,
) -> None:
    ict, gauge, _ = _members(RE)
    sampler = ShotSampler(
        [gauge], ict.acq_timestamp, clock_name="U_ICT", shot_timeout=5.0
    )

    async def scenario():
        await sampler.prepare(3)
        await sampler.kickoff()
        status = sampler.complete()
        await asyncio.sleep(0.02)
        set_mock_value(ict.acq_timestamp, 101.0)
        await asyncio.sleep(0.02)
        assert sampler.sampled == 1
        await sampler.cancel_step()
        await status  # settled, not failed
        rows = [row async for row in sampler.collect()]
        # re-armed for the retake
        await sampler.prepare(1)
        await sampler.kickoff()
        status = sampler.complete()
        await asyncio.sleep(0.02)
        set_mock_value(ict.acq_timestamp, 102.0)
        await status
        return rows, [row async for row in sampler.collect()]

    dropped, retaken = _run(RE, scenario)
    assert dropped == []
    assert [r["data"]["u_ict-acq_timestamp"] for r in retaken] == [102.0]


def test_a_plugin_cameras_clock_rides_as_its_own_column(RE: RunEngine) -> None:
    """The clock need not be a member (a camera in D): its stamp column is added."""
    _, gauge, bins = _members(RE)
    clock = soft_signal_rw(float, 50.0, name="uc_cam-acq_timestamp")
    _run(RE, lambda: clock.connect(mock=True))
    sampler = ShotSampler([gauge, bins], clock, clock_name="UC_Cam")

    async def scenario():
        await sampler.prepare(1)
        keys = set(await sampler.describe_collect())
        await sampler.kickoff()
        status = sampler.complete()
        await asyncio.sleep(0.02)
        await clock.set(51.0)
        await status
        return keys, [row async for row in sampler.collect()]

    keys, rows = _run(RE, scenario)
    assert keys == {"u_gauge-pressure", "bin_number", "uc_cam-acq_timestamp"}
    assert rows[0]["data"]["uc_cam-acq_timestamp"] == 51.0


def test_prepare_and_kickoff_order_is_enforced(RE: RunEngine) -> None:
    ict, gauge, _ = _members(RE)
    sampler = ShotSampler([gauge], ict.acq_timestamp, clock_name="U_ICT")

    async def scenario():
        with pytest.raises(RuntimeError, match="before prepare"):
            await sampler.kickoff()
        with pytest.raises(RuntimeError, match="before kickoff"):
            await sampler.complete()
        with pytest.raises(ValueError, match="at least one shot"):
            await sampler.prepare(0)

    _run(RE, scenario)
