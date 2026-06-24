"""Tests for the strict single-shot acquisition path.

No free-running firer here: the fake device fires **only** when the plan's
``fire_shot`` callable executes, proving the plan owns every shot.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.exceptions import (
    GeecsQuiescenceTimeoutError,
    GeecsTriggerTimeoutError,
)
from geecs_bluesky.plans.single_shot import geecs_confirm_quiescent
from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice
from tests.fake_server_helpers import (
    BackgroundFakeServers,
    connect_devices,
    disconnect_devices,
)


class _StubTsDevice:
    """Minimal device exposing ``last_acq_timestamp`` for quiescence tests.

    ``advancing=True`` returns a new value on every read (trigger still
    firing); ``advancing=False`` returns a constant (trigger stopped).
    """

    def __init__(self, advancing: bool) -> None:
        self._advancing = advancing
        self._t = 1000.0

    @property
    def last_acq_timestamp(self) -> float:
        if self._advancing:
            self._t += 1.0
        return self._t


def _drive(plan) -> None:
    """Step a plan generator to completion, ignoring yielded messages.

    Treats ``bps.sleep`` as instantaneous so quiescence logic is exercised
    deterministically without real time passing.
    """
    try:
        plan.send(None)
        while True:
            plan.send(None)
    except StopIteration:
        pass


def test_confirm_quiescent_returns_when_trigger_stopped() -> None:
    """A static acq_timestamp (trigger stopped) → returns without raising."""
    dev = _StubTsDevice(advancing=False)
    _drive(geecs_confirm_quiescent([dev], quiet_s=0.5, timeout_s=5.0, poll_s=0.1))


def test_confirm_quiescent_raises_when_trigger_still_firing() -> None:
    """An advancing acq_timestamp (still free-running) → raises after timeout."""
    dev = _StubTsDevice(advancing=True)
    with pytest.raises(GeecsQuiescenceTimeoutError):
        _drive(geecs_confirm_quiescent([dev], quiet_s=0.5, timeout_s=1.0, poll_s=0.1))


def test_confirm_quiescent_no_sync_devices_is_noop() -> None:
    """No device exposes last_acq_timestamp → returns immediately."""
    _drive(geecs_confirm_quiescent([object()], quiet_s=0.5))


@contextmanager
def _setup_scan() -> Iterator[
    tuple[FakeGeecsDevice, GeecsMotor, GeecsGenericDetector, RunEngine, list]
]:
    fake = FakeGeecsDevice(
        name="U_Combined",
        variables={"Position (mm)": 0.0, "Sig": 1.0, "acq_timestamp": 1000.0},
    )
    with BackgroundFakeServers(fake) as server:
        host, port = server.endpoint

        motor = GeecsMotor("U_Combined", "Position (mm)", host, port, name="scan_motor")
        cam = GeecsGenericDetector("U_Combined", ["Sig"], host, port, name="cam")
        cam.configure_shot_id(rep_rate_hz=1.0)

        events: list[dict] = []
        RE = RunEngine()
        RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)
        connect_devices(RE, motor, cam)
        try:
            yield fake, motor, cam, RE, events
        finally:
            disconnect_devices(RE, motor, cam)


@pytest.mark.fake_server
def test_strict_scan_fires_each_shot_itself() -> None:
    """One fire → one complete row; no free-running trigger required."""
    with _setup_scan() as (fake, motor, cam, RE, events):
        fire_count = 0

        def fire():
            nonlocal fire_count
            fire_count += 1
            fake.fire_shot()
            yield from bps.null()

        RE(
            geecs_step_scan(
                motor=motor,
                positions=[0.0, 1.0],
                detectors=[cam],
                shots_per_step=2,
                fire_shot=fire,
            )
        )

        assert fire_count == 4
        assert len(events) == 4
        # Plan-owned shots are consecutive: the trigger only ticks when fired
        assert [ev["data"]["cam-shot_id"] for ev in events] == [1, 2, 3, 4]
        for ev in events:
            assert ev["data"]["cam-valid"] is True
            assert ev["data"]["cam-shot_offset"] == 0


@pytest.mark.fake_server
def test_strict_scan_setup_trigger_runs_once_before_shots() -> None:
    """setup_trigger (arm + confirm) runs once at scan start, not per shot."""
    with _setup_scan() as (fake, _motor, cam, RE, events):
        calls = {"setup": 0, "fire": 0}

        def setup():
            calls["setup"] += 1
            yield from bps.null()

        def fire():
            calls["fire"] += 1
            fake.fire_shot()
            yield from bps.null()

        RE(
            geecs_step_scan(
                motor=None,
                positions=[None],
                detectors=[cam],
                shots_per_step=3,
                setup_trigger=setup,
                fire_shot=fire,
            )
        )

        assert calls["setup"] == 1, "setup_trigger must run exactly once"
        assert calls["fire"] == 3
        assert len(events) == 3
        # Start doc records that the plan fired its own shots
        assert all(ev["data"]["cam-valid"] for ev in events)


@pytest.mark.fake_server
def test_strict_scan_hard_fails_when_device_misses_the_shot() -> None:
    """A device not responding to the plan's own shot aborts the scan."""
    with _setup_scan() as (_fake, motor, cam, RE, events):
        cam._trigger_timeout = 0.5  # keep the failure fast

        def dead_fire():
            # Trigger source broken: nothing fires, the camera never advances
            yield from bps.null()

        with pytest.raises(Exception) as excinfo:
            RE(
                geecs_step_scan(
                    motor=motor,
                    positions=[0.0],
                    detectors=[cam],
                    shots_per_step=1,
                    fire_shot=dead_fire,
                )
            )
        # The timeout names the offending device through the status failure
        assert "U_Combined" in str(excinfo.value) or isinstance(
            excinfo.value.__cause__, GeecsTriggerTimeoutError
        )
        assert events == []
