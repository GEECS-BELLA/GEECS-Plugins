"""Tests for the strict single-shot acquisition path (CA-mock devices).

No free-running pacer here: the mock device's ``acq_timestamp`` advances
**only** when the plan's ``fire_shot`` callable executes, proving the plan
owns every shot.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine
from bluesky.utils import FailedStatus

from geecs_bluesky.exceptions import (
    GeecsQuiescenceTimeoutError,
    GeecsTriggerTimeoutError,
)
from geecs_bluesky.plans.single_shot import geecs_confirm_quiescent, geecs_single_shot
from geecs_bluesky.plans.step_scan import geecs_step_scan

pytest.importorskip("aioca")

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaGenericDetector, CaMotor  # noqa: E402
from tests.ca_mock_helpers import connect_mock, follow_setpoint  # noqa: E402


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
def _setup_scan() -> Iterator[tuple[CaMotor, CaGenericDetector, RunEngine, list]]:
    motor = CaMotor("U_Combined", "Position (mm)", name="scan_motor")
    cam = CaGenericDetector("U_Combined", ["Sig"], name="cam")
    cam.configure_shot_id(rep_rate_hz=1.0)

    events: list[dict] = []
    RE = RunEngine()
    RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)
    connect_mock(RE, motor, cam)
    follow_setpoint(motor)
    set_mock_value(cam.acq_timestamp, 1000.0)
    yield motor, cam, RE, events


def _make_fire(cam: CaGenericDetector, counter: dict) -> object:
    """Fire hook: advancing acq_timestamp IS the shot (runs in the RE loop)."""
    state = {"t": 1000.0}

    def fire():
        counter["fire"] = counter.get("fire", 0) + 1
        state["t"] += 1.0
        set_mock_value(cam.acq_timestamp, state["t"])
        yield from bps.null()

    return fire


def test_strict_scan_fires_each_shot_itself() -> None:
    """One fire → one complete row; no free-running trigger required."""
    with _setup_scan() as (motor, cam, RE, events):
        calls: dict = {}
        RE(
            geecs_step_scan(
                motor=motor,
                positions=[0.0, 1.0],
                detectors=[cam],
                shots_per_step=2,
                fire_shot=_make_fire(cam, calls),
            )
        )

        assert calls["fire"] == 4
        assert len(events) == 4
        # Plan-owned shots are consecutive: the trigger only ticks when fired
        assert [ev["data"]["cam-shot_id"] for ev in events] == [1, 2, 3, 4]
        for ev in events:
            assert ev["data"]["cam-valid"] is True
            assert ev["data"]["cam-shot_offset"] == 0


def test_strict_scan_setup_trigger_runs_once_before_shots() -> None:
    """setup_trigger (arm + confirm) runs once at scan start, not per shot."""
    with _setup_scan() as (_motor, cam, RE, events):
        calls: dict = {}

        def setup():
            calls["setup"] = calls.get("setup", 0) + 1
            yield from bps.null()

        RE(
            geecs_step_scan(
                motor=None,
                positions=[None],
                detectors=[cam],
                shots_per_step=3,
                setup_trigger=setup,
                fire_shot=_make_fire(cam, calls),
            )
        )

        assert calls["setup"] == 1, "setup_trigger must run exactly once"
        assert calls["fire"] == 3
        assert len(events) == 3
        assert all(ev["data"]["cam-valid"] for ev in events)


def test_strict_scan_hard_fails_when_device_misses_the_shot() -> None:
    """A device not responding to the plan's own shot aborts the scan."""
    with _setup_scan() as (motor, cam, RE, events):
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


# ---------------------------------------------------------------------------
# Bounded refire (max_refires) — recovery from the no-frame case
# ---------------------------------------------------------------------------


def _single_shot_run(devices, fire, **kwargs):
    """Wrap geecs_single_shot in an open/close run so create/save are legal."""
    yield from bps.open_run()
    yield from geecs_single_shot(devices, fire, **kwargs)
    yield from bps.close_run()


def test_single_shot_refires_after_missed_frame(caplog) -> None:
    """A no-frame first fire is retried; the second fire completes the row."""
    with _setup_scan() as (_motor, cam, RE, events):
        cam._trigger_timeout = 0.5  # keep the missed attempt fast
        calls = {"fire": 0}
        state = {"t": 1000.0}

        def flaky_fire():
            calls["fire"] += 1
            if calls["fire"] >= 2:  # first fire drops the frame (no advance)
                state["t"] += 1.0
                set_mock_value(cam.acq_timestamp, state["t"])
            yield from bps.null()

        with caplog.at_level(logging.WARNING, logger="geecs_bluesky.plans.single_shot"):
            RE(_single_shot_run([cam], flaky_fire))

        assert calls["fire"] == 2
        assert len(events) == 1
        assert events[0]["data"]["cam-valid"] is True
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == "geecs_bluesky.plans.single_shot"
        ]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "U_Combined" in msg  # names the device that produced no frame
        assert "attempt 1 of 3" in msg


def test_single_shot_refire_exhaustion_propagates() -> None:
    """All fires missed → FailedStatus propagates after max_refires+1 fires."""
    with _setup_scan() as (_motor, cam, RE, events):
        cam._trigger_timeout = 0.5
        calls = {"fire": 0}

        def dead_fire():
            calls["fire"] += 1
            yield from bps.null()  # nothing ever frames

        with pytest.raises(FailedStatus) as excinfo:
            RE(_single_shot_run([cam], dead_fire, max_refires=2))

        assert calls["fire"] == 3  # max_refires + 1 physical fires
        assert isinstance(excinfo.value.__cause__, GeecsTriggerTimeoutError)
        assert excinfo.value.__cause__.device_name == "U_Combined"
        assert events == []  # a failed shot records nothing


def test_single_shot_partial_miss_drains_orphan_frame() -> None:
    """Device A frames every fire, B misses the first: one row, one shot.

    Pins the strict-semantics argument for refire: A's orphan frame from the
    failed first fire is re-baselined/drained by the second attempt's
    ``trigger()``, so the recorded row holds the *second* fire's frame for
    both devices — never a mix of two physical shots.
    """
    cam_a = CaGenericDetector("U_CamA", ["Sig"], name="cam_a")
    cam_b = CaGenericDetector("U_CamB", ["Sig"], name="cam_b")
    cam_a.configure_shot_id(rep_rate_hz=1.0)
    cam_b.configure_shot_id(rep_rate_hz=1.0)
    events: list[dict] = []
    RE = RunEngine()
    RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)
    connect_mock(RE, cam_a, cam_b)
    for cam in (cam_a, cam_b):
        set_mock_value(cam.acq_timestamp, 1000.0)
        cam._trigger_timeout = 0.5

    calls = {"fire": 0}

    def fire():
        calls["fire"] += 1
        t = 1000.0 + calls["fire"]
        set_mock_value(cam_a.sig, float(calls["fire"]))  # payload tags the fire
        set_mock_value(cam_a.acq_timestamp, t)  # A frames every fire
        if calls["fire"] >= 2:  # B drops the first frame
            set_mock_value(cam_b.sig, float(calls["fire"]))
            set_mock_value(cam_b.acq_timestamp, t)
        yield from bps.null()

    RE(_single_shot_run([cam_a, cam_b], fire))

    assert calls["fire"] == 2
    assert len(events) == 1
    data = events[0]["data"]
    # Both devices' recorded frame is the second (successful) physical fire —
    # A's orphan frame (t=1001, sig=1.0) was drained, not read into the row.
    assert data["cam_a-acq_timestamp"] == 1002.0
    assert data["cam_b-acq_timestamp"] == 1002.0
    assert data["cam_a-sig"] == 2.0
    assert data["cam_b-sig"] == 2.0
    assert data["cam_a-shot_id"] == data["cam_b-shot_id"]
    assert data["cam_a-valid"] is True
    assert data["cam_b-valid"] is True


def test_single_shot_clean_shot_fires_once_no_warning(caplog) -> None:
    """No timeout → exactly one fire and no refire warnings (default path)."""
    with _setup_scan() as (_motor, cam, RE, events):
        calls: dict = {}

        with caplog.at_level(logging.WARNING, logger="geecs_bluesky.plans.single_shot"):
            RE(_single_shot_run([cam], _make_fire(cam, calls)))

        assert calls["fire"] == 1
        assert len(events) == 1
        assert events[0]["data"]["cam-valid"] is True
        assert not [
            r
            for r in caplog.records
            if r.name == "geecs_bluesky.plans.single_shot"
            and r.levelno >= logging.WARNING
        ]
