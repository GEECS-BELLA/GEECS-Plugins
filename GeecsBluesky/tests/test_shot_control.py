"""Unit tests for shot control arm/disarm — no real hardware required.

Covers:
- ShotController.set_state: empty-string values skipped, per-state dispatch
- geecs_step_scan: arm called after move, disarm after shots, per step
"""

from __future__ import annotations

from typing import Any

import pytest
from bluesky import RunEngine
from ophyd_async.core import AsyncStatus

from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.shot_controller import ShotController

pytest.importorskip("aioca")

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaGenericDetector, CaMotor  # noqa: E402
from tests.ca_mock_helpers import (  # noqa: E402
    connect_mock,
    follow_setpoint,
    start_pacer,
)

# Shot control config matching the real U_DG645_ShotControl YAML
SHOT_CONTROL_VARS = {
    "Trigger.ExecuteSingleShot": {
        "OFF": "",
        "SCAN": "",
        "SINGLESHOT": "on",
        "STANDBY": "",
    },
    "Trigger.Source": {
        "OFF": "Single shot external rising edges",
        "SCAN": "External rising edges",
        "STANDBY": "External rising edges",
    },
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class _MockSetter:
    """Records set() calls; no network needed."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[Any] = []

    def set(self, value: Any) -> AsyncStatus:
        self.calls.append(value)

        async def _noop() -> None:
            pass

        return AsyncStatus(_noop())


def _make_controller_with_mock_setters() -> tuple[
    RunEngine, ShotController, dict[str, _MockSetter]
]:
    """Build a ShotController with injected mock setters (no network)."""
    config = ShotControlConfig(
        device="U_DG645_ShotControl", variables=SHOT_CONTROL_VARS
    )
    mock_setters = {var: _MockSetter(var) for var in SHOT_CONTROL_VARS}
    return RunEngine(), ShotController(config, mock_setters), mock_setters


# ---------------------------------------------------------------------------
# _UdpSetter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ShotController.set_state
# ---------------------------------------------------------------------------


class TestSetTriggerState:
    def test_scan_state_skips_empty_variables(self) -> None:
        """SCAN state: Trigger.ExecuteSingleShot (empty) must be skipped."""
        re, controller, setters = _make_controller_with_mock_setters()
        re(controller.set_state("SCAN"))

        assert setters["Trigger.ExecuteSingleShot"].calls == []
        assert setters["Trigger.Source"].calls == ["External rising edges"]

    def test_standby_state_skips_empty_variables(self) -> None:
        """STANDBY state: same empty-value skipping as SCAN."""
        re, controller, setters = _make_controller_with_mock_setters()
        re(controller.set_state("STANDBY"))

        assert setters["Trigger.ExecuteSingleShot"].calls == []
        assert setters["Trigger.Source"].calls == ["External rising edges"]

    def test_singleshot_sets_execute_variable(self) -> None:
        """SINGLESHOT: ExecuteSingleShot gets 'on'; Source has no SINGLESHOT entry."""
        re, controller, setters = _make_controller_with_mock_setters()
        re(controller.set_state("SINGLESHOT"))

        assert setters["Trigger.ExecuteSingleShot"].calls == ["on"]
        assert setters["Trigger.Source"].calls == []

    def test_off_state_sets_source(self) -> None:
        """OFF state: Source set to single-shot mode string."""
        re, controller, setters = _make_controller_with_mock_setters()
        re(controller.set_state("OFF"))

        assert setters["Trigger.ExecuteSingleShot"].calls == []
        assert setters["Trigger.Source"].calls == ["Single shot external rising edges"]

    def test_state_with_no_setter_for_variable_is_skipped(self) -> None:
        """A state whose variable has no setter is skipped, not an error."""
        re, controller, setters = _make_controller_with_mock_setters()
        controller._setters.pop("Trigger.Source")
        re(controller.set_state("SCAN"))  # must not raise
        assert setters["Trigger.ExecuteSingleShot"].calls == []


# ---------------------------------------------------------------------------
# geecs_step_scan arm/disarm ordering
# ---------------------------------------------------------------------------


def _scan_pair() -> tuple[CaMotor, CaGenericDetector]:
    motor = CaMotor("U_Combined", "Position (mm)", name="test_motor")
    det = CaGenericDetector("U_Combined", ["Signal"], name="test_det")
    det.configure_shot_id(rep_rate_hz=1.0)
    return motor, det


class TestStepScanArmDisarmOrdering:
    def test_arm_disarm_ordering(self) -> None:
        """arm runs after move, disarm runs after shots — verified per step.

        With 2 positions × 2 shots:
          step 1: arm(events=0) → 2 shots → disarm(events=2)
          step 2: arm(events=2) → 2 shots → disarm(events=4)
        """
        motor, det = _scan_pair()
        events: list[dict] = []
        arm_at: list[int] = []
        disarm_at: list[int] = []

        def mock_arm():
            arm_at.append(len(events))
            yield from []

        def mock_disarm():
            disarm_at.append(len(events))
            yield from []

        RE = RunEngine()
        RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)
        connect_mock(RE, motor, det)
        follow_setpoint(motor)
        set_mock_value(det.acq_timestamp, 1000.0)
        pacer = start_pacer(RE, [(det, 1000.0)], initial_delay=0.2, interval=0.15)
        try:
            RE(
                geecs_step_scan(
                    motor=motor,
                    positions=[0.0, 1.0],
                    detectors=[det],
                    shots_per_step=2,
                    arm_trigger=mock_arm,
                    disarm_trigger=mock_disarm,
                )
            )
        finally:
            pacer.cancel()

        assert len(events) == 4, f"Expected 4 events, got {len(events)}"
        assert arm_at == [0, 2], f"arm called at wrong event counts: {arm_at}"
        assert disarm_at == [2, 4], f"disarm called at wrong event counts: {disarm_at}"

    def test_no_arm_disarm_still_collects_events(self) -> None:
        """arm_trigger=None runs normally — backward compat with internal trigger."""
        motor, det = _scan_pair()
        events: list[dict] = []
        RE = RunEngine()
        RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)
        connect_mock(RE, motor, det)
        follow_setpoint(motor)
        set_mock_value(det.acq_timestamp, 1000.0)
        pacer = start_pacer(RE, [(det, 1000.0)], initial_delay=0.2, interval=0.15)
        try:
            RE(
                geecs_step_scan(
                    motor=motor,
                    positions=[0.0],
                    detectors=[det],
                    shots_per_step=3,
                )
            )
        finally:
            pacer.cancel()

        assert len(events) == 3
