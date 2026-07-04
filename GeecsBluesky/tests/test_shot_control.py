"""Unit tests for shot control arm/disarm — no real hardware required.

Covers:
- UdpSetter: string/numeric values sent correctly over fake UDP
- ShotController.set_state: empty-string values skipped, per-state dispatch
- geecs_step_scan: arm called after move, disarm after shots, per step
"""

from __future__ import annotations

from typing import Any

import pytest
from bluesky import RunEngine
from ophyd_async.core import AsyncStatus

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner
from geecs_bluesky.shot_controller import ShotController, UdpSetter
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer
from geecs_bluesky.transport.udp_client import GeecsUdpClient
from tests.fake_server_helpers import (
    BackgroundFakeServers,
    connect_devices,
    disconnect_devices,
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


@pytest.mark.fake_server
class TestUdpSetter:
    async def test_set_string_value(self) -> None:
        """set() delivers a string value to the fake device."""
        device = FakeGeecsDevice(
            name="U_DG645_ShotControl",
            variables={"Trigger.Source": "Single shot external rising edges"},
        )
        async with FakeGeecsServer(device) as srv:
            udp = GeecsUdpClient(srv.host, srv.port, device_name="U_DG645_ShotControl")
            await udp.connect()
            setter = UdpSetter(udp, "Trigger.Source")
            await setter.set("External rising edges")
            assert device.variables["Trigger.Source"] == "External rising edges"
            await udp.close()

    async def test_set_returns_async_status(self) -> None:
        device = FakeGeecsDevice(
            name="U_DG645_ShotControl",
            variables={"Trigger.Source": "Single shot external rising edges"},
        )
        async with FakeGeecsServer(device) as srv:
            udp = GeecsUdpClient(srv.host, srv.port, device_name="U_DG645_ShotControl")
            await udp.connect()
            setter = UdpSetter(udp, "Trigger.Source")
            status = setter.set("External rising edges")
            assert isinstance(status, AsyncStatus)
            await status
            await udp.close()

    async def test_set_numeric_value_sent_as_string(self) -> None:
        """Numeric values are stringified; the fake server coerces back to float."""
        device = FakeGeecsDevice(
            name="U_DG645_ShotControl",
            variables={"Delay": 0.0},
        )
        async with FakeGeecsServer(device) as srv:
            udp = GeecsUdpClient(srv.host, srv.port, device_name="U_DG645_ShotControl")
            await udp.connect()
            setter = UdpSetter(udp, "Delay")
            await setter.set(0.001)
            assert device.variables["Delay"] == pytest.approx(0.001)
            await udp.close()


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

    def test_no_controller_is_noop(self) -> None:
        """A scanner without a shot controller arms as an empty plan."""
        scanner = BlueskyScanner.__new__(BlueskyScanner)
        scanner._RE = RunEngine()
        scanner._shot_control = None
        scanner._shot_controller = None
        scanner._RE(scanner._arm_trigger())  # must not raise


# ---------------------------------------------------------------------------
# geecs_step_scan arm/disarm ordering
# ---------------------------------------------------------------------------


@pytest.fixture
def combined_device() -> FakeGeecsDevice:
    return FakeGeecsDevice(
        name="U_Combined",
        variables={
            "Position (mm)": 0.0,
            "Signal": 1.0,
            "acq_timestamp": 1000.0,
        },
    )


@pytest.mark.fake_server
class TestStepScanArmDisarmOrdering:
    def test_arm_disarm_ordering(self, combined_device: FakeGeecsDevice) -> None:
        """arm runs after move, disarm runs after shots — verified per step.

        With 2 positions × 2 shots:
          step 1: arm(events=0) → 2 shots → disarm(events=2)
          step 2: arm(events=2) → 2 shots → disarm(events=4)
        """
        with BackgroundFakeServers(
            combined_device,
            fire=lambda devices: devices[0].fire_shot(),
            interval=0.15,
        ) as server:
            host, port = server.endpoint

            motor = GeecsMotor(
                "U_Combined", "Position (mm)", host, port, name="test_motor"
            )
            det = GeecsGenericDetector(
                "U_Combined", ["Signal"], host, port, name="test_det"
            )

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
            RE.subscribe(
                lambda name, doc: events.append(doc) if name == "event" else None
            )

            connect_devices(RE, motor, det)
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
                disconnect_devices(RE, motor, det)

        assert len(events) == 4, f"Expected 4 events, got {len(events)}"
        assert arm_at == [0, 2], f"arm called at wrong event counts: {arm_at}"
        assert disarm_at == [2, 4], f"disarm called at wrong event counts: {disarm_at}"

    def test_no_arm_disarm_still_collects_events(
        self, combined_device: FakeGeecsDevice
    ) -> None:
        """arm_trigger=None runs normally — backward compat with internal trigger."""
        with BackgroundFakeServers(
            combined_device,
            fire=lambda devices: devices[0].fire_shot(),
            interval=0.15,
        ) as server:
            host, port = server.endpoint

            motor = GeecsMotor(
                "U_Combined", "Position (mm)", host, port, name="test_motor2"
            )
            det = GeecsGenericDetector(
                "U_Combined", ["Signal"], host, port, name="test_det2"
            )

            events: list[dict] = []
            RE = RunEngine()
            RE.subscribe(
                lambda name, doc: events.append(doc) if name == "event" else None
            )
            connect_devices(RE, motor, det)
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
                disconnect_devices(RE, motor, det)

        assert len(events) == 3
