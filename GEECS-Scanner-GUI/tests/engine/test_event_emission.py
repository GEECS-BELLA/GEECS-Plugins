"""Tests for event emission from DeviceCommandExecutor.

Exercises the on_event callback wiring.  All tests are network-free:
FakeScanDevice duck-types real hardware.

The ScanStepExecutor / ScanManager / ScanLifecycleStateMachine emission
tests that used to live here were deleted with the legacy scan engine (G1
of the greenfield cutover); the event vocabulary itself is pinned in
GeecsBluesky's suite and in ``tests/engine/test_event_shims.py``.
"""

from __future__ import annotations

from typing import List

import pytest

from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceCommandFailed,
    GeecsDeviceCommandRejected,
    GeecsDeviceExeTimeout,
)
from geecs_scanner.engine.device_command_executor import DeviceCommandExecutor
from geecs_scanner.engine.scan_events import (
    DeviceCommandEvent,
    ScanEvent,
)
from tests.engine.conftest import FakeScanDevice


# ---------------------------------------------------------------------------
# Extra fake devices (extend FakeScanDevice with get() and always-fail modes)
# ---------------------------------------------------------------------------


class _AlwaysRejectDevice(FakeScanDevice):
    def set(self, variable, value, **kw):
        self._set_call_count += 1
        raise GeecsDeviceCommandRejected(self.name, f"set {variable}")


class _AlwaysTimeoutDevice(FakeScanDevice):
    def set(self, variable, value, **kw):
        self._set_call_count += 1
        raise GeecsDeviceExeTimeout(self.name, f"set {variable}", timeout=1.0)

    def get(self, variable):
        raise GeecsDeviceExeTimeout(self.name, f"get {variable}", timeout=1.0)


class _AlwaysFailDevice(FakeScanDevice):
    def set(self, variable, value, **kw):
        self._set_call_count += 1
        raise GeecsDeviceCommandFailed(self.name, f"set {variable}", "injected")

    def get(self, variable):
        raise GeecsDeviceCommandFailed(self.name, f"get {variable}", "injected")


class _HappyGetDevice(FakeScanDevice):
    def get(self, variable):
        return 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cmd_events(events: List[ScanEvent]) -> List[DeviceCommandEvent]:
    return [e for e in events if isinstance(e, DeviceCommandEvent)]


# ---------------------------------------------------------------------------
# DeviceCommandExecutor — set() outcomes
# ---------------------------------------------------------------------------


class TestDeviceCommandSetEvents:
    def _exe(self, events: List) -> DeviceCommandExecutor:
        return DeviceCommandExecutor(
            max_retries=3, retry_delay=0.0, on_event=events.append
        )

    def test_accepted_emits_event(self):
        events: List[ScanEvent] = []
        self._exe(events).set(FakeScanDevice("D"), "V", 1.0)

        cmds = _cmd_events(events)
        assert len(cmds) == 1
        assert cmds[0].device == "D"
        assert cmds[0].variable == "V"
        assert cmds[0].outcome == "accepted"

    def test_rejected_exhausted_emits_rejected_outcome(self):
        events: List[ScanEvent] = []
        exe = self._exe(events)
        with pytest.raises(Exception):
            exe.set(_AlwaysRejectDevice("D"), "V", 1.0)

        outcomes = {e.outcome for e in _cmd_events(events)}
        assert "rejected" in outcomes

    def test_timeout_emits_timeout_outcome(self):
        events: List[ScanEvent] = []
        exe = self._exe(events)
        with pytest.raises(Exception):
            exe.set(_AlwaysTimeoutDevice("D"), "V", 1.0)

        outcomes = {e.outcome for e in _cmd_events(events)}
        assert "timeout" in outcomes

    def test_failed_emits_failed_outcome(self):
        events: List[ScanEvent] = []
        exe = self._exe(events)
        with pytest.raises(Exception):
            exe.set(_AlwaysFailDevice("D"), "V", 1.0)

        outcomes = {e.outcome for e in _cmd_events(events)}
        assert "failed" in outcomes

    def test_event_carries_device_and_variable_names(self):
        events: List[ScanEvent] = []
        self._exe(events).set(FakeScanDevice("MyDevice"), "Position.X", 42.0)

        ev = _cmd_events(events)[0]
        assert ev.device == "MyDevice"
        assert ev.variable == "Position.X"

    def test_event_timestamp_is_positive(self):
        events: List[ScanEvent] = []
        self._exe(events).set(FakeScanDevice("D"), "V", 0.0)
        assert _cmd_events(events)[0].timestamp > 0

    def test_no_event_emitted_when_no_callback(self):
        # Regression: executor must not crash when on_event is None
        exe = DeviceCommandExecutor(max_retries=1, retry_delay=0.0)
        exe.set(FakeScanDevice("D"), "V", 1.0)  # must not raise


# ---------------------------------------------------------------------------
# DeviceCommandExecutor — get() outcomes
# ---------------------------------------------------------------------------


class TestDeviceCommandGetEvents:
    def _exe(self, events: List) -> DeviceCommandExecutor:
        return DeviceCommandExecutor(on_event=events.append)

    def test_get_accepted_emits_event(self):
        events: List[ScanEvent] = []
        self._exe(events).get(_HappyGetDevice("D"), "V")

        cmds = _cmd_events(events)
        assert cmds
        assert cmds[0].outcome == "accepted"

    def test_get_timeout_emits_timeout_outcome(self):
        events: List[ScanEvent] = []
        exe = self._exe(events)
        with pytest.raises(Exception):
            exe.get(_AlwaysTimeoutDevice("D"), "V")

        outcomes = {e.outcome for e in _cmd_events(events)}
        assert "timeout" in outcomes

    def test_get_failed_emits_failed_outcome(self):
        events: List[ScanEvent] = []
        exe = self._exe(events)
        with pytest.raises(Exception):
            exe.get(_AlwaysFailDevice("D"), "V")

        outcomes = {e.outcome for e in _cmd_events(events)}
        assert "failed" in outcomes
