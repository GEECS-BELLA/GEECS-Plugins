"""Tests for Block 6 event emission from DeviceCommandExecutor and ScanStepExecutor.

Exercises the on_event callback wiring added in Block 6.  All tests are
network-free: FakeScanDevice duck-types real hardware, and execute_step is
patched out so the loop tests exercise only the event-emission scaffolding.
"""

from __future__ import annotations

import threading
from typing import List
from unittest.mock import MagicMock

import pytest

from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceCommandFailed,
    GeecsDeviceCommandRejected,
    GeecsDeviceExeTimeout,
)
from geecs_scanner.engine.device_command_executor import DeviceCommandExecutor
from geecs_scanner.engine.models.scan_options import ScanOptions
from geecs_scanner.engine.scan_events import (
    DeviceCommandEvent,
    ScanEvent,
    ScanStepEvent,
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


def _step_events(events: List[ScanEvent]) -> List[ScanStepEvent]:
    return [e for e in events if isinstance(e, ScanStepEvent)]


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


# ---------------------------------------------------------------------------
# ScanStepExecutor — execute_scan_loop event sequence
# ---------------------------------------------------------------------------


def _make_loop_executor(monkeypatch, n_steps: int = 2, shots_per_step: int = 5):
    """Build a ScanStepExecutor with execute_step patched to a counter increment."""
    from geecs_scanner.engine.scan_executor import ScanStepExecutor

    stop_event = threading.Event()
    pause_event = threading.Event()
    pause_event.set()

    shot_counter = [0]
    data_logger = MagicMock()
    data_logger.get_current_shot.side_effect = lambda: shot_counter[0]

    executor = ScanStepExecutor(
        device_manager=MagicMock(),
        data_logger=data_logger,
        scan_data_manager=MagicMock(),
        options=ScanOptions(),
        stop_scanning_thread_event=stop_event,
        pause_scan_event=pause_event,
    )
    executor.cmd_executor = DeviceCommandExecutor()

    events: List[ScanEvent] = []
    executor.on_event = events.append

    def _fake_step(step, index):
        shot_counter[0] += shots_per_step

    monkeypatch.setattr(executor, "execute_step", _fake_step)

    steps = [{"variables": {}, "is_composite": False, "wait_time": 0.1}] * n_steps
    executor.execute_scan_loop(steps)
    return events


class TestScanLoopEventSequence:
    def test_two_events_per_step(self, monkeypatch):
        events = _make_loop_executor(monkeypatch, n_steps=3)
        assert len(_step_events(events)) == 6  # started + completed × 3 steps

    def test_started_precedes_completed(self, monkeypatch):
        events = _make_loop_executor(monkeypatch, n_steps=2)
        phases = [e.phase for e in _step_events(events)]
        assert phases == ["started", "completed", "started", "completed"]

    def test_step_index_increments(self, monkeypatch):
        events = _make_loop_executor(monkeypatch, n_steps=3)
        started = [e for e in _step_events(events) if e.phase == "started"]
        assert [e.step_index for e in started] == [0, 1, 2]

    def test_total_steps_on_every_event(self, monkeypatch):
        events = _make_loop_executor(monkeypatch, n_steps=4)
        assert all(e.total_steps == 4 for e in _step_events(events))

    def test_shots_completed_progresses(self, monkeypatch):
        events = _make_loop_executor(monkeypatch, n_steps=2, shots_per_step=5)
        phases_shots = [(e.phase, e.shots_completed) for e in _step_events(events)]
        # started[0]=0, completed[0]=5, started[1]=5, completed[1]=10
        assert phases_shots[0] == ("started", 0)
        assert phases_shots[1] == ("completed", 5)
        assert phases_shots[2] == ("started", 5)
        assert phases_shots[3] == ("completed", 10)

    def test_pre_set_stop_event_skips_all_steps(self):
        """Loop should emit no step events when stop is already set on entry."""
        from geecs_scanner.engine.scan_executor import ScanStepExecutor

        stop_event = threading.Event()
        stop_event.set()
        pause_event = threading.Event()
        pause_event.set()

        data_logger = MagicMock()
        data_logger.get_current_shot.return_value = 0

        executor = ScanStepExecutor(
            device_manager=MagicMock(),
            data_logger=data_logger,
            scan_data_manager=MagicMock(),
            options=ScanOptions(),
            stop_scanning_thread_event=stop_event,
            pause_scan_event=pause_event,
        )
        executor.cmd_executor = DeviceCommandExecutor()

        events: List[ScanEvent] = []
        executor.on_event = events.append

        steps = [{"variables": {}, "is_composite": False, "wait_time": 0.0}] * 3
        executor.execute_scan_loop(steps)

        assert _step_events(events) == []

    def test_on_event_none_does_not_raise(self, monkeypatch):
        """Executor with no on_event callback must complete without error."""
        from geecs_scanner.engine.scan_executor import ScanStepExecutor

        stop_event = threading.Event()
        pause_event = threading.Event()
        pause_event.set()

        data_logger = MagicMock()
        data_logger.get_current_shot.return_value = 0

        executor = ScanStepExecutor(
            device_manager=MagicMock(),
            data_logger=data_logger,
            scan_data_manager=MagicMock(),
            options=ScanOptions(),
            stop_scanning_thread_event=stop_event,
            pause_scan_event=pause_event,
        )
        executor.cmd_executor = DeviceCommandExecutor()
        # on_event deliberately left as None (the default)

        monkeypatch.setattr(executor, "execute_step", lambda step, idx: None)

        steps = [{"variables": {}, "is_composite": False, "wait_time": 0.0}]
        executor.execute_scan_loop(steps)  # must not raise
