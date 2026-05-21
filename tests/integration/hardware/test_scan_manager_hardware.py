"""Hardware integration tests for Block 6 event emission.

These tests exercise the full scan event stream against live GEECS hardware.
They are gated by the ``GEECS_HARDWARE_AVAILABLE`` environment variable so they
are never executed in CI or on developer machines that are not connected to the
beamline network.

Run in the lab::

    export GEECS_HARDWARE_AVAILABLE=1
    pytest -m "integration and hardware" \\
           tests/integration/hardware/test_scan_manager_hardware.py -v

Hardware requirements
---------------------
- ``U_DG645_ShotControl``  — shot control / trigger device
- ``UC_TopView``           — detector (``acq_timestamp`` subscribed as scalar)
- ``U_ESP_JetXYZ``         — scan device (``Position.Axis 1``)
- Shot control config YAML at the experiment's ``timing_configs/`` path
- A save-element YAML that subscribes ``UC_TopView``

Expected event sequence for a 3-step scan (4→5, step 0.5)
----------------------------------------------------------
    ScanLifecycleEvent(INITIALIZING, total_shots=N)
    ScanLifecycleEvent(RUNNING)
    ScanStepEvent(step_index=0, phase="started")
    ScanStepEvent(step_index=0, phase="completed")
    ScanStepEvent(step_index=1, phase="started")
    ScanStepEvent(step_index=1, phase="completed")
    ScanStepEvent(step_index=2, phase="started")
    ScanStepEvent(step_index=2, phase="completed")
    ScanLifecycleEvent(DONE)
"""

from __future__ import annotations

import time
from typing import List

import pytest
import yaml

from geecs_scanner.engine import (
    ScanLifecycleEvent,
    ScanManager,
    ScanState,
    ScanStepEvent,
)
from geecs_scanner.engine.models.scan_options import ScanOptions
from geecs_scanner.engine.scan_events import ScanEvent
from geecs_scanner.utils import ApplicationPaths

pytestmark = [pytest.mark.integration, pytest.mark.hardware]

# ---------------------------------------------------------------------------
# Hardware configuration (Undulator experiment, HTU-Normal timing)
# ---------------------------------------------------------------------------

_EXPERIMENT = "Undulator"
_SHOT_CONTROL_YAML = "HTU-Normal.yaml"
_SCAN_DEVICE = "U_ESP_JetXYZ"
_SCAN_VARIABLE = "Position.Axis 1"
_SCAN_START = 4.0
_SCAN_END = 5.0
_SCAN_STEP = 0.5
_WAIT_TIME = 2.5  # seconds per step
_REP_RATE = 1.0  # Hz
_MASTER_CONTROL_IP = "192.168.7.203"

# Save-element YAML that subscribes UC_TopView (acq_timestamp).
# Adjust path if the experiment config layout differs.
_SAVE_ELEMENT_YAML = "UC_TopView.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shot_control_info(hardware_available):
    """Load the shot control YAML for the Undulator experiment."""
    app_paths = ApplicationPaths(experiment=_EXPERIMENT)
    yaml_path = app_paths.exp_shot_control / _SHOT_CONTROL_YAML
    if not yaml_path.exists():
        pytest.skip(f"Shot control config not found: {yaml_path}")
    with open(yaml_path) as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def save_element_path(hardware_available):
    """Return a Path to the UC_TopView save-element YAML."""
    app_paths = ApplicationPaths(experiment=_EXPERIMENT)
    yaml_path = app_paths.exp_save_devices / _SAVE_ELEMENT_YAML
    if not yaml_path.exists():
        pytest.skip(f"Save element config not found: {yaml_path}")
    return yaml_path


@pytest.fixture(scope="module")
def manager_with_events(hardware_available, shot_control_info):
    """Return a (ScanManager, events_list) pair ready for scan execution."""
    options = ScanOptions(rep_rate_hz=_REP_RATE, master_control_ip=_MASTER_CONTROL_IP)

    events: List[ScanEvent] = []
    mgr = ScanManager(
        experiment_dir=_EXPERIMENT,
        shot_control_information=shot_control_info,
        options=options,
        on_event=events.append,
    )
    return mgr, events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lifecycle(events: List[ScanEvent]) -> List[ScanLifecycleEvent]:
    return [e for e in events if isinstance(e, ScanLifecycleEvent)]


def _steps(events: List[ScanEvent]) -> List[ScanStepEvent]:
    return [e for e in events if isinstance(e, ScanStepEvent)]


def _wait_for_scan(mgr: ScanManager, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while mgr.is_scanning_active():
        if time.time() > deadline:
            mgr.stop_scanning_thread()
            pytest.fail(f"Scan did not finish within {timeout}s")
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScanLifecycleEvents:
    def test_initializing_running_done_sequence(
        self, manager_with_events, save_element_path
    ):
        mgr, events = manager_with_events
        events.clear()

        scan_config = {
            "device_var": f"{_SCAN_DEVICE}:{_SCAN_VARIABLE}",
            "start": _SCAN_START,
            "end": _SCAN_END,
            "step": _SCAN_STEP,
            "wait_time": _WAIT_TIME,
        }

        ok = mgr.reinitialize(config_path=save_element_path)
        assert ok, "ScanManager failed to reinitialize — check device connectivity"

        mgr.start_scan_thread(scan_config=scan_config)
        _wait_for_scan(mgr)

        states = [e.state for e in _lifecycle(events)]
        assert ScanState.INITIALIZING in states, f"No INITIALIZING event. Got: {states}"
        assert ScanState.RUNNING in states, f"No RUNNING event. Got: {states}"
        terminal = {ScanState.DONE, ScanState.ABORTED}
        assert any(s in terminal for s in states), (
            f"No terminal event (DONE/ABORTED). Got: {states}"
        )

    def test_initializing_carries_total_shots(
        self, manager_with_events, save_element_path
    ):
        mgr, events = manager_with_events
        events.clear()

        scan_config = {
            "device_var": f"{_SCAN_DEVICE}:{_SCAN_VARIABLE}",
            "start": _SCAN_START,
            "end": _SCAN_END,
            "step": _SCAN_STEP,
            "wait_time": _WAIT_TIME,
        }

        mgr.reinitialize(config_path=save_element_path)
        mgr.start_scan_thread(scan_config=scan_config)
        _wait_for_scan(mgr)

        init_events = [
            e for e in _lifecycle(events) if e.state == ScanState.INITIALIZING
        ]
        assert init_events, "No INITIALIZING event found"
        assert init_events[0].total_shots > 0, (
            "INITIALIZING event must carry total_shots > 0"
        )

    def test_state_transitions_are_ordered(
        self, manager_with_events, save_element_path
    ):
        """INITIALIZING must precede RUNNING; RUNNING must precede DONE/ABORTED."""
        mgr, events = manager_with_events
        events.clear()

        scan_config = {
            "device_var": f"{_SCAN_DEVICE}:{_SCAN_VARIABLE}",
            "start": _SCAN_START,
            "end": _SCAN_END,
            "step": _SCAN_STEP,
            "wait_time": _WAIT_TIME,
        }

        mgr.reinitialize(config_path=save_element_path)
        mgr.start_scan_thread(scan_config=scan_config)
        _wait_for_scan(mgr)

        states = [e.state for e in _lifecycle(events)]
        idx = {s: states.index(s) for s in states if s in states}

        if ScanState.INITIALIZING in idx and ScanState.RUNNING in idx:
            assert idx[ScanState.INITIALIZING] < idx[ScanState.RUNNING]

        running_idx = idx.get(ScanState.RUNNING)
        for terminal in (ScanState.DONE, ScanState.ABORTED):
            if terminal in idx and running_idx is not None:
                assert running_idx < idx[terminal]


class TestScanStepEvents:
    def test_step_events_emitted_for_each_step(
        self, manager_with_events, save_element_path
    ):
        """3-step scan (4→5 step 0.5) should emit 6 ScanStepEvents."""
        mgr, events = manager_with_events
        events.clear()

        scan_config = {
            "device_var": f"{_SCAN_DEVICE}:{_SCAN_VARIABLE}",
            "start": _SCAN_START,
            "end": _SCAN_END,
            "step": _SCAN_STEP,
            "wait_time": _WAIT_TIME,
        }

        mgr.reinitialize(config_path=save_element_path)
        mgr.start_scan_thread(scan_config=scan_config)
        _wait_for_scan(mgr)

        step_evts = _steps(events)
        assert len(step_evts) == 6, (
            f"Expected 6 ScanStepEvents (3 steps × 2 phases), got {len(step_evts)}"
        )

    def test_started_before_completed_per_step(
        self, manager_with_events, save_element_path
    ):
        mgr, events = manager_with_events
        events.clear()

        scan_config = {
            "device_var": f"{_SCAN_DEVICE}:{_SCAN_VARIABLE}",
            "start": _SCAN_START,
            "end": _SCAN_END,
            "step": _SCAN_STEP,
            "wait_time": _WAIT_TIME,
        }

        mgr.reinitialize(config_path=save_element_path)
        mgr.start_scan_thread(scan_config=scan_config)
        _wait_for_scan(mgr)

        phases = [e.phase for e in _steps(events)]
        assert phases == [
            "started",
            "completed",
            "started",
            "completed",
            "started",
            "completed",
        ]

    def test_shots_completed_is_non_decreasing(
        self, manager_with_events, save_element_path
    ):
        mgr, events = manager_with_events
        events.clear()

        scan_config = {
            "device_var": f"{_SCAN_DEVICE}:{_SCAN_VARIABLE}",
            "start": _SCAN_START,
            "end": _SCAN_END,
            "step": _SCAN_STEP,
            "wait_time": _WAIT_TIME,
        }

        mgr.reinitialize(config_path=save_element_path)
        mgr.start_scan_thread(scan_config=scan_config)
        _wait_for_scan(mgr)

        shots = [e.shots_completed for e in _steps(events)]
        for prev, curr in zip(shots, shots[1:]):
            assert curr >= prev, f"shots_completed went backwards: {shots}"
