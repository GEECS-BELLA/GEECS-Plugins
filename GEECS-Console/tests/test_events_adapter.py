"""Fake ScanEvents -> Qt signal emissions (same-thread, hermetic)."""

import pytest

from fake_events import (
    ScanDialogEvent,
    ScanErrorEvent,
    ScanLifecycleEvent,
    ScanRestoreFailedEvent,
    ScanStepEvent,
    _Request,
)
from geecs_console.events_adapter import ScanEventsAdapter


@pytest.fixture
def adapter(qapp):
    return ScanEventsAdapter()


@pytest.fixture
def recorded(adapter):
    record = {
        "state": [],
        "totals": [],
        "scan_numbers": [],
        "progress": [],
        "error": [],
        "log": [],
    }
    adapter.state_changed.connect(record["state"].append)
    adapter.totals_known.connect(record["totals"].append)
    adapter.scan_number_known.connect(record["scan_numbers"].append)
    adapter.progress.connect(lambda *args: record["progress"].append(args))
    adapter.error.connect(record["error"].append)
    adapter.log_line.connect(record["log"].append)
    return record


class TestLifecycle:
    def test_state_and_totals(self, adapter, recorded):
        adapter.handle(ScanLifecycleEvent(state="initializing", total_shots=50))
        adapter.handle(ScanLifecycleEvent(state="running"))
        assert recorded["state"] == ["initializing", "running"]
        assert recorded["totals"] == [50]  # only the INITIALIZING event carries totals

    def test_scan_number_emitted_only_when_present(self, adapter, recorded):
        # Before the folder claim the engine emits None — no signal.
        adapter.handle(ScanLifecycleEvent(state="initializing", total_shots=50))
        assert recorded["scan_numbers"] == []
        # After the claim every lifecycle emission carries the number.
        adapter.handle(ScanLifecycleEvent(state="running", scan_number=42))
        adapter.handle(ScanLifecycleEvent(state="done", scan_number=42))
        assert recorded["scan_numbers"] == [42, 42]

    def test_scan_number_absent_attribute_is_tolerated(self, adapter, recorded):
        """Older engines without the field must not crash the adapter."""

        class ScanLifecycleEvent:  # no scan_number attribute at all
            state = "running"
            total_shots = 0

        adapter.handle(ScanLifecycleEvent())
        assert recorded["scan_numbers"] == []
        assert recorded["state"] == ["running"]

    def test_enum_valued_state_uses_value(self, adapter, recorded):
        from enum import Enum

        class ScanState(str, Enum):
            DONE = "done"

        adapter.handle(ScanLifecycleEvent(state=ScanState.DONE))
        assert recorded["state"] == ["done"]


class TestStepAndErrors:
    def test_step_progress_tuple(self, adapter, recorded):
        adapter.handle(
            ScanStepEvent(
                step_index=2, total_steps=5, shots_completed=30, phase="completed"
            )
        )
        assert recorded["progress"] == [(2, 5, 30)]
        assert "step 3/5 completed" in recorded["log"][0]

    def test_error_event(self, adapter, recorded):
        adapter.handle(ScanErrorEvent(message="device died", recoverable=False))
        assert recorded["error"] == ["device died"]

    def test_restore_failure_logged(self, adapter, recorded):
        adapter.handle(ScanRestoreFailedEvent(device="U_JetXYZ", message="timeout"))
        assert recorded["log"] == ["restore failed: U_JetXYZ: timeout"]

    def test_dialog_event_logged_not_crashed(self, adapter, recorded):
        adapter.handle(ScanDialogEvent(request=_Request(exc=RuntimeError("boom"))))
        assert "operator question" in recorded["log"][0]

    def test_unknown_event_falls_back_to_class_name(self, adapter, recorded):
        class SomethingNewEvent:
            pass

        adapter.handle(SomethingNewEvent())
        assert recorded["log"] == ["SomethingNewEvent"]


class TestNarrationDedupe:
    """The log tail is narration: consecutive identical lines collapse
    (Sam's 2026-08-19 Scan004 showed 'scan running' x2 and the final step
    line x2); the data signals always emit."""

    def test_rerun_running_with_scan_number_narrates_the_number(
        self, adapter, recorded
    ):
        adapter.handle(ScanLifecycleEvent(state="running"))
        adapter.handle(ScanLifecycleEvent(state="running", scan_number=4))
        assert recorded["log"] == ["scan running", "scan running (Scan004)"]
        # The data signal is not deduped.
        assert recorded["state"] == ["running", "running"]

    def test_identical_lifecycle_narrates_once(self, adapter, recorded):
        adapter.handle(ScanLifecycleEvent(state="running"))
        adapter.handle(ScanLifecycleEvent(state="running"))
        assert recorded["log"] == ["scan running"]
        assert recorded["state"] == ["running", "running"]

    def test_repeated_final_step_event_narrates_once(self, adapter, recorded):
        adapter.handle(
            ScanStepEvent(
                step_index=0, total_steps=1, shots_completed=10, phase="completed"
            )
        )
        adapter.handle(
            ScanStepEvent(
                step_index=0, total_steps=1, shots_completed=10, phase="completed"
            )
        )
        assert recorded["log"] == ["step 1/1 completed (10 shots)"]
        # Progress still emitted for both events.
        assert recorded["progress"] == [(0, 1, 10), (0, 1, 10)]

    def test_alternating_lines_are_not_suppressed(self, adapter, recorded):
        adapter.handle(
            ScanStepEvent(
                step_index=0, total_steps=1, shots_completed=1, phase="completed"
            )
        )
        adapter.handle(
            ScanStepEvent(
                step_index=0, total_steps=1, shots_completed=2, phase="completed"
            )
        )
        assert recorded["log"] == [
            "step 1/1 completed (1 shots)",
            "step 1/1 completed (2 shots)",
        ]
