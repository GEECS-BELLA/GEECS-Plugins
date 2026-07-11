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
        "progress": [],
        "error": [],
        "log": [],
    }
    adapter.state_changed.connect(record["state"].append)
    adapter.totals_known.connect(record["totals"].append)
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
