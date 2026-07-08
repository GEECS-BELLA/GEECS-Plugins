"""Pin the re-export shims: geecs_scanner event types ARE geecs_bluesky's.

The event vocabulary moved down to ``geecs_bluesky.events`` (engine
consolidation milestone); ``geecs_scanner.engine.scan_events`` and
``geecs_scanner.engine.dialog_request`` re-export the same class objects so
every existing import path — and every ``isinstance`` check across the
GUI/engine boundary — keeps working verbatim.
"""

from __future__ import annotations

import geecs_bluesky.events as engine_events
from geecs_scanner.engine import dialog_request, scan_events


def test_scan_event_classes_are_identical_objects() -> None:
    for name in (
        "ScanState",
        "ScanEvent",
        "ScanLifecycleEvent",
        "ScanStepEvent",
        "DeviceCommandEvent",
        "ScanErrorEvent",
        "ScanRestoreFailedEvent",
        "ScanDialogEvent",
    ):
        assert getattr(scan_events, name) is getattr(engine_events, name), name


def test_dialog_request_is_the_same_class_object() -> None:
    assert dialog_request.DialogRequest is engine_events.DialogRequest


def test_engine_package_reexports_survive() -> None:
    from geecs_scanner import engine

    assert engine.ScanLifecycleEvent is engine_events.ScanLifecycleEvent
    assert engine.ScanState is engine_events.ScanState


def test_legacy_residue_stays_in_the_shim() -> None:
    """DEVICE_COMMAND_ERRORS / escalate_device_error (geecs_python_api deps)."""
    assert len(dialog_request.DEVICE_COMMAND_ERRORS) == 3
    assert dialog_request.escalate_device_error(RuntimeError("x"), None) is True


def test_dialog_request_semantics_preserved() -> None:
    """Field defaults exactly as before the move (positional exc, containers)."""
    exc = RuntimeError("boom")
    request = dialog_request.DialogRequest(exc)
    assert request.exc is exc
    assert request.context is None
    assert request.title is None
    assert request.continue_label is None
    assert request.abort_label is None
    assert request.abort == [False]
    assert not request.response_event.is_set()
