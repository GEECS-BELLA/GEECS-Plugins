"""Tests for geecs_log_triage.classifier."""

from __future__ import annotations

from datetime import datetime

import pytest

from geecs_data_utils.scan_log_loader import LogEntry, Severity
from geecs_log_triage.classifier import classify
from geecs_log_triage.schemas import Classification


def _entry(message: str, level: Severity = Severity.ERROR) -> LogEntry:
    return LogEntry(
        timestamp=datetime(2026, 5, 7, 10, 0, 0),
        level=level,
        logger_name="geecs_scanner.scan_manager",
        thread_name="MainThread",
        shot_id="-",
        message=message,
    )


# ---------------------------------------------------------------------------
# Exception-type map lookups (explicit exc_type argument)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_type, expected",
    [
        ("KeyError", Classification.BUG_CANDIDATE),
        ("AttributeError", Classification.BUG_CANDIDATE),
        ("TypeError", Classification.BUG_CANDIDATE),
        ("ValueError", Classification.BUG_CANDIDATE),
        ("AssertionError", Classification.BUG_CANDIDATE),
        ("ActionError", Classification.CONFIG_ISSUE),
        ("ConflictingScanElements", Classification.CONFIG_ISSUE),
        ("ValidationError", Classification.CONFIG_ISSUE),
        ("GeecsDeviceInstantiationError", Classification.HARDWARE_ISSUE),
        ("ConnectionRefusedError", Classification.HARDWARE_ISSUE),
        ("TimeoutError", Classification.HARDWARE_ISSUE),
        ("FileNotFoundError", Classification.OPERATOR_ERROR),
        ("PermissionError", Classification.OPERATOR_ERROR),
    ],
)
def test_classification_map_exc_type(exc_type: str, expected: Classification):
    e = _entry("some message")
    assert classify(e, exception_type=exc_type) == expected


# ---------------------------------------------------------------------------
# Message-body fallback (exc_type absent or unknown)
# ---------------------------------------------------------------------------


def test_classify_uncaught_exception_in_message():
    e = _entry("Uncaught exception in scan thread")
    assert classify(e) == Classification.BUG_CANDIDATE


def test_classify_subscription_failed_in_message():
    e = _entry("subscription failed for device XYZ")
    assert classify(e) == Classification.HARDWARE_ISSUE


def test_classify_scan_aborted_in_message():
    e = _entry("scan aborted by operator request")
    assert classify(e) == Classification.OPERATOR_ERROR


def test_classify_exc_type_in_message_body():
    """Exception type token embedded in message body triggers map lookup."""
    e = _entry("Failed to connect: ConnectionRefusedError")
    assert classify(e) == Classification.HARDWARE_ISSUE


# ---------------------------------------------------------------------------
# Unknown fallback
# ---------------------------------------------------------------------------


def test_classify_unknown_when_no_match():
    e = _entry("something completely unrecognised happened")
    assert classify(e) == Classification.UNKNOWN


def test_classify_unknown_exc_type_falls_back_to_message():
    e = _entry("something completely unrecognised happened")
    assert classify(e, exception_type="SomeObscureError") == Classification.UNKNOWN


# ---------------------------------------------------------------------------
# #621: NotConnectedError + the expected-soft-drop guard
# ---------------------------------------------------------------------------


def test_notconnectederror_is_a_hardware_issue():
    """A bare CA connect failure classifies as hardware, never unknown."""
    entry = _entry("device connect failed")
    assert classify(entry, "NotConnectedError") is Classification.HARDWARE_ISSUE


def test_engine_tolerated_soft_drop_is_expected_condition():
    """The engine's own soft-tier drop WARNING (GeecsBluesky
    session.telemetry) marks a tolerated-by-design condition — it must not
    file as a per-scan hardware issue even though the record carries a
    NotConnectedError traceback (the guard precedes the type map)."""
    entry = _entry(
        "Dropping background-telemetry device U_GhostLowPowerWFS: "
        "unreachable at scan start (soft tier — never aborts the scan)"
    )
    assert classify(entry, "NotConnectedError") is Classification.EXPECTED_CONDITION
    assert classify(entry) is Classification.EXPECTED_CONDITION


def test_unrelated_hardware_error_not_downranked():
    """The guard keys on the engine's exact tolerated-drop phrase — a real
    hardware failure mentioning a device stays a hardware issue."""
    entry = _entry("device not responding: U_Amp4_IR_input")
    assert classify(entry, "NotConnectedError") is Classification.HARDWARE_ISSUE


def test_dotted_exception_type_normalizes_for_the_map():
    """Real tracebacks print non-builtins fully qualified — the map lookup
    must normalize (#680 review finding: the exact-key match was dead code
    for every record the harvester can produce)."""
    entry = _entry("device connect failed")
    assert (
        classify(entry, "ophyd_async.core._utils.NotConnectedError")
        is Classification.HARDWARE_ISSUE
    )


def test_dotted_type_end_to_end_through_the_fingerprint_extractor():
    """The honest pipeline test: a traceback rendered the way ophyd-async
    renders it → signature extraction → classification."""
    from geecs_log_triage.fingerprint import _extract_traceback_signature

    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "connector.py", line 10, in connect\n'
        "    await device.connect()\n"
        "ophyd_async.core._utils.NotConnectedError: device connect failed\n"
    )
    exc_type = _extract_traceback_signature(traceback_text)[0]
    assert exc_type == "ophyd_async.core._utils.NotConnectedError"
    assert classify(_entry("boom"), exc_type) is Classification.HARDWARE_ISSUE


def test_expected_drop_phrase_matches_the_engine_source():
    """Bidirectional pin of the cross-package phrase coupling: when the
    sibling GeecsBluesky checkout is present (the monorepo layout), the
    engine's SOFT_TELEMETRY_DROP_MSG must contain the classifier's guard
    phrase — a wording pass in either package fails this test instead of
    silently reverting ghost devices to per-scan noise."""
    from pathlib import Path

    session_py = (
        Path(__file__).resolve().parents[2]
        / "GeecsBluesky"
        / "geecs_bluesky"
        / "session.py"
    )
    if not session_py.is_file():
        pytest.skip("standalone checkout — no sibling GeecsBluesky source")
    source = session_py.read_text(encoding="utf-8")
    assert "SOFT_TELEMETRY_DROP_MSG" in source
    assert "soft tier — never aborts the scan" in source
