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
