"""Tests for geecs_log_triage.fingerprint — normalization stability and hash consistency."""

from __future__ import annotations

import textwrap
from datetime import datetime

from geecs_data_utils.scan_log_loader import LogEntry, Severity
from geecs_log_triage.fingerprint import fingerprint_for, normalize_message
from geecs_log_triage.schemas import Classification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    message: str,
    level: Severity = Severity.ERROR,
    logger_name: str = "geecs_scanner.scan_manager",
    traceback: str | None = None,
    shot_id: str = "-",
) -> LogEntry:
    return LogEntry(
        timestamp=datetime(2026, 5, 7, 10, 0, 0),
        level=level,
        logger_name=logger_name,
        thread_name="MainThread",
        shot_id=shot_id,
        message=message,
        traceback=traceback,
    )


# ---------------------------------------------------------------------------
# normalize_message
# ---------------------------------------------------------------------------


def test_normalize_strips_digit_runs():
    assert "N" in normalize_message("scan 42 failed")
    assert "42" not in normalize_message("scan 42 failed")


def test_normalize_strips_absolute_path():
    norm = normalize_message("/home/user/data/scan37/file.txt")
    assert "/home/user/data/scan37/" not in norm
    assert "file.txt" in norm


def test_normalize_strips_hex_tokens():
    norm = normalize_message("device 0xdeadbeef not responding")
    assert "0xdeadbeef" not in norm
    assert "X" in norm


def test_normalize_strips_trailing_punct():
    norm = normalize_message("failed to connect:")
    assert not norm.endswith(":")


def test_normalize_same_message_different_scan_numbers():
    """Two messages differing only in scan/shot number should normalize identically."""
    n1 = normalize_message("Scan 10 aborted after shot 5")
    n2 = normalize_message("Scan 99 aborted after shot 7")
    assert n1 == n2


# ---------------------------------------------------------------------------
# fingerprint_for — no traceback path
# ---------------------------------------------------------------------------


def test_fingerprint_no_traceback_same_message_same_hash():
    e1 = _make_entry("Device timeout on port 5555")
    e2 = _make_entry("Device timeout on port 6666")
    fp1 = fingerprint_for(e1)
    fp2 = fingerprint_for(e2)
    assert fp1.fingerprint_hash == fp2.fingerprint_hash


def test_fingerprint_no_traceback_different_message_different_hash():
    e1 = _make_entry("Device timeout")
    e2 = _make_entry("Scan aborted by operator")
    fp1 = fingerprint_for(e1)
    fp2 = fingerprint_for(e2)
    assert fp1.fingerprint_hash != fp2.fingerprint_hash


def test_fingerprint_hash_length():
    e = _make_entry("some message")
    fp = fingerprint_for(e)
    assert len(fp.fingerprint_hash) == 12


# ---------------------------------------------------------------------------
# fingerprint_for — traceback path
# ---------------------------------------------------------------------------

_KEYERROR_TB = textwrap.dedent(
    """\
    Traceback (most recent call last):
      File "/path/to/geecs_scanner/scan_manager.py", line 100, in run_scan
        result = device.get_value()
      File "/path/to/geecs_scanner/device.py", line 55, in get_value
        return self.cache[key]
    KeyError: 'pressure'
    """
)

_KEYERROR_TB_DIFFERENT_LINE = textwrap.dedent(
    """\
    Traceback (most recent call last):
      File "/path/to/geecs_scanner/scan_manager.py", line 200, in run_scan
        result = device.get_value()
      File "/path/to/geecs_scanner/device.py", line 55, in get_value
        return self.cache[key]
    KeyError: 'pressure'
    """
)


def test_fingerprint_traceback_same_type_file_func_same_hash():
    """Same exception type and user frame → same hash despite different line numbers."""
    e1 = _make_entry("Uncaught exception", traceback=_KEYERROR_TB)
    e2 = _make_entry("Uncaught exception", traceback=_KEYERROR_TB_DIFFERENT_LINE)
    fp1 = fingerprint_for(e1)
    fp2 = fingerprint_for(e2)
    assert fp1.fingerprint_hash == fp2.fingerprint_hash


def test_fingerprint_traceback_exception_type_extracted():
    e = _make_entry("Uncaught exception", traceback=_KEYERROR_TB)
    fp = fingerprint_for(e)
    assert fp.exception_type == "KeyError"


def test_fingerprint_traceback_sample_traceback_stored():
    e = _make_entry("Uncaught exception", traceback=_KEYERROR_TB)
    fp = fingerprint_for(e)
    assert fp.sample_traceback is not None
    assert "KeyError" in fp.sample_traceback


def test_fingerprint_classification_stored():
    e = _make_entry("some message")
    fp = fingerprint_for(e, classification=Classification.BUG_CANDIDATE)
    assert fp.classification == Classification.BUG_CANDIDATE


def test_fingerprint_traceback_different_exception_type_different_hash():
    attr_tb = _KEYERROR_TB.replace("KeyError: 'pressure'", "AttributeError: 'foo'")
    e1 = _make_entry("error", traceback=_KEYERROR_TB)
    e2 = _make_entry("error", traceback=attr_tb)
    fp1 = fingerprint_for(e1)
    fp2 = fingerprint_for(e2)
    assert fp1.fingerprint_hash != fp2.fingerprint_hash
