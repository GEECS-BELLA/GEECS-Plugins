"""Tests for geecs_data_utils.scan_log_loader (parser/loader layer)."""

from __future__ import annotations

import textwrap

import pytest

from geecs_data_utils.scan_log_loader import (
    HEADER_RE,
    Severity,
    parse_lines,
    parse_scan_log,
)


# ---------------------------------------------------------------------------
# Sample log text (matches the format string from logging_setup.py)
# ---------------------------------------------------------------------------

SIMPLE_LINE = (
    "2026-05-07 10:15:30.123 INFO geecs_scanner.scan_manager "
    "[MainThread] shot=- - Scan started"
)

ERROR_LINE = (
    "2026-05-07 10:16:00.456 ERROR geecs_scanner.scan_manager "
    "[ScanThread] shot=42 - Device timeout"
)

WARNING_LINE = (
    "2026-05-07 10:15:45.789 WARNING geecs_scanner.device_manager "
    "[MainThread] shot=- - Retrying connection"
)

TRACEBACK_BLOCK = textwrap.dedent(
    """\
    2026-05-07 10:17:00.001 ERROR geecs_scanner.scan_manager [ScanThread] shot=- - Uncaught exception
    Traceback (most recent call last):
      File "/path/to/geecs_scanner/scan_manager.py", line 100, in run_scan
        result = device.get_value()
      File "/path/to/geecs_scanner/device.py", line 55, in get_value
        return self.cache[key]
    KeyError: 'pressure'
    """
)


# ---------------------------------------------------------------------------
# Header regex
# ---------------------------------------------------------------------------


def test_header_re_matches_simple_line():
    m = HEADER_RE.match(SIMPLE_LINE)
    assert m is not None
    assert m.group("level") == "INFO"
    assert m.group("name") == "geecs_scanner.scan_manager"
    assert m.group("thread") == "MainThread"
    assert m.group("shot") == "-"
    assert m.group("msg") == "Scan started"


def test_header_re_matches_error_with_shot_id():
    m = HEADER_RE.match(ERROR_LINE)
    assert m is not None
    assert m.group("level") == "ERROR"
    assert m.group("shot") == "42"
    assert m.group("msg") == "Device timeout"


def test_header_re_no_match_on_traceback_continuation():
    assert HEADER_RE.match("Traceback (most recent call last):") is None
    assert HEADER_RE.match('  File "/foo.py", line 1, in bar') is None
    assert HEADER_RE.match("KeyError: 'pressure'") is None


# ---------------------------------------------------------------------------
# parse_lines — basic cases
# ---------------------------------------------------------------------------


def test_parse_lines_single_entry():
    entries = parse_lines([SIMPLE_LINE])
    assert len(entries) == 1
    e = entries[0]
    assert e.level == Severity.INFO
    assert e.logger_name == "geecs_scanner.scan_manager"
    assert e.thread_name == "MainThread"
    assert e.shot_id == "-"
    assert e.message == "Scan started"
    assert e.traceback is None


def test_parse_lines_multiple_entries():
    entries = parse_lines([SIMPLE_LINE, ERROR_LINE, WARNING_LINE])
    assert len(entries) == 3
    levels = [e.level for e in entries]
    assert levels == [Severity.INFO, Severity.ERROR, Severity.WARNING]


def test_parse_lines_multiline_traceback():
    entries = parse_lines(TRACEBACK_BLOCK.splitlines(keepends=True))
    assert len(entries) == 1
    e = entries[0]
    assert e.level == Severity.ERROR
    assert e.message == "Uncaught exception"
    assert e.traceback is not None
    assert "KeyError" in e.traceback
    assert "scan_manager.py" in e.traceback


def test_parse_lines_orphan_continuation_before_first_header():
    """Lines before the first header should be silently dropped."""
    lines = [
        "orphan line with no header\n",
        SIMPLE_LINE + "\n",
    ]
    entries = parse_lines(lines)
    assert len(entries) == 1


def test_parse_lines_empty():
    assert parse_lines([]) == []


def test_parse_lines_records_source_file(tmp_path):
    fake_path = tmp_path / "scan.log"
    entries = parse_lines([SIMPLE_LINE], source_file=fake_path)
    assert entries[0].source_file == fake_path
    assert entries[0].line_number == 1


# ---------------------------------------------------------------------------
# parse_scan_log — file I/O
# ---------------------------------------------------------------------------


def test_parse_scan_log_reads_file(tmp_path):
    log = tmp_path / "scan.log"
    log.write_text(SIMPLE_LINE + "\n" + ERROR_LINE + "\n", encoding="utf-8")
    entries = parse_scan_log(log)
    assert len(entries) == 2
    assert entries[0].level == Severity.INFO
    assert entries[1].level == Severity.ERROR


def test_parse_scan_log_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_scan_log(tmp_path / "nonexistent.log")


def test_parse_scan_log_sets_source_file(tmp_path):
    log = tmp_path / "scan.log"
    log.write_text(SIMPLE_LINE + "\n", encoding="utf-8")
    entries = parse_scan_log(log)
    assert entries[0].source_file == log


# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------


def test_severity_level_no_ordering():
    assert Severity.DEBUG.level_no < Severity.INFO.level_no
    assert Severity.INFO.level_no < Severity.WARNING.level_no
    assert Severity.WARNING.level_no < Severity.ERROR.level_no
    assert Severity.ERROR.level_no < Severity.CRITICAL.level_no
