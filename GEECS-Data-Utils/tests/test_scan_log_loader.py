"""Tests for the scan.log parser and its format constants."""

from __future__ import annotations

import logging

from geecs_data_utils import (
    SCAN_LOG_DATEFMT,
    SCAN_LOG_FORMAT,
    parse_lines,
    parse_scan_log,
)
from geecs_data_utils.scan_log_loader import HEADER_RE


def _format_one_record() -> str:
    """Render a single LogRecord through the canonical scan-log formatter."""
    formatter = logging.Formatter(SCAN_LOG_FORMAT, datefmt=SCAN_LOG_DATEFMT)
    record = logging.LogRecord(
        name="geecs_bluesky.demo",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.threadName = "bluesky-scan"
    record.shot_id = "-"
    return formatter.format(record)


def test_format_constants_match_header_regex() -> None:
    """A line built from SCAN_LOG_FORMAT must match the parser's HEADER_RE.

    This is the contract guarding writers (legacy scanner, Bluesky backend)
    against drift from the reader.
    """
    line = _format_one_record()
    assert HEADER_RE.match(line) is not None


def test_format_constants_roundtrip_fields() -> None:
    line = _format_one_record()
    [entry] = parse_lines([line])
    assert entry.logger_name == "geecs_bluesky.demo"
    assert entry.thread_name == "bluesky-scan"
    assert entry.shot_id == "-"
    assert entry.message == "hello world"
    assert entry.level.value == "INFO"


def test_parse_scan_log_from_disk(tmp_path) -> None:
    log = tmp_path / "scan.log"
    log.write_text(_format_one_record() + "\n", encoding="utf-8")
    [entry] = parse_scan_log(log)
    assert entry.message == "hello world"
    assert entry.source_file == log
