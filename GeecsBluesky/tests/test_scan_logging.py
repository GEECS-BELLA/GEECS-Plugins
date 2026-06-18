"""Tests for the per-scan scan.log file (geecs_bluesky.scan_logging)."""

from __future__ import annotations

import logging

import pytest

from geecs_bluesky.scan_logging import SCAN_LOG_FILENAME, scan_log_file
from geecs_data_utils import parse_scan_log


def test_writes_scan_log_in_folder(tmp_path) -> None:
    with scan_log_file(tmp_path) as log_path:
        assert log_path == tmp_path / SCAN_LOG_FILENAME
        logging.getLogger("geecs_bluesky.demo").info("hello scan")
    assert (tmp_path / SCAN_LOG_FILENAME).is_file()


def test_roundtrip_parses_with_triage_loader(tmp_path) -> None:
    """A record written through the handler parses back via the triage loader.

    This pins the writer/parser contract: the format string and HEADER_RE both
    live in geecs_data_utils, and this test fails if they ever drift.
    """
    with scan_log_file(tmp_path):
        logging.getLogger("geecs_bluesky.demo").warning("something happened")

    entries = parse_scan_log(tmp_path / SCAN_LOG_FILENAME)
    msgs = {e.message: e for e in entries}
    assert "something happened" in msgs
    rec = msgs["something happened"]
    assert rec.level.value == "WARNING"
    assert rec.logger_name == "geecs_bluesky.demo"
    assert rec.shot_id == "-"  # default injected when no shot context
    assert rec.thread_name  # populated from the record


def test_captures_tracebacks(tmp_path) -> None:
    with scan_log_file(tmp_path):
        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger("geecs_bluesky.demo").exception("caught it")

    entries = parse_scan_log(tmp_path / SCAN_LOG_FILENAME)
    rec = next(e for e in entries if e.message == "caught it")
    assert rec.traceback is not None
    assert "ValueError: boom" in rec.traceback


def test_handler_detached_after_context(tmp_path) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    with scan_log_file(tmp_path):
        assert len(root.handlers) == len(before) + 1
    # Handler removed even on clean exit.
    assert list(root.handlers) == before


def test_handler_detached_on_exception(tmp_path) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    with pytest.raises(RuntimeError):
        with scan_log_file(tmp_path):
            raise RuntimeError("scan blew up")
    assert list(root.handlers) == before


def test_missing_folder_skips_file(tmp_path) -> None:
    """A missing scan folder disables file logging but does not raise."""
    missing = tmp_path / "does_not_exist"
    root = logging.getLogger()
    before = list(root.handlers)
    with scan_log_file(missing) as log_path:
        assert log_path is None
        logging.getLogger("geecs_bluesky.demo").info("still runs")
    # No handler added, no folder created.
    assert list(root.handlers) == before
    assert not missing.exists()


def test_respects_level_threshold(tmp_path) -> None:
    """Records below the file's level are not written."""
    with scan_log_file(tmp_path, level=logging.WARNING):
        logging.getLogger("geecs_bluesky.demo").info("debug-ish, dropped")
        logging.getLogger("geecs_bluesky.demo").error("kept")

    entries = parse_scan_log(tmp_path / SCAN_LOG_FILENAME)
    msgs = {e.message for e in entries}
    assert "kept" in msgs
    assert "debug-ish, dropped" not in msgs
