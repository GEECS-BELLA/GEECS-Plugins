"""Tests for the shared per-scan scan.log helper (Gate-2 follow-up).

Covers the extracted :mod:`geecs_bluesky.scan_log` helper itself and the GUI
bridge's delegation (behavior regression — identical file, format, and
scan-id stamping).  CI-safe: no CA needed.  The headless
:class:`GeecsSession` attachment tests (mock CA backends) live in
``test_scan_log_session.py``.
"""

from __future__ import annotations

import logging
import threading

import pytest

from geecs_bluesky.scan_log import scan_log

module_logger = logging.getLogger("geecs_bluesky.tests.scan_log")


def test_scan_log_writes_stamped_lines(tmp_path) -> None:
    folder = tmp_path / "Scan007"
    folder.mkdir()
    with scan_log(7, str(folder)):
        module_logger.info("hello from the scan")
    content = (folder / "scan.log").read_text()
    assert "scan Scan007: starting" in content
    assert "hello from the scan" in content
    assert "scan Scan007: finished" in content
    # Every record is stamped with the scan id by the context filter.
    assert "scan=Scan007" in content


def test_scan_log_noop_without_claim(tmp_path) -> None:
    with scan_log(None, None):
        module_logger.info("unclaimed")
    assert list(tmp_path.iterdir()) == []


def test_scan_log_missing_folder_warns_and_skips(tmp_path, caplog) -> None:
    missing = tmp_path / "ScanNNN"
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.scan_log"):
        with scan_log(9, str(missing)):
            module_logger.info("nowhere to write")
    assert not missing.exists()  # never created (scan-folder invariant)
    assert "skipping scan.log" in caplog.text


def test_scan_log_detaches_and_restores_levels(tmp_path) -> None:
    folder = tmp_path / "Scan008"
    folder.mkdir()
    pkg_logger = logging.getLogger("geecs_bluesky")
    before_level = pkg_logger.level
    before_handlers = list(pkg_logger.handlers)
    with pytest.raises(RuntimeError, match="mid-scan"):
        with scan_log(8, str(folder)):
            raise RuntimeError("mid-scan failure")
    assert pkg_logger.level == before_level
    assert pkg_logger.handlers == before_handlers
    # Post-exit records do not land in the file.
    module_logger.info("after the scan")
    assert "after the scan" not in (folder / "scan.log").read_text()


def test_bridge_scan_log_delegates_to_shared_helper(tmp_path) -> None:
    """Regression: BlueskyScanner._scan_log behavior is unchanged."""
    from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner

    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._device_lock = threading.Lock()

    folder = tmp_path / "Scan012"
    folder.mkdir()
    with scanner._scan_log(12, str(folder)):
        module_logger.info("bridge line")
    content = (folder / "scan.log").read_text()
    assert "scan Scan012: starting" in content
    assert "bridge line" in content
    assert "scan=Scan012" in content
    assert "scan Scan012: finished" in content

    # The no-claim no-op is preserved too.
    with scanner._scan_log(None, None):
        module_logger.info("no claim")
