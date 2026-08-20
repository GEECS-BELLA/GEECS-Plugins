"""Tests for the shared per-scan scan.log helper (Gate-2 follow-up).

Covers the extracted :mod:`geecs_bluesky.scan_log` helper itself.  CI-safe:
no CA needed.  The headless :class:`GeecsSession` attachment tests (mock CA
backends) live in ``test_scan_log_session.py``.
"""

from __future__ import annotations

import logging

import pytest

from geecs_bluesky.scan_log import (
    begin_pre_scan_capture,
    discard_pre_scan_capture,
    scan_log,
)

module_logger = logging.getLogger("geecs_bluesky.tests.scan_log")


@pytest.fixture(autouse=True)
def _no_leftover_buffer():
    """Every test starts and ends without a pending pre-scan buffer."""
    discard_pre_scan_capture()
    yield
    discard_pre_scan_capture()


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
    root = logging.getLogger()
    before_level = root.level
    before_handlers = list(root.handlers)
    with pytest.raises(RuntimeError, match="mid-scan"):
        with scan_log(8, str(folder)):
            raise RuntimeError("mid-scan failure")
    assert root.level == before_level
    assert root.handlers == before_handlers
    # Post-exit records do not land in the file.
    module_logger.info("after the scan")
    assert "after the scan" not in (folder / "scan.log").read_text()


def test_scan_log_captures_foreign_namespaces(tmp_path) -> None:
    """scan.log records the whole process story (root attach), not an
    allowlist — RunEngine state changes and ophyd connect failures included."""
    folder = tmp_path / "Scan010"
    folder.mkdir()
    with scan_log(10, str(folder)):
        logging.getLogger("bluesky").info("Change state from 'idle' -> 'running'")
        logging.getLogger("ophyd_async.core").warning("NotConnectedError: ca://x")
        logging.getLogger("geecs_data_utils.tiled_export").info("Wrote s-file")
    content = (folder / "scan.log").read_text()
    assert "idle' -> 'running'" in content
    assert "NotConnectedError" in content
    assert "Wrote s-file" in content
    # Foreign records are stamped with the scan id too.
    assert content.count("scan=Scan010") >= 5


def test_scan_log_quiets_transport_chatter(tmp_path) -> None:
    """httpx / mysql.connector INFO chatter stays out; their WARNINGs land."""
    folder = tmp_path / "Scan014"
    folder.mkdir()
    with scan_log(14, str(folder)):
        logging.getLogger("httpx").info("HTTP Request: POST /api/v1/metadata")
        logging.getLogger("mysql.connector").info("plugin_name: sha2")
        logging.getLogger("httpx").warning("retrying after timeout")
    content = (folder / "scan.log").read_text()
    assert "HTTP Request" not in content
    assert "plugin_name" not in content
    assert "retrying after timeout" in content


def test_pre_scan_buffer_flushes_into_scan_log(tmp_path) -> None:
    """Records emitted between submission and the folder claim open the file."""
    folder = tmp_path / "Scan011"
    folder.mkdir()
    begin_pre_scan_capture()
    module_logger.info("reinitialised from ScanRequest")
    logging.getLogger("ophyd_async.core").warning("telemetry device dropped")
    with scan_log(11, str(folder)):
        module_logger.info("mid-scan line")
    content = (folder / "scan.log").read_text()
    # Buffered lines appear, stamped, and BEFORE the "starting" banner.
    assert "reinitialised from ScanRequest" in content
    assert "telemetry device dropped" in content
    assert content.index("reinitialised") < content.index("Scan011: starting")
    assert "scan=Scan011" in content.splitlines()[0]
    # The buffer handler is gone from the root logger after the attach.
    assert not any(
        type(h).__name__ == "PreScanLogBuffer" for h in logging.getLogger().handlers
    )


def test_pre_scan_buffer_discarded_on_no_claim(tmp_path) -> None:
    """A buffer with no scan.log home never leaks into a later scan's file."""
    begin_pre_scan_capture()
    module_logger.info("from the unclaimed attempt")
    with scan_log(None, None):
        pass  # save_data=False path — discards the buffer
    folder = tmp_path / "Scan012"
    folder.mkdir()
    with scan_log(12, str(folder)):
        module_logger.info("second scan line")
    content = (folder / "scan.log").read_text()
    assert "from the unclaimed attempt" not in content
    assert "second scan line" in content


def test_begin_pre_scan_capture_supersedes_previous_buffer(tmp_path) -> None:
    """A re-submission's buffer replaces the stale one, lines and handler both."""
    begin_pre_scan_capture()
    module_logger.info("stale submission line")
    begin_pre_scan_capture()
    module_logger.info("fresh submission line")
    buffers = [
        h
        for h in logging.getLogger().handlers
        if type(h).__name__ == "PreScanLogBuffer"
    ]
    assert len(buffers) == 1
    folder = tmp_path / "Scan013"
    folder.mkdir()
    with scan_log(13, str(folder)):
        pass
    content = (folder / "scan.log").read_text()
    assert "stale submission line" not in content
    assert "fresh submission line" in content
