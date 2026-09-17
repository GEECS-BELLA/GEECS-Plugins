"""ScanLogFile: a root-logger file handler for the span of one run."""

from __future__ import annotations

import logging

from geecs_bluesky.scan_log import ScanLogFile


def _folder(tmp_path):
    folder = tmp_path / "Scan001"
    folder.mkdir()
    return folder


def test_scan_log_writes_stamped_lines(tmp_path) -> None:
    log = ScanLogFile()
    path = log.open(1, _folder(tmp_path))
    logging.getLogger("geecs_bluesky.test").info("hello scan")
    log.close()
    text = path.read_text()
    assert "scan=Scan001" in text and "hello scan" in text
    assert "starting (dir=" in text and "finished" in text
    assert not log.is_open and log.path is None


def test_missing_folder_warns_and_skips(tmp_path, caplog) -> None:
    log = ScanLogFile()
    with caplog.at_level(logging.WARNING):
        assert log.open(3, tmp_path / "Scan003") is None
    assert "does not exist" in caplog.text
    assert not log.is_open
    log.close()  # a no-op


def test_detaches_and_restores_the_root_level(tmp_path) -> None:
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.WARNING)
    try:
        log = ScanLogFile()
        log.open(1, _folder(tmp_path))
        assert root.level == logging.INFO
        handlers = list(root.handlers)
        log.close()
        assert root.level == logging.WARNING
        assert len(root.handlers) == len(handlers) - 1
    finally:
        root.setLevel(before)


def test_captures_foreign_namespaces_and_quiets_transport_chatter(tmp_path) -> None:
    log = ScanLogFile()
    path = log.open(1, _folder(tmp_path))
    logging.getLogger("bluesky").info("RE state change")
    logging.getLogger("httpx").info("GET /x 200")
    logging.getLogger("httpx").warning("retrying")
    log.close()
    text = path.read_text()
    assert "RE state change" in text
    assert "GET /x 200" not in text and "retrying" in text


def test_reopen_closes_the_previous_file(tmp_path) -> None:
    log = ScanLogFile()
    first = log.open(1, _folder(tmp_path))
    second_folder = tmp_path / "Scan002"
    second_folder.mkdir()
    second = log.open(2, second_folder)
    logging.getLogger("x").info("second only")
    log.close()
    assert "second only" not in first.read_text()
    assert "second only" in second.read_text()
