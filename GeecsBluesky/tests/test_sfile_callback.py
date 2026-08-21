"""Tests for legacy scalar s-file export callback."""

from __future__ import annotations

import logging

from geecs_bluesky import sfile_callback
from geecs_bluesky.sfile_callback import SFileExportCallback


def test_callback_exports_matching_stop_document(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def record_export(run_uid: str, scan_number: int) -> object:
        calls.append((run_uid, scan_number))
        return None

    monkeypatch.setattr(sfile_callback, "_export_scalar_files", record_export)

    callback = SFileExportCallback()
    callback("start", {"uid": "run-a", "scan_number": 17})
    callback("start", {"uid": "run-b", "scan_number": 18})
    # Stop the EARLIER run first: an implementation that ignores the uid and
    # exports "the latest start" fails here (PR #635 review finding 2).
    callback("stop", {"uid": "stop-a", "run_start": "run-a", "exit_status": "success"})
    callback("stop", {"uid": "stop-b", "run_start": "run-b", "exit_status": "success"})

    assert calls == [("run-a", 17), ("run-b", 18)]


def test_callback_skips_aborted_and_failed_runs(monkeypatch, caplog) -> None:
    """Parity with the post-RE() call sites: no s-file for a non-success run."""
    calls: list[tuple[str, int]] = []

    def record_export(run_uid: str, scan_number: int) -> object:
        calls.append((run_uid, scan_number))
        return None

    monkeypatch.setattr(sfile_callback, "_export_scalar_files", record_export)

    callback = SFileExportCallback()
    callback("start", {"uid": "run-a", "scan_number": 30})
    callback("start", {"uid": "run-b", "scan_number": 31})
    with caplog.at_level(logging.INFO, logger=sfile_callback.__name__):
        callback(
            "stop", {"uid": "stop-a", "run_start": "run-a", "exit_status": "abort"}
        )
        callback("stop", {"uid": "stop-b", "run_start": "run-b", "exit_status": "fail"})

    assert calls == []
    assert "exit_status='abort'" in caplog.text
    assert "exit_status='fail'" in caplog.text
    # State hygiene: the skipped runs' entries are released.
    assert callback._starts == {}


def test_callback_swallowing_export_failure_logs_warning(monkeypatch, caplog) -> None:
    def fail_export(run_uid: str, scan_number: int) -> object:
        raise RuntimeError(f"boom {run_uid} {scan_number}")

    monkeypatch.setattr(sfile_callback, "_export_scalar_files", fail_export)

    callback = SFileExportCallback()
    callback("start", {"uid": "run-a", "scan_number": 23})

    with caplog.at_level(logging.WARNING, logger=sfile_callback.__name__):
        callback(
            "stop", {"uid": "stop-a", "run_start": "run-a", "exit_status": "success"}
        )

    assert "Could not export legacy scalar files for scan 23 (uid=run-a)" in caplog.text


def test_callback_skips_run_without_scan_number(monkeypatch, caplog) -> None:
    calls: list[tuple[str, int]] = []

    def record_export(run_uid: str, scan_number: int) -> object:
        calls.append((run_uid, scan_number))
        return None

    monkeypatch.setattr(sfile_callback, "_export_scalar_files", record_export)

    callback = SFileExportCallback()
    callback("start", {"uid": "run-a"})

    with caplog.at_level(logging.WARNING, logger=sfile_callback.__name__):
        callback(
            "stop", {"uid": "stop-a", "run_start": "run-a", "exit_status": "success"}
        )

    assert calls == []
    assert (
        "Skipping legacy scalar file export for run run-a: "
        "start document has no scan_number"
    ) in caplog.text
