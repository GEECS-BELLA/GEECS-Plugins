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
    callback("stop", {"uid": "stop-b", "run_start": "run-b"})

    assert calls == [("run-b", 18)]


def test_callback_swallowing_export_failure_logs_warning(monkeypatch, caplog) -> None:
    def fail_export(run_uid: str, scan_number: int) -> object:
        raise RuntimeError(f"boom {run_uid} {scan_number}")

    monkeypatch.setattr(sfile_callback, "_export_scalar_files", fail_export)

    callback = SFileExportCallback()
    callback("start", {"uid": "run-a", "scan_number": 23})

    with caplog.at_level(logging.WARNING, logger=sfile_callback.__name__):
        callback("stop", {"uid": "stop-a", "run_start": "run-a"})

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
        callback("stop", {"uid": "stop-a", "run_start": "run-a"})

    assert calls == []
    assert (
        "Skipping legacy scalar file export for run run-a: "
        "start document has no scan_number"
    ) in caplog.text
