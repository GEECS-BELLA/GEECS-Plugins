"""Data written vs settled for analysis: ``run_closed_evidence`` and ``scan_is_settled``."""

from __future__ import annotations

import os
import time
from pathlib import Path

from geecs_data_utils.data.sfile import (
    ROWLESS_SETTLE_SECONDS,
    SETTLE_SECONDS,
    run_closed_evidence,
    scan_data_txt_path_for,
    scan_is_settled,
    sfile_path_for_scan,
)
from geecs_data_utils.testing import make_scan_folder, scan_info_text, write_closed


def _scan(tmp_path: Path, **kwargs) -> Path:
    return make_scan_folder(tmp_path / "26_0310" / "scans" / "Scan004", **kwargs)


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_a_scan_still_being_written_is_not_settled(tmp_path):
    scan = _scan(tmp_path, scan_info={"ScanEndInfo": ""})
    assert not scan_is_settled(scan)  # claimed, no close yet
    # the table being written now: its data is written, it is not settled
    write_closed(scan, age=0.0)
    assert run_closed_evidence(scan) is not None
    assert not scan_is_settled(scan)
    assert scan_is_settled(scan, now=time.time() + SETTLE_SECONDS + 1)


def test_an_analysis_write_does_not_unsettle_a_scan(tmp_path):
    scan = _scan(tmp_path, scan_info={"ScanEndInfo": "success"})
    write_closed(scan)
    _age(scan / "ScanInfoScan004.ini", 60)
    assert scan_is_settled(scan)
    # analysis merges its scalars into the s-file and writes its own tree
    sfile_path_for_scan(scan).write_text("Shotnumber\tx\n1\t2\n")
    out = scan.parent.parent / "analysis" / "Scan004" / "A"
    out.mkdir(parents=True)
    (out / "summary.png").write_bytes(b"")
    (scan / "analysis_status").mkdir()
    (scan / "analysis_status" / "A.yaml").write_text("status: done\n")
    assert scan_is_settled(scan)


def test_a_scan_with_no_rows_settles(tmp_path):
    # the native scanner fills ScanEndInfo at the stop and writes no table
    scan = _scan(tmp_path, scan_info={"ScanEndInfo": "abort: no shots"})
    assert run_closed_evidence(scan) is None
    assert not scan_is_settled(scan)  # a table may still come
    assert not scan_is_settled(scan, now=time.time() + SETTLE_SECONDS + 1)
    assert scan_is_settled(scan, now=time.time() + ROWLESS_SETTLE_SECONDS + 1)


def test_an_unclosed_scan_without_a_table_never_settles(tmp_path):
    # a scan the scanner never closed (empty ScanEndInfo, no table): not settled
    scan = _scan(tmp_path, scan_info=scan_info_text({"ScanEndInfo": ""}))
    assert not scan_is_settled(scan, now=time.time() + 10 * ROWLESS_SETTLE_SECONDS)


def test_a_legacy_scan_settles_on_its_table(tmp_path):
    # Master Control: ScanEndInfo always empty; the table is the close
    scan = _scan(tmp_path, scan_info={"ScanEndInfo": ""})
    write_closed(scan)
    _age(scan / "ScanInfoScan004.ini", 60)
    assert scan_data_txt_path_for(scan).is_file()
    assert scan_is_settled(scan)


def test_settled_reads_only_and_refuses_a_missing_folder(tmp_path):
    missing = tmp_path / "26_0310" / "scans" / "Scan009"
    assert not scan_is_settled(missing)
    assert not missing.exists()
