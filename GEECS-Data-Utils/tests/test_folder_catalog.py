"""Tests for the scan-folder ScanCatalog and its merge with a primary catalog."""

from __future__ import annotations

import os
from datetime import date, datetime

import pytest

from geecs_data_utils.folder_catalog import (
    FolderScanCatalog,
    MergedScanCatalog,
    folder_uid,
    parse_folder_uid,
)
from geecs_data_utils.scan_frame import scan_frame
from geecs_data_utils.tiled_catalog import (
    CatalogStatus,
    RunDetail,
    RunSummary,
    resolve_scan_folder,
)

DAY = date(2026, 9, 25)
LABVIEW_OFFSET = 2082844800.0


def _labview(hour: int, minute: int, day: date = DAY) -> float:
    return datetime(day.year, day.month, day.day, hour, minute).timestamp() + (
        LABVIEW_OFFSET
    )


def _day_dir(base, experiment="Thomson", day=DAY):
    return (
        base
        / experiment
        / f"Y{day.year}"
        / f"{day.month:02d}-{day.strftime('%b')}"
        / f"{day.strftime('%y_%m%d')}"
    )


def _scan(
    base,
    number,
    *,
    parameter="HTT-Aerotech Position.Axis1",
    start=500,
    end=1300,
    step=50,
    shots=20,
    end_info="",
    sfile_time=None,
    sfile="analysis",
    experiment="Thomson",
):
    day_dir = _day_dir(base, experiment)
    scan_dir = day_dir / "scans" / f"Scan{number:03d}"
    scan_dir.mkdir(parents=True)
    (scan_dir / f"ScanInfoScan{number:03d}.ini").write_text(
        "[Scan Info]\n"
        f'Scan No = "{number}"\n'
        'ScanStartInfo = "drive frog scan"\n'
        f'Scan Parameter = "{parameter}"\n'
        f"Start = {start}\nEnd = {end}\nStep size = {step}\n"
        f"Shots per step = {shots}\n"
        f'ScanEndInfo = "{end_info}"\n'
    )
    header = [
        f"{parameter} Alias:drive grating 2 trans",
        "HTT-C23_2_MagSpec2 centroidx",
        "DateTime Timestamp",
        "Shotnumber",
        "Bin #",
    ]
    stamp = sfile_time if sfile_time is not None else _labview(15, 18)
    rows = [f"{start}\t{600 + i}\t{stamp + i}\t{i + 1}\t1" for i in range(3)]
    text = "\t".join(header) + "\n" + "\n".join(rows) + "\n"
    if sfile == "analysis":
        (day_dir / "analysis").mkdir(exist_ok=True)
        (day_dir / "analysis" / f"s{number}.txt").write_text(text)
    elif sfile == "raw":
        (scan_dir / f"ScanDataScan{number:03d}.txt").write_text(text)
    return scan_dir


def _tree(path):
    return sorted(p.relative_to(path) for p in path.rglob("*"))


class TestUid:
    def test_round_trip(self):
        uid = folder_uid("Magnet Test Bench", DAY, 7)
        assert parse_folder_uid(uid) == ("Magnet Test Bench", DAY, 7)

    @pytest.mark.parametrize(
        "uid", ["a1b2-uuid", "folder:", "folder:Thomson:nope:3", "folder::2026-09-25:1"]
    )
    def test_malformed_is_none(self, uid):
        assert parse_folder_uid(uid) is None


class TestFolderScanCatalog:
    def test_lists_folders_newest_first(self, tmp_path):
        _scan(tmp_path, 1, sfile_time=_labview(11, 7))
        _scan(tmp_path, 2, sfile_time=_labview(15, 18))
        runs = FolderScanCatalog(tmp_path).list_runs("Thomson", DAY)
        assert [r.scan_number for r in runs] == [2, 1]
        first = runs[0]
        assert first.uid == "folder:Thomson:2026-09-25:2"
        assert first.mode == "1D"
        assert first.shots == 17 * 20
        assert first.exit_status == "success"
        assert first.experiment == "Thomson"
        assert first.description == "drive frog scan"
        assert datetime.fromtimestamp(first.start_time).hour == 15

    def test_start_time_off_the_folder_day_falls_back_to_noon(self, tmp_path):
        scan_dir = _scan(tmp_path, 1, sfile_time=_labview(9, 0, date(2020, 1, 1)))
        stale = datetime(2020, 1, 1, 9).timestamp()
        os.utime(scan_dir / "ScanInfoScan001.ini", (stale, stale))
        catalog = FolderScanCatalog(tmp_path)
        (run,) = catalog.list_runs("Thomson", DAY)
        assert datetime.fromtimestamp(run.start_time) == datetime(2026, 9, 25, 12)
        assert catalog.load_run(run.uid).start_doc["time_approximate"] is True

    def test_no_sfile_dates_from_the_ini_mtime_as_approximate(self, tmp_path):
        scan_dir = _scan(tmp_path, 1, sfile=None)
        started = datetime(2026, 9, 25, 10, 30).timestamp()
        os.utime(scan_dir / "ScanInfoScan001.ini", (started, started))
        catalog = FolderScanCatalog(tmp_path)
        (run,) = catalog.list_runs("Thomson", DAY)
        assert run.start_time == started
        assert catalog.load_run(run.uid).start_doc["time_approximate"] is True

    def test_scan_log_outranks_the_ini_mtime(self, tmp_path):
        scan_dir = _scan(tmp_path, 1, sfile=None)
        (scan_dir / "scan.log").write_text(
            "2026-09-25 08:15:00.123 INFO geecs_bluesky.scan_log "
            "[bluesky-run-engine] scan=Scan001 - scan Scan001: starting\n"
        )
        (run,) = FolderScanCatalog(tmp_path).list_runs("Thomson", DAY)
        assert datetime.fromtimestamp(run.start_time).replace(microsecond=0) == (
            datetime(2026, 9, 25, 8, 15)
        )

    def test_noscan_has_no_motor(self, tmp_path):
        _scan(tmp_path, 1, parameter="Shotnumber", start=1, end=50, step=1, shots=1)
        (run,) = FolderScanCatalog(tmp_path).list_runs("Thomson", DAY)
        assert run.mode == "NOSCAN"
        assert run.shots == 50

    def test_scan_without_any_sfile_is_unfinished(self, tmp_path):
        _scan(tmp_path, 1, sfile=None)
        (run,) = FolderScanCatalog(tmp_path).list_runs("Thomson", DAY)
        assert run.exit_status is None

    def test_raw_scandata_alone_is_closed_and_dates_the_scan(self, tmp_path):
        # run_closed_evidence accepts ScanDataScanNNN.txt; so does the catalog.
        _scan(tmp_path, 1, sfile="raw")
        (run,) = FolderScanCatalog(tmp_path).list_runs("Thomson", DAY)
        assert run.exit_status == "success"
        assert datetime.fromtimestamp(run.start_time).hour == 15

    @pytest.mark.parametrize(
        "end_info, status",
        [
            ("Fail: stage", "fail"),
            ("aborted", "abort"),
            ("Success", "success"),
            ("operator went home", "unknown"),
        ],
    )
    def test_end_info_wins(self, tmp_path, end_info, status):
        _scan(tmp_path, 1, end_info=end_info)
        (run,) = FolderScanCatalog(tmp_path).list_runs("Thomson", DAY)
        assert run.exit_status == status

    def test_load_run_start_doc_matches_the_sfile_column(self, tmp_path):
        scan_dir = _scan(tmp_path, 5)
        detail = FolderScanCatalog(tmp_path).load_run(folder_uid("Thomson", DAY, 5))
        assert detail.data is None
        assert detail.start_doc["motors"] == [
            "HTT-Aerotech Position.Axis1 Alias:drive grating 2 trans"
        ]
        assert resolve_scan_folder(detail, DAY) == scan_dir
        frame = scan_frame(detail, scan_dir).frame
        assert detail.start_doc["motors"][0] in frame.columns
        assert len(frame) == 3

    @pytest.mark.parametrize("uid", ["not-a-folder-uid", "folder:Thomson:2026-09-25:9"])
    def test_load_run_unknown_is_keyerror(self, tmp_path, uid):
        _scan(tmp_path, 1)
        with pytest.raises(KeyError):
            FolderScanCatalog(tmp_path).load_run(uid)

    def test_missing_day_is_empty_and_creates_nothing(self, tmp_path):
        _scan(tmp_path, 1)
        before = _tree(tmp_path)
        catalog = FolderScanCatalog(tmp_path)
        assert catalog.list_runs("Thomson", date(2026, 9, 26)) == []
        assert catalog.list_runs("Undulator", DAY) == []
        with pytest.raises(KeyError):
            catalog.load_run("folder:Thomson:2026-09-26:1")
        assert _tree(tmp_path) == before

    def test_probe(self, tmp_path):
        assert FolderScanCatalog(tmp_path).probe().ok
        assert not FolderScanCatalog(tmp_path / "gone").probe().ok


class TestFinishedScanCache:
    @staticmethod
    def _count_reads(monkeypatch):
        import geecs_data_utils.folder_catalog as module

        calls = []
        real = module.read_scan_info_file

        def counting(path):
            calls.append(path)
            return real(path)

        monkeypatch.setattr(module, "read_scan_info_file", counting)
        return calls

    def test_finished_day_relists_without_reading_files(self, tmp_path, monkeypatch):
        _scan(tmp_path, 1)
        _scan(tmp_path, 2)
        calls = self._count_reads(monkeypatch)
        catalog = FolderScanCatalog(tmp_path)
        first = catalog.list_runs("Thomson", DAY)
        assert len(calls) == 2
        assert catalog.list_runs("Thomson", DAY) == first
        catalog.load_run(folder_uid("Thomson", DAY, 1))
        assert len(calls) == 2

    def test_unfinished_scan_is_reread_until_it_finishes(self, tmp_path):
        scan_dir = _scan(tmp_path, 1, sfile=None)
        catalog = FolderScanCatalog(tmp_path)
        (run,) = catalog.list_runs("Thomson", DAY)
        assert run.exit_status is None
        # Master Control finishes a scan by writing its s-files.
        (scan_dir / "ScanDataScan001.txt").write_text("Shotnumber\n1\n")
        (run,) = catalog.list_runs("Thomson", DAY)
        assert run.exit_status == "success"

    def test_cached_documents_are_copies(self, tmp_path):
        _scan(tmp_path, 1)
        catalog = FolderScanCatalog(tmp_path)
        uid = folder_uid("Thomson", DAY, 1)
        catalog.load_run(uid).start_doc["motors"].append("mutated")
        assert catalog.load_run(uid).start_doc["motors"] == [
            "HTT-Aerotech Position.Axis1 Alias:drive grating 2 trans"
        ]

    def test_files_beside_the_scan_folders_are_ignored(self, tmp_path):
        scan_dir = _scan(tmp_path, 1)
        (scan_dir.parent / "Scan002").write_text("not a folder")
        (scan_dir.parent / "notes").mkdir()
        runs = FolderScanCatalog(tmp_path).list_runs("Thomson", DAY)
        assert [r.scan_number for r in runs] == [1]


def _summary(uid, number, hour):
    return RunSummary(
        uid=uid,
        scan_number=number,
        start_time=datetime(2026, 9, 25, hour).timestamp(),
        mode="1D",
        shots=10,
        exit_status="success",
        experiment="Thomson",
    )


class _Primary:
    def __init__(self, runs=(), fail=False):
        self.runs = list(runs)
        self.fail = fail
        self.loaded = []

    def probe(self):
        return CatalogStatus(ok=True, label="tiled: fake")

    def list_runs(self, experiment, day):
        if self.fail:
            raise ConnectionError("tiled down")
        return list(self.runs)

    def load_run(self, uid):
        self.loaded.append(uid)
        return RunDetail(summary=_summary(uid, 1, 9))


class TestMergedScanCatalog:
    def test_primary_run_claims_its_scan_number(self, tmp_path):
        _scan(tmp_path, 1, sfile_time=_labview(10, 0))
        _scan(tmp_path, 2, sfile_time=_labview(11, 0))
        primary = _Primary([_summary("uuid-2", 2, 11)])
        runs = MergedScanCatalog(primary, FolderScanCatalog(tmp_path)).list_runs(
            "Thomson", DAY
        )
        assert [r.uid for r in runs] == ["uuid-2", "folder:Thomson:2026-09-25:1"]

    def test_primary_outage_degrades_to_folders(self, tmp_path):
        _scan(tmp_path, 1)
        merged = MergedScanCatalog(_Primary(fail=True), FolderScanCatalog(tmp_path))
        assert [r.scan_number for r in merged.list_runs("Thomson", DAY)] == [1]

    def test_primary_outage_with_no_folders_raises(self, tmp_path):
        merged = MergedScanCatalog(_Primary(fail=True), FolderScanCatalog(tmp_path))
        with pytest.raises(ConnectionError):
            merged.list_runs("Thomson", DAY)

    def test_claimed_folders_are_never_read(self, tmp_path, monkeypatch):
        _scan(tmp_path, 1)
        _scan(tmp_path, 2)
        calls = TestFinishedScanCache._count_reads(monkeypatch)
        primary = _Primary([_summary("uuid-1", 1, 9), _summary("uuid-2", 2, 10)])
        runs = MergedScanCatalog(primary, FolderScanCatalog(tmp_path)).list_runs(
            "Thomson", DAY
        )
        assert [r.uid for r in runs] == ["uuid-2", "uuid-1"]
        assert calls == []

    def test_share_error_degrades_to_the_primary(self, tmp_path, monkeypatch):
        folders = FolderScanCatalog(tmp_path)

        def unreadable(*args, **kwargs):
            raise PermissionError("share blip")

        monkeypatch.setattr(folders, "list_runs", unreadable)
        primary = _Primary([_summary("uuid-1", 1, 9)])
        runs = MergedScanCatalog(primary, folders).list_runs("Thomson", DAY)
        assert [r.uid for r in runs] == ["uuid-1"]

    def test_both_down_raises_the_primary_error(self, tmp_path, monkeypatch):
        folders = FolderScanCatalog(tmp_path)

        def unreadable(*args, **kwargs):
            raise PermissionError("share blip")

        monkeypatch.setattr(folders, "list_runs", unreadable)
        merged = MergedScanCatalog(_Primary(fail=True), folders)
        with pytest.raises(ConnectionError):
            merged.list_runs("Thomson", DAY)

    def test_load_run_routes_by_uid(self, tmp_path):
        _scan(tmp_path, 1)
        primary = _Primary()
        merged = MergedScanCatalog(primary, FolderScanCatalog(tmp_path))
        assert merged.load_run("folder:Thomson:2026-09-25:1").data is None
        assert primary.loaded == []
        merged.load_run("uuid-7")
        assert primary.loaded == ["uuid-7"]
