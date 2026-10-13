"""The scan-folder facts' one home (#1061 rows 3.6/3.7/3.8).

Each class pins the ONE behaviour chosen where the consumers used to
disagree; the old divergent behaviour is named in each docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geecs_data_utils.scan_info import (
    ScanInfoRecord,
    analysis_folder_for,
    classify_end_info,
    end_info_reason,
    is_noscan,
    parse_scan_folder_name,
    scan_folder_name,
)
from geecs_data_utils.scan_paths import ScanPaths
from geecs_data_utils.type_defs import ScanTag


class TestFolderName:
    """Lenient parse, padded write (was: \\d+ in the logbook, \\d{3,} in
    Data-Utils/ScanAnalysis, any ``Scan*`` prefix in LogTriage)."""

    @pytest.mark.parametrize(
        "name, number",
        [("Scan042", 42), ("Scan7", 7), ("Scan0007", 7), ("Scan1000", 1000)],
    )
    def test_parses(self, name, number):
        assert parse_scan_folder_name(name) == number

    @pytest.mark.parametrize(
        "name", ["ScanFoo", "Scan", "Scan007_old", "ScanDataScan007.txt", "scan007"]
    )
    def test_refuses(self, name):
        assert parse_scan_folder_name(name) is None

    def test_canonical_name_is_padded(self):
        assert scan_folder_name(7) == "Scan007"
        assert scan_folder_name(1000) == "Scan1000"

    def test_scan_paths_accepts_an_unpadded_folder(self, tmp_path):
        folder = tmp_path / "Undulator" / "Y2026" / "09-Sep" / "26_0911" / "scans"
        (folder / "Scan7").mkdir(parents=True)
        assert ScanPaths(folder=folder / "Scan7").get_tag().number == 7


class TestAnalysisFolder:
    """One pure rule (was: four unvalidated copies + core_sink's checks)."""

    def test_is_the_scans_sibling(self, tmp_path):
        scan = tmp_path / "26_0911" / "scans" / "Scan002"
        assert (
            analysis_folder_for(scan) == tmp_path / "26_0911" / "analysis" / "Scan002"
        )
        assert not (tmp_path / "26_0911").exists()  # pure: nothing touched

    @pytest.mark.parametrize("rel", ["raw/Scan002", "scans/notascan"])
    def test_shape_is_always_checked(self, tmp_path, rel):
        with pytest.raises(ValueError):
            analysis_folder_for(tmp_path / rel)

    def test_validate_requires_the_scan_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analysis_folder_for(tmp_path / "scans" / "Scan001", validate=True)
        assert not (tmp_path / "scans").exists()

    def test_scan_paths_getter_creates_nothing(self, tmp_path):
        """``ScanPaths.analysis_folder()`` replaced ``get_analysis_folder()``,
        which ran ``os.makedirs`` on the analysis tree."""
        base = tmp_path / "data"
        tag = ScanTag(year=2026, month=9, day=11, number=3, experiment="Undulator")
        scan = ScanPaths.get_scan_folder_path(tag, base_directory=base)
        scan.mkdir(parents=True)
        before = sorted(tmp_path.rglob("*"))
        paths = ScanPaths(tag=tag, base_directory=base)
        assert paths.analysis_folder() == scan.parent.parent / "analysis" / "Scan003"
        assert ScanPaths.get_scan_analysis_folder_path(tag, base) == (
            paths.analysis_folder()
        )
        assert sorted(tmp_path.rglob("*")) == before


class TestScanInfoRecord:
    """Finite floats only (was: the folder catalog accepted inf/nan)."""

    def test_types_the_ini(self):
        record = ScanInfoRecord.from_ini(
            {
                "Scan Parameter": " U_S1H Current ",
                "Start": "-1.0",
                "End": "1",
                "Step size": "0.5",
                "Shots per step": "2.0",
                "Background": "false",
                "ScanStartInfo": "",
                "ScanEndInfo": "",
            }
        )
        assert record.parameter == "U_S1H Current"
        assert (record.start, record.end, record.step_size) == (-1.0, 1.0, 0.5)
        assert record.shots_per_step == 2
        assert record.background is False
        assert record.purpose is None
        assert record.end_info == ""

    @pytest.mark.parametrize("raw", ["inf", "-inf", "nan", "1e400", "abc", ""])
    def test_unusable_numbers_are_none(self, raw):
        record = ScanInfoRecord.from_ini({"Start": raw, "Shots per step": raw})
        assert record.start is None and record.shots_per_step is None

    def test_empty_ini_is_all_none(self):
        assert ScanInfoRecord.from_ini({}) == ScanInfoRecord()


class TestNoscan:
    """One vocabulary (was: ScanAnalysis knew only noscan/shotnumber)."""

    @pytest.mark.parametrize(
        "parameter", [None, "", "  ", "Shotnumber", "NOSCAN", "none", " None "]
    )
    def test_noscan(self, parameter):
        assert is_noscan(parameter)
        assert ScanInfoRecord(parameter=parameter).noscan

    @pytest.mark.parametrize("parameter", ["U_S1H Current", "Bin #", "nonesuch"])
    def test_a_real_parameter(self, parameter):
        assert not is_noscan(parameter)


class TestClassifyEndInfo:
    """One verdict for the logbook and the portal (was: ``failed/aborted/
    incomplete`` vs ``fail/abort/None``, and an empty ScanEndInfo was
    always ``incomplete`` in the logbook but ``success`` once the run
    closed in the folder catalog)."""

    @pytest.mark.parametrize(
        "end_info, status",
        [
            ("success", "success"),
            ("Success", "success"),
            ("fail: TimeoutError(...)", "failed"),
            ("Fail: stage", "failed"),
            ("abort", "aborted"),
            ("aborted", "aborted"),
            ("abort: RunEngine stopped by user", "aborted"),
            ("cancelled by the interlock", "unknown"),
            ("successful-ish", "unknown"),
        ],
    )
    def test_end_info_wins_over_the_files(self, end_info, status):
        assert classify_end_info(end_info, closed=False) == status
        assert classify_end_info(end_info, closed=True) == status

    @pytest.mark.parametrize("end_info", [None, "", "   "])
    def test_empty_is_decided_by_the_run_closed_evidence(self, end_info):
        assert classify_end_info(end_info, closed=False) == "incomplete"
        assert classify_end_info(end_info, closed=True) == "success"

    @pytest.mark.parametrize(
        "end_info, reason",
        [
            ("fail: TimeoutError('x')", "TimeoutError('x')"),
            ("abort: RunEngine stopped by user", "RunEngine stopped by user"),
            ("abort", "abort"),
            ("success", None),
            ("cancelled", None),
            (None, None),
        ],
    )
    def test_reason(self, end_info, reason):
        assert end_info_reason(end_info) == reason


def test_module_creates_nothing(tmp_path: Path, monkeypatch):
    """Every helper is read-only (the repo's scan-folder invariant)."""

    def explode(*args, **kwargs):
        raise AssertionError("scan_info must never create a path")

    monkeypatch.setattr(Path, "mkdir", explode)
    monkeypatch.setattr("os.makedirs", explode)
    scan = tmp_path / "scans" / "Scan001"
    analysis_folder_for(scan)
    with pytest.raises(FileNotFoundError):
        analysis_folder_for(scan, validate=True)


def test_copy_fresh_sfile_creates_only_the_day_analysis_folder(tmp_path):
    """The one writer the old creating getter served: day-level analysis/ only."""
    from geecs_data_utils.scan_data import ScanData

    base = tmp_path / "data"
    tag = ScanTag(year=2026, month=9, day=11, number=3, experiment="Undulator")
    scan = ScanPaths.get_scan_folder_path(tag, base_directory=base)
    scan.mkdir(parents=True)
    (scan / "ScanDataScan003.txt").write_text("Shotnumber\n1\n")
    sd = ScanData(paths=ScanPaths(tag=tag, base_directory=base))
    sd.copy_fresh_sfile_to_analysis()
    analysis = scan.parent.parent / "analysis"
    assert (analysis / "s3.txt").is_file()
    assert not (analysis / "Scan003").exists()
