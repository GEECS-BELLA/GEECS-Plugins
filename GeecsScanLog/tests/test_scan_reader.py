"""The reader's contract, including the scan-folder invariant."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from geecs_scan_log.scan_reader import read_day, scan_status

DAY = date(2026, 9, 11)


class TestReadDay:
    """Reading a day of scan folders."""

    def test_lists_scans_in_number_order(self, share: Path) -> None:
        """Only ScanNNN directories count, ordered by number."""
        day = read_day(DAY, "Undulator", base_directory=share)
        assert day.exists is True
        assert [s.number for s in day.scans] == [1, 6, 31]

    def test_parses_scan_info(self, share: Path) -> None:
        """ScanInfo fields land on the summary, quotes stripped."""
        scan = read_day(DAY, "Undulator", base_directory=share).scans[0]
        assert scan.parameter == "U_S1H Current"
        assert scan.start == -1.0 and scan.end == 1.0
        assert scan.shots_per_step == 2
        assert scan.plan == "scan" and scan.scanner == "bluesky"
        assert scan.background is False
        assert scan.purpose.startswith("807 phase 1 acceptance")
        assert scan.status == "success"
        assert scan.failure_reason is None
        assert scan.devices == ["UC_Amp4_IR_input"]

    def test_surfaces_the_failure_reason(self, share: Path) -> None:
        """A failed scan carries its ScanEndInfo text, prefix stripped."""
        scan = next(
            s
            for s in read_day(DAY, "Undulator", base_directory=share).scans
            if s.number == 6
        )
        assert scan.status == "failed"
        assert scan.failure_reason.startswith("<AsyncStatus")
        assert "uc_amp4_ir_input-hdf-capture" in scan.failure_reason

    def test_folder_without_scan_info_is_incomplete(self, share: Path) -> None:
        """A bare folder reports incomplete rather than erroring."""
        scan = next(
            s
            for s in read_day(DAY, "Undulator", base_directory=share).scans
            if s.number == 31
        )
        assert scan.status == "incomplete"
        assert scan.has_scan_info is False
        assert scan.parameter is None
        assert scan.devices == []

    def test_counts_failures(self, share: Path) -> None:
        """The day summary tallies failed scans for the header."""
        assert read_day(DAY, "Undulator", base_directory=share).failed == 1

    def test_missing_day_is_empty_not_an_error(self, tmp_path: Path) -> None:
        """A date that never ran is an ordinary empty day."""
        day = read_day(date(2019, 1, 1), "Undulator", base_directory=tmp_path)
        assert day.exists is False
        assert day.scans == []


class TestScanStatus:
    """The status classifier."""

    def test_success(self) -> None:
        """An exact success string classifies as success."""
        assert scan_status("success", True) == "success"

    def test_failure_prefix(self) -> None:
        """Anything starting with 'fail' classifies as failed."""
        assert scan_status("fail: TimeoutError(...)", True) == "failed"

    def test_no_scan_info_beats_everything(self) -> None:
        """Without a ScanInfo file the scan is incomplete."""
        assert scan_status("success", False) == "incomplete"

    def test_unrecognised_end_info(self) -> None:
        """An end string we do not recognise is reported as unknown."""
        assert scan_status("aborted by operator", True) == "unknown"


class TestScanFolderCreationInvariant:
    """The logbook is a consumer of scan folders, never a producer.

    Mirrors ``ScanAnalysis/tests/test_task_queue.py`` and the ImageAnalysis
    equivalents. Reading a day that does not exist must not bring the day
    or scan folders into being — a transient share blip would otherwise
    plant empty directories that orphan the real data when it resolves.
    """

    def test_reading_absent_day_creates_nothing(self, tmp_path: Path) -> None:
        """No directory appears under the share root."""
        before = set(tmp_path.rglob("*"))
        read_day(date(2026, 9, 12), "Undulator", base_directory=tmp_path)
        assert set(tmp_path.rglob("*")) == before

    def test_reader_never_calls_mkdir(self, tmp_path: Path, monkeypatch) -> None:
        """Any mkdir during a read is a bug; fail loudly if one appears."""

        def explode(*args, **kwargs):
            raise AssertionError("the logbook must never create scan folders")

        monkeypatch.setattr(Path, "mkdir", explode)
        read_day(date(2026, 9, 12), "Undulator", base_directory=tmp_path)


class TestCampaigns:
    """Consecutive scans sharing a parameter and purpose group into a run.

    A busy day is a handful of campaigns rather than a hundred unrelated
    scans, and the grouping is derived from what the scanner already wrote
    — nobody declares a campaign, so nobody can forget to.
    """

    def test_groups_consecutive_matching_scans(self, share: Path) -> None:
        """Each distinct (parameter, purpose) run becomes one campaign."""
        day = read_day(DAY, "Undulator", base_directory=share)
        spans = [(c.span, len(c.scans)) for c in day.campaigns]
        # Scan001 (U_S1H), Scan006 (Shotnumber), Scan031 (no scan info)
        assert spans == [("Scan001", 1), ("Scan006", 1), ("Scan031", 1)]

    def test_a_run_collapses_to_one_campaign(self, make_run) -> None:
        """Twenty identical scans are one campaign, not twenty."""
        root = make_run(20)
        campaigns = read_day(DAY, "Undulator", base_directory=root).campaigns
        assert len(campaigns) == 1
        assert campaigns[0].span == "Scan001–Scan020"
        assert len(campaigns[0].scans) == 20

    def test_campaign_reports_failures_inside_it(self, share: Path) -> None:
        """A failure inside a campaign is visible on the campaign itself."""
        day = read_day(DAY, "Undulator", base_directory=share)
        failing = [c for c in day.campaigns if c.failed]
        assert len(failing) == 1
        assert failing[0].scans[0].number == 6
