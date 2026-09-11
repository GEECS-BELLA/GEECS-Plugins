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
        assert [s.number for s in day.scans] == [1, 6, 31, 40]

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
        assert scan_status("cancelled by the interlock", True) == "unknown"

    def test_abort_is_its_own_outcome(self) -> None:
        """RE.abort(), Ctrl-C and the queueserver stop all land here.

        The scanner writes ``exit_status`` verbatim, so an aborted run
        reads ``abort: <reason>``. Reporting that as "unknown" hid the
        reason and kept it out of the day's tally.
        """
        assert scan_status("abort", True) == "aborted"
        assert scan_status("abort: RunEngine stopped by user", True) == "aborted"

    def test_empty_end_info_is_incomplete_not_unknown(self) -> None:
        """ScanEndInfo = "" means not finalised, not unrecognised.

        The scanner writes it empty when the folder is claimed and fills
        it at the stop document. It is the most common state on the real
        share, so calling it "unknown" painted most of a day amber.
        """
        assert scan_status("", True) == "incomplete"
        assert scan_status("   ", True) == "incomplete"
        assert scan_status(None, True) == "incomplete"


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
        # Scan001/Scan040 share (U_S1H, same purpose) but are not adjacent;
        # Scan031 has no ScanInfo at all.
        assert spans == [
            ("Scan001", 1),
            ("Scan006", 1),
            ("Scan031", 1),
            ("Scan040", 1),
        ]

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


class TestCaching:
    """Scan folders are memoised on their modification time.

    Over a VPN-mounted share each ScanInfo open is a round trip, so
    re-reading a finished day on every request dominated the page.
    """

    def test_second_read_uses_the_cache(self, share: Path) -> None:
        """Re-reading a day hits the cache rather than the share."""
        from geecs_scan_log.scan_reader import _read_scan_cached

        _read_scan_cached.cache_clear()
        read_day(DAY, "Undulator", base_directory=share)
        assert _read_scan_cached.cache_info().hits == 0
        read_day(DAY, "Undulator", base_directory=share)
        assert _read_scan_cached.cache_info().hits == 4

    def test_in_place_finalisation_busts_the_cache(self, share: Path) -> None:
        """A scan that fails AFTER being cached must stop reading as running.

        The scanner finalises an outcome by rewriting ScanInfo in place
        (``ScanInfoCallback.on_stop`` -> ``path.open("w")``). Truncating an
        existing inode changes no directory entry, so the *folder's* mtime
        does not move. A cache keyed on the folder therefore served the
        pre-stop empty ``ScanEndInfo`` forever — losing the failure reason
        this view exists to surface, and worst for failed scans, which
        write no s-file that might otherwise have disturbed the folder.
        """
        import os
        from geecs_scan_log.scan_reader import _read_scan_cached

        _read_scan_cached.cache_clear()
        scans = share / "Undulator" / "Y2026" / "09-Sep" / "26_0911" / "scans"
        running = scans / "Scan040"
        folder_mtime_before = running.stat().st_mtime

        first = read_day(DAY, "Undulator", base_directory=share)
        assert first.scans[3].status == "incomplete"

        # Finalise exactly as the scanner does: rewrite the same file.
        ini = running / "ScanInfoScan040.ini"
        body = ini.read_text().replace(
            'ScanEndInfo = ""', "ScanEndInfo = \"fail: TimeoutError('no shot')\""
        )
        with ini.open("w") as handle:
            handle.write(body)
        os.utime(running, (folder_mtime_before, folder_mtime_before))

        assert running.stat().st_mtime == folder_mtime_before, (
            "folder mtime moved; this test would pass for the wrong reason"
        )

        again = read_day(DAY, "Undulator", base_directory=share)
        assert again.scans[3].status == "failed"
        assert "TimeoutError" in again.scans[3].failure_reason

    def test_a_changed_folder_misses_the_cache(self, share: Path) -> None:
        """A scan still being written is never served stale."""
        import os
        from geecs_scan_log.scan_reader import _read_scan_cached

        _read_scan_cached.cache_clear()
        first = read_day(DAY, "Undulator", base_directory=share)
        assert first.scans[2].status == "incomplete"

        # Scan031 finishes: ScanInfo appears and the folder mtime moves.
        bare = (
            share / "Undulator" / "Y2026" / "09-Sep" / "26_0911" / "scans" / "Scan031"
        )
        (bare / "ScanInfoScan031.ini").write_text(
            '[Scan Info]\nScan No = 31\nScanEndInfo = "success"\n'
        )
        os.utime(bare, (0, 0))  # force a distinct mtime

        again = read_day(DAY, "Undulator", base_directory=share)
        assert again.scans[2].status == "success"


class TestLeanReads:
    """One directory listing answers both per-scan questions.

    Globbing for ScanInfo and then listing devices costs two round trips
    per scan on an SMB share; ``scan_contents`` does one. Measured 1.85x
    on cold days (403 -> 218 ms per scan).
    """

    def test_one_listing_answers_every_question(self, share: Path) -> None:
        """A populated scan yields ini, its stat, the log and the devices."""
        from geecs_scan_log.scan_reader import scan_contents

        folder = share / "Undulator" / "Y2026" / "09-Sep" / "26_0911" / "scans"
        got = scan_contents(folder / "Scan001")
        assert got.ini_path is not None
        assert got.ini_path.endswith("ScanInfoScan001.ini")
        assert got.ini_mtime is not None and got.ini_size
        assert got.log_path is not None and got.log_path.endswith("scan.log")
        assert got.devices == ("UC_Amp4_IR_input",)

    def test_bare_folder_yields_only_its_log(self, share: Path) -> None:
        """A folder with no ScanInfo still reports the log it does have."""
        from geecs_scan_log.scan_reader import scan_contents

        folder = share / "Undulator" / "Y2026" / "09-Sep" / "26_0911" / "scans"
        got = scan_contents(folder / "Scan031")
        assert got.ini_path is None and got.ini_mtime is None
        assert got.log_path is not None
        assert got.devices == ()

    def test_missing_folder_is_reported_not_raised(self, tmp_path: Path) -> None:
        """An absent folder returns empties rather than exploding a day."""
        from geecs_scan_log.scan_reader import scan_contents

        assert scan_contents(tmp_path / "nope") == (None, None, None, None, ())


class TestEmptyRuns:
    """Folders with no ScanInfo group together and say so."""

    def test_metadata_free_run_is_labelled(self, make_run, tmp_path) -> None:
        """A run of bare folders reports itself as metadata-free."""
        scans = tmp_path / "Undulator" / "Y2026" / "09-Sep" / "26_0911" / "scans"
        scans.mkdir(parents=True)
        for n in range(1, 6):
            folder = scans / f"Scan{n:03d}"
            folder.mkdir()
            (folder / "scan.log").write_text("")

        campaigns = read_day(DAY, "Undulator", base_directory=tmp_path).campaigns
        assert len(campaigns) == 1
        assert campaigns[0].is_empty_run is True
        assert len(campaigns[0].scans) == 5

    def test_a_real_run_is_not(self, share) -> None:
        """A run with a scan parameter is never labelled metadata-free."""
        campaigns = read_day(DAY, "Undulator", base_directory=share).campaigns
        real = [c for c in campaigns if c.parameter]
        assert real and all(c.is_empty_run is False for c in real)


class TestStartTime:
    """When a scan ran, and how honest the view is about knowing."""

    def test_prefers_the_scan_log(self, share: Path) -> None:
        """A scan with a log reports the log's own first timestamp."""
        scan = read_day(DAY, "Undulator", base_directory=share).scans[0]
        assert scan.started is not None
        assert scan.started.strftime("%H:%M:%S") == "08:10:35"
        assert scan.started_approximate is False

    def test_falls_back_to_scaninfo_mtime(self, make_run) -> None:
        """An archive scan with no scan.log still shows a time, flagged.

        Much of the archive predates scan.log. Dropping the fallback
        entirely traded a wrong answer for no answer at all — a whole 2025
        day rendered every time as an em dash.
        """
        root = make_run(1)
        # make_run writes ScanInfo but no scan.log — the archive shape.
        scan = read_day(DAY, "Undulator", base_directory=root).scans[0]
        assert scan.started is not None
        assert scan.started_approximate is True

    def test_no_sources_means_no_claim(self, tmp_path: Path) -> None:
        """With neither log nor ScanInfo, the time is absent, not invented."""
        scans = tmp_path / "Undulator" / "Y2026" / "09-Sep" / "26_0911" / "scans"
        (scans / "Scan001").mkdir(parents=True)

        scan = read_day(DAY, "Undulator", base_directory=tmp_path).scans[0]
        assert scan.started is None
        assert scan.started_approximate is False
