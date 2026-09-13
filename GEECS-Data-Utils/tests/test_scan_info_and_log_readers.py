"""The two shared readers: ScanInfo parsing and a scan log's start time.

Both are consumed by ``ScanPaths`` and by the scan logbook, so they are
pinned here in the package that owns them rather than only indirectly from
a consumer's suite.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from geecs_data_utils.scan_log_loader import first_log_timestamp
from geecs_data_utils.scan_paths import read_scan_info_file

SCAN_INFO = """[Scan Info]
Scan No = 5
ScanStartInfo = "807 phase 1 acceptance"
Scan Parameter = "U_S1H Current"
Start = -1.0
ScanEndInfo = ""
Background = false
"""

#: Transcribed from a real scan.log so the fixture exercises the real
#: header regex rather than an invented shape that would never match.
LOG_LINE = (
    "2026-09-11 00:08:13.905 INFO geecs_bluesky.scan_log "
    "[bluesky-run-engine] scan=Scan001 - scan Scan001: starting\n"
)


class TestReadScanInfoFile:
    """Parsing ``ScanInfoScanNNN.ini``."""

    def test_parses_and_strips_quotes(self, tmp_path: Path) -> None:
        """Values keep their content and lose their surrounding quotes."""
        ini = tmp_path / "ScanInfoScan005.ini"
        ini.write_text(SCAN_INFO)
        got = read_scan_info_file(ini)
        assert got["Scan Parameter"] == "U_S1H Current"
        assert got["ScanStartInfo"] == "807 phase 1 acceptance"
        assert got["ScanEndInfo"] == ""
        assert got["Scan No"] == "5"

    def test_preserves_key_case(self, tmp_path: Path) -> None:
        """Keys are not lowercased; consumers index by the written name."""
        ini = tmp_path / "s.ini"
        ini.write_text(SCAN_INFO)
        assert "Shots per step" not in read_scan_info_file(ini)
        assert "ScanStartInfo" in read_scan_info_file(ini)

    def test_missing_file_reads_empty(self, tmp_path: Path) -> None:
        """An absent file is empty, never an exception."""
        assert read_scan_info_file(tmp_path / "nope.ini") == {}

    def test_sectionless_file_reads_empty(self, tmp_path: Path) -> None:
        """A file without [Scan Info] is empty, never an exception."""
        bad = tmp_path / "bad.ini"
        bad.write_text("[Other]\nx = 1\n")
        assert read_scan_info_file(bad) == {}

    def test_malformed_file_reads_empty(self, tmp_path: Path) -> None:
        """Garbage is empty too: one bad scan must not end a day's read."""
        bad = tmp_path / "bad.ini"
        bad.write_text("this is not an ini at all\n===\n")
        assert read_scan_info_file(bad) == {}


class TestFirstLogTimestamp:
    """The honest start time for a scan."""

    def test_returns_the_first_records_timestamp(self, tmp_path: Path) -> None:
        """The first parsable header wins."""
        log = tmp_path / "scan.log"
        log.write_text(LOG_LINE)
        assert first_log_timestamp(log) == datetime(2026, 9, 11, 0, 8, 13, 905000)

    def test_skips_leading_unparsable_lines(self, tmp_path: Path) -> None:
        """Continuation or banner lines before the first record are ignored."""
        log = tmp_path / "scan.log"
        log.write_text("a banner line\n  indented continuation\n" + LOG_LINE)
        assert first_log_timestamp(log) == datetime(2026, 9, 11, 0, 8, 13, 905000)

    def test_missing_file_is_none(self, tmp_path: Path) -> None:
        """An absent log yields None rather than raising."""
        assert first_log_timestamp(tmp_path / "nope.log") is None

    def test_empty_and_headerless_are_none(self, tmp_path: Path) -> None:
        """A log with no parsable header yields None."""
        for body in ("", "nothing parsable here\n"):
            log = tmp_path / "scan.log"
            log.write_text(body)
            assert first_log_timestamp(log) is None
