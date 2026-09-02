"""Hermetic tests for the shared ``analysis_status/`` reader (#682).

Hand-written YAML stands in for the writer here; the writer/reader
contract itself is pinned in ScanAnalysis's suite
(``tests/test_analysis_status_contract.py``), where the real writer is
importable.  These tests pin the tolerance union the reader promises
its consumers and the read-only discipline.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from geecs_data_utils import AnalysisStatus, read_analysis_statuses
from geecs_data_utils.analysis_status import (
    STATUS_DIR_NAME,
    STATUS_FIELDS,
    analysis_status_dir,
    parse_status_timestamp,
    read_analysis_status,
)


def _status_dir(tmp_path: Path) -> Path:
    scan_folder = tmp_path / "scans" / "Scan007"
    status_dir = scan_folder / STATUS_DIR_NAME
    status_dir.mkdir(parents=True)
    return status_dir


def _write(status_dir: Path, name: str, document) -> Path:
    path = status_dir / name
    path.write_text(yaml.safe_dump(document))
    return path


class TestReadOnly:
    def test_missing_scan_folder_reads_empty_and_creates_nothing(self, tmp_path):
        scan_folder = tmp_path / "scans" / "Scan099"
        assert read_analysis_statuses(scan_folder) == {}
        # THE invariant: analysis-side code never plants a scan folder.
        assert not scan_folder.exists()
        assert list(tmp_path.iterdir()) == []

    def test_missing_status_dir_reads_empty_and_creates_nothing(self, tmp_path):
        scan_folder = tmp_path / "scans" / "Scan007"
        scan_folder.mkdir(parents=True)
        assert read_analysis_statuses(scan_folder) == {}
        assert list(scan_folder.iterdir()) == []

    def test_empty_status_dir_reads_empty(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        assert read_analysis_statuses(status_dir.parent) == {}

    def test_analysis_status_dir_is_pure(self, tmp_path):
        scan_folder = tmp_path / "scans" / "Scan007"
        assert analysis_status_dir(scan_folder) == scan_folder / "analysis_status"
        assert not scan_folder.exists()


class TestFullRecord:
    def test_writer_shape_reads_field_for_field(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        document = {
            "analyzer_id": "topview_baseline",
            "priority": 7,
            "state": "done",
            "error": None,
            "claimed_by": "runner-1",
            "claimed_at": "2026-08-22T10:00:00+00:00",
            "last_heartbeat": "2026-08-22T10:05:00+00:00",
            "display_files": ["/share/analysis/Scan007/UC_TopView/summary.png"],
        }
        assert tuple(document) == STATUS_FIELDS
        _write(status_dir, "topview_baseline.yaml", document)
        # Every writer key must be surfaced by the record (review finding
        # on #750: STATUS_FIELDS alone could grow without the dataclass).
        assert set(STATUS_FIELDS) <= {
            f.name for f in dataclasses.fields(AnalysisStatus)
        }

        records = read_analysis_statuses(status_dir.parent)
        assert list(records) == ["topview_baseline"]
        record = records["topview_baseline"]
        assert record.readable
        assert record.task_id == "topview_baseline"
        assert record.analyzer_id == "topview_baseline"
        assert record.priority == 7
        assert record.state == "done"
        assert record.error is None
        assert record.claimed_by == "runner-1"
        assert record.claimed_at == datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc)
        assert record.last_heartbeat == datetime(
            2026, 8, 22, 10, 5, tzinfo=timezone.utc
        )
        assert record.display_files == (
            "/share/analysis/Scan007/UC_TopView/summary.png",
        )

    def test_heartbeat_age(self):
        stamp = datetime(2026, 8, 22, 10, 5, tzinfo=timezone.utc)
        record = AnalysisStatus(task_id="t", last_heartbeat=stamp)
        assert record.heartbeat_age_s(stamp + timedelta(seconds=90)) == 90.0
        # A naive ``now`` is UTC, like the stamps.
        assert record.heartbeat_age_s(datetime(2026, 8, 22, 10, 6)) == 60.0
        assert AnalysisStatus(task_id="t").heartbeat_age_s() is None
        # Default ``now`` is the wall clock: a 2026 stamp reads as positive age.
        assert record.heartbeat_age_s() > 0

    def test_record_is_frozen(self):
        record = AnalysisStatus(task_id="t")
        try:
            record.state = "done"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("AnalysisStatus must be frozen")


class TestDirectoryListing:
    def test_sorted_by_filename_and_non_yaml_entries_ignored(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        _write(status_dir, "zeta.yaml", {"state": "queued"})
        _write(status_dir, "alpha.yaml", {"state": "done"})
        # The queue's own readers glob *.yaml only — a .yml it would never
        # run must not surface as a task (review finding on #750).
        _write(status_dir, "phantom.yml", {"state": "queued"})
        (status_dir / "alpha.claim").write_text("runner-1 2026-08-22T10:00:00+00:00\n")
        (status_dir / "tmpabc123.tmp").write_text("state: claimed\n")
        (status_dir / "notes.txt").write_text("state: claimed\n")
        (status_dir / "subdir").mkdir()

        records = read_analysis_statuses(status_dir.parent)
        assert list(records) == ["alpha", "zeta"]
        assert records["alpha"].state == "done"
        assert records["zeta"].state == "queued"

    def test_partial_document_reads_with_defaults(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        _write(
            status_dir, "magspec.yaml", {"analyzer_id": "magspec", "state": "queued"}
        )
        record = read_analysis_statuses(status_dir.parent)["magspec"]
        assert record == AnalysisStatus(
            task_id="magspec", analyzer_id="magspec", state="queued"
        )
        assert record.display_files == ()
        assert record.heartbeat_age_s() is None

    def test_empty_file_reads_as_a_readable_record_of_nones(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        (status_dir / "blank.yaml").write_text("")
        record = read_analysis_statuses(status_dir.parent)["blank"]
        assert record.readable
        assert record == AnalysisStatus(task_id="blank")


class TestTolerance:
    def test_torn_yaml_is_unreadable_not_fatal(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        (status_dir / "torn.yaml").write_text("{ not: [valid")
        _write(status_dir, "fine.yaml", {"state": "done"})
        records = read_analysis_statuses(status_dir.parent)
        assert records["fine"].state == "done"
        torn = records["torn"]
        assert not torn.readable
        assert torn.unreadable  # the parser's message, for the caller to show
        assert torn.state is None

    def test_non_mapping_document_is_unreadable(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        (status_dir / "list.yaml").write_text("- queued\n- done\n")
        record = read_analysis_statuses(status_dir.parent)["list"]
        assert record.unreadable == "not a mapping"

    def test_directory_with_status_suffix_is_unreadable(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        (status_dir / "oops.yaml").mkdir()
        record = read_analysis_statuses(status_dir.parent)["oops"]
        assert not record.readable

    def test_odd_field_types_degrade_the_field_not_the_entry(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        _write(
            status_dir,
            "odd.yaml",
            {
                "state": "claimed",
                "priority": "high",
                "last_heartbeat": ["not", "a", "string"],
                "claimed_at": "yesterday-ish",
                "display_files": "not-a-list.png",
            },
        )
        record = read_analysis_statuses(status_dir.parent)["odd"]
        assert record.readable
        assert record.state == "claimed"
        assert record.priority is None
        assert record.last_heartbeat is None
        assert record.heartbeat_age_s() is None
        assert record.claimed_at is None
        assert record.display_files == ()  # a scalar is not a file list

    def test_display_files_keeps_only_strings(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        _write(
            status_dir,
            "mixed.yaml",
            {"state": "done", "display_files": ["a.png", 7, None, ["b.png"], "c.png"]},
        )
        record = read_analysis_statuses(status_dir.parent)["mixed"]
        assert record.display_files == ("a.png", "c.png")

    def test_priority_coercion(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        _write(status_dir, "s.yaml", {"priority": "12"})
        _write(status_dir, "f.yaml", {"priority": 3.0})
        _write(status_dir, "b.yaml", {"priority": True})
        _write(status_dir, "n.yaml", {"priority": None})
        records = read_analysis_statuses(status_dir.parent)
        assert records["s"].priority == 12
        assert records["f"].priority == 3
        assert records["b"].priority is None  # a bool is not a priority
        assert records["n"].priority is None

    def test_non_string_scalars_stringify(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        _write(status_dir, "num.yaml", {"state": 1, "error": 2.5, "claimed_by": None})
        record = read_analysis_statuses(status_dir.parent)["num"]
        assert record.state == "1"
        assert record.error == "2.5"
        assert record.claimed_by is None

    def test_unknown_fields_are_ignored(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        _write(status_dir, "extra.yaml", {"state": "done", "future_field": {"x": 1}})
        record = read_analysis_statuses(status_dir.parent)["extra"]
        assert record.readable and record.state == "done"

    def test_single_file_reader(self, tmp_path):
        status_dir = _status_dir(tmp_path)
        path = _write(status_dir, "one.yaml", {"state": "failed", "error": "boom"})
        record = read_analysis_status(path)
        assert (record.task_id, record.state, record.error) == ("one", "failed", "boom")
        assert not read_analysis_status(tmp_path / "nowhere.yaml").readable


class TestTimestamps:
    def test_iso_with_offset(self):
        assert parse_status_timestamp("2026-08-22T10:00:00+00:00") == datetime(
            2026, 8, 22, 10, 0, tzinfo=timezone.utc
        )

    def test_naive_is_utc(self):
        assert parse_status_timestamp("2026-08-22T10:00:00") == datetime(
            2026, 8, 22, 10, 0, tzinfo=timezone.utc
        )

    def test_writer_stamp_form_round_trips(self):
        # The writer stamps ``datetime.now(timezone.utc).isoformat()``.
        now = datetime.now(timezone.utc)
        assert parse_status_timestamp(now.isoformat()) == now

    def test_unquoted_stamp_already_a_datetime(self, tmp_path):
        # A hand-edited file whose stamp YAML parsed into a datetime.
        status_dir = _status_dir(tmp_path)
        (status_dir / "hand.yaml").write_text(
            "state: claimed\nlast_heartbeat: 2026-08-22 10:05:00\n"
        )
        record = read_analysis_statuses(status_dir.parent)["hand"]
        assert record.last_heartbeat == datetime(
            2026, 8, 22, 10, 5, tzinfo=timezone.utc
        )

    def test_garbage_is_none(self):
        for value in (None, "", "yesterday", 42, ["2026-08-22"], {"t": 1}):
            assert parse_status_timestamp(value) is None, value
