"""Unit tests for scan_analysis.task_queue.

All tests use tmp_path and simple mock objects — no real data or network required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List

import yaml

from scan_analysis.task_queue import (
    CLAIM_STALE_AFTER_SECONDS,
    STATUS_DIR_NAME,
    TaskStatus,
    _is_stale,
    build_worklist,
    extract_scan_number,
    read_statuses,
    update_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_analyzer(analyzer_id: str, priority: int = 50) -> SimpleNamespace:
    """Minimal analyzer stub with the attributes task_queue inspects."""
    return SimpleNamespace(id=analyzer_id, priority=priority, device_name=analyzer_id)


def _write_status(status_dir: Path, ts: TaskStatus) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{ts.analyzer_id}.yaml"
    path.write_text(yaml.safe_dump(ts.to_dict()))


# ---------------------------------------------------------------------------
# extract_scan_number
# ---------------------------------------------------------------------------


class TestExtractScanNumber:
    def test_valid_lowercase(self):
        assert extract_scan_number("s042.txt") == 42

    def test_valid_uppercase(self):
        assert extract_scan_number("S001.txt") == 1

    def test_leading_zeros_stripped(self):
        assert extract_scan_number("s007.txt") == 7

    def test_non_sfile_name(self):
        assert extract_scan_number("scan042.txt") is None

    def test_wrong_extension(self):
        assert extract_scan_number("s042.csv") is None

    def test_empty_string(self):
        assert extract_scan_number("") is None

    def test_extra_suffix(self):
        assert extract_scan_number("s042_extra.txt") is None


# ---------------------------------------------------------------------------
# TaskStatus serialisation
# ---------------------------------------------------------------------------


class TestTaskStatusRoundtrip:
    def test_to_dict_contains_all_fields(self):
        ts = TaskStatus(analyzer_id="cam_a", priority=10, state="queued")
        d = ts.to_dict()
        assert d["analyzer_id"] == "cam_a"
        assert d["priority"] == 10
        assert d["state"] == "queued"

    def test_from_file_roundtrip(self, tmp_path):
        ts = TaskStatus(
            analyzer_id="cam_b",
            priority=5,
            state="done",
            claimed_by="host:123",
            display_files=["/tmp/out.png"],
        )
        path = tmp_path / "cam_b.yaml"
        path.write_text(yaml.safe_dump(ts.to_dict()))

        loaded = TaskStatus.from_file(path)
        assert loaded.analyzer_id == "cam_b"
        assert loaded.priority == 5
        assert loaded.state == "done"
        assert loaded.claimed_by == "host:123"
        assert loaded.display_files == ["/tmp/out.png"]

    def test_from_file_defaults_missing_fields(self, tmp_path):
        path = tmp_path / "minimal.yaml"
        path.write_text(yaml.safe_dump({"analyzer_id": "x", "state": "queued"}))

        loaded = TaskStatus.from_file(path)
        assert loaded.priority == 0
        assert loaded.claimed_by is None
        assert loaded.display_files is None


# ---------------------------------------------------------------------------
# _is_stale
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class TestIsStale:
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def _claimed(self, seconds_ago: float, *, no_heartbeat: bool = False) -> TaskStatus:
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc) - timedelta(
            seconds=seconds_ago
        )
        return TaskStatus(
            analyzer_id="a",
            priority=0,
            state="claimed",
            claimed_at=_iso(ts),
            last_heartbeat=None if no_heartbeat else _iso(ts),
        )

    def test_queued_is_never_stale(self):
        ts = TaskStatus(analyzer_id="a", priority=0, state="queued")
        assert not _is_stale(ts, self.now)

    def test_done_is_never_stale(self):
        ts = TaskStatus(analyzer_id="a", priority=0, state="done")
        assert not _is_stale(ts, self.now)

    def test_fresh_claimed_is_not_stale(self):
        ts = self._claimed(seconds_ago=10)
        assert not _is_stale(ts, self.now)

    def test_old_claimed_is_stale(self):
        ts = self._claimed(seconds_ago=CLAIM_STALE_AFTER_SECONDS + 1)
        assert _is_stale(ts, self.now)

    def test_exactly_at_threshold_is_not_stale(self):
        ts = self._claimed(seconds_ago=CLAIM_STALE_AFTER_SECONDS)
        assert not _is_stale(ts, self.now)

    def test_no_heartbeat_falls_back_to_claimed_at(self):
        # claimed_at is old, no heartbeat — should be stale
        ts = self._claimed(seconds_ago=CLAIM_STALE_AFTER_SECONDS + 1, no_heartbeat=True)
        assert _is_stale(ts, self.now)

    def test_no_timestamps_at_all_is_stale(self):
        ts = TaskStatus(
            analyzer_id="a",
            priority=0,
            state="claimed",
            claimed_at=None,
            last_heartbeat=None,
        )
        assert _is_stale(ts, self.now)


# ---------------------------------------------------------------------------
# read_statuses
# ---------------------------------------------------------------------------


class TestReadStatuses:
    def test_missing_status_dir_returns_empty(self, tmp_path):
        scan_folder = tmp_path / "scan"
        scan_folder.mkdir()
        assert read_statuses(scan_folder) == []

    def test_empty_status_dir_returns_empty(self, tmp_path):
        scan_folder = tmp_path / "scan"
        (scan_folder / STATUS_DIR_NAME).mkdir(parents=True)
        assert read_statuses(scan_folder) == []

    def test_reads_all_yaml_files(self, tmp_path):
        scan_folder = tmp_path / "scan"
        status_dir = scan_folder / STATUS_DIR_NAME
        for i, state in enumerate(["queued", "done", "failed"]):
            _write_status(
                status_dir,
                TaskStatus(analyzer_id=f"analyzer_{i}", priority=i, state=state),
            )

        statuses = read_statuses(scan_folder)
        assert len(statuses) == 3
        states = {s.state for s in statuses}
        assert states == {"queued", "done", "failed"}

    def test_ignores_non_yaml_files(self, tmp_path):
        scan_folder = tmp_path / "scan"
        status_dir = scan_folder / STATUS_DIR_NAME
        status_dir.mkdir(parents=True)
        (status_dir / "notes.txt").write_text("not a status file")
        _write_status(
            status_dir, TaskStatus(analyzer_id="a", priority=0, state="queued")
        )

        statuses = read_statuses(scan_folder)
        assert len(statuses) == 1


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    def test_creates_new_file(self, tmp_path):
        scan_folder = tmp_path / "scan"
        scan_folder.mkdir()
        update_status(scan_folder, "cam_a", state="queued", priority=10)

        statuses = read_statuses(scan_folder)
        assert len(statuses) == 1
        assert statuses[0].state == "queued"
        assert statuses[0].priority == 10

    def test_updates_state_of_existing_file(self, tmp_path):
        scan_folder = tmp_path / "scan"
        status_dir = scan_folder / STATUS_DIR_NAME
        _write_status(
            status_dir, TaskStatus(analyzer_id="cam_a", priority=5, state="queued")
        )

        update_status(scan_folder, "cam_a", state="done", display_files=["/out.png"])

        loaded = TaskStatus.from_file(status_dir / "cam_a.yaml")
        assert loaded.state == "done"
        assert loaded.display_files == ["/out.png"]

    def test_partial_update_preserves_other_fields(self, tmp_path):
        scan_folder = tmp_path / "scan"
        status_dir = scan_folder / STATUS_DIR_NAME
        _write_status(
            status_dir,
            TaskStatus(
                analyzer_id="cam_a",
                priority=5,
                state="claimed",
                claimed_by="host:99",
            ),
        )

        # Only update last_heartbeat — claimed_by should be preserved
        update_status(scan_folder, "cam_a", last_heartbeat="2025-01-01T00:00:00+00:00")

        loaded = TaskStatus.from_file(status_dir / "cam_a.yaml")
        assert loaded.claimed_by == "host:99"
        assert loaded.last_heartbeat == "2025-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# build_worklist
# ---------------------------------------------------------------------------


class TestBuildWorklist:
    """Tests for build_worklist using tmp_path and monkeypatching ScanPaths."""

    def _setup(
        self,
        tmp_path: Path,
        monkeypatch,
        statuses: List[TaskStatus],
        scan_number: int = 1,
    ) -> tuple:
        """Create a fake scan folder with status files; patch ScanPaths."""
        from geecs_data_utils import ScanTag

        scan_folder = tmp_path / f"Scan{scan_number:03d}"
        scan_folder.mkdir()
        status_dir = scan_folder / STATUS_DIR_NAME
        for ts in statuses:
            _write_status(status_dir, ts)

        tag = ScanTag(year=2025, month=1, day=1, number=scan_number, experiment="Test")

        import scan_analysis.task_queue as tq

        monkeypatch.setattr(
            tq.ScanPaths,
            "get_scan_folder_path",
            staticmethod(lambda tag, base_directory=None: scan_folder),
        )

        return tag, scan_folder

    def test_queued_task_is_included(self, tmp_path, monkeypatch):
        analyzer = _make_analyzer("cam_a", priority=10)
        ts = TaskStatus(analyzer_id="cam_a", priority=10, state="queued")
        tag, _ = self._setup(tmp_path, monkeypatch, [ts])

        worklist = build_worklist([tag], [analyzer])
        assert len(worklist) == 1
        assert worklist[0][2] is analyzer

    def test_done_task_excluded_by_default(self, tmp_path, monkeypatch):
        analyzer = _make_analyzer("cam_a")
        ts = TaskStatus(analyzer_id="cam_a", priority=10, state="done")
        tag, _ = self._setup(tmp_path, monkeypatch, [ts])

        worklist = build_worklist([tag], [analyzer])
        assert worklist == []

    def test_done_task_included_with_rerun_completed(self, tmp_path, monkeypatch):
        analyzer = _make_analyzer("cam_a")
        ts = TaskStatus(analyzer_id="cam_a", priority=10, state="done")
        tag, _ = self._setup(tmp_path, monkeypatch, [ts])

        worklist = build_worklist([tag], [analyzer], rerun_completed=True)
        assert len(worklist) == 1

    def test_failed_task_excluded_by_default(self, tmp_path, monkeypatch):
        analyzer = _make_analyzer("cam_a")
        ts = TaskStatus(analyzer_id="cam_a", priority=10, state="failed")
        tag, _ = self._setup(tmp_path, monkeypatch, [ts])

        assert build_worklist([tag], [analyzer]) == []

    def test_failed_task_included_with_rerun_failed(self, tmp_path, monkeypatch):
        analyzer = _make_analyzer("cam_a")
        ts = TaskStatus(analyzer_id="cam_a", priority=10, state="failed")
        tag, _ = self._setup(tmp_path, monkeypatch, [ts])

        worklist = build_worklist([tag], [analyzer], rerun_failed=True)
        assert len(worklist) == 1

    def test_stale_claimed_task_included(self, tmp_path, monkeypatch):
        analyzer = _make_analyzer("cam_a")
        old_hb = (
            datetime.now(timezone.utc)
            - timedelta(seconds=CLAIM_STALE_AFTER_SECONDS + 60)
        ).isoformat()
        ts = TaskStatus(
            analyzer_id="cam_a",
            priority=10,
            state="claimed",
            last_heartbeat=old_hb,
        )
        tag, _ = self._setup(tmp_path, monkeypatch, [ts])

        worklist = build_worklist([tag], [analyzer])
        assert len(worklist) == 1

    def test_fresh_claimed_task_excluded(self, tmp_path, monkeypatch):
        analyzer = _make_analyzer("cam_a")
        fresh_hb = datetime.now(timezone.utc).isoformat()
        ts = TaskStatus(
            analyzer_id="cam_a",
            priority=10,
            state="claimed",
            last_heartbeat=fresh_hb,
        )
        tag, _ = self._setup(tmp_path, monkeypatch, [ts])

        assert build_worklist([tag], [analyzer]) == []

    def test_no_status_file_means_included(self, tmp_path, monkeypatch):
        """An analyzer with no status file yet should be included."""
        analyzer = _make_analyzer("cam_new")
        tag, _ = self._setup(tmp_path, monkeypatch, [])  # no status files at all

        worklist = build_worklist([tag], [analyzer])
        assert len(worklist) == 1

    def test_sorted_by_priority_then_scan_number(self, tmp_path, monkeypatch):
        from geecs_data_utils import ScanTag
        import scan_analysis.task_queue as tq

        # Two scan folders with two analyzers each
        folders = {}
        for num in [1, 2]:
            folder = tmp_path / f"Scan{num:03d}"
            folder.mkdir()
            folders[num] = folder

        def fake_get_folder(tag, base_directory=None):
            return folders[tag.number]

        monkeypatch.setattr(
            tq.ScanPaths, "get_scan_folder_path", staticmethod(fake_get_folder)
        )

        hi_prio = _make_analyzer("hi", priority=0)
        lo_prio = _make_analyzer("lo", priority=100)

        tags = [
            ScanTag(year=2025, month=1, day=1, number=1, experiment="Test"),
            ScanTag(year=2025, month=1, day=1, number=2, experiment="Test"),
        ]
        worklist = build_worklist(tags, [hi_prio, lo_prio])

        priorities = [item[0] for item in worklist]
        assert priorities == sorted(priorities), "worklist not sorted by priority"
        # high-priority analyzer should come before low-priority for same scan
        ids = [item[2].id for item in worklist]
        assert ids.index("hi") < ids.index("lo")
