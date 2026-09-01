"""Pin the ``analysis_status/`` writer against the shared reader (#682).

``TaskStatus.to_dict()`` in ``task_queue.py`` is the authoritative shape of
the per-task YAML files; ``geecs_data_utils.analysis_status`` is the one
read-side view every other package consumes them through (GEECS-MCP's
``get_scan_analysis`` today).  A stale prose copy of the shape once
shipped a dead parser downstream (the #675 review; the #679 drift
class) — so this suite, which can import both sides, reads what the REAL
writer produces through the shared reader and fails the moment they
disagree.  When the writer grows a field: extend ``STATUS_FIELDS`` +
``AnalysisStatus`` in GEECS-Data-Utils in the same PR.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml
from geecs_data_utils import ScanTag
from geecs_data_utils import analysis_status as shared

import scan_analysis.task_queue as tq
from scan_analysis.task_queue import (
    STATUS_DIR_NAME,
    TaskStatus,
    analyzer_task_id,
    init_status_for_scan,
    update_status,
)


def _scan_folder(tmp_path: Path) -> Path:
    scan_folder = tmp_path / "scans" / "Scan042"
    scan_folder.mkdir(parents=True)
    return scan_folder


class TestShapeContract:
    def test_status_dir_name_agrees(self):
        assert STATUS_DIR_NAME == shared.STATUS_DIR_NAME

    def test_to_dict_keys_are_exactly_the_shared_fields(self):
        """The drift pin: a new writer key fails here until the reader knows it."""
        record = TaskStatus(analyzer_id="a", priority=1, state="queued")
        assert tuple(record.to_dict()) == shared.STATUS_FIELDS

    def test_writer_states_are_the_documented_ones(self):
        # The reader passes unknown states through; this keeps the two
        # packages' vocabularies from drifting silently.
        written = {"queued", "claimed", "done", "failed", "no_data"}
        assert written == set(shared.STATUS_STATES)


class TestRealWriterOutput:
    def test_update_status_full_record_reads_field_for_field(self, tmp_path):
        scan_folder = _scan_folder(tmp_path)
        claimed_at = datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc)
        heartbeat = datetime(2026, 8, 22, 10, 5, tzinfo=timezone.utc)
        update_status(
            scan_folder,
            "topview_baseline",
            priority=7,
            state="failed",
            error="no s-file columns for UC_MagSpec",
            claimed_by="runner-1",
            claimed_at=claimed_at.isoformat(),  # the writer's own stamp form
            last_heartbeat=heartbeat.isoformat(),
            display_files=[str(tmp_path / "analysis" / "Scan042" / "a.png")],
        )

        records = shared.read_analysis_statuses(scan_folder)
        assert list(records) == ["topview_baseline"]
        record = records["topview_baseline"]
        assert record.readable
        assert record.task_id == "topview_baseline"
        assert record.analyzer_id == "topview_baseline"
        assert record.priority == 7
        assert record.state == "failed"
        assert record.error == "no s-file columns for UC_MagSpec"
        assert record.claimed_by == "runner-1"
        assert record.claimed_at == claimed_at
        assert record.last_heartbeat == heartbeat
        assert record.display_files == (
            str(tmp_path / "analysis" / "Scan042" / "a.png"),
        )
        assert record.heartbeat_age_s(heartbeat) == 0.0

    def test_init_status_for_scan_queued_record(self, tmp_path, monkeypatch):
        scan_folder = _scan_folder(tmp_path)
        monkeypatch.setattr(
            tq.ScanPaths,
            "get_scan_folder_path",
            staticmethod(lambda tag, base_directory=None: scan_folder),
        )
        tag = ScanTag(year=2026, month=8, day=22, number=42, experiment="Test")
        analyzer = SimpleNamespace(id="pin_check", priority=3)
        init_status_for_scan(tag, [analyzer], base_directory=tmp_path)

        task_id = analyzer_task_id(analyzer)
        record = shared.read_analysis_statuses(scan_folder)[task_id]
        assert record == shared.AnalysisStatus(
            task_id=task_id, analyzer_id=task_id, priority=3, state="queued"
        )

    def test_shared_reader_agrees_with_the_writers_own_reader(self, tmp_path):
        """Field-for-field agreement with ``TaskStatus.from_file``."""
        scan_folder = _scan_folder(tmp_path)
        heartbeat = datetime.now(timezone.utc).isoformat()
        update_status(
            scan_folder,
            "magspec",
            priority=2,
            state="claimed",
            claimed_by="runner-2",
            claimed_at=heartbeat,
            last_heartbeat=heartbeat,
        )
        path = scan_folder / STATUS_DIR_NAME / "magspec.yaml"
        own = TaskStatus.from_file(path)
        ours = shared.read_analysis_status(path)
        assert ours.analyzer_id == own.analyzer_id
        assert ours.priority == own.priority
        assert ours.state == own.state
        assert ours.error == own.error
        assert ours.claimed_by == own.claimed_by
        assert ours.claimed_at == tq._parse_ts(own.claimed_at)
        assert ours.last_heartbeat == tq._parse_ts(own.last_heartbeat)
        assert list(ours.display_files) == (own.display_files or [])

    def test_claim_lock_is_invisible_to_the_reader(self, tmp_path):
        scan_folder = _scan_folder(tmp_path)
        update_status(scan_folder, "magspec", state="queued")
        assert tq.try_acquire_claim(scan_folder, "magspec", owner="runner-1")
        try:
            assert list(shared.read_analysis_statuses(scan_folder)) == ["magspec"]
        finally:
            tq.release_claim(scan_folder, "magspec")

    def test_to_dict_document_reads_the_same_as_the_shared_reader(self, tmp_path):
        # The literal to_dict() → yaml.safe_dump path the writer takes,
        # without going through update_status's merge.
        status_dir = _scan_folder(tmp_path) / STATUS_DIR_NAME
        status_dir.mkdir()
        original = TaskStatus(
            analyzer_id="done_task",
            priority=0,
            state="done",
            display_files=["/x/one.png", "/x/two.png"],
        )
        (status_dir / "done_task.yaml").write_text(yaml.safe_dump(original.to_dict()))
        record = shared.read_analysis_statuses(status_dir.parent)["done_task"]
        assert record.display_files == ("/x/one.png", "/x/two.png")
        assert (record.state, record.priority, record.error) == ("done", 0, None)
        assert record.claimed_at is None and record.last_heartbeat is None
