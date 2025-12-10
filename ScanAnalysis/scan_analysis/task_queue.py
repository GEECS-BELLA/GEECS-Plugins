"""
Lightweight task queue for scan analysis.

Single-app assumptions:
- No cross-process locking (safe to run one app). Future multi-app can add lock files.
- Per-scan status files stored under `<scan_folder>/analysis_status/<analyzer_id>.yaml`.
- Worklist sorted by (priority, scan_number).

States: queued -> claimed -> done/failed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from geecs_data_utils import ScanPaths, ScanTag
from scan_analysis.config.analyzer_factory import create_analyzer
from scan_analysis.config.config_loader import load_experiment_config

logger = logging.getLogger(__name__)

STATUS_DIR_NAME = "analysis_status"


@dataclass
class TaskStatus:
    """Status record for a single analyzer on a specific scan."""

    analyzer_id: str
    priority: int
    state: str  # queued, claimed, done, failed
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialize this status to a dict for YAML storage."""
        return {
            "analyzer_id": self.analyzer_id,
            "priority": self.priority,
            "state": self.state,
            "error": self.error,
        }

    @staticmethod
    def from_file(path: Path) -> "TaskStatus":
        """Create a TaskStatus from a YAML file on disk."""
        data = yaml.safe_load(path.read_text()) or {}
        return TaskStatus(
            analyzer_id=data["analyzer_id"],
            priority=int(data["priority"]),
            state=data.get("state", "queued"),
            error=data.get("error"),
        )


def _status_dir(scan_folder: Path) -> Path:
    return scan_folder / STATUS_DIR_NAME


def _status_path(scan_folder: Path, analyzer_id: str) -> Path:
    return _status_dir(scan_folder) / f"{analyzer_id}.yaml"


def init_status_for_scan(
    scan_tag: ScanTag,
    analyzers: Iterable,
    *,
    base_directory: Optional[Path] = None,
) -> None:
    """
    Create queued status files for all analyzers in a scan folder if missing.

    Idempotent for existing status files.
    """
    scan_folder = ScanPaths.get_scan_folder_path(
        tag=scan_tag, base_directory=base_directory
    )
    status_dir = _status_dir(scan_folder)
    status_dir.mkdir(parents=True, exist_ok=True)

    for analyzer in analyzers:
        analyzer_id = getattr(
            analyzer, "id", getattr(analyzer, "device_name", "unknown")
        )
        # If analyzer objects have no id, fall back to class name + device
        analyzer_id = (
            analyzer_id
            or f"{analyzer.__class__.__name__}_{getattr(analyzer, 'device_name', '')}"
        )
        path = _status_path(scan_folder, analyzer_id)
        if path.exists():
            continue
        ts = TaskStatus(
            analyzer_id=analyzer_id,
            priority=getattr(analyzer, "priority", 100),
            state="queued",
        )
        path.write_text(yaml.safe_dump(ts.to_dict()))


def read_statuses(scan_folder: Path) -> List[TaskStatus]:
    """Read all status files in a scan folder."""
    status_dir = _status_dir(scan_folder)
    if not status_dir.exists():
        return []
    statuses: List[TaskStatus] = []
    for f in status_dir.glob("*.yaml"):
        try:
            statuses.append(TaskStatus.from_file(f))
        except Exception as exc:  # pragma: no cover - log and skip
            logger.warning("Failed to read status %s: %s", f, exc)
    return statuses


def update_status(
    scan_folder: Path,
    analyzer_id: str,
    *,
    priority: int,
    state: str,
    error: Optional[str] = None,
) -> None:
    """Update status file for a given analyzer in a scan folder."""
    status_dir = _status_dir(scan_folder)
    status_dir.mkdir(parents=True, exist_ok=True)
    path = _status_path(scan_folder, analyzer_id)
    ts = TaskStatus(
        analyzer_id=analyzer_id, priority=priority, state=state, error=error
    )
    path.write_text(yaml.safe_dump(ts.to_dict()))


def build_worklist(
    scan_tags: Iterable[ScanTag],
    analyzers: Iterable,
    *,
    base_directory: Optional[Path] = None,
    rerun_completed: bool = False,
    rerun_failed: bool = True,
) -> List[tuple[int, ScanTag, object]]:
    """
    Build a list of (priority, scan_tag, analyzer) for tasks with state=queued.

    Sorted by (priority, scan number).
    """
    work: List[tuple[int, ScanTag, object]] = []
    for tag in scan_tags:
        scan_folder = ScanPaths.get_scan_folder_path(
            tag=tag, base_directory=base_directory
        )
        statuses = {s.analyzer_id: s for s in read_statuses(scan_folder)}
        for analyzer in analyzers:
            analyzer_id = getattr(
                analyzer, "id", getattr(analyzer, "device_name", "unknown")
            )
            analyzer_id = (
                analyzer_id
                or f"{analyzer.__class__.__name__}_{getattr(analyzer, 'device_name', '')}"
            )
            st = statuses.get(analyzer_id)
            include = st is None or st.state == "queued"
            if st and st.state == "done" and rerun_completed:
                include = True
            if st and st.state == "failed" and rerun_failed:
                include = True
            if include:
                priority = getattr(analyzer, "priority", 100)
                work.append((priority, tag, analyzer))
    return sorted(work, key=lambda item: (item[0], item[1].number))


def run_worklist(
    worklist: List[tuple[int, ScanTag, object]],
    *,
    base_directory: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    """
    Run analyzers on the given worklist (single-app; no locking).

    Updates status files to claimed/done/failed.
    If dry_run=True, skip analyzer execution but update status as done.
    """
    for priority, tag, analyzer in worklist:
        scan_folder = ScanPaths.get_scan_folder_path(
            tag=tag, base_directory=base_directory
        )
        analyzer_id = getattr(
            analyzer, "id", getattr(analyzer, "device_name", "unknown")
        )
        analyzer_id = (
            analyzer_id
            or f"{analyzer.__class__.__name__}_{getattr(analyzer, 'device_name', '')}"
        )

        update_status(scan_folder, analyzer_id, priority=priority, state="claimed")
        try:
            if not dry_run:
                analyzer.run_analysis(tag)
            update_status(scan_folder, analyzer_id, priority=priority, state="done")
        except Exception as exc:  # pragma: no cover - log failure and continue
            logger.exception("Analyzer %s failed on %s: %s", analyzer_id, tag, exc)
            update_status(
                scan_folder,
                analyzer_id,
                priority=priority,
                state="failed",
                error=str(exc),
            )


def load_analyzers_from_config(
    experiment: str, *, config_dir: Optional[Path] = None
) -> List[object]:
    """Load analyzers (sorted by priority) for an experiment config."""
    cfg = load_experiment_config(experiment, config_dir=config_dir)
    return [create_analyzer(a) for a in cfg.get_analyzers_by_priority()]
