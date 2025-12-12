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
import os
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from geecs_data_utils import ScanPaths, ScanTag
from scan_analysis.config.analyzer_factory import create_analyzer
from scan_analysis.config.config_loader import load_experiment_config

logger = logging.getLogger(__name__)

STATUS_DIR_NAME = "analysis_status"
HEARTBEAT_INTERVAL_SECONDS = 30
CLAIM_STALE_AFTER_SECONDS = 180


@dataclass
class TaskStatus:
    """Status record for a single analyzer on a specific scan."""

    analyzer_id: str
    priority: int
    state: str  # queued, claimed, done, failed
    error: Optional[str] = None
    claimed_by: Optional[str] = None
    claimed_at: Optional[str] = None
    last_heartbeat: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialize this status to a dict for YAML storage."""
        return {
            "analyzer_id": self.analyzer_id,
            "priority": self.priority,
            "state": self.state,
            "error": self.error,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at,
            "last_heartbeat": self.last_heartbeat,
        }

    @staticmethod
    def from_file(path: Path) -> "TaskStatus":
        """Create a TaskStatus from a YAML file on disk."""
        data = yaml.safe_load(path.read_text()) or {}
        return TaskStatus(
            analyzer_id=data.get("analyzer_id", ""),
            priority=int(data.get("priority", 0)),
            state=data.get("state", "queued"),
            error=data.get("error"),
            claimed_by=data.get("claimed_by"),
            claimed_at=data.get("claimed_at"),
            last_heartbeat=data.get("last_heartbeat"),
        )


def _status_dir(scan_folder: Path) -> Path:
    return scan_folder / STATUS_DIR_NAME


def _status_path(scan_folder: Path, analyzer_id: str) -> Path:
    return _status_dir(scan_folder) / f"{analyzer_id}.yaml"


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _is_stale(status: TaskStatus, now: datetime) -> bool:
    if status.state != "claimed":
        return False
    last = _parse_ts(status.last_heartbeat) or _parse_ts(status.claimed_at)
    if last is None:
        return True
    return (now - last).total_seconds() > CLAIM_STALE_AFTER_SECONDS


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
    priority: Optional[int] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    claimed_by: Optional[str] = None,
    claimed_at: Optional[str] = None,
    last_heartbeat: Optional[str] = None,
) -> None:
    """Update status file for a given analyzer in a scan folder."""
    status_dir = _status_dir(scan_folder)
    status_dir.mkdir(parents=True, exist_ok=True)
    path = _status_path(scan_folder, analyzer_id)
    if path.exists():
        current = TaskStatus.from_file(path)
    else:
        current = TaskStatus(
            analyzer_id=analyzer_id,
            priority=priority or 0,
            state=state or "queued",
        )

    current.priority = priority if priority is not None else current.priority
    current.state = state if state is not None else current.state
    current.error = error if error is not None else current.error
    current.claimed_by = claimed_by if claimed_by is not None else current.claimed_by
    current.claimed_at = claimed_at if claimed_at is not None else current.claimed_at
    current.last_heartbeat = (
        last_heartbeat if last_heartbeat is not None else current.last_heartbeat
    )

    ts = current
    path.write_text(yaml.safe_dump(ts.to_dict()))


def reset_status_for_scan(
    scan_tag: ScanTag,
    analyzers: Iterable,
    *,
    base_directory: Optional[Path] = None,
    states_to_reset: tuple[str, ...] = ("done", "failed", "claimed"),
    rerun_only_ids: Optional[Iterable[str]] = None,
    rerun_skip_ids: Optional[Iterable[str]] = None,
) -> None:
    """
    Reset status files to queued for a scan, useful before re-running analyses.

    Parameters
    ----------
    scan_tag : ScanTag
        Target scan to reset.
    analyzers : Iterable
        Analyzer objects to consider (used to derive ids).
    base_directory : Path, optional
        Root directory override.
    states_to_reset : tuple[str, ...]
        Only statuses in this set will be reset to queued.
    rerun_only_ids : Iterable[str], optional
        If provided, only analyzer ids in this set are reset.
    rerun_skip_ids : Iterable[str], optional
        Analyzer ids to skip when resetting.
    """
    only_ids = set(rerun_only_ids) if rerun_only_ids is not None else None
    skip_ids = set(rerun_skip_ids) if rerun_skip_ids is not None else set()
    scan_folder = ScanPaths.get_scan_folder_path(
        tag=scan_tag, base_directory=base_directory
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
        if analyzer_id in skip_ids:
            continue
        if only_ids is not None and analyzer_id not in only_ids:
            continue
        st = statuses.get(analyzer_id)
        if st and st.state in states_to_reset:
            update_status(
                scan_folder,
                analyzer_id,
                priority=st.priority,
                state="queued",
                error=None,
                claimed_by=None,
                claimed_at=None,
                last_heartbeat=None,
            )


def build_worklist(
    scan_tags: Iterable[ScanTag],
    analyzers: Iterable,
    *,
    base_directory: Optional[Path] = None,
    rerun_completed: bool = False,
    rerun_failed: bool = True,
    rerun_claimed: bool = True,
    rerun_only_ids: Optional[Iterable[str]] = None,
    rerun_skip_ids: Optional[Iterable[str]] = None,
) -> List[tuple[int, ScanTag, object]]:
    """
    Build a list of (priority, scan_tag, analyzer) for tasks with state=queued.

    Sorted by (priority, scan number).
    """
    only_ids = set(rerun_only_ids) if rerun_only_ids is not None else None
    skip_ids = set(rerun_skip_ids) if rerun_skip_ids is not None else set()
    work: List[tuple[int, ScanTag, object]] = []
    now = datetime.now(timezone.utc)
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
            eligible_for_rerun = analyzer_id not in skip_ids and (
                only_ids is None or analyzer_id in only_ids
            )
            if st and st.state == "done" and rerun_completed and eligible_for_rerun:
                include = True
            if st and st.state == "failed" and rerun_failed and eligible_for_rerun:
                include = True
            if st and st.state == "claimed" and rerun_claimed and eligible_for_rerun:
                include = _is_stale(st, now)
            if st and st.state == "claimed" and rerun_completed and eligible_for_rerun:
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

        logger.info(
            "run_worklist: claiming scan=%s analyzer=%s priority=%s dry_run=%s",
            tag,
            analyzer_id,
            priority,
            dry_run,
        )
        owner = f"{socket.gethostname()}:{os.getpid()}"
        now_iso = datetime.now(timezone.utc).isoformat()
        update_status(
            scan_folder,
            analyzer_id,
            priority=priority,
            state="claimed",
            claimed_by=owner,
            claimed_at=now_iso,
            last_heartbeat=now_iso,
        )
        stop_event = threading.Event()
        hb_thread = threading.Thread(
            target=_heartbeat_updater,
            args=(scan_folder, analyzer_id, priority, stop_event),
            daemon=True,
        )
        hb_thread.start()
        try:
            if not dry_run:
                analyzer.run_analysis(tag)
            update_status(
                scan_folder,
                analyzer_id,
                priority=priority,
                state="done",
                error=None,
                claimed_by=None,
                claimed_at=None,
                last_heartbeat=None,
            )
            logger.info(
                "run_worklist: completed scan=%s analyzer=%s priority=%s",
                tag,
                analyzer_id,
                priority,
            )
        except Exception as exc:  # pragma: no cover - log failure and continue
            logger.exception("Analyzer %s failed on %s: %s", analyzer_id, tag, exc)
            update_status(
                scan_folder,
                analyzer_id,
                priority=priority,
                state="failed",
                error=str(exc),
                claimed_by=None,
                claimed_at=None,
                last_heartbeat=None,
            )
        finally:
            stop_event.set()
            hb_thread.join(timeout=HEARTBEAT_INTERVAL_SECONDS)


def load_analyzers_from_config(
    experiment: str, *, config_dir: Optional[Path] = None
) -> List[object]:
    """Load analyzers (sorted by priority) for an experiment config."""
    cfg = load_experiment_config(experiment, config_dir=config_dir)
    return [create_analyzer(a) for a in cfg.get_analyzers_by_priority()]


def _heartbeat_updater(
    scan_folder: Path,
    analyzer_id: str,
    priority: int,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(HEARTBEAT_INTERVAL_SECONDS):
        hb = datetime.now(timezone.utc).isoformat()
        update_status(
            scan_folder,
            analyzer_id,
            priority=priority,
            state="claimed",
            last_heartbeat=hb,
        )
