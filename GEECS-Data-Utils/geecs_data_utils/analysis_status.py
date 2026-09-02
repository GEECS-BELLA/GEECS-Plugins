"""Tolerant reader for a scan folder's ``analysis_status/`` task files (#682).

ScanAnalysis's task queue (``scan_analysis.task_queue``) writes one YAML
per (scan, analyzer) at ``<scan_folder>/analysis_status/<task_id>.yaml``;
``TaskStatus.to_dict()`` there is THE authoritative shape.  This module
is the one *schema-light* read-side view of those files for consumers
outside ScanAnalysis (GEECS-MCP's ``get_scan_analysis`` today; code that
needs the typed ``TaskStatus`` — e.g. to feed ``claim_is_active`` — keeps
using ``task_queue.read_statuses``, as MCP's ``run_tools`` does) — light on
purpose: it names the writer's documented fields (:data:`STATUS_FIELDS`)
and coerces each one tolerantly instead of validating a contract, so a
torn write mid-heartbeat, an odd field type, or a field this reader has
never heard of degrades one entry (or is ignored) and never fails the
caller.

The drift pin lives with the writer: ScanAnalysis's suite
(``tests/test_analysis_status_contract.py``) reads what the real writer
produces through this module and fails the moment ``TaskStatus.to_dict()``
and :data:`STATUS_FIELDS` disagree — the #679-style drift class (a stale
prose copy of the schema once shipped a dead parser downstream).

Strictly read-only (the repo's scan-folder invariant): a missing scan
folder or ``analysis_status/`` directory reads as empty; nothing is ever
created.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

#: The status directory's name inside a scan folder (the writer's
#: ``STATUS_DIR_NAME`` — the contract test asserts they agree).
STATUS_DIR_NAME = "analysis_status"

#: File suffixes read as status files — ``.yaml`` only, exactly what the
#: queue's own readers (``read_statuses``/``build_worklist``) glob: a
#: ``.yml`` the queue would never run must not show up as a task here.
#: Everything else in the directory (the ``.claim`` lock files, ``.tmp``
#: atomic-write leftovers) is ignored.
STATUS_FILE_SUFFIXES = (".yaml",)

#: The keys ``TaskStatus.to_dict()`` writes, in the writer's order.  When
#: the writer grows a field, the contract test fails until this tuple and
#: :class:`AnalysisStatus` catch up.
STATUS_FIELDS = (
    "analyzer_id",
    "priority",
    "state",
    "error",
    "claimed_by",
    "claimed_at",
    "last_heartbeat",
    "display_files",
)

#: The states the writer produces.  Documentation only — the reader
#: passes an unknown state through rather than rejecting the entry.
STATUS_STATES = ("queued", "claimed", "done", "failed", "no_data")


@dataclass(frozen=True)
class AnalysisStatus:
    """One task's status, as tolerantly read from its YAML file.

    Every writer field is present with the loosest useful type; an
    absent or malformed field is ``None`` (``()`` for ``display_files``),
    never an exception.  ``unreadable`` is set — and every other field is
    left at its default — when the file itself could not be parsed (a
    torn write, a non-mapping document, an undecodable byte).
    """

    #: The file stem — the writer's task id (``analyzer_task_id``).
    task_id: str
    state: Optional[str] = None
    error: Optional[str] = None
    analyzer_id: Optional[str] = None
    priority: Optional[int] = None
    claimed_by: Optional[str] = None
    #: Parsed timestamps, always tz-aware (the writer stamps UTC ISO-8601
    #: strings and assumes UTC when a stamp is naive — mirrored here).
    claimed_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    #: The writer's list with non-string entries dropped; a non-list
    #: value reads as empty.
    display_files: tuple[str, ...] = ()
    #: Why the file could not be parsed; ``None`` for a readable file.
    unreadable: Optional[str] = None

    @property
    def readable(self) -> bool:
        """True when the file parsed (``unreadable`` is ``None``)."""
        return self.unreadable is None

    def heartbeat_age_s(self, now: Optional[datetime] = None) -> Optional[float]:
        """Seconds since ``last_heartbeat``; ``None`` without a parsable stamp.

        Parameters
        ----------
        now : datetime, optional
            Reference time; defaults to ``datetime.now(timezone.utc)``.
            A naive value is taken as UTC, like the stamps themselves.
        """
        if self.last_heartbeat is None:
            return None
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return (now - self.last_heartbeat).total_seconds()


def analysis_status_dir(scan_folder: Path) -> Path:
    """The status directory path for ``scan_folder`` (pure — no I/O)."""
    return Path(scan_folder) / STATUS_DIR_NAME


def parse_status_timestamp(value: Any) -> Optional[datetime]:
    """Parse a writer timestamp tolerantly; ``None`` when it is not one.

    Accepts the writer's ISO-8601 strings and — for a hand-edited file
    whose unquoted stamp YAML already turned into a ``datetime`` — datetime
    objects.  Naive values are taken as UTC (the writer's own
    ``_parse_ts`` convention).  Anything else (empty, other types,
    unparsable text) is ``None``.
    """
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and value:
        try:
            stamp = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    else:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _as_str(value: Any) -> Optional[str]:
    """``None`` stays ``None``; strings pass through; other scalars stringify."""
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _as_int(value: Any) -> Optional[int]:
    """The writer's ``int`` field, or ``None`` when it does not read as one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_file_list(value: Any) -> tuple[str, ...]:
    """A list's string entries; anything that is not a list reads as empty."""
    if not isinstance(value, list):
        return ()
    return tuple(name for name in value if isinstance(name, str))


def read_analysis_status(path: Path) -> AnalysisStatus:
    """Read one status file; a file that cannot be parsed is ``unreadable``.

    The whole load-and-coerce sits inside one guard so that one odd YAML
    on a writable share degrades that one entry, never the caller.
    """
    task_id = Path(path).stem
    try:
        document = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(document, dict):
            raise ValueError("not a mapping")
        return AnalysisStatus(
            task_id=task_id,
            state=_as_str(document.get("state")),
            error=_as_str(document.get("error")),
            analyzer_id=_as_str(document.get("analyzer_id")),
            priority=_as_int(document.get("priority")),
            claimed_by=_as_str(document.get("claimed_by")),
            claimed_at=parse_status_timestamp(document.get("claimed_at")),
            last_heartbeat=parse_status_timestamp(document.get("last_heartbeat")),
            display_files=_as_file_list(document.get("display_files")),
        )
    except Exception as exc:  # a torn write mid-heartbeat is not our error
        return AnalysisStatus(task_id=task_id, unreadable=str(exc))


def read_analysis_statuses(scan_folder: Path) -> dict[str, AnalysisStatus]:
    """Read every status file under ``scan_folder/analysis_status/``.

    Parameters
    ----------
    scan_folder : Path
        The ``scans/Scan<NNN>/`` folder.  Never created: a missing scan
        folder or status directory reads as empty.

    Returns
    -------
    dict[str, AnalysisStatus]
        Keyed by task id (the file stem), in sorted filename order.
    """
    status_dir = analysis_status_dir(scan_folder)
    statuses: dict[str, AnalysisStatus] = {}
    if not status_dir.is_dir():
        return statuses
    for entry in sorted(status_dir.iterdir()):
        if entry.suffix not in STATUS_FILE_SUFFIXES:
            continue
        statuses[entry.stem] = read_analysis_status(entry)
    return statuses


__all__ = [
    "STATUS_DIR_NAME",
    "STATUS_FIELDS",
    "STATUS_FILE_SUFFIXES",
    "STATUS_STATES",
    "AnalysisStatus",
    "analysis_status_dir",
    "parse_status_timestamp",
    "read_analysis_status",
    "read_analysis_statuses",
]
