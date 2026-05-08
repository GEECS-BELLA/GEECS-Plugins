"""Walk scan folders for a date range and aggregate errors into a `TriageReport`.

The harvester is the seam between the deterministic per-scan tooling
(parser/fingerprint/classifier) and downstream consumers (CLI, Stage 2 LLM,
notebooks).

Depends on :mod:`geecs_data_utils` for:

- :class:`geecs_data_utils.ScanPaths` - GEECS folder convention awareness.
- :func:`geecs_data_utils.load_scan_log` - reads `scan.log` from a folder.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from geecs_data_utils import LogEntry, ScanPaths, load_scan_log
from geecs_log_triage.classifier import classify
from geecs_log_triage.fingerprint import (
    _extract_traceback_signature,
    fingerprint_for,
)
from geecs_log_triage.schemas import (
    ErrorOccurrence,
    Severity,
    TriageReport,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _level_at_least(entry: LogEntry, min_level: Severity) -> bool:
    """Return True if `entry.level` is at or above `min_level`."""
    return entry.level.level_no >= min_level.level_no


def _scan_id_from_folder(folder: Path) -> str:
    """Return the `Scan{NNN}` identifier for a scan folder.

    Falls back to the basename if the folder name doesn't match the convention.
    """
    name = folder.name
    return name if name.startswith("Scan") else folder.name


def _occurrences_from_entries(
    entries: Iterable[LogEntry],
    scan_folder: Path,
    min_level: Severity,
) -> tuple[list[ErrorOccurrence], int]:
    """Build `ErrorOccurrence` records from parsed entries (filtered by level).

    Returns
    -------
    occurrences : list[ErrorOccurrence]
        One per matching log entry.
    seen_count : int
        Total entries seen (pre-filter), useful for report stats.
    """
    occs: list[ErrorOccurrence] = []
    seen = 0
    scan_id = _scan_id_from_folder(scan_folder)

    for entry in entries:
        seen += 1
        if not _level_at_least(entry, min_level):
            continue

        exc_type: Optional[str] = None
        if entry.traceback:
            exc_type, _, _ = _extract_traceback_signature(entry.traceback)

        klass = classify(entry, exception_type=exc_type)
        fp = fingerprint_for(entry, classification=klass)

        occs.append(
            ErrorOccurrence(
                fingerprint=fp,
                entry=entry,
                scan_id=scan_id,
                scan_folder=scan_folder,
            )
        )

    return occs, seen


def _group_by_fingerprint(
    occurrences: Iterable[ErrorOccurrence],
) -> dict[str, list[ErrorOccurrence]]:
    """Group occurrences by `fingerprint.fingerprint_hash`."""
    out: dict[str, list[ErrorOccurrence]] = {}
    for occ in occurrences:
        out.setdefault(occ.fingerprint.fingerprint_hash, []).append(occ)
    return out


# --------------------------------------------------------------------------
# Public harvest API
# --------------------------------------------------------------------------


def harvest_scan_folder(
    scan_folder: Path | str,
    *,
    min_level: Severity = Severity.ERROR,
) -> TriageReport:
    """Harvest a single scan folder and return a `TriageReport`.

    Useful for ad-hoc inspection of a known-bad scan.

    Parameters
    ----------
    scan_folder : Path or str
        Path to the scan folder containing a `scan.log` file.
    min_level : Severity, optional
        Lowest severity to include. Defaults to ERROR.

    Returns
    -------
    TriageReport
        Report covering this single scan.
    """
    folder = Path(scan_folder)
    try:
        entries = load_scan_log(folder)
    except FileNotFoundError:
        entries = []

    occurrences, seen = _occurrences_from_entries(entries, folder, min_level)

    if entries:
        report_date = entries[-1].timestamp.date()
    else:
        report_date = datetime.now(tz=timezone.utc).date()

    return TriageReport(
        date_range=(report_date, report_date),
        experiment=None,
        min_level=min_level,
        total_scans_examined=1,
        total_log_entries=seen,
        total_errors=len(occurrences),
        grouped=_group_by_fingerprint(occurrences),
        generated_at=datetime.now(tz=timezone.utc),
    )


def day_folder_for(target: date, experiment: str) -> Path:
    """Return the day folder (parent of ``scans/``) for *target* and *experiment*.

    Parameters
    ----------
    target : date
        The day to resolve.
    experiment : str
        Experiment name.

    Returns
    -------
    Path
        ``{base}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}`` — the folder that
        contains ``scans/`` and where ``triage.md`` should be written.
    """
    if ScanPaths.paths_config is None:
        ScanPaths.reload_paths_config(default_experiment=experiment)
    base = Path(ScanPaths.paths_config.base_path)
    return (
        base
        / experiment
        / f"Y{target.year}"
        / f"{target.month:02d}-{target.strftime('%B')}"
        / f"{target.strftime('%y_%m%d')}"
    )


def _iter_scan_folders_for_date(
    target: date,
    experiment: str,
) -> list[Path]:
    """Return scan folders for a given date and experiment.

    Uses :class:`geecs_data_utils.ScanPaths` configuration to resolve the
    base directory; iterates `Scan*` folders inside the daily ``scans/``
    directory. Returns an empty list if the daily directory does not exist.
    """
    daily_dir = day_folder_for(target, experiment) / "scans"

    if not daily_dir.is_dir():
        logger.info(
            "No scans directory for %s on %s: %s", experiment, target, daily_dir
        )
        return []

    return sorted(
        p for p in daily_dir.iterdir() if p.is_dir() and p.name.startswith("Scan")
    )


def harvest_scan(
    target: date,
    experiment: str,
    scan_number: int,
    *,
    min_level: Severity = Severity.ERROR,
) -> TriageReport:
    """Harvest a single scan by number for a given date.

    Parameters
    ----------
    target : date
        The day that contains the scan.
    experiment : str
        Experiment name.
    scan_number : int
        Scan number (e.g. ``42`` matches ``Scan042`` or ``Scan42``).
    min_level : Severity, optional
        Lowest severity to keep. Defaults to ERROR.

    Returns
    -------
    TriageReport
        Report for that single scan. `total_scans_examined` is 0 if not found.
    """
    folders = _iter_scan_folders_for_date(target, experiment)
    padded = f"Scan{scan_number:03d}"
    unpadded = f"Scan{scan_number}"
    matching = [f for f in folders if f.name in (padded, unpadded)]

    if not matching:
        logger.warning(
            "Scan %s not found for %s on %s", scan_number, experiment, target
        )
        return TriageReport(
            date_range=(target, target),
            experiment=experiment,
            min_level=min_level,
            total_scans_examined=0,
            total_log_entries=0,
            total_errors=0,
            grouped={},
            generated_at=datetime.now(tz=timezone.utc),
        )

    folder = matching[0]
    try:
        entries = load_scan_log(folder)
    except FileNotFoundError:
        entries = []

    occurrences, seen = _occurrences_from_entries(entries, folder, min_level)
    return TriageReport(
        date_range=(target, target),
        experiment=experiment,
        min_level=min_level,
        total_scans_examined=1,
        total_log_entries=seen,
        total_errors=len(occurrences),
        grouped=_group_by_fingerprint(occurrences),
        generated_at=datetime.now(tz=timezone.utc),
    )


def harvest_date(
    target: date,
    experiment: str,
    *,
    min_level: Severity = Severity.ERROR,
) -> TriageReport:
    """Harvest all scans for a single date.

    Parameters
    ----------
    target : date
        The day to harvest.
    experiment : str
        Experiment name (used by `ScanPaths` for folder layout).
    min_level : Severity, optional
        Lowest severity to keep. Defaults to ERROR.

    Returns
    -------
    TriageReport
        Aggregated report for the day.
    """
    all_occs: list[ErrorOccurrence] = []
    total_entries = 0
    examined = 0

    for folder in _iter_scan_folders_for_date(target, experiment):
        examined += 1
        try:
            entries = load_scan_log(folder)
        except FileNotFoundError:
            logger.debug("No scan.log in %s; skipping", folder)
            continue

        occs, seen = _occurrences_from_entries(entries, folder, min_level)
        all_occs.extend(occs)
        total_entries += seen

    return TriageReport(
        date_range=(target, target),
        experiment=experiment,
        min_level=min_level,
        total_scans_examined=examined,
        total_log_entries=total_entries,
        total_errors=len(all_occs),
        grouped=_group_by_fingerprint(all_occs),
        generated_at=datetime.now(tz=timezone.utc),
    )


def harvest_date_range(
    start: date,
    end: date,
    experiment: str,
    *,
    min_level: Severity = Severity.ERROR,
) -> TriageReport:
    """Harvest a range of dates inclusive on both ends.

    Parameters
    ----------
    start, end : date
        Inclusive date range.
    experiment : str
        Experiment name.
    min_level : Severity, optional
        Lowest severity to keep. Defaults to ERROR.

    Returns
    -------
    TriageReport
        Aggregated across all dates in the range.
    """
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")

    all_occs: list[ErrorOccurrence] = []
    total_entries = 0
    examined = 0

    cur = start
    while cur <= end:
        day_report = harvest_date(cur, experiment, min_level=min_level)
        examined += day_report.total_scans_examined
        total_entries += day_report.total_log_entries
        for occs in day_report.grouped.values():
            all_occs.extend(occs)
        cur = cur + timedelta(days=1)

    return TriageReport(
        date_range=(start, end),
        experiment=experiment,
        min_level=min_level,
        total_scans_examined=examined,
        total_log_entries=total_entries,
        total_errors=len(all_occs),
        grouped=_group_by_fingerprint(all_occs),
        generated_at=datetime.now(tz=timezone.utc),
    )
