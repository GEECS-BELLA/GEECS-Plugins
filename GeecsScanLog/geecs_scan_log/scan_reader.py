"""Read a day's scan folders into :mod:`~geecs_scan_log.models` views.

A day is a *query*: :func:`read_day` lists whatever ``ScanNNN`` folders
exist under the date's ``scans/`` directory at the moment it is called.
Nothing has to be created first, and a scan that finished seconds ago is
simply there on the next call.

Read-only by construction
-------------------------
This module is analysis-side code under the repository's scan-folder
invariant (see the root ``CLAUDE.md``). It therefore never:

- constructs ``ScanPaths(read_mode=False)``,
- calls ``mkdir`` on any path,
- treats a missing folder as something to create.

A missing day or scan folder is reported as absent, never repaired. The
invariant is pinned by ``tests/test_scan_reader.py``.
"""

from __future__ import annotations

import logging
import re
from configparser import ConfigParser, Error as ConfigParserError
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

from geecs_data_utils import ScanPaths

from geecs_scan_log.models import DaySummary, ScanStatus, ScanSummary

logger = logging.getLogger(__name__)

#: ``ScanNNN`` directories, the only thing treated as a scan.
_SCAN_DIR = re.compile(r"^Scan(\d{3,})$")

#: Files that live in a scan folder but are not per-device subdirectories.
_NON_DEVICE_SUFFIXES = {".ini", ".txt", ".tdms", ".tdms_index", ".log"}

#: Subdirectories that are not devices.
_NON_DEVICE_DIRS = {"analysis_status"}


def scan_status(end_info: Optional[str], has_scan_info: bool) -> ScanStatus:
    """Classify a scan from its ``ScanEndInfo``.

    The classification is deliberately shallow: it reports what the file
    says rather than inferring intent. A folder without ``ScanInfo`` is
    ``"incomplete"`` — which covers a scan still running, one aborted
    early, and development churn alike, because those are not separable
    from the folder alone.

    Parameters
    ----------
    end_info : str or None
        The raw ``ScanEndInfo`` value, or ``None`` when absent.
    has_scan_info : bool
        Whether a ``ScanInfo`` file was found at all.

    Returns
    -------
    ScanStatus
        ``"success"``, ``"failed"``, ``"incomplete"`` or ``"unknown"``.
    """
    if not has_scan_info:
        return "incomplete"
    if not end_info:
        return "unknown"
    text = end_info.strip()
    if text.lower() == "success":
        return "success"
    if text.lower().startswith("fail"):
        return "failed"
    return "unknown"


def _failure_reason(end_info: Optional[str], status: ScanStatus) -> Optional[str]:
    """Return the failure text without its ``fail:`` prefix, if failed."""
    if status != "failed" or not end_info:
        return None
    return re.sub(r"^fail:\s*", "", end_info.strip(), flags=re.IGNORECASE) or None


def _as_float(raw: Optional[str]) -> Optional[float]:
    """Parse a float from a ScanInfo value, returning None when unusable."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _as_int(raw: Optional[str]) -> Optional[int]:
    """Parse an int from a ScanInfo value, returning None when unusable."""
    value = _as_float(raw)
    return None if value is None else int(value)


def _as_bool(raw: Optional[str]) -> Optional[bool]:
    """Parse a bool from a ScanInfo value, returning None when unusable."""
    if raw is None:
        return None
    text = raw.strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def read_scan_info(scan_folder: Path) -> dict[str, str]:
    """Parse ``ScanInfoScanNNN.ini`` from a scan folder.

    Parameters
    ----------
    scan_folder : Path
        An existing ``ScanNNN`` directory.

    Returns
    -------
    dict of str to str
        The ``Scan Info`` section with surrounding quotes stripped, or an
        empty dict when the file is absent or unparsable. A malformed file
        is logged and treated as absent rather than raised: one bad scan
        must not take down a whole day's view.
    """
    matches = sorted(scan_folder.glob("ScanInfo*.ini"))
    if not matches:
        return {}

    parser = ConfigParser()
    parser.optionxform = str
    try:
        parser.read(matches[0])
        return {k: v.strip("'\"") for k, v in parser.items("Scan Info")}
    except (ConfigParserError, OSError, UnicodeDecodeError) as exc:
        logger.warning("unreadable ScanInfo in %s: %s", scan_folder, exc)
        return {}


def list_devices(scan_folder: Path) -> list[str]:
    """Return the per-device subdirectory names inside a scan folder."""
    try:
        return sorted(
            child.name
            for child in scan_folder.iterdir()
            if child.is_dir()
            and child.name not in _NON_DEVICE_DIRS
            and child.suffix not in _NON_DEVICE_SUFFIXES
        )
    except OSError as exc:
        logger.warning("cannot list devices in %s: %s", scan_folder, exc)
        return []


def read_scan(scan_folder: Path, number: int) -> ScanSummary:
    """Build a :class:`ScanSummary` from one existing scan folder.

    Parameters
    ----------
    scan_folder : Path
        An existing ``ScanNNN`` directory. Never created here.
    number : int
        The scan number parsed from the folder name.

    Returns
    -------
    ScanSummary
        The derived view. A folder with no ``ScanInfo`` yields a summary
        with ``has_scan_info=False`` and status ``"incomplete"`` rather
        than an error.
    """
    info = read_scan_info(scan_folder)
    has_info = bool(info)
    end_info = info.get("ScanEndInfo")
    status = scan_status(end_info, has_info)

    try:
        started = datetime.fromtimestamp(scan_folder.stat().st_mtime)
    except OSError:
        started = None

    return ScanSummary(
        number=number,
        started=started,
        parameter=info.get("Scan Parameter") or None,
        start=_as_float(info.get("Start")),
        end=_as_float(info.get("End")),
        step_size=_as_float(info.get("Step size")),
        shots_per_step=_as_int(info.get("Shots per step")),
        mode=info.get("ScanMode") or None,
        plan=info.get("Plan") or None,
        scanner=info.get("Scanner") or None,
        trigger_profile=info.get("Trigger profile") or None,
        background=_as_bool(info.get("Background")),
        purpose=info.get("ScanStartInfo") or None,
        status=status,
        failure_reason=_failure_reason(end_info, status),
        devices=list_devices(scan_folder),
        has_scan_info=has_info,
    )


def read_day(
    when: date,
    experiment: str,
    base_directory: Optional[Union[Path, str]] = None,
) -> DaySummary:
    """List every scan folder present for one date.

    Parameters
    ----------
    when : date
        The date to read.
    experiment : str
        The experiment whose share to read, e.g. ``"Undulator"``.
    base_directory : Path or str, optional
        Override the configured data-share root. Used by tests.

    Returns
    -------
    DaySummary
        Scans ordered by number. When the day's ``scans/`` directory does
        not exist the summary has ``exists=False`` and no scans — an empty
        day, which is an ordinary state and not an error.

    Notes
    -----
    The day folder is *never* created. ``ScanPaths.get_daily_scan_folder``
    only builds a path; this function checks it and reports absence.
    """
    tag = ScanPaths.get_scan_tag(
        when.year, when.month, when.day, number=0, experiment=experiment
    )
    folder = ScanPaths.get_daily_scan_folder(tag=tag, base_directory=base_directory)

    summary = DaySummary(
        day=when, experiment=experiment, folder=str(folder), exists=folder.is_dir()
    )
    if not summary.exists:
        logger.info("no scans directory for %s: %s", when.isoformat(), folder)
        return summary

    scans: list[ScanSummary] = []
    try:
        children = sorted(folder.iterdir())
    except OSError as exc:
        logger.warning("cannot list %s: %s", folder, exc)
        return summary

    for child in children:
        if not child.is_dir():
            continue
        match = _SCAN_DIR.match(child.name)
        if not match:
            continue
        scans.append(read_scan(child, int(match.group(1))))

    summary.scans = sorted(scans, key=lambda s: s.number)
    return summary
