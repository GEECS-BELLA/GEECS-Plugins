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
import os
import re
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser, Error as ConfigParserError
from datetime import date, datetime
from functools import lru_cache
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

#: Concurrent folder reads, layered on top of the per-scan savings in
#: :func:`scan_contents`. The work is latency-bound, not CPU-bound: every
#: listing and open is an SMB round trip, ~50 ms over VPN. A 108-scan day
#: measured 27 s serially with the original four-round-trip reader and
#: 5.6 s with sixteen in flight. Python releases the GIL across file I/O,
#: so threads are the right tool and a process pool would only add
#: pickling cost. Sixteen is the figure measured against a genuinely cold
#: share; higher counts appeared better only because the OS page cache was
#: warm from the previous run.
_READ_WORKERS = 16

#: How many scan summaries to remember across requests.
_CACHE_SIZE = 4096


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


def scan_contents(scan_folder: Path) -> tuple[Optional[str], list[str]]:
    """Return the ScanInfo path and device names from ONE directory listing.

    The obvious implementation — ``glob("ScanInfo*.ini")`` to find the file,
    then ``iterdir()`` to list devices — costs two directory listings per
    scan. On a VPN-mounted SMB share each of those is a round trip, and a
    day view does it once per scan. One ``os.scandir`` answers both
    questions, which measured 1.85x faster on cold days (403 -> 218 ms per
    scan across untouched August dates).

    Parameters
    ----------
    scan_folder : Path
        An existing ``ScanNNN`` directory.

    Returns
    -------
    tuple of (str or None, list of str)
        The ScanInfo file path if present, and the sorted per-device
        subdirectory names.
    """
    ini: Optional[str] = None
    devices: list[str] = []
    try:
        with os.scandir(scan_folder) as entries:
            for entry in entries:
                name = entry.name
                if entry.is_dir():
                    if name not in _NON_DEVICE_DIRS:
                        devices.append(name)
                elif name.startswith("ScanInfo") and name.endswith(".ini"):
                    ini = entry.path
    except OSError as exc:
        logger.warning("cannot list %s: %s", scan_folder, exc)
        return None, []
    return ini, sorted(devices)


def parse_scan_info(ini_path: Optional[str]) -> dict[str, str]:
    """Parse a ``ScanInfoScanNNN.ini`` file.

    Parameters
    ----------
    ini_path : str or None
        The file to read, as found by :func:`scan_contents`.

    Returns
    -------
    dict of str to str
        The ``Scan Info`` section with surrounding quotes stripped, or an
        empty dict when the file is absent or unparsable. A malformed file
        is logged and treated as absent rather than raised: one bad scan
        must not take down a whole day's view.
    """
    if not ini_path:
        return {}
    parser = ConfigParser()
    parser.optionxform = str
    try:
        parser.read(ini_path)
        return {k: v.strip("'\"") for k, v in parser.items("Scan Info")}
    except (ConfigParserError, OSError, UnicodeDecodeError) as exc:
        logger.warning("unreadable ScanInfo at %s: %s", ini_path, exc)
        return {}


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

    Notes
    -----
    Memoised on the folder's modification time — see
    :func:`_read_scan_cached`. A folder that cannot be stat-ed is read
    directly rather than cached.
    """
    try:
        mtime = scan_folder.stat().st_mtime
    except OSError:
        return _read_scan_uncached(scan_folder, number, None)
    return _read_scan_cached(str(scan_folder), number, mtime)


@lru_cache(maxsize=_CACHE_SIZE)
def _read_scan_cached(folder: str, number: int, mtime: float) -> ScanSummary:
    """Read one scan, memoised on the folder's modification time.

    A finished scan folder never changes, so re-reading it on every day
    view is pure cost — and over a VPN-mounted share that cost dominates
    the page. Keying on ``mtime`` keeps the cache honest: a scan still
    being written bumps its folder time and misses the cache, so a running
    scan is never served stale.

    Parameters
    ----------
    folder : str
        The scan folder as a string, so the cache key is hashable.
    number : int
        The scan number.
    mtime : float
        The folder's modification time. Part of the key; not read as data.

    Returns
    -------
    ScanSummary
        The derived view.
    """
    return _read_scan_uncached(Path(folder), number, mtime)


def _read_scan_uncached(
    scan_folder: Path, number: int, mtime: Optional[float]
) -> ScanSummary:
    """Build a :class:`ScanSummary` from disk, bypassing the cache.

    Takes ``mtime`` from the caller when it already has it — ``read_day``
    gets it free from the parent's ``scandir`` — so a scan costs one
    directory listing and one file read, with no extra ``stat``.
    """
    ini_path, devices = scan_contents(scan_folder)
    info = parse_scan_info(ini_path)
    has_info = bool(info)
    end_info = info.get("ScanEndInfo")
    status = scan_status(end_info, has_info)

    if mtime is None:
        try:
            mtime = scan_folder.stat().st_mtime
        except OSError:
            mtime = None
    started = datetime.fromtimestamp(mtime) if mtime is not None else None

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
        devices=devices,
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

    # One scandir of the day folder yields the scan names AND their mtimes,
    # so no per-scan stat is needed downstream.
    targets: list[tuple[Path, int, Optional[float]]] = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                match = _SCAN_DIR.match(entry.name)
                if not match or not entry.is_dir():
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    mtime = None
                targets.append((Path(entry.path), int(match.group(1)), mtime))
    except OSError as exc:
        logger.warning("cannot list %s: %s", folder, exc)
        return summary

    if not targets:
        return summary

    def _one(item: tuple[Path, int, Optional[float]]) -> ScanSummary:
        path, number, mtime = item
        if mtime is None:
            return _read_scan_uncached(path, number, None)
        return _read_scan_cached(str(path), number, mtime)

    # Concurrency on top of the leaner per-scan read: see _READ_WORKERS.
    workers = min(_READ_WORKERS, len(targets))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scans = list(pool.map(_one, targets))

    summary.scans = sorted(scans, key=lambda s: s.number)
    return summary
