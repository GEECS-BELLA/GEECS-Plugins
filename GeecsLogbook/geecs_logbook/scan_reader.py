"""Read a day's scan folders into :mod:`~geecs_logbook.models` views.

A day is a *query*: :func:`read_day` lists whatever ``ScanNNN`` folders
exist under the date's ``scans/`` directory at the moment it is called.
Nothing has to be created first, and a scan that finished seconds ago is
simply there on the next call.

What this module owns, and what it borrows
------------------------------------------
It owns the *logbook's* view of a scan: the status classification and the
day/campaign shaping. The parsing primitives live one layer down in
``geecs_data_utils``, which already owns scan folders, so there is one
surface to fix when the formats change:

- ``read_scan_info_file`` — the ``[Scan Info]`` parse, shared with
  ``ScanPaths.load_scan_info``.
- ``first_log_timestamp`` — when the scan actually ran, from ``scan.log``.

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
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional, Union

from geecs_data_utils import ScanPaths
from geecs_data_utils.scan_log_loader import first_log_timestamp
from geecs_data_utils.scan_paths import read_scan_info_file

from geecs_logbook.models import DaySummary, ScanStatus, ScanSummary

logger = logging.getLogger(__name__)

#: ``ScanNNN`` directories. Zero padding is conventional but not
#: guaranteed — ``geecs_log_triage.harvester`` resolves unpadded folders
#: too, and a view whose premise is "whatever scan folders exist" must not
#: make one invisible.
_SCAN_DIR = re.compile(r"^Scan(\d+)$")

#: Subdirectories inside a scan folder that are not devices. ``analysis/``
#: appears here as well as beside ``scans/`` — ScanAnalysis writes one
#: inside the scan folder too (GEECS-Data-Utils' folder convention shows
#: both). The overlap costs us a device literally named ``analysis``; no
#: such device exists, and the alternative is listing an output directory
#: as a device on every analysed scan.
_NON_DEVICE_DIRS = {"analysis_status", "analysis"}

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


class ScanContents(NamedTuple):
    """What one directory listing of a scan folder tells us.

    Attributes
    ----------
    ini_path : str or None
        The ``ScanInfoScanNNN.ini`` file, when present.
    ini_mtime : float or None
        Its modification time — part of the cache key, because the scanner
        finalises the outcome by rewriting this file *in place* and that
        does not move the folder's own timestamp.
    ini_size : int or None
        Its size, for the same reason.
    log_path : str or None
        The ``scan.log`` file, when present.
    devices : tuple of str
        Per-device subdirectory names.
    """

    ini_path: Optional[str]
    ini_mtime: Optional[float]
    ini_size: Optional[int]
    log_path: Optional[str]
    devices: tuple[str, ...]


def scan_status(end_info: Optional[str], has_scan_info: bool) -> ScanStatus:
    """Classify a scan from its ``ScanEndInfo``.

    Reports what the files say rather than inferring intent.

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

    Notes
    -----
    An **empty** ``ScanEndInfo`` is ``incomplete``, not ``unknown``. The
    scanner writes ``ScanEndInfo = ""`` when the folder is claimed and
    fills it in at the stop document, so an empty value means *not
    finalised* — a scan still running, or one that died before its stop
    document. It is the most common state on the real share (37 of 49
    ScanInfo files across four sampled days), so classifying it as
    "unrecognised" painted most of a day amber.
    """
    if not has_scan_info:
        return "incomplete"
    if end_info is None or not end_info.strip():
        return "incomplete"
    text = end_info.strip()
    lowered = text.lower()
    if lowered == "success":
        return "success"
    if lowered.startswith("fail"):
        return "failed"
    if lowered.startswith("abort"):
        # `RE.abort()`, Ctrl-C, and the queueserver stop the console and
        # GEECS-MCP both expose all reach on_stop as exit_status="abort".
        # Its reason is as worth surfacing as a failure's.
        return "aborted"
    return "unknown"


def _failure_reason(end_info: Optional[str], status: ScanStatus) -> Optional[str]:
    """Return the outcome text without its verdict prefix, when not clean."""
    if status not in {"failed", "aborted"} or not end_info:
        return None
    return (
        re.sub(r"^(fail|abort)[a-z]*:\s*", "", end_info.strip(), flags=re.IGNORECASE)
        or None
    )


def _as_float(raw: Optional[str]) -> Optional[float]:
    """Parse a finite float from a ScanInfo value, else ``None``."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    # inf/nan are unusable downstream and would escape int() as OverflowError
    # or ValueError; a whole day must not 503 over one malformed field.
    return value if -1e308 < value < 1e308 else None


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


def scan_contents(scan_folder: Path) -> ScanContents:
    """Describe a scan folder from ONE directory listing.

    The obvious implementation — ``glob("ScanInfo*.ini")`` to find the
    file, ``iterdir()`` to list devices, ``stat()`` for freshness — costs
    several round trips per scan. On a VPN-mounted SMB share each is ~50 ms
    and a day view pays it once per scan. One ``os.scandir`` answers all of
    it, which measured 1.85x faster on cold days (403 -> 218 ms per scan).

    Parameters
    ----------
    scan_folder : Path
        An existing ``ScanNNN`` directory.

    Returns
    -------
    ScanContents
        Empty fields when the folder cannot be listed — reported, never
        raised, so one unreadable scan does not take down a day.
    """
    ini_path = ini_mtime = ini_size = log_path = None
    devices: list[str] = []
    try:
        with os.scandir(scan_folder) as entries:
            for entry in entries:
                name = entry.name
                if entry.is_dir():
                    if name not in _NON_DEVICE_DIRS:
                        devices.append(name)
                elif name.startswith("ScanInfo") and name.endswith(".ini"):
                    ini_path = entry.path
                    try:
                        info = entry.stat()
                        ini_mtime, ini_size = info.st_mtime, info.st_size
                    except OSError:
                        pass
                elif name == "scan.log":
                    log_path = entry.path
    except OSError as exc:
        logger.warning("cannot list %s: %s", scan_folder, exc)
        return ScanContents(None, None, None, None, ())
    return ScanContents(ini_path, ini_mtime, ini_size, log_path, tuple(sorted(devices)))


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
    contents = scan_contents(scan_folder)
    args = (
        str(scan_folder),
        number,
        contents.ini_path,
        contents.ini_mtime,
        contents.ini_size,
        contents.log_path,
        contents.devices,
    )
    if contents.ini_path and contents.ini_mtime is None:
        # The listing found the file but could not stat it — a flaky share.
        # Caching on the path alone would pin this scan's state for the life
        # of the process, so read it uncached instead.
        return _summarize(*args)
    return _read_scan_cached(*args)


@lru_cache(maxsize=_CACHE_SIZE)
def _read_scan_cached(
    folder: str,
    number: int,
    ini_path: Optional[str],
    ini_mtime: Optional[float],
    ini_size: Optional[int],
    log_path: Optional[str],
    devices: tuple[str, ...],
) -> ScanSummary:
    """Memoise :func:`_summarize` on the ScanInfo file's identity.

    Only the file reads are cached; the caller has already paid for the
    directory listing that produced these arguments, and that listing is
    what makes the key honest.

    The key is the ScanInfo file's own ``(mtime, size)``, never the
    folder's. The scanner finalises a scan by rewriting that file **in
    place** (``path.open("w")``), which changes no directory entry and so
    leaves the folder's timestamp untouched. Keying on the folder served a
    running scan's empty ``ScanEndInfo`` forever — losing exactly the
    failure reason this view exists to surface.
    """
    return _summarize(folder, number, ini_path, ini_mtime, ini_size, log_path, devices)


def _summarize(
    folder: str,
    number: int,
    ini_path: Optional[str],
    ini_mtime: Optional[float],
    ini_size: Optional[int],
    log_path: Optional[str],
    devices: tuple[str, ...],
) -> ScanSummary:
    """Build a summary from an already-listed scan folder, uncached."""
    info = read_scan_info_file(ini_path) if ini_path else {}
    has_info = bool(info)
    end_info = info.get("ScanEndInfo")
    status = scan_status(end_info, has_info)

    # scan.log is the honest start. Archive scans predate it, so fall back
    # to the ScanInfo file's own mtime — the scan's *end*, minutes out
    # rather than the folder's hours — and flag it rather than implying a
    # precision it does not have. Never the folder's mtime.
    started = first_log_timestamp(log_path) if log_path else None
    approximate = False
    if started is None and ini_mtime is not None:
        started = datetime.fromtimestamp(ini_mtime)
        approximate = True

    return ScanSummary(
        number=number,
        started=started,
        started_approximate=approximate,
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
        devices=list(devices),
        has_scan_info=has_info,
    )


#: ``YY_MMDD`` day folders inside a month folder.
_DAY_DIR = re.compile(r"^(\d{2})_(\d{2})(\d{2})$")


def month_folder(
    first: date, experiment: str, base_directory: Optional[Union[Path, str]] = None
) -> Path:
    """Return the month directory holding a month's day folders.

    ``{base}/{experiment}/Y2026/09-Sep`` — derived from the same builder
    the day reader uses, so the two cannot disagree about the layout.
    """
    tag = ScanPaths.get_scan_tag(
        first.year, first.month, 1, number=0, experiment=experiment
    )
    return ScanPaths.get_daily_scan_folder(
        tag=tag, base_directory=base_directory
    ).parent.parent


def days_with_folders(
    first: date,
    experiment: str,
    base_directory: Optional[Union[Path, str]] = None,
) -> Optional[set[date]]:
    """Return the dates in ``first``'s month that have a day folder.

    **One** directory listing of the month folder — never a walk of
    thirty day folders. A day folder exists because something wrote into
    it (a scan, an analysis pass), which is what a calendar wants to mark.

    Returns ``None`` when the experiment directory itself is missing: the
    share is not mounted where this service expects it, which is a
    different fact from "nothing happened this month" and the caller
    should say so. A month folder that does not exist is an empty set.
    """
    folder = month_folder(first, experiment, base_directory)
    if not folder.parent.parent.is_dir():  # {base}/{experiment}
        logger.warning("experiment directory missing: %s", folder.parent.parent)
        return None
    found: set[date] = set()
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                match = _DAY_DIR.match(entry.name)
                if not match or not entry.is_dir():
                    continue
                yy, mm, dd = (int(g) for g in match.groups())
                # Only this month's shape; a stray folder is not a day.
                if 2000 + yy != first.year or mm != first.month:
                    continue
                try:
                    found.add(date(first.year, mm, dd))
                except ValueError:
                    continue
    except FileNotFoundError:
        return found
    except OSError as exc:
        logger.warning("cannot list %s: %s", folder, exc)
        return None
    return found


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

    targets: list[tuple[Path, int]] = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                match = _SCAN_DIR.match(entry.name)
                if match and entry.is_dir():
                    targets.append((Path(entry.path), int(match.group(1))))
    except OSError as exc:
        logger.warning("cannot list %s: %s", folder, exc)
        return summary

    if not targets:
        return summary

    # Concurrency on top of the leaner per-scan read: see _READ_WORKERS.
    workers = min(_READ_WORKERS, len(targets))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scans = list(pool.map(lambda item: read_scan(*item), targets))

    summary.scans = sorted(scans, key=lambda s: s.number)
    return summary
