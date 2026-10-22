"""Read a day's scan folders into :mod:`~geecs_logbook.models` views.

A day is a *query*: :func:`read_day` lists whatever ``ScanNNN`` folders
exist under the date's ``scans/`` directory at the moment it is called.
Nothing has to be created first, and a scan that finished seconds ago is
simply there on the next call.

What this module owns, and what it borrows
------------------------------------------
It owns the *logbook's* view of a scan: the day shaping and the summary.
The scan-folder facts live one layer down in ``geecs_data_utils``, which
already owns scan folders, so there is one surface to fix when the formats
change — and the portal's folder catalog reads the same facts the same way:

- ``geecs_data_utils.scan_info`` — the ``ScanNNN`` name, the ``[Scan Info]``
  parse and its typed record, and the outcome from ``ScanEndInfo`` plus
  the run-closed evidence (``classify_end_info``).
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
from geecs_data_utils.data.sfile import (
    header_for_variable,
    read_sfile_head,
    scan_data_txt_path_for,
    sfile_candidates,
    sfile_column,
    sfile_path_for_scan,
)
from geecs_data_utils.scan_info import (
    ScanInfoRecord,
    classify_end_info,
    end_info_reason,
    parse_scan_folder_name,
    read_scan_info_sections,
    scan_log_variables,
)
from geecs_data_utils.scan_log_loader import first_log_timestamp

from geecs_logbook.models import DaySummary, ScanSummary

logger = logging.getLogger(__name__)

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
READ_WORKERS = 16

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
    scan_data : bool
        Whether ``ScanDataScanNNN.txt`` is present — written when the run
        closes, so it is run-closed evidence.
    """

    ini_path: Optional[str]
    ini_mtime: Optional[float]
    ini_size: Optional[int]
    log_path: Optional[str]
    devices: tuple[str, ...]
    scan_data: bool = False


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
    scan_data = False
    scan_data_name = scan_data_txt_path_for(scan_folder).name
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
                elif name == scan_data_name:
                    scan_data = True
    except OSError as exc:
        logger.warning("cannot list %s: %s", scan_folder, exc)
        return ScanContents(None, None, None, None, ())
    return ScanContents(
        ini_path, ini_mtime, ini_size, log_path, tuple(sorted(devices)), scan_data
    )


def _run_closed(
    scan_folder: Path, contents: ScanContents, sfiles: Optional[frozenset[str]]
) -> bool:
    """Whether the run closed: ``ScanDataScanNNN.txt``, else the analysis s-file.

    The same evidence as ``geecs_data_utils.data.sfile.run_closed_evidence``,
    from listings already in hand: the scan folder's own, and *sfiles* —
    the names in the day's ``analysis/`` folder, listed once per day by
    :func:`read_day`. Without that listing it costs one stat.
    """
    if contents.scan_data:
        return True
    try:
        sfile = sfile_path_for_scan(scan_folder)
    except ValueError:  # not a canonical scans/ScanNNN folder
        return False
    if sfiles is not None:
        return sfile.name in sfiles
    try:
        return sfile.is_file()
    except OSError:
        return False


def read_scan(
    scan_folder: Path, number: int, *, sfiles: Optional[frozenset[str]] = None
) -> ScanSummary:
    """Build a :class:`ScanSummary` from one existing scan folder.

    Parameters
    ----------
    scan_folder : Path
        An existing ``ScanNNN`` directory. Never created here.
    number : int
        The scan number parsed from the folder name.
    sfiles : frozenset of str, optional
        The file names in the day's ``analysis/`` folder, when the caller
        listed it (see :func:`_run_closed`).

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
        _run_closed(scan_folder, contents, sfiles),
    )
    if contents.ini_path and contents.ini_mtime is None:
        # The listing found the file but could not stat it — a flaky share.
        # Caching on the path alone would pin this scan's state for the life
        # of the process, so read it uncached instead.
        summary = _summarize(*args)
    else:
        summary = _read_scan_cached(*args)
    # The s-file is found from listings already in hand and kept on the
    # summary, so the day's lookup reads it without walking the folder again.
    sfile = sfile_for(scan_folder, scan_data=contents.scan_data, sfiles=sfiles)
    update: dict = {"sfile": str(sfile) if sfile else None}
    if summary.scan_log_variables:
        # The values live in the s-file, written at the run's close and in
        # place — so they are keyed on that file, not on the ini, and copied
        # onto the (cached, shared) summary rather than written into it.
        row = first_row(sfile) if sfile else None
        update["values_state"] = values_state(row, summary.status)
        if update["values_state"] == "ok":
            header = list(row)
            values = {}
            for name in summary.scan_log_variables:
                column = sfile_column(header_for_variable(name), header)
                values[name] = (row.get(column) or None) if column else None
            update["start_values"] = values
    return summary.model_copy(update=update)


def values_state(row: Optional[dict[str, str]], status: str) -> str:
    """Why a scan's first-shot values are or are not available (``ScanSummary.values_state``)."""
    if row is None:
        return "recording" if status == "incomplete" else "no_sfile"
    return "ok" if row else "no_shots"


def sfile_for(
    scan_folder: Path,
    *,
    scan_data: Optional[bool] = None,
    sfiles: Optional[frozenset[str]] = None,
) -> Optional[Path]:
    """The scan's s-file: the first of ``sfile_candidates`` that exists.

    Answered from the listings the caller already holds — *sfiles* (the
    day's ``analysis/`` names) and *scan_data* (``ScanDataScanNNN.txt`` in
    the folder's own listing); a missing listing costs one stat.
    """
    for path in sfile_candidates(scan_folder):
        known = (
            (path.name in sfiles if sfiles is not None else None)
            if path.parent.name == "analysis"
            else scan_data
        )
        if known is None:
            try:
                known = path.is_file()
            except OSError:
                known = False
        if known:
            return path
    return None


def first_row(path: Path) -> Optional[dict[str, str]]:
    """An s-file's first data row as ``{column: value}``.

    ``{}`` when the file has a header and no row (the run ended before its
    first shot); ``None`` when it cannot be read. Two lines, cached on the
    file's ``(mtime, size)`` — the file is written in place at the close,
    so a read that raced the write is replaced by the next request's.
    """
    try:
        info = path.stat()
        header, first = _sfile_head_cached(str(path), info.st_mtime, info.st_size)
    except OSError as exc:
        logger.warning("cannot read %s: %s", path, exc)
        return None
    if not first:
        return {}
    return dict(zip(header, first + ("",) * (len(header) - len(first))))


@lru_cache(maxsize=_CACHE_SIZE)
def _sfile_head_cached(
    path: str, mtime: float, size: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Memoise the two-line read on the file's identity (see :func:`first_row`)."""
    header, first = read_sfile_head(Path(path))
    return tuple(header), tuple(first)


@lru_cache(maxsize=_CACHE_SIZE)
def _read_scan_cached(
    folder: str,
    number: int,
    ini_path: Optional[str],
    ini_mtime: Optional[float],
    ini_size: Optional[int],
    log_path: Optional[str],
    devices: tuple[str, ...],
    closed: bool,
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
    failure reason this view exists to surface. ``closed`` is in the key
    for the same reason: Master Control ends a scan by writing its
    s-files, never by rewriting the ini.
    """
    return _summarize(
        folder, number, ini_path, ini_mtime, ini_size, log_path, devices, closed
    )


def _summarize(
    folder: str,
    number: int,
    ini_path: Optional[str],
    ini_mtime: Optional[float],
    ini_size: Optional[int],
    log_path: Optional[str],
    devices: tuple[str, ...],
    closed: bool,
) -> ScanSummary:
    """Build a summary from an already-listed scan folder, uncached."""
    sections = read_scan_info_sections(ini_path) if ini_path else {}
    info = sections.get("Scan Info", {})
    record = ScanInfoRecord.from_ini(info)
    status = classify_end_info(record.end_info, closed=closed)

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
        parameter=record.parameter,
        start=record.start,
        end=record.end,
        step_size=record.step_size,
        shots_per_step=record.shots_per_step,
        mode=record.mode,
        plan=record.plan,
        scanner=record.scanner,
        trigger_profile=record.trigger_profile,
        background=record.background,
        purpose=record.purpose,
        status=status,
        failure_reason=end_info_reason(record.end_info),
        devices=list(devices),
        has_scan_info=bool(info),
        folder_name=Path(folder).name,
        scan_log_variables=scan_log_variables(sections),
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
    found: set[date] = set()
    try:
        folder = month_folder(first, experiment, base_directory)
        if not folder.parent.parent.is_dir():  # {base}/{experiment}
            logger.warning("experiment directory missing: %s", folder.parent.parent)
            return None
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
        return found  # no month folder: nothing happened this month
    except OSError as exc:  # EACCES, EIO, a wedged mount — the share, not the month
        logger.warning("cannot list the month folder for %s: %s", first, exc)
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
                number = parse_scan_folder_name(entry.name)
                if number is not None and entry.is_dir():
                    targets.append((Path(entry.path), number))
    except OSError as exc:
        logger.warning("cannot list %s: %s", folder, exc)
        return summary

    if not targets:
        return summary

    # The run-closed evidence's second file, for every scan at once: one
    # listing of the day's analysis/ folder (empty when it does not exist;
    # None — a stat per scan — when the share cannot list it).
    sfiles: Optional[frozenset[str]]
    try:
        with os.scandir(folder.parent / "analysis") as entries:
            sfiles = frozenset(entry.name for entry in entries)
    except FileNotFoundError:
        sfiles = frozenset()
    except OSError:
        sfiles = None

    # Concurrency on top of the leaner per-scan read: see READ_WORKERS.
    workers = min(READ_WORKERS, len(targets))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scans = list(pool.map(lambda item: read_scan(*item, sfiles=sfiles), targets))

    summary.scans = sorted(scans, key=lambda s: s.number)
    return summary
