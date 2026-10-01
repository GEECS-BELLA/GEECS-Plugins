"""The scan folders on the share as a ``ScanCatalog`` — the pre-Bluesky era.

:class:`~geecs_data_utils.tiled_catalog.TiledScanCatalog` sees only the
runs the Bluesky worker recorded.  Every scan LabVIEW Master Control ever
took — and every experiment that still runs on it — exists only as a
``scans/ScanNNN`` folder with a ``ScanInfo`` ini and an s-file.  Two
implementations of the same protocol close that gap:

- :class:`FolderScanCatalog` — a day's ``scans/ScanNNN`` folders, one
  listing row per ``ScanInfo`` ini.  A loaded run carries a start document
  **synthesized** from the ini (the keys :mod:`geecs_data_utils.tiled_schema`
  reads: ``motors``, ``num_points``, ``shots_per_step``, ``scan_folder``)
  and **no event table** (``data=None``): the per-shot scalars are the
  s-file, which :func:`geecs_data_utils.scan_frame.scan_frame` already reads
  for "a legacy scan with no run".  One reader of the s-file, not two.
- :class:`MergedScanCatalog` — a primary catalog (Tiled) plus the folders:
  a day lists every primary run, then each folder whose scan number no
  primary run of that experiment claims.  Folders are canonical for the
  day's *existence* (owner ruling 2026-09-13, ``GEECS-DataPortal/CLAUDE.md``);
  a scan that has both keeps its richer Tiled run.

Folder run uids are ``folder:{experiment}:{YYYY-MM-DD}:{number}`` —
self-describing, so :meth:`FolderScanCatalog.load_run` needs no listing
first.  Read-only throughout: nothing here creates a path (repo
scan-folder invariant).
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from geecs_data_utils.data.sfile import (
    _SCAN_FOLDER_RE,
    scan_data_txt_path_for,
    sfile_path_for_scan,
)
from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
from geecs_data_utils.scan_log_loader import first_log_timestamp
from geecs_data_utils.scan_paths import daily_scan_folder, read_scan_info_file
from geecs_data_utils.tiled_catalog import (
    CatalogStatus,
    RunDetail,
    RunSummary,
    ScanCatalog,
    summary_from_metadata,
)

logger = logging.getLogger(__name__)

#: Prefix that marks a uid as a folder run (never a Bluesky uuid).
FOLDER_UID_PREFIX = "folder:"

#: Finished scans remembered per catalog (a year of busy days is ~10k).
_FINISHED_CACHE_SIZE = 16384

#: The s-file column Master Control stamps each shot's wall time into.
_DATETIME_COLUMN = "DateTime Timestamp"

#: ``Scan Parameter`` values that mean "no variable was stepped".
_NOSCAN_PARAMETERS = {"", "shotnumber", "noscan", "none"}


def folder_uid(experiment: str, day: date, number: int) -> str:
    """Build a folder run's uid.

    Parameters
    ----------
    experiment : str
        The experiment directory name.
    day : datetime.date
        The scan's day.
    number : int
        The day-scoped scan number.

    Returns
    -------
    str
        ``folder:{experiment}:{YYYY-MM-DD}:{number}``.
    """
    return f"{FOLDER_UID_PREFIX}{experiment}:{day.isoformat()}:{number}"


def parse_folder_uid(uid: str) -> Optional[tuple[str, date, int]]:
    """Split a folder uid into ``(experiment, day, number)``.

    Parameters
    ----------
    uid : str
        Any run uid.

    Returns
    -------
    tuple or None
        The parts, or ``None`` when *uid* is not a well-formed folder uid.
    """
    if not uid.startswith(FOLDER_UID_PREFIX):
        return None
    # rsplit: an experiment name may itself contain a colon-free space
    # ("Magnet Test Bench"); the day and number never contain a colon.
    parts = uid[len(FOLDER_UID_PREFIX) :].rsplit(":", 2)
    if len(parts) != 3 or not parts[0]:
        return None
    try:
        return parts[0], date.fromisoformat(parts[1]), int(parts[2])
    except ValueError:
        return None


def _float(raw: Optional[str]) -> Optional[float]:
    try:
        return float(raw) if raw not in (None, "") else None
    except ValueError:
        return None


def _sfile_head(scan_dir: Path) -> tuple[list[str], list[str], bool]:
    """Read an s-file's header and first data row — two lines, never the file.

    Tries the analysis s-file first (the table the portal plots), then the
    scanner's ``ScanDataScanNNN.txt`` — the two files
    :func:`~geecs_data_utils.data.sfile.run_closed_evidence` accepts as
    proof the run closed, in the same helpers' paths.  Opening one *is*
    that evidence, so the closure verdict costs no extra stat.

    Returns
    -------
    tuple
        ``(header, first_row, closed)``; empty lists and ``False`` when
        neither file is readable.
    """
    for path in (sfile_path_for_scan(scan_dir), scan_data_txt_path_for(scan_dir)):
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                header = handle.readline().rstrip("\r\n").split("\t")
                first = handle.readline().rstrip("\r\n").split("\t")
        except OSError:
            continue
        return header, first, True
    return [], [], False


def _on_day(epoch: Optional[float], day: date) -> bool:
    if epoch is None:
        return False
    try:
        return datetime.fromtimestamp(epoch).date() == day
    except (OverflowError, OSError, ValueError):
        return False


def _start_time(
    scan_dir: Path, number: int, header: list[str], first: list[str], day: date
) -> tuple[float, bool]:
    """When the scan started, and whether that time is only approximate.

    The first shot's ``DateTime Timestamp`` when the s-file has one; else
    ``scan.log``'s first record; else the ScanInfo ini's mtime (the
    logbook's ladder — Master Control writes the ini as a scan starts).
    Every rung must land on the folder's own day, because the portal
    re-bases a run's scan folder on it; local noon is the last resort and,
    like the ini mtime, is flagged approximate.
    """
    if _DATETIME_COLUMN in header:
        index = header.index(_DATETIME_COLUMN)
        raw = _float(first[index]) if index < len(first) else None
        if raw is not None and raw > LABVIEW_EPOCH_OFFSET:
            epoch = raw - LABVIEW_EPOCH_OFFSET
            if _on_day(epoch, day):
                return epoch, False
    logged = first_log_timestamp(scan_dir / "scan.log")
    if logged is not None and _on_day(logged.timestamp(), day):
        return logged.timestamp(), False
    try:
        mtime = (scan_dir / f"ScanInfoScan{number:03d}.ini").stat().st_mtime
    except OSError:
        mtime = None
    if _on_day(mtime, day):
        return mtime, True
    return datetime(day.year, day.month, day.day, 12).timestamp(), True


def _scan_variable_column(parameter: str, header: list[str]) -> str:
    """The s-file column that records *parameter* (it may carry an alias).

    Master Control writes the ini's ``Scan Parameter`` as ``Device Variable``
    and the s-file column as ``Device Variable Alias:<alias>`` when the
    variable has one.
    """
    if parameter in header:
        return parameter
    for column in header:
        if column.startswith(f"{parameter} Alias:"):
            return column
    return parameter


def _exit_status(end_info: str, closed: bool) -> Optional[str]:
    """``ScanEndInfo`` when it says something, else the run-closed evidence.

    The ``ScanEndInfo`` words follow the logbook's ``scan_status``
    (``success`` / ``fail…`` / ``abort…``; any other text is ``unknown``,
    never promoted to success).  Master Control leaves ``ScanEndInfo``
    empty on every scan, finished or not, so there an empty value falls
    back to the s-file evidence (:func:`_sfile_head`) — the portal needs a
    finished verdict to cache and to serve a completed run as immutable.
    With neither, the scan reads as still running (``None``).
    """
    text = end_info.strip().lower()
    if text == "success":
        return "success"
    if text.startswith("fail"):
        return "fail"
    if text.startswith("abort"):
        return "abort"
    if text:
        return "unknown"
    return "success" if closed else None


def folder_start_doc(
    scan_dir: Path, experiment: str, day: date, number: int
) -> tuple[dict, dict]:
    """Synthesize a run's start and stop documents from its scan folder.

    Parameters
    ----------
    scan_dir : Path
        The existing ``scans/ScanNNN`` folder.
    experiment : str
        The experiment directory name.
    day : datetime.date
        The folder's day.
    number : int
        The scan number.

    Returns
    -------
    tuple of (dict, dict)
        ``(start_doc, stop_doc)`` in the keys ``tiled_schema`` reads; the
        stop document is empty while the scan looks unfinished.
    """
    info = read_scan_info_file(scan_dir / f"ScanInfoScan{number:03d}.ini")
    header, first, closed = _sfile_head(scan_dir)
    started, approximate = _start_time(scan_dir, number, header, first, day)
    start_doc: dict = {
        "scan_number": number,
        "experiment": experiment,
        "time": started,
        "time_approximate": approximate,
        "description": info.get("ScanStartInfo", ""),
        "scan_folder": str(scan_dir),
        "source": "folder",
    }
    parameter = info.get("Scan Parameter", "").strip()
    start = _float(info.get("Start"))
    end = _float(info.get("End"))
    step = _float(info.get("Step size"))
    shots_per_step = _float(info.get("Shots per step"))
    if parameter.lower() not in _NOSCAN_PARAMETERS:
        start_doc["motors"] = [_scan_variable_column(parameter, header)]
        start_doc["plan_pattern"] = "inner_product"
    if None not in (start, end, step, shots_per_step) and step:
        start_doc["num_points"] = int(round(abs(end - start) / abs(step))) + 1
        start_doc["shots_per_step"] = int(shots_per_step)
    status = _exit_status(info.get("ScanEndInfo", ""), closed)
    stop_doc = {"exit_status": status} if status else {}
    return start_doc, stop_doc


class FolderScanCatalog:
    """A day's ``scans/ScanNNN`` folders as runs (see the module docstring).

    A **finished** scan's documents are remembered (bounded LRU, keyed by
    its folder): nothing that feeds them changes once a scan has ended, so a
    finished day lists from one directory read and no file opens.  The
    share's pathology is file-operation count, and the portal re-lists the
    day on every scan page (the neighbour buttons).  An unfinished scan —
    no ``exit_status`` yet — is re-read on every call, so it flips to
    finished the moment its ``ScanEndInfo`` or analysis s-file appears.

    Parameters
    ----------
    base_path : Path, optional
        The data root; defaults to the ``GeecsPathsConfig`` base path.
        Tests pass a tmp tree.
    """

    def __init__(self, base_path: Optional[Path] = None):
        self._base_path = base_path
        self._lock = threading.Lock()
        self._finished: OrderedDict[str, tuple[dict, dict]] = OrderedDict()

    def _documents(
        self, scan_dir: Path, experiment: str, day: date, number: int
    ) -> tuple[dict, dict]:
        """Return a scan's documents, from memory once the scan has finished."""
        key = str(scan_dir)
        with self._lock:
            hit = self._finished.get(key)
            if hit is not None:
                self._finished.move_to_end(key)
        if hit is None:
            hit = folder_start_doc(scan_dir, experiment, day, number)
            if hit[1].get("exit_status"):
                with self._lock:
                    self._finished[key] = hit
                    while len(self._finished) > _FINISHED_CACHE_SIZE:
                        self._finished.popitem(last=False)
        # Copies: a consumer mutating its start_doc must not edit the cache.
        return copy.deepcopy(hit)

    def _scans_dir(self, experiment: str, day: date) -> Optional[Path]:
        folder = daily_scan_folder(experiment, base_path=self._base_path, day=day)
        return folder if folder is not None and folder.is_dir() else None

    def probe(self) -> CatalogStatus:
        """Report whether the data root is reachable; never raises.

        Returns
        -------
        CatalogStatus
            ``ok`` when the base path is a readable directory.
        """
        from geecs_data_utils.scan_paths import ScanPaths

        base = self._base_path
        if base is None and ScanPaths.paths_config is not None:
            base = Path(ScanPaths.paths_config.base_path)
        try:
            if base is not None and base.is_dir():
                return CatalogStatus(ok=True, label=f"folders: {base}")
        except OSError as exc:
            return CatalogStatus(ok=False, label=f"folders: {exc}")
        return CatalogStatus(ok=False, label="folders: data root unavailable")

    def list_runs(
        self, experiment: str, day: date, *, skip: frozenset[int] = frozenset()
    ) -> list[RunSummary]:
        """Return *day*'s scan folders for *experiment*, newest first.

        Parameters
        ----------
        experiment : str
            The experiment directory name (config default when empty).
        day : datetime.date
            The folder day.
        skip : frozenset of int, optional
            Scan numbers to leave out **without reading their files** —
            the merge passes the numbers its primary already listed, so a
            Bluesky day costs one directory listing, not a read per scan.

        Returns
        -------
        list of RunSummary
            One row per ``ScanNNN`` folder; empty when the day has none.
        """
        scans_dir = self._scans_dir(experiment, day)
        if scans_dir is None:
            return []
        # .../{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/scans
        resolved_experiment = scans_dir.parents[3].name
        summaries = []
        # scandir: the entry type comes with the listing, so telling a
        # ScanNNN folder from a file costs no per-entry stat on the share.
        with os.scandir(scans_dir) as entries:
            folders = sorted(
                (int(match.group("number")), Path(entry.path))
                for entry in entries
                if (match := _SCAN_FOLDER_RE.match(entry.name))
                and int(match.group("number")) not in skip
                and entry.is_dir()
            )
        for number, scan_dir in folders:
            start_doc, stop_doc = self._documents(
                scan_dir, resolved_experiment, day, number
            )
            summaries.append(
                summary_from_metadata(
                    folder_uid(resolved_experiment, day, number), start_doc, stop_doc
                )
            )
        summaries.sort(key=lambda s: (s.start_time, s.scan_number or 0), reverse=True)
        return summaries

    def load_run(self, uid: str) -> RunDetail:
        """Load one folder run: synthesized documents, no event table.

        Parameters
        ----------
        uid : str
            A folder uid (:func:`folder_uid`).

        Returns
        -------
        RunDetail
            ``data=None`` — the s-file is the scan's scalar table and
            :func:`~geecs_data_utils.scan_frame.scan_frame` reads it.

        Raises
        ------
        KeyError
            A uid that is not a folder uid, or whose folder does not exist.
        """
        parts = parse_folder_uid(uid)
        if parts is None:
            raise KeyError(f"not a folder run uid: {uid!r}")
        experiment, day, number = parts
        scans_dir = self._scans_dir(experiment, day)
        scan_dir = None if scans_dir is None else scans_dir / f"Scan{number:03d}"
        if scan_dir is None or not scan_dir.is_dir():
            raise KeyError(f"no scan folder for {uid!r}")
        start_doc, stop_doc = self._documents(scan_dir, experiment, day, number)
        return RunDetail(
            summary=summary_from_metadata(uid, start_doc, stop_doc),
            start_doc=start_doc,
            stop_doc=stop_doc,
            data=None,
        )


class MergedScanCatalog:
    """A primary catalog with the day's uncaptured scan folders appended.

    Parameters
    ----------
    primary : ScanCatalog
        The run catalog (Tiled).
    folders : FolderScanCatalog
        The scan folders.
    """

    def __init__(self, primary: ScanCatalog, folders: FolderScanCatalog):
        self._primary = primary
        self._folders = folders

    def probe(self) -> CatalogStatus:
        """Delegate to the primary — the folders are the fallback, not the state.

        Returns
        -------
        CatalogStatus
            The primary catalog's probe.
        """
        return self._primary.probe()

    def list_runs(self, experiment: str, day: date) -> list[RunSummary]:
        """Primary runs plus every folder no primary run claims, newest first.

        The primary answers first, and its scan numbers are skipped in the
        folder listing unread.  Each side degrades to the other: a primary
        outage lists the folders alone, a share error (``OSError`` — an SMB
        blip, a permissions glitch) lists the primary alone, both logged.
        Only when both fail does the primary's error propagate, keeping the
        front-end's outage report for a day that truly has no answer.

        Parameters
        ----------
        experiment : str
            Experiment name.
        day : datetime.date
            The day.

        Returns
        -------
        list of RunSummary
            The merged listing.
        """
        try:
            primary_runs = self._primary.list_runs(experiment, day)
        except Exception as exc:
            primary_error: Optional[Exception] = exc
            primary_runs = []
        else:
            primary_error = None
        claimed = frozenset(
            run.scan_number for run in primary_runs if run.scan_number is not None
        )
        try:
            folder_runs = self._folders.list_runs(experiment, day, skip=claimed)
        except OSError:
            if primary_error is not None:
                raise primary_error from None
            logger.warning(
                "scan folders unreadable — listing the primary catalog only",
                exc_info=True,
            )
            return primary_runs
        if primary_error is not None:
            if not folder_runs:
                raise primary_error
            logger.warning(
                "primary catalog unavailable — listing scan folders only",
                exc_info=primary_error,
            )
        merged = primary_runs + folder_runs
        merged.sort(key=lambda s: s.start_time, reverse=True)
        return merged

    def load_run(self, uid: str) -> RunDetail:
        """Route a folder uid to the folders and every other uid to the primary.

        Parameters
        ----------
        uid : str
            The run uid.

        Returns
        -------
        RunDetail
            The loaded run.
        """
        if uid.startswith(FOLDER_UID_PREFIX):
            return self._folders.load_run(uid)
        return self._primary.load_run(uid)
