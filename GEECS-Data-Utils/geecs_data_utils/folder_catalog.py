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
import time
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

from geecs_data_utils.data.sfile import sfile_column, sfile_head_for_scan
from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
from geecs_data_utils.scan_info import (
    ScanInfoRecord,
    classify_end_info,
    finite_float,
    parse_scan_folder_name,
    read_scan_info_file,
    scan_folder_name,
)
from geecs_data_utils.scan_log_loader import first_log_timestamp
from geecs_data_utils.scan_paths import current_paths_config, daily_scan_folder
from geecs_data_utils.tiled_catalog import (
    CatalogStatus,
    CatalogUnreachable,
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

#: :func:`~geecs_data_utils.scan_info.classify_end_info`'s outcome in the
#: stop document's ``exit_status`` words — the vocabulary every Tiled run
#: carries, so a folder run and a Bluesky run read alike.  ``incomplete``
#: has no stop document yet.
_EXIT_STATUS = {
    "success": "success",
    "failed": "fail",
    "aborted": "abort",
    "unknown": "unknown",
    "incomplete": None,
}


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


def _on_day(epoch: Optional[float], day: date) -> bool:
    if epoch is None:
        return False
    try:
        return datetime.fromtimestamp(epoch).date() == day
    except (OverflowError, OSError, ValueError):
        return False


def _start_time(
    scan_dir: Path, header: list[str], first: list[str], day: date
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
        raw = finite_float(first[index]) if index < len(first) else None
        if raw is not None and raw > LABVIEW_EPOCH_OFFSET:
            epoch = raw - LABVIEW_EPOCH_OFFSET
            if _on_day(epoch, day):
                return epoch, False
    logged = first_log_timestamp(scan_dir / "scan.log")
    if logged is not None and _on_day(logged.timestamp(), day):
        return logged.timestamp(), False
    try:
        mtime = (scan_dir / f"ScanInfo{scan_dir.name}.ini").stat().st_mtime
    except OSError:
        mtime = None
    if _on_day(mtime, day):
        return mtime, True
    return datetime(day.year, day.month, day.day, 12).timestamp(), True


def _scan_variable_column(parameter: str, header: list[str]) -> str:
    """The s-file column that records *parameter* (it may carry an alias)."""
    return sfile_column(parameter, header) or parameter


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
    info = ScanInfoRecord.from_ini(
        read_scan_info_file(scan_dir / f"ScanInfo{scan_dir.name}.ini")
    )
    header, first, closed = sfile_head_for_scan(scan_dir)
    started, approximate = _start_time(scan_dir, header, first, day)
    start_doc: dict = {
        "scan_number": number,
        "experiment": experiment,
        "time": started,
        "time_approximate": approximate,
        "description": info.purpose or "",
        "scan_folder": str(scan_dir),
        "source": "folder",
    }
    start, end, step = info.start, info.end, info.step_size
    if not info.noscan and info.parameter:
        start_doc["motors"] = [_scan_variable_column(info.parameter, header)]
        start_doc["plan_pattern"] = "inner_product"
    if (
        start is not None
        and end is not None
        and step
        and info.shots_per_step is not None
    ):
        start_doc["num_points"] = int(round(abs(end - start) / abs(step))) + 1
        start_doc["shots_per_step"] = info.shots_per_step
    status = _EXIT_STATUS[classify_end_info(info.end_info, closed=closed)]
    stop_doc = {"exit_status": status} if status else {}
    return start_doc, stop_doc


def _scan_folders(scans_dir: Path) -> dict[int, Path]:
    """A day's scan folders by number, from one ``os.scandir``.

    The entry type comes with the listing, so telling a ``ScanNNN`` folder
    from a file costs no per-entry stat on the share.  Should both
    ``Scan007`` and ``Scan7`` exist, the canonical (padded) name wins.
    """
    found: dict[int, Path] = {}
    with os.scandir(scans_dir) as entries:
        for entry in entries:
            number = parse_scan_folder_name(entry.name)
            if number is None or not entry.is_dir():
                continue
            if number not in found or entry.name == scan_folder_name(number):
                found[number] = Path(entry.path)
    return found


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

    def _data_root(self) -> Optional[Path]:
        """The data root: the constructor's, else ``config.ini``'s.

        Through :func:`~geecs_data_utils.scan_paths.current_paths_config`,
        which looks again for a root that was missing when the process
        started, so the catalog recovers when the share comes back.
        """
        if self._base_path is not None:
            return self._base_path
        config = current_paths_config()
        return Path(config.base_path) if config is not None else None

    def _scans_dir(self, experiment: str, day: date) -> Optional[Path]:
        """The day's ``scans/`` folder; ``None`` when the day has none.

        Raises
        ------
        OSError
            The data root itself is unreachable (an unmounted share, or none
            configured): that is an outage, never an empty day.
        """
        root = self._data_root()
        folder = daily_scan_folder(experiment, base_path=root, day=day)
        if folder is not None and folder.is_dir():
            return folder
        if root is None or not root.is_dir():
            raise OSError(
                f"the data root {str(root) if root else '(none in config.ini [Paths])'} "
                "is unreachable: is the data share mounted?"
            )
        return None

    def probe(self) -> CatalogStatus:
        """Report whether the data root is reachable; never raises.

        Returns
        -------
        CatalogStatus
            ``ok`` when the base path is a readable directory.
        """
        base = self._data_root()
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

        Raises
        ------
        OSError
            The data root is unreachable (see :meth:`_scans_dir`).
        """
        scans_dir = self._scans_dir(experiment, day)
        if scans_dir is None:
            return []
        # .../{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/scans
        resolved_experiment = scans_dir.parents[3].name
        summaries = []
        folders = _scan_folders(scans_dir)
        for number, scan_dir in sorted(folders.items()):
            if number in skip:
                continue
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
        try:
            scans_dir = self._scans_dir(experiment, day)
        except OSError:  # the root unreachable is still "no folder" here
            scans_dir = None
        scan_dir: Optional[Path] = None
        if scans_dir is not None:
            scan_dir = scans_dir / scan_folder_name(number)
            if not scan_dir.is_dir():
                # An unpadded folder (``Scan7``) lists under its number too.
                try:
                    scan_dir = _scan_folders(scans_dir).get(number)
                except OSError:  # a share blip is still "no folder" here
                    scan_dir = None
        if scan_dir is None:
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

    A primary that cannot be reached (:class:`CatalogUnreachable`, or any
    ``ConnectionError``/``TimeoutError``) is not asked again for
    ``retry_after`` seconds: an unreachable Tiled costs a connect timeout
    and its retries (about a minute) on every call otherwise, and a page
    that lists a day would wait that long each time. Meanwhile listings
    come from the folders alone and a primary uid's ``load_run`` raises the
    remembered error at once (an outage, as the protocol says). A primary
    that answers but fails one query (one day's malformed run) is not
    skipped: that error stays with its call.

    Parameters
    ----------
    primary : ScanCatalog
        The run catalog (Tiled).
    folders : FolderScanCatalog
        The scan folders.
    retry_after : float, default 300
        Seconds a failed primary is skipped before it is tried again.
    clock : callable, optional
        Monotonic seconds; ``time.monotonic`` by default (tests inject one).
    """

    def __init__(
        self,
        primary: ScanCatalog,
        folders: FolderScanCatalog,
        *,
        retry_after: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._primary = primary
        self._folders = folders
        self._retry_after = retry_after
        self._clock = clock
        # When the primary was last found unreachable, and how; cleared by
        # its next success.
        self._primary_failure: Optional[tuple[float, Exception]] = None

    def _skipping(self) -> Optional[Exception]:
        """The remembered unreachable-primary error while its window lasts."""
        failure = self._primary_failure
        if failure is not None and self._clock() - failure[0] < self._retry_after:
            return failure[1]
        return None

    def _unreachable(self, exc: Exception) -> bool:
        """Remember *exc* if it says the primary cannot be reached at all."""
        if isinstance(exc, (CatalogUnreachable, ConnectionError, TimeoutError)):
            self._primary_failure = (self._clock(), exc)
            return True
        self._primary_failure = None  # it answered, if only with an error
        return False

    def _primary_listing(
        self, experiment: str, day: date
    ) -> tuple[list[RunSummary], Optional[Exception], bool]:
        """The primary's runs, or ``([], error, fresh)``.

        ``fresh`` is false while a remembered failure skips the primary.
        """
        skipped = self._skipping()
        if skipped is not None:
            return [], skipped, False
        try:
            runs = self._primary.list_runs(experiment, day)
        except Exception as exc:  # noqa: BLE001 — primary catalog fails over
            self._unreachable(exc)
            return [], exc, True
        self._primary_failure = None
        return runs, None, True

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
        outage lists the folders alone (and the primary is skipped for
        ``retry_after`` seconds), a share error (``OSError`` — an SMB
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
        primary_runs, primary_error, fresh = self._primary_listing(experiment, day)
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
            if fresh:
                logger.warning(
                    "primary catalog unavailable — listing scan folders only%s",
                    f" (not asked again for {self._retry_after:.0f} s)"
                    if self._primary_failure is not None
                    else "",
                    exc_info=primary_error,
                )
            else:
                logger.info(
                    "primary catalog skipped after a recent failure — "
                    "listing scan folders only"
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
        skipped = self._skipping()
        if skipped is not None:
            raise skipped
        try:
            detail = self._primary.load_run(uid)
        except Exception as exc:
            self._unreachable(exc)
            raise
        self._primary_failure = None
        return detail
