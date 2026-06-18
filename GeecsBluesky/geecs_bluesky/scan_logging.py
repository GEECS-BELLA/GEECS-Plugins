"""Per-scan ``scan.log`` file for Bluesky scans.

The legacy GEECS scanner wrote a per-scan log file to
``scans/ScanNNN/scan.log`` that proved invaluable for debugging and for users
passing along records of issues — a ``triage`` CLI/skill parses these logs.  In
``use_bluesky`` mode the legacy ``ScanManager`` (which owned that file) is
bypassed, so :class:`~geecs_bluesky.scanner_bridge.BlueskyScanner` restores the
file here.

A :func:`scan_log_file` context manager attaches a :class:`logging.FileHandler`
to the **root** logger for the duration of a scan, so the file captures not only
``geecs_bluesky.*`` but also ``bluesky.*`` and ``ophyd_async.*`` — a complete
debug trail for the run.  The line format is imported from
``geecs_data_utils`` (the same module that owns the triage parser) so the
writer and parser cannot drift.

This is a **consumer** of the scan folder: it writes ``scan.log`` into an
already-claimed ``scans/ScanNNN/`` folder but never creates the folder itself
(the cross-package "analysis code never creates scan folders" invariant).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

#: Default per-scan log file name (matches the legacy scanner and the triage
#: loader's ``load_scan_log`` default).
SCAN_LOG_FILENAME = "scan.log"


class _ShotIdDefaultFilter(logging.Filter):
    """Inject a default ``shot_id`` so the ``shot=%(shot_id)s`` field resolves.

    The shared scan-log format references ``%(shot_id)s``, which is not a
    standard ``LogRecord`` attribute.  Records emitted outside any shot context
    (most of them, in the Bluesky backend) would raise ``KeyError`` at format
    time without this.  We default to ``"-"`` — the same sentinel the legacy
    scanner and the triage parser expect for "no shot".
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Set ``record.shot_id`` to ``"-"`` when absent; always keep the record."""
        if not hasattr(record, "shot_id"):
            record.shot_id = "-"
        return True


@contextmanager
def scan_log_file(
    scan_folder: str | Path,
    *,
    level: int = logging.INFO,
    filename: str = SCAN_LOG_FILENAME,
) -> Iterator[Path | None]:
    """Capture root-logger output to ``<scan_folder>/scan.log`` for a scan.

    Attaches a :class:`logging.FileHandler` on entry and detaches + closes it on
    exit, even if the body raises — so a per-scan log is always finalised.  The
    handler is added to the root logger, so it captures ``geecs_bluesky.*``,
    ``bluesky.*``, and ``ophyd_async.*`` records at *level* or above.

    Parameters
    ----------
    scan_folder:
        The already-claimed ``scans/ScanNNN/`` folder.  Must exist — this
        function never creates it (cross-package scan-folder invariant).  If it
        is missing or ``None``, logging-to-file is skipped (a warning is logged
        to the existing handlers) and the body still runs.
    level:
        Minimum level the file captures.  Defaults to ``INFO`` — the full
        ``geecs_bluesky`` trail plus any ``bluesky`` / ``ophyd_async`` warnings
        and errors, without the flood of ophyd-async DEBUG.
    filename:
        Log file name within *scan_folder*.  Defaults to ``"scan.log"`` to match
        the legacy scanner and ``geecs_data_utils.load_scan_log``.

    Yields
    ------
    pathlib.Path or None
        The path of the log file being written, or ``None`` when logging-to-file
        was skipped.
    """
    # Import here so geecs_bluesky stays importable even if geecs_data_utils is
    # momentarily unavailable; the format lives with the triage parser.
    from geecs_data_utils import SCAN_LOG_DATEFMT, SCAN_LOG_FORMAT

    folder = Path(scan_folder) if scan_folder is not None else None
    if folder is None or not folder.is_dir():
        logger.warning("Scan folder %s unavailable; per-scan log file disabled", folder)
        yield None
        return

    log_path = folder / filename
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(SCAN_LOG_FORMAT, datefmt=SCAN_LOG_DATEFMT))
    handler.addFilter(_ShotIdDefaultFilter())

    root = logging.getLogger()
    # The root logger's own level gates what reaches handlers; lower it if it
    # would otherwise filter out records this file wants (it is restored below).
    prev_root_level = root.level
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    root.addHandler(handler)
    logger.info("Per-scan log started: %s", log_path)
    try:
        yield log_path
    finally:
        logger.info("Per-scan log finished: %s", log_path)
        root.removeHandler(handler)
        handler.close()
        root.setLevel(prev_root_level)
