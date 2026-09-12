"""Read and parse `scan.log` files written by `geecs_scanner.logging_setup`.

The per-scan log format is owned by
``GEECS-Scanner-GUI/geecs_scanner/logging_setup.py::attach_scan_log``::

    "%(asctime)s.%(msecs)03d %(levelname)s %(name)s [%(threadName)s] "
    "shot=%(shot_id)s - %(message)s"

with ``datefmt="%Y-%m-%d %H:%M:%S"``. Multi-line tracebacks (or any
continuation text) appear as additional lines that do **not** start with a
header matching :data:`HEADER_RE`; those are aggregated into the previous
record's ``traceback`` field.

This module deliberately lives in `geecs_data_utils` (not in a triage-specific
package) so that any consumer — interactive notebooks, plotting helpers,
debugging scripts, or higher-level triage tooling — can read scan logs with
the same data model.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Union

from pydantic import BaseModel, ConfigDict


# --------------------------------------------------------------------------
# Public types
# --------------------------------------------------------------------------


class Severity(str, Enum):
    """Standard Python `logging` severity levels, as strings.

    Stored as strings rather than ints so JSON round-trips are readable.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @property
    def level_no(self) -> int:
        """Return the numeric `logging` level for this severity."""
        return {
            Severity.DEBUG: 10,
            Severity.INFO: 20,
            Severity.WARNING: 30,
            Severity.ERROR: 40,
            Severity.CRITICAL: 50,
        }[self]


class LogEntry(BaseModel):
    """A single parsed log record from a `scan.log` file.

    Attributes
    ----------
    timestamp : datetime
        Record timestamp parsed from `%(asctime)s.%(msecs)03d`.
    level : Severity
        Severity level.
    logger_name : str
        Python logger name (e.g., `geecs_scanner.scan_manager`).
    thread_name : str
        Originating thread name.
    shot_id : str
        Shot identifier from the log context (`-` if unset).
    message : str
        Single-line message body.
    traceback : str, optional
        Multi-line traceback text appended to the record, if any. Lines
        that immediately follow a header line and that don't themselves
        match the header regex are accumulated here.
    source_file : Path, optional
        Path to the `scan.log` this entry was parsed from.
    line_number : int, optional
        1-based line number of the header line within `source_file`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime
    level: Severity
    logger_name: str
    thread_name: str
    shot_id: str
    message: str
    traceback: Optional[str] = None
    source_file: Optional[Path] = None
    line_number: Optional[int] = None


# --------------------------------------------------------------------------
# Header regex
# --------------------------------------------------------------------------
#
# Captures:
#   ts    - "YYYY-MM-DD HH:MM:SS.mmm"
#   level - DEBUG / INFO / WARNING / ERROR / CRITICAL
#   name  - logger name (no whitespace, dots allowed)
#   thread- thread name (within square brackets)
#   shot  - context token (no whitespace; literal "-" if unset).  The legacy
#           engine wrote ``shot=<n>``; the Bluesky stack writes
#           ``scan=ScanNNN`` (``geecs_bluesky/scan_log.py``) — both are
#           accepted, and the capture keeps its historical name ``shot``
#           so downstream consumers are unaffected.
#   msg   - rest of the line
#
HEADER_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+"
    r"(?P<name>\S+)\s+"
    r"\[(?P<thread>[^\]]+)\]\s+"
    r"(?:shot|scan)=(?P<shot>\S+)\s+-\s+"
    r"(?P<msg>.*)$"
)


def _parse_header(line: str) -> Optional[dict]:
    """Return header field dict if `line` matches HEADER_RE, else None."""
    m = HEADER_RE.match(line.rstrip("\n"))
    if not m:
        return None
    return m.groupdict()


def _materialize(
    fields: dict,
    traceback_lines: list[str],
    source_file: Optional[Path],
    line_number: Optional[int],
) -> LogEntry:
    """Convert a header field dict + traceback buffer into a `LogEntry`."""
    ts = datetime.strptime(fields["ts"], "%Y-%m-%d %H:%M:%S.%f")
    tb = "\n".join(traceback_lines).rstrip() if traceback_lines else None
    return LogEntry(
        timestamp=ts,
        level=Severity(fields["level"]),
        logger_name=fields["name"],
        thread_name=fields["thread"],
        shot_id=fields["shot"],
        message=fields["msg"],
        traceback=tb,
        source_file=source_file,
        line_number=line_number,
    )


# --------------------------------------------------------------------------
# Public parse / load API
# --------------------------------------------------------------------------


def parse_lines(
    lines: Iterable[str],
    source_file: Optional[Path] = None,
) -> list[LogEntry]:
    """Parse an iterable of raw log lines into `LogEntry` records.

    Multi-line continuations (lines that don't match `HEADER_RE`) are
    appended to the immediately preceding record's traceback. Orphan
    continuation lines that appear before the first header line are
    silently skipped (typically noise from log truncation).

    Parameters
    ----------
    lines : Iterable[str]
        Raw lines from a `scan.log` file. Trailing newlines are tolerated.
    source_file : Path, optional
        Path to record on each emitted `LogEntry.source_file`.

    Returns
    -------
    list[LogEntry]
        Parsed records in file order.
    """
    out: list[LogEntry] = []
    cur_fields: Optional[dict] = None
    cur_tb: list[str] = []
    cur_lineno: Optional[int] = None

    for idx, raw in enumerate(lines, start=1):
        header = _parse_header(raw)
        if header is not None:
            if cur_fields is not None:
                out.append(_materialize(cur_fields, cur_tb, source_file, cur_lineno))
            cur_fields = header
            cur_tb = []
            cur_lineno = idx
        else:
            if cur_fields is not None:
                cur_tb.append(raw.rstrip("\n"))
            # Else: orphan continuation; silently skip.

    if cur_fields is not None:
        out.append(_materialize(cur_fields, cur_tb, source_file, cur_lineno))

    return out


def first_log_timestamp(scan_log_path: Union[Path, str]) -> Optional[datetime]:
    """Return the timestamp of a scan log's first record, or ``None``.

    Reads only until the first parsable header rather than the whole file,
    so a caller iterating a day of scans pays one short read per scan
    instead of parsing every line of every log.

    This is the honest answer to "when did this scan run". A scan folder's
    modification time is not: any later pass that writes into the folder
    (the analysis task queue's ``analysis_status/``, for one) moves it, and
    it has been observed more than an hour off the real start.

    Parameters
    ----------
    scan_log_path : Path or str
        Filesystem path to the ``scan.log`` file. Need not exist.

    Returns
    -------
    datetime or None
        The first record's timestamp, or ``None`` when the file is
        missing, empty, or has no parsable header line. Never raises:
        callers iterating a day must not be taken down by one bad log.
    """
    path = Path(scan_log_path)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = _parse_header(line)
                if fields:
                    return datetime.strptime(fields["ts"], "%Y-%m-%d %H:%M:%S.%f")
    except (OSError, ValueError):
        return None
    return None


def parse_scan_log(scan_log_path: Union[Path, str]) -> list[LogEntry]:
    """Parse a `scan.log` file from disk.

    Parameters
    ----------
    scan_log_path : Path or str
        Filesystem path to the `scan.log` file.

    Returns
    -------
    list[LogEntry]
        Parsed records, with `source_file` populated.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    p = Path(scan_log_path)
    if not p.is_file():
        raise FileNotFoundError(f"scan log not found: {p}")
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        return parse_lines(fh, source_file=p)


def load_scan_log(
    scan_folder: Union[Path, str],
    filename: str = "scan.log",
) -> list[LogEntry]:
    """Load and parse the per-scan log file from a scan folder.

    Convenience wrapper for the common case where you have a scan folder
    (e.g., resolved via :class:`ScanPaths`) and want to read its
    `scan.log` file.

    Parameters
    ----------
    scan_folder : Path or str
        Path to the scan folder. The function looks for `<scan_folder>/<filename>`.
    filename : str, optional
        Log file name within the scan folder. Defaults to ``"scan.log"`` to
        match :func:`geecs_scanner.logging_setup.attach_scan_log`.

    Returns
    -------
    list[LogEntry]
        Parsed log entries. Returns an empty list if the file exists but is
        empty.

    Raises
    ------
    FileNotFoundError
        If `<scan_folder>/<filename>` does not exist.
    """
    folder = Path(scan_folder)
    return parse_scan_log(folder / filename)
