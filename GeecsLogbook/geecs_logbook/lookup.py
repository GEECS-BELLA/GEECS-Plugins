"""The day's variable lookup: any recorded column's first-shot value, scan by scan.

The day page's "Look up a variable" panel asks one question: *what was X at
the start of each of today's scans?* Every scan's s-file already holds the
answer in its first row — every logged variable is in every shot's row — so
this module reads two lines per scan (:func:`~geecs_logbook.scan_reader.first_row`,
cached on each file's identity) and reshapes them into a table. Nothing is
stored; nothing is read until the panel asks.

Columns are named as the s-file names them (``"Device Variable"``); a
LabVIEW column carrying an alias (``"Device Variable Alias:<alias>"``) is
listed under its plain name, so the same variable lines up across days
written by either scanner. A scan with no s-file yet (still running, or it
never closed) has no values and says so.

Read-only, like everything that reads the share here: nothing is created.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from geecs_data_utils.data.sfile import header_for_variable, sfile_column
from pydantic import BaseModel, Field

from geecs_logbook.models import KIT_STATE, DaySummary, ScanStatus
from geecs_logbook.scan_reader import _READ_WORKERS, first_row

#: The suffix Master Control appends to an aliased variable's column.
_ALIAS = " Alias:"


class LookupColumn(BaseModel):
    """One column some scan of the day recorded."""

    name: str = Field(description='The s-file column, "Device Variable"')
    scans: int = Field(description="How many of the day's scans recorded it")
    listed: bool = Field(
        description="On some scan's scan-log list (the scanner's shared list)"
    )


class LookupRow(BaseModel):
    """One scan of the day and its first-shot values for the asked columns."""

    number: int
    label: str
    started: Optional[str] = Field(default=None, description="HH:MM:SS")
    status: ScanStatus
    state: str = Field(description="The status as a kit word, for the page's dot")
    parameter: Optional[str] = None
    recorded: bool = Field(description="Whether the scan has an s-file yet")
    values: dict[str, Optional[str]] = Field(default_factory=dict)


class DayLookup(BaseModel):
    """The lookup table for one day."""

    day: str
    columns: list[LookupColumn] = Field(
        default_factory=list, description="Every column any scan recorded"
    )
    default_columns: list[str] = Field(
        default_factory=list,
        description="The scan-log lists' variables, in list order (the panel's first view)",
    )
    shown: list[str] = Field(default_factory=list, description="The columns in rows")
    rows: list[LookupRow] = Field(default_factory=list)


def _plain(column: str) -> str:
    cut = column.find(_ALIAS)
    return column[:cut] if cut >= 0 else column


def day_lookup(summary: DaySummary, asked: Optional[list[str]] = None) -> DayLookup:
    """Build the day's table: every column, and the *asked* ones' values per scan.

    Parameters
    ----------
    summary : DaySummary
        The day as :func:`~geecs_logbook.scan_reader.read_day` listed it.
    asked : list of str, optional
        Columns to fill (``"Device Variable"``). ``None`` means the day's
        scan-log lists — what the scanner asked every scan to show.
    """
    default: list[str] = []
    for scan in summary.scans:
        for name in scan.scan_log_variables:
            column = header_for_variable(name)
            if column not in default:
                default.append(column)
    # An empty ?col= (every column removed) asks for none, not the defaults.
    shown = list(dict.fromkeys(c for c in asked if c)) if asked is not None else default

    folder = Path(summary.folder)
    targets = [folder / (s.folder_name or s.label) for s in summary.scans]
    rows_by_scan: list[Optional[dict[str, str]]] = []
    if targets:
        with ThreadPoolExecutor(max_workers=min(_READ_WORKERS, len(targets))) as pool:
            rows_by_scan = list(pool.map(first_row, targets))

    counts: dict[str, int] = {}
    rows = []
    for scan, row in zip(summary.scans, rows_by_scan):
        header = list(row) if row else []
        for column in dict.fromkeys(_plain(c) for c in header):
            counts[column] = counts.get(column, 0) + 1
        values: dict[str, Optional[str]] = {}
        for name in shown:
            column = sfile_column(name, header)
            values[name] = (row.get(column) or None) if row and column else None
        rows.append(
            LookupRow(
                number=scan.number,
                label=scan.label,
                started=scan.started.strftime("%H:%M:%S") if scan.started else None,
                status=scan.status,
                state=KIT_STATE[scan.status],
                parameter=scan.parameter,
                recorded=row is not None,
                values=values,
            )
        )
    listed = set(default)
    columns = [
        LookupColumn(name=name, scans=n, listed=name in listed)
        for name, n in sorted(counts.items(), key=lambda kv: kv[0].lower())
    ]
    return DayLookup(
        day=summary.day.isoformat(),
        columns=columns,
        default_columns=default,
        shown=shown,
        rows=rows,
    )
