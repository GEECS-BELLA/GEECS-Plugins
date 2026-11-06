"""The day page's scan filters: each scan's s-file summary, on demand.

The day list loads from the catalog alone (fast).  Only when asked — the
page's **Conditions & scan filters** button, or a link that carries rules
— does the portal read every scan's s-file (``geecs_data_utils.scan_summary``)
to show its conditions and judge it against the rules.  The reads run a
few at a time (the share's latency, not the CPU, is the cost) and a
summary is kept in memory keyed on the s-file's modification time, so a
second look is instant and a re-analysed scan reads afresh.

Read-only: a lookup never creates a folder.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Sequence

from geecs_data_utils.data.sfile import sfile_path_for_scan
from geecs_data_utils.scan_info import scan_folder_name
from geecs_data_utils.scan_paths import daily_scan_folder
from geecs_data_utils.scan_summary import (
    ScanRule,
    ScanSummary,
    scan_verdict,
    summarize_scan,
)

__all__ = ["DayScans", "parse_rules"]

#: Scans read at once over the share.
_READERS = 8
#: Summaries kept in memory (a busy day is ~100 scans).
_KEPT = 4000


def parse_rules(text: str) -> list[ScanRule]:
    """The ``where`` query (a JSON list of rules) as rules; ``""`` is none.

    Raises ``ValueError`` (pydantic's) on a malformed list.
    """
    import json

    if not text.strip():
        return []
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError("where: a JSON list of rules")
    return [ScanRule.model_validate(r) for r in raw]


class DayScans:
    """Summaries of a day's scans, cached by s-file modification time.

    Parameters
    ----------
    scans_folder : callable, optional
        ``(experiment, day) -> scans/ folder or None`` — the data root's
        daily folder by default; tests point it at a temporary tree.
    """

    def __init__(
        self,
        scans_folder: Optional[Callable[[str, date], Optional[Path]]] = None,
    ) -> None:
        self._scans_folder = scans_folder or (
            lambda experiment, day: daily_scan_folder(experiment=experiment, day=day)
        )
        self._kept: "OrderedDict[tuple, Optional[ScanSummary]]" = OrderedDict()
        self._lock = threading.Lock()

    def summary(self, folder: Path, columns: Sequence[str]) -> Optional[ScanSummary]:
        """One scan's summary (``None``: no s-file), from memory when unchanged."""
        sfile = sfile_path_for_scan(folder)
        try:
            stamp = sfile.stat().st_mtime_ns
        except OSError:
            return None
        key = (str(folder), stamp, tuple(sorted(columns)))
        with self._lock:
            if key in self._kept:
                self._kept.move_to_end(key)
                return self._kept[key]
        found = summarize_scan(folder, columns)
        with self._lock:
            self._kept[key] = found
            while len(self._kept) > _KEPT:
                self._kept.popitem(last=False)
        return found

    def summaries(
        self, experiment: str, day: date, numbers: Sequence[int], columns: Sequence[str]
    ) -> dict[int, Optional[ScanSummary]]:
        """Each numbered scan's summary, read a few at a time."""
        scans = self._scans_folder(experiment, day)
        if scans is None or not numbers:
            return {n: None for n in numbers}
        with ThreadPoolExecutor(max_workers=_READERS) as pool:
            found = pool.map(
                lambda n: self.summary(scans / scan_folder_name(n), columns), numbers
            )
            return dict(zip(numbers, found))


def run_json(summary: Optional[ScanSummary], rules: Sequence[ScanRule]) -> dict:
    """One scan's part of the day's scans route."""
    if summary is None:
        return {
            "verdict": scan_verdict(None, rules) if rules else "missing",
            "shots": None,
            "conditions": [],
            "values": {},
        }
    return {
        "verdict": scan_verdict(summary, rules),
        "shots": summary.shots,
        "conditions": [
            {"column": c.column, "label": c.label, "mixed": c.mixed, "detail": c.detail}
            for c in summary.conditions
        ],
        "values": {
            column: (None if value.count == 0 else value.center)
            for column, value in summary.values.items()
        },
    }
