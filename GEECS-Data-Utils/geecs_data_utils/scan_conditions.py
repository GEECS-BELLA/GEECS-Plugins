"""Scan conditions: the labels of a scalar-columns job's condition columns.

A condition column (a ``ScalarJob`` column with ``cases``) stores, per
shot, the number of the rule that matched — 0 for the first case, 1 for
the next, the last number for ``otherwise``.  The words those numbers
stand for are written beside the job's scalar sidecar, one small JSON file
per job and scan::

    analysis/ScanNNN/ScanNNN_<job id>_labels.json
    {"schema_version": 1, "job": "<job id>",
     "columns": {"target in beam": ["Al foil", "Cu foil", "nothing"]}}

so a page reads them from the scan's own analysis folder (no configs tree)
and they are the words in force when the scan was analysed.  This module
is the one definition of that file — the writer (ScanAnalysis' scalar
job) and the readers (the Data Portal) both import it — and of the
per-scan summary: which label the scan's shots carry, and whether they
disagree.

Nothing here creates a directory: the writer's parent must exist.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "LABELS_SUFFIX",
    "ScanCondition",
    "labels_path",
    "read_condition_labels",
    "scan_conditions",
    "write_condition_labels",
]

#: The file-name tail of a job's labels file.
LABELS_SUFFIX = "_labels.json"


def labels_path(analysis_scan_dir: Path, scan_name: str, job_id: str) -> Path:
    """Where a job's labels for one scan live (``ScanNNN_<job>_labels.json``)."""
    return Path(analysis_scan_dir) / f"{scan_name}_{job_id}{LABELS_SUFFIX}"


def write_condition_labels(
    path: Path, job_id: str, columns: Mapping[str, list[str]]
) -> Path:
    """Replace one job's labels file atomically; its folder must already exist."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"{path.parent} does not exist")
    body = {
        "schema_version": 1,
        "job": job_id,
        "columns": {name: list(labels) for name, labels in columns.items()},
    }
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(body, out, indent=1)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return path


def read_condition_labels(analysis_scan_dir: Path) -> dict[str, list[str]]:
    """Every condition column's labels found in a scan's analysis folder.

    Tolerant: a missing folder is no conditions, and an unreadable or
    malformed file is skipped with a log line.  Files are read in name
    order, so when two jobs label one column the later name wins.
    """
    folder = Path(analysis_scan_dir)
    if not folder.is_dir():
        return {}
    found: dict[str, list[str]] = {}
    for path in sorted(folder.glob(f"*{LABELS_SUFFIX}")):
        try:
            columns = json.loads(path.read_text(encoding="utf-8"))["columns"]
            for name, labels in columns.items():
                if isinstance(name, str) and isinstance(labels, list):
                    found[name] = [str(label) for label in labels]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("skipping condition labels %s: %s", path, exc)
    return found


@dataclass(frozen=True)
class ScanCondition:
    """One condition column over a scan's shots.

    Attributes
    ----------
    column : str
        The s-file column.
    counts : dict[str, int]
        Shots per label, in the labels' order, labels no shot carries left out.
    unknown : int
        Shots with no value (an input was missing).
    """

    column: str
    counts: dict[str, int] = field(default_factory=dict)
    unknown: int = 0

    @property
    def mixed(self) -> bool:
        """Whether the scan's shots carry more than one label."""
        return len(self.counts) > 1

    @property
    def label(self) -> str:
        """The scan's label: the one label, ``mixed``, or ``unknown``."""
        if not self.counts:
            return "unknown"
        if self.mixed:
            return "mixed"
        return next(iter(self.counts))

    @property
    def detail(self) -> str:
        """The split, e.g. ``Al foil 40 · nothing 25 · unknown 2`` (empty if one label)."""
        if not self.mixed and not (self.counts and self.unknown):
            return ""
        parts = [f"{label} {n}" for label, n in self.counts.items()]
        if self.unknown:
            parts.append(f"unknown {self.unknown}")
        return " · ".join(parts)


def scan_conditions(
    frame: pd.DataFrame, labels: Mapping[str, list[str]]
) -> list[ScanCondition]:
    """Summarize each labelled column *frame* has, in the labels' order.

    A stored number with no label (a job edited since) reads ``value N``.
    """
    out = []
    for column, names in labels.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        known = values[np.isfinite(values)]
        counts: dict[str, int] = {}
        for code in sorted(set(known.tolist())):
            n = int((known == code).sum())
            index = int(code)
            name = (
                names[index]
                if index == code and 0 <= index < len(names)
                else f"value {code:g}"
            )
            counts[name] = counts.get(name, 0) + n
        out.append(
            ScanCondition(
                column=column, counts=counts, unknown=int(len(values) - len(known))
            )
        )
    return out
