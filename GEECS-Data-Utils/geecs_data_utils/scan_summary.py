"""One scan's s-file at a glance, and the day list's scan rules over it.

The Data Portal's day page filters scans by what is in each one's s-file:
a condition column's label (``APL is in``) or a column's mean over the
scan's shots (``charge between 10 and 20``).  :func:`summarize_scan` reads
one scan's s-file and condition labels; :func:`scan_verdict` judges a
summary against the rules — one definition, so the page and the JSON
listing (and a notebook) agree.

A verdict is one of:

``pass``     every rule holds;
``mixed``    no rule fails, but a condition holds for some of the scan's
             shots only (its shots disagree);
``fail``     a rule does not hold;
``missing``  the scan lacks a rule's column (no s-file, a condition no job
             wrote, a column never recorded).

Read-only: nothing is created.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field, model_validator

from geecs_data_utils.data.binning import ColumnSummary, summarize_columns
from geecs_data_utils.data.sfile import read_sfile, sfile_path_for_scan
from geecs_data_utils.scan_conditions import (
    ScanCondition,
    read_condition_labels,
    scan_conditions,
)
from geecs_data_utils.scan_info import analysis_folder_for

__all__ = [
    "ScanRule",
    "ScanSummary",
    "Verdict",
    "rule_verdict",
    "scan_verdict",
    "summarize_scan",
]

Verdict = Literal["pass", "mixed", "fail", "missing"]


class ScanRule(BaseModel):
    """One rule a scan must meet: a condition's label, or a column's mean in a range."""

    column: str = Field(min_length=1, description="The s-file column.")
    op: Literal["is", "is_not", "between"] = Field(
        description="is / is_not a condition's label, or the column's mean between."
    )
    label: Optional[str] = Field(None, description="The label, for is / is_not.")
    low: Optional[float] = Field(
        None, description="Lower bound (inclusive); none: no limit."
    )
    high: Optional[float] = Field(
        None, description="Upper bound (inclusive); none: no limit."
    )

    @model_validator(mode="after")
    def _complete(self) -> "ScanRule":
        if self.op in ("is", "is_not") and not self.label:
            raise ValueError(f"{self.column!r}: '{self.op}' needs a label")
        if self.op == "between":
            if self.low is None and self.high is None:
                raise ValueError(f"{self.column!r}: 'between' needs a bound")
            if self.low is not None and self.high is not None and self.low > self.high:
                raise ValueError(f"{self.column!r}: low above high")
        return self


@dataclass(frozen=True)
class ScanSummary:
    """A scan's s-file at a glance: its shots, columns, conditions and asked-for means."""

    shots: int
    columns: frozenset[str]
    conditions: tuple[ScanCondition, ...] = ()
    labels: dict[str, list[str]] = field(default_factory=dict)
    values: dict[str, ColumnSummary] = field(default_factory=dict)

    def condition(self, column: str) -> Optional[ScanCondition]:
        """The condition column *column*, if the scan has it."""
        return next((c for c in self.conditions if c.column == column), None)


def summarize_scan(
    scan_folder: Path, columns: Sequence[str] = ()
) -> Optional[ScanSummary]:
    """One scan's summary (``None``: no s-file yet), with the means of *columns*."""
    path = sfile_path_for_scan(Path(scan_folder))
    if not path.is_file():
        return None
    frame = read_sfile(path)
    labels = read_condition_labels(analysis_folder_for(Path(scan_folder)))
    return ScanSummary(
        shots=len(frame),
        columns=frozenset(str(c) for c in frame.columns),
        conditions=tuple(scan_conditions(frame, labels)),
        labels={c: names for c, names in labels.items() if c in frame.columns},
        values=summarize_columns(frame, [c for c in columns if c in frame.columns]),
    )


def rule_verdict(summary: Optional[ScanSummary], rule: ScanRule) -> Verdict:
    """How *summary* meets one *rule*."""
    if summary is None:
        return "missing"
    if rule.op == "between":
        value = summary.values.get(rule.column)
        if value is None or value.count == 0 or math.isnan(value.center):
            return "missing"
        low = -math.inf if rule.low is None else rule.low
        high = math.inf if rule.high is None else rule.high
        return "pass" if low <= value.center <= high else "fail"
    condition = summary.condition(rule.column)
    if condition is None:
        return "missing"
    labels = set(condition.counts)
    has = rule.label in labels
    if rule.op == "is":
        if not has:
            return "fail"
        return "mixed" if len(labels) > 1 else "pass"
    # is_not
    if not has:
        return "pass"
    return "fail" if labels == {rule.label} else "mixed"


def scan_verdict(summary: Optional[ScanSummary], rules: Sequence[ScanRule]) -> Verdict:
    """How *summary* meets every rule: a fail wins, then missing, then mixed."""
    verdicts = {rule_verdict(summary, rule) for rule in rules}
    if "fail" in verdicts:
        return "fail"
    if "missing" in verdicts:
        return "missing"
    if "mixed" in verdicts:
        return "mixed"
    return "pass"
