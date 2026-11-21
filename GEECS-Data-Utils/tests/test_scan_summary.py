"""A scan's s-file at a glance, and the day list's scan rules over it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from geecs_data_utils.scan_conditions import labels_path, write_condition_labels
from geecs_data_utils.scan_summary import (
    ScanRule,
    rule_verdict,
    scan_verdict,
    summarize_scan,
)


def make_scan(root: Path, number: int, frame: pd.DataFrame | None, labels=None) -> Path:
    """A day's scans/ScanNNN folder, its analysis s-file and condition labels."""
    day = root / "data" / "Y2026" / "11-Nov" / "26_1105"
    folder = day / "scans" / f"Scan{number:03d}"
    folder.mkdir(parents=True)
    if frame is not None:
        (day / "analysis").mkdir(exist_ok=True)
        frame.to_csv(day / "analysis" / f"s{number}.txt", sep="\t", index=False)
    if labels:
        out = day / "analysis" / f"Scan{number:03d}"
        out.mkdir()
        write_condition_labels(
            labels_path(out, f"Scan{number:03d}", "job"), "job", labels
        )
    return folder


LABELS = {"APL": ["in", "out"]}


@pytest.fixture
def scans(tmp_path):
    def frame(apl, q):
        return pd.DataFrame(
            {"Shotnumber": range(1, len(apl) + 1), "APL": apl, "charge": q}
        )

    return {
        "in": make_scan(tmp_path, 1, frame([0, 0], [10.0, 12.0]), LABELS),
        "out": make_scan(tmp_path, 2, frame([1, 1], [30.0, 30.0]), LABELS),
        "mixed": make_scan(tmp_path, 3, frame([0, 1], [5.0, 7.0]), LABELS),
        "nolabels": make_scan(tmp_path, 4, frame([0, 0], [1.0, 1.0])),
        "nosfile": make_scan(tmp_path, 5, None),
    }


def test_summary(scans):
    s = summarize_scan(scans["mixed"], ["charge", "absent"])
    assert s.shots == 2 and "charge" in s.columns
    assert s.condition("APL").mixed
    assert s.values["charge"].center == pytest.approx(6.0)
    assert "absent" not in s.values
    assert summarize_scan(scans["nosfile"]) is None


@pytest.mark.parametrize(
    ("scan", "rule", "verdict"),
    [
        ("in", {"op": "is", "label": "in"}, "pass"),
        ("out", {"op": "is", "label": "in"}, "fail"),
        ("mixed", {"op": "is", "label": "in"}, "mixed"),
        ("in", {"op": "is_not", "label": "in"}, "fail"),
        ("out", {"op": "is_not", "label": "in"}, "pass"),
        ("mixed", {"op": "is_not", "label": "in"}, "mixed"),
        ("nolabels", {"op": "is", "label": "in"}, "missing"),
        ("nosfile", {"op": "is", "label": "in"}, "missing"),
    ],
)
def test_condition_rules(scans, scan, rule, verdict):
    s = summarize_scan(scans[scan])
    assert rule_verdict(s, ScanRule(column="APL", **rule)) == verdict


def test_number_rules(scans):
    rule = ScanRule(column="charge", op="between", low=10, high=20)
    assert rule_verdict(summarize_scan(scans["in"], ["charge"]), rule) == "pass"
    assert rule_verdict(summarize_scan(scans["out"], ["charge"]), rule) == "fail"
    one_sided = ScanRule(column="charge", op="between", low=25)
    assert rule_verdict(summarize_scan(scans["out"], ["charge"]), one_sided) == "pass"
    absent = ScanRule(column="nope", op="between", low=0)
    assert rule_verdict(summarize_scan(scans["in"], ["nope"]), absent) == "missing"


def test_all_rules_together(scans):
    rules = [
        ScanRule(column="APL", op="is", label="in"),
        ScanRule(column="charge", op="between", high=8),
    ]
    assert scan_verdict(summarize_scan(scans["mixed"], ["charge"]), rules) == "mixed"
    assert scan_verdict(summarize_scan(scans["in"], ["charge"]), rules) == "fail"
    assert scan_verdict(summarize_scan(scans["in"], ["charge"]), []) == "pass"


def test_incomplete_rules_are_refused():
    with pytest.raises(ValidationError, match="needs a label"):
        ScanRule(column="APL", op="is")
    with pytest.raises(ValidationError, match="needs a bound"):
        ScanRule(column="charge", op="between")
    with pytest.raises(ValidationError, match="low above high"):
        ScanRule(column="charge", op="between", low=2, high=1)
