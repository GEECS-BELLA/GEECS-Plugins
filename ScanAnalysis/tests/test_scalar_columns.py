"""The ``scalar_columns`` job: formulas over s-file columns, merged back into the s-file."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import ScalarJob

from scan_analysis import base
from scan_analysis.analyzers.common.scalar_columns import ScalarColumnsAnalyzer
from scan_analysis.base import DataUnavailableWarning
from scan_analysis.config.diagnostic_factory import create_scan_analyzer

TAG = ScanTag(year=2026, month=3, day=10, number=12, experiment="Test")


def job(*columns) -> ScalarJob:
    return ScalarJob.model_validate(
        {"kind": "scalar_columns", "columns": list(columns)}
    )


MM = {"name": "Cam x (mm)", "expr": "x * 0.5", "inputs": {"x": "Cam centroid x"}}
RATIO = {"name": "ratio", "expr": "a * b / c", "inputs": {"a": "A", "b": "B", "c": "C"}}


def build_scan(base_dir: Path) -> tuple[Path, Path]:
    """A completed noscan with its raw ScanData table and its s-file."""
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    scan.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan012.ini").write_text(
        '[Scan Info]\nScan No = "12"\nScan Parameter = "noscan"\n'
    )
    rows = pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3, 4],
            "Bin #": [1, 1, 1, 1],
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [2.0, 2.0, 2.0, 2.0],
            "C": [1.0, 0.0, 4.0, 8.0],
        }
    )
    rows.to_csv(scan / "ScanDataScan012.txt", sep="\t", index=False)
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    sfile = analysis / "s12.txt"
    rows.assign(**{"Cam centroid x": [10, 20, "nan", 40]}).to_csv(
        sfile, sep="\t", index=False
    )
    return scan, sfile


def run(monkeypatch, base_dir: Path, doc: ScalarJob):
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    analyzer = create_scan_analyzer(doc, id="Derived")
    try:
        return analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()


def test_the_factory_routes_the_job_to_its_runner():
    analyzer = create_scan_analyzer(job(MM), id="Derived")
    assert isinstance(analyzer, ScalarColumnsAnalyzer)
    assert analyzer.id == "Derived"
    assert analyzer.priority == 1000


def test_columns_are_computed_and_merged_into_the_sfile(tmp_path, monkeypatch):
    scan, sfile = build_scan(tmp_path)
    before = set(p.relative_to(tmp_path) for p in scan.rglob("*"))
    labels = run(monkeypatch, tmp_path, job(MM, RATIO))
    out = pd.read_csv(sfile, sep="\t")
    np.testing.assert_allclose(out["Cam x (mm)"], [5, 10, np.nan, 20])
    # 2*2/0 is not finite: a blank cell, never a failed run
    np.testing.assert_allclose(out["ratio"], [2, np.nan, 1.5, 1])
    assert list(out["A"]) == [1, 2, 3, 4]  # every other column kept
    assert "wrote 2 column(s) to s12.txt" in labels[0]
    assert "'ratio' (3 of 4 shots)" in labels[0]
    # nothing new in the scans tree; the sidecar lives under analysis/
    assert set(p.relative_to(tmp_path) for p in scan.rglob("*")) == before
    assert list((sfile.parent / "Scan012").glob("*Derived*"))


def test_a_later_column_reads_an_earlier_one(tmp_path, monkeypatch):
    _, sfile = build_scan(tmp_path)
    twice = {"name": "twice", "expr": "m * 2", "inputs": {"m": "Cam x (mm)"}}
    run(monkeypatch, tmp_path, job(MM, twice))
    np.testing.assert_allclose(
        pd.read_csv(sfile, sep="\t")["twice"], [10, 20, np.nan, 40]
    )


def test_rerunning_replaces_its_own_columns(tmp_path, monkeypatch):
    _, sfile = build_scan(tmp_path)
    run(monkeypatch, tmp_path, job(MM))
    run(monkeypatch, tmp_path, job({**MM, "expr": "x * 2"}))
    out = pd.read_csv(sfile, sep="\t")
    assert list(out.columns).count("Cam x (mm)") == 1
    assert out["Cam x (mm)"].iloc[0] == 20


def test_a_missing_input_is_no_data_with_a_hint(tmp_path, monkeypatch):
    _, sfile = build_scan(tmp_path)
    before = sfile.read_bytes()
    typo = {"name": "y", "expr": "x", "inputs": {"x": "Cam centroid  x"}}
    with pytest.raises(DataUnavailableWarning, match="did you mean 'Cam centroid x'"):
        run(monkeypatch, tmp_path, job(MM, typo))
    assert sfile.read_bytes() == before  # nothing written


def test_a_scanner_column_is_never_overwritten(tmp_path, monkeypatch):
    _, sfile = build_scan(tmp_path)
    before = sfile.read_bytes()
    clash = {"name": "A", "expr": "a * 2", "inputs": {"a": "A"}}
    with pytest.raises(ValueError, match="scanner's own columns"):
        run(monkeypatch, tmp_path, job(clash))
    assert sfile.read_bytes() == before


def test_a_constant_formula_fills_every_shot():
    table = pd.DataFrame({"Shotnumber": [1, 2]})
    out = ScalarColumnsAnalyzer(job({"name": "k", "expr": "2 * pi"})).compute(table)
    np.testing.assert_allclose(out["k"], [2 * np.pi] * 2)


def compute(*columns, **data):
    table = pd.DataFrame(
        {"Shotnumber": range(1, len(next(iter(data.values()))) + 1), **data}
    )
    return ScalarColumnsAnalyzer(job(*columns)).compute(table)


def test_comparisons_and_logic_are_element_wise():
    out = compute(
        {"name": "in", "expr": "4 <= x <= 6", "inputs": {"x": "X"}},
        {
            "name": "both",
            "expr": "x > 1 and not y == 0",
            "inputs": {"x": "X", "y": "Y"},
        },
        {"name": "any", "expr": "x > 5 or y > 5", "inputs": {"x": "X", "y": "Y"}},
        X=[3.0, 4.0, 6.0, 7.0, np.nan],
        Y=[0.0, 1.0, 9.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(out["in"], [0, 1, 1, 0, np.nan])
    np.testing.assert_array_equal(out["both"], [0, 1, 1, 0, np.nan])
    # a missing input is unknown, even where the other side would decide it
    np.testing.assert_array_equal(out["any"], [0, 0, 1, 1, np.nan])


TARGET = {
    "name": "target",
    "inputs": {"x": "Stage"},
    "cases": [
        {"when": "4 <= x <= 6", "label": "Al foil"},
        {"when": "5 <= x <= 12", "label": "Cu foil"},
    ],
    "otherwise": "nothing",
}


def test_cases_take_the_first_rule_that_holds():
    out = compute(TARGET, Stage=[5.0, 8.0, 20.0, np.nan, "bad"])
    # 5 matches both rules: the first wins
    np.testing.assert_array_equal(out["target"], [0, 1, 2, np.nan, np.nan])


def test_a_condition_writes_its_labels_beside_the_sidecar(tmp_path, monkeypatch):
    from geecs_data_utils.scan_conditions import read_condition_labels

    scan, sfile = build_scan(tmp_path)
    cond = {**TARGET, "inputs": {"x": "A"}}
    run(monkeypatch, tmp_path, job(MM, cond))
    np.testing.assert_array_equal(pd.read_csv(sfile, sep="\t")["target"], [2, 2, 2, 0])
    analysis = sfile.parent / "Scan012"
    assert (analysis / "Scan012_Derived_labels.json").is_file()
    assert read_condition_labels(analysis) == {
        "target": ["Al foil", "Cu foil", "nothing"]
    }


def test_a_job_without_conditions_writes_no_labels(tmp_path, monkeypatch):
    _, sfile = build_scan(tmp_path)
    run(monkeypatch, tmp_path, job(MM))
    assert not list((sfile.parent / "Scan012").glob("*_labels.json"))


def test_the_task_queue_runs_it_under_its_own_data_root(tmp_path):
    """A host's base_directory reaches the analyzer: the scan read is the one
    whose status record is written (a worker over another root once read and
    wrote the configured share instead)."""
    from scan_analysis.task_queue import read_statuses, run_task

    scan, sfile = build_scan(tmp_path)
    analyzer = create_scan_analyzer(job(MM), id="Derived")
    try:
        state = run_task(1000, TAG, analyzer, base_directory=tmp_path)
    finally:
        analyzer.cleanup()
    assert state == "done"
    np.testing.assert_allclose(
        pd.read_csv(sfile, sep="\t")["Cam x (mm)"], [5, 10, np.nan, 20]
    )
    assert [s.state for s in read_statuses(scan)] == ["done"]


def test_column_functions_reduce_over_the_scan():
    out = compute(
        {"name": "centred", "expr": "x - mean(x)", "inputs": {"x": "X"}},
        {
            "name": "scaled",
            "expr": "(x - min(x)) / (max(x) - min(x))",
            "inputs": {"x": "X"},
        },
        {"name": "above", "expr": "x > median(x)", "inputs": {"x": "X"}},
        {"name": "spread", "expr": "std(x) + 0 * x", "inputs": {"x": "X"}},
        X=[1.0, 2.0, 3.0, np.nan, 6.0],
    )
    np.testing.assert_allclose(out["centred"], [-2, -1, 0, np.nan, 3])
    np.testing.assert_allclose(out["scaled"], [0, 0.2, 0.4, np.nan, 1])
    np.testing.assert_array_equal(out["above"], [0, 0, 1, np.nan, 1])
    np.testing.assert_allclose(out["spread"][0], np.std([1, 2, 3, 6], ddof=1))


def test_a_column_with_no_values_reduces_to_blank():
    out = compute(
        {"name": "c", "expr": "x - mean(x)", "inputs": {"x": "X"}}, X=[np.nan, np.nan]
    )
    assert out["c"].isna().all()
