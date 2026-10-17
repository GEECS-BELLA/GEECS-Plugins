"""The core route writes the golden trees (the retired wrappers' products, pinned) and keeps the contract."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisRecipe

import scan_analysis.base as base
from scan_analysis.base import DataUnavailableWarning
from scan_analysis import core_workers
from scan_analysis.core_analyzer import CoreScanAnalyzer, write_scalars_into_rows
from geecs_analysis.recipe import is_line
from scan_analysis.route_compare import (
    GOLDEN_RTOL,
    compare_snapshots,
    snapshot_analysis_tree,
)

TAG = ScanTag(year=2026, month=1, day=1, number=1, experiment="Test")
PARAM_COLUMN = "U_Motor Position Alias:motor"
SORT_COLUMN = "U_Charge Value"
SHOTS = 6


def recipe(kind="beam", *, mode="per_shot", sort=None, **scan) -> AnalysisRecipe:
    """The v3 recipe: ``beam``/``line`` measure; ``standard``/``trace`` none."""
    line = kind in {"line", "trace"}
    if line:
        source = {"kind": "line", "loading": {"data_type": "npy"}, "label": "x vs y"}
        figure, first = {"fig": {"dpi": 30}}, {"kind": "waterfall"}
    else:
        source = {"kind": "camera"}
        figure = {"imshow": {"vmin": 0}, "fig": {"dpi": 30}}
        first = {"kind": "image_grid", "panel_size": [6.0, 6.0]}
    if mode == "per_bin":
        scan["average_frames_first"] = True
    return AnalysisRecipe.model_validate(
        {
            "device": "Spec" if line else "Camera",
            "output_name": "Diag",
            "input": {**source, "file_tail": ".npy"},
            "measure": {"kind": kind if kind in {"beam", "line"} else "none"},
            "scan": scan,
            "figure": figure,
            "summaries": [{**first, **(sort or {})}, {"kind": "average"}],
        }
    )


def build_scan(
    base_dir: Path, *, line: bool, noscan: bool, device_files=True, single_bin=False
):
    """A completed scan in the GEECS layout, with its s-file and ScanInfo."""
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    device = scan / ("Spec" if line else "Camera")
    (device if device_files else scan).mkdir(parents=True)  # fixture acquisition
    parameter = "noscan" if noscan else "U_Motor:Position"
    (scan / "ScanInfoScan001.ini").write_text(
        "[Scan Info]\n"
        'Scan No = "1"\n'
        f'Scan Parameter = "{parameter}"\n'
        'Start = "1"\nEnd = "3"\nStep size = "1"\nShots per step = "2"\n'
    )
    if device_files:
        for shot in range(1, SHOTS + 1):
            if line:
                x = np.linspace(0.0, 10.0, 50)
                y = (1 + shot / 10) * np.exp(-((x - 2 - 0.8 * shot) ** 2) / 2)
                np.save(
                    device / f"Scan001_Spec_{shot:03d}.npy", np.column_stack([x, y])
                )
            else:
                yy, xx = np.mgrid[:24, :24]
                image = 200 * np.exp(-((xx - 6 - shot) ** 2 + (yy - 12) ** 2) / 8) + 5
                np.save(device / f"Scan001_Camera_{shot:03d}.npy", image)
    rows = pd.DataFrame(
        {
            "Shotnumber": range(1, SHOTS + 1),
            "Bin #": [1] * SHOTS
            if noscan or single_bin
            else [1 + i // 2 for i in range(SHOTS)],
            PARAM_COLUMN: [1.0 + i // 2 + 0.05 * (i % 2) for i in range(SHOTS)],
            SORT_COLUMN: [30.0, 10.0, 50.0, 20.0, 60.0, 40.0],
        }
    )
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    rows.to_csv(analysis / "s1.txt", sep="\t", index=False)
    return scan


def run(
    monkeypatch, tmp_path, label, doc, *, noscan, device_files=True, single_bin=False
):
    """One core run of ``doc`` over a fresh scan under ``tmp_path / label``."""
    base_dir = tmp_path / label
    line = is_line(doc)
    scan = build_scan(
        base_dir,
        line=line,
        noscan=noscan,
        device_files=device_files,
        single_bin=single_bin,
    )
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    analyzer = CoreScanAnalyzer(doc, id="Diag", priority=1)
    raw_before = sorted(p.relative_to(scan) for p in scan.rglob("*"))
    try:
        display = analyzer.run_analysis(TAG)
    finally:
        plan = getattr(analyzer, "last_plan", None)
        analyzer.cleanup()
    assert sorted(p.relative_to(scan) for p in scan.rglob("*")) == raw_before
    return scan, display, plan


def snapshot(scan: Path) -> dict:
    """Every analysis output, decoded by the shared route-comparison rules."""
    return snapshot_analysis_tree(scan.parent.parent / "analysis")


def relative_display(scan: Path, display) -> list[str]:
    analysis = scan.parent.parent / "analysis"
    return sorted(Path(p).relative_to(analysis).as_posix() for p in display)


CASES = [
    pytest.param("beam", "per_shot", False, None, id="beam-per_shot-scan"),
    pytest.param("beam", "per_bin", False, None, id="beam-per_bin-scan"),
    pytest.param("beam", "per_shot", True, None, id="beam-noscan"),
    pytest.param("line", "per_shot", False, None, id="line-per_shot-scan"),
    pytest.param("standard", "per_shot", False, None, id="standard-per_shot-scan"),
    pytest.param("trace", "per_shot", False, None, id="trace-per_shot-scan"),
    pytest.param(
        "line",
        "per_shot",
        True,
        {"sort_key": "U_Charge"},
        id="line-noscan-sorted",
    ),
]


@pytest.mark.parametrize("kind,mode,noscan,sort", CASES)
def test_the_core_route_writes_the_golden_tree(
    tmp_path, monkeypatch, golden, request, kind, mode, noscan, sort
):
    """Every file, HDF5 payload, s-file column, sidecar and display file: exact.

    The golden is the tree the retired legacy wrapper wrote for this case
    (captured from the core while the two matched, #1039; the wrapper's last
    state is tag ``legacy-analysis-final``); the core sums in scalar-row
    order, so against its golden even the noscan average is exact.
    """
    doc = recipe(kind, mode=mode, sort=sort)
    core_scan, core_display, plan = run(
        monkeypatch, tmp_path, "core", doc, noscan=noscan
    )
    core = snapshot(core_scan)
    display = relative_display(core_scan, core_display)
    expected = golden(f"core_route/{request.node.callspec.id}", core, display=display)
    assert sorted(core) == sorted(expected.files)
    assert any(name.endswith(".png") for name in core)
    if not noscan:
        # Figures name the scan by the cleaned ScanInfo string, as legacy did.
        assert plan.position_label == "U_Motor Position"
    assert any(name.endswith(".h5") for name in core)
    assert (
        compare_snapshots(expected.files, core, average_ulps=0, rtol=GOLDEN_RTOL) == []
    )
    assert display == expected.extra["display"]
    assert core_display


def test_single_bin_scan_keeps_the_cleaned_parameter_label(tmp_path, monkeypatch):
    _, _, plan = run(
        monkeypatch, tmp_path, "core", recipe(), noscan=False, single_bin=True
    )
    assert [p.identifier for p in plan.singles] == [1] and not plan.summary
    assert plan.position_label == "U_Motor Position"


def test_scalars_persist_without_products_when_save_is_off(
    tmp_path, monkeypatch, golden
):
    doc = recipe(save=False)
    core_scan, core_display, _ = run(monkeypatch, tmp_path, "core", doc, noscan=False)
    core = snapshot(core_scan)
    expected = golden("core_route/save-off", core)
    assert (
        sorted(core) == sorted(expected.files) == ["Scan001/Scan001_Diag.txt", "s1.txt"]
    )
    assert (
        compare_snapshots(expected.files, core, average_ulps=0, rtol=GOLDEN_RTOL) == []
    )
    assert "Diag_x_CoM" in core["s1.txt"][1].columns
    assert core_display == []


def test_missing_device_folder_is_no_data(tmp_path, monkeypatch):
    with pytest.raises(DataUnavailableWarning):
        run(monkeypatch, tmp_path, "core", recipe(), noscan=False, device_files=False)
    assert not list(tmp_path.rglob("*.h5"))


def test_missing_sfile_returns_none(tmp_path, monkeypatch):
    scan = build_scan(tmp_path, line=False, noscan=False)
    (scan.parent.parent / "analysis" / "s1.txt").unlink()
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    analyzer = CoreScanAnalyzer(recipe(), id="Diag", priority=1)
    assert analyzer.run_analysis(TAG) is None
    assert not list(tmp_path.rglob("*.h5"))


# Sort values by shot: 30, 10, 50, 20, 60, 40 (mean 35, population std 17.08).
SORT_CASES = [
    pytest.param({}, [2, 4, 1, 6, 3, 5], id="sigma-default-3-keeps-all"),
    pytest.param({"sort_sigma": 1.0}, [4, 1, 6, 3], id="sigma-1-drops-10-and-60"),
    pytest.param(
        {"sort_sigma": 1.0, "sort_bounds": [15, 45]},
        [4, 1, 6],
        id="bounds-override-sigma",
    ),
]


@pytest.mark.parametrize("options,expected_shots", SORT_CASES)
def test_sorted_waterfall_rows_follow_legacy_sigma_and_bounds_rules(
    tmp_path, monkeypatch, options, expected_shots
):
    doc = recipe("line", sort={"sort_key": "U_Charge", **options})
    _, display, plan = run(monkeypatch, tmp_path, "core", doc, noscan=True)
    assert [p.identifier for p in plan.summary] == expected_shots
    assert [p.position for p in plan.summary] == sorted(
        [30.0, 10.0, 50.0, 20.0, 60.0, 40.0][n - 1] for n in expected_shots
    )
    assert plan.position_label == SORT_COLUMN
    assert [p.identifier for p in plan.singles] == ["average"]
    assert len(display) == 1 and display[0].endswith("Spec_summary_waterfall.png")


@pytest.mark.parametrize("merge_refused", [False, True])
def test_sorting_by_an_own_output_column_survives_a_refused_merge(
    tmp_path, monkeypatch, merge_refused
):
    """Legacy wrote its scalars into the in-memory rows before rendering."""
    if merge_refused:
        monkeypatch.setattr(base, "merge_sfile", lambda *args, **kwargs: None)
    doc = recipe("line", sort={"sort_key": "Diag_CoM"})
    scan, _, plan = run(monkeypatch, tmp_path, "core", doc, noscan=True)
    assert plan.position_label == "Diag_CoM"
    positions = [p.position for p in plan.summary]
    assert len(positions) == SHOTS and positions == sorted(positions)
    sidecar = scan.parent.parent / "analysis" / "Scan001" / "Scan001_Diag.txt"
    assert "Diag_CoM" in pd.read_csv(sidecar, sep="\t").columns
    sfile = pd.read_csv(scan.parent.parent / "analysis" / "s1.txt", sep="\t")
    assert ("Diag_CoM" in sfile.columns) is not merge_refused


def test_contract_attributes_and_cleanup():
    analyzer = CoreScanAnalyzer(recipe(), id="MyDiag", priority=7)
    assert (analyzer.id, analyzer.priority, analyzer.device_name) == (
        "MyDiag",
        7,
        "Camera",
    )
    analyzer.auxiliary_data = pd.DataFrame({"Shotnumber": [1]})
    analyzer.display_contents = ["x"]
    analyzer.cleanup()
    assert analyzer.auxiliary_data is None and analyzer.display_contents == []
    assert analyzer.last_plan is None


@pytest.mark.parametrize("kind,mode,noscan,sort", CASES[:4])
def test_pooled_run_writes_the_identical_tree(
    tmp_path, monkeypatch, caplog, kind, mode, noscan, sort
):
    """workers > 1 is a throughput knob: every file equals the serial run's, exactly."""
    import logging

    monkeypatch.setattr(core_workers, "MIN_UNITS_FOR_POOL", 1)
    doc = recipe(kind, mode=mode, sort=sort)
    serial_scan, serial_display, _ = run(
        monkeypatch, tmp_path, "serial", doc, noscan=noscan
    )
    pooled_recipe = doc.model_copy(
        update={"scan": doc.scan.model_copy(update={"workers": 2})}
    )
    with caplog.at_level(logging.INFO, logger="scan_analysis.core_analyzer"):
        pooled_scan, pooled_display, _ = run(
            monkeypatch, tmp_path, "pooled", pooled_recipe, noscan=noscan
        )
    assert any(r.getMessage().endswith("units, 2 workers") for r in caplog.records)
    serial, pooled = snapshot(serial_scan), snapshot(pooled_scan)
    assert sorted(serial) == sorted(pooled)
    assert compare_snapshots(serial, pooled, average_ulps=0) == []
    assert relative_display(pooled_scan, pooled_display) == relative_display(
        serial_scan, serial_display
    )


def test_host_cap_bounds_the_recipe_request(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setattr(core_workers, "MIN_UNITS_FOR_POOL", 1)
    monkeypatch.setattr(CoreScanAnalyzer, "worker_cap", 1)
    doc = recipe(workers=8)
    with caplog.at_level(logging.INFO, logger="scan_analysis.core_analyzer"):
        run(monkeypatch, tmp_path, "capped", doc, noscan=True)
    assert any(
        r.getMessage().endswith(f"{SHOTS} units, 1 worker") for r in caplog.records
    )


def test_a_fold_error_raises_only_after_the_scalars_persist(tmp_path, monkeypatch):
    """A result the products cannot fold never costs the run its scalar columns."""
    from scan_analysis import core_analyzer

    def refuse(self, outcome):
        raise ValueError("Averaged results must have matching units")

    monkeypatch.setattr(core_analyzer.ProductCollector, "add", refuse)
    with pytest.raises(ValueError, match="matching units"):
        run(monkeypatch, tmp_path, "core", recipe(), noscan=False)
    scan = next(
        p for p in (tmp_path / "core").rglob("Scan001") if p.parent.name == "scans"
    )
    tree = snapshot(scan)
    assert "Scan001/Scan001_Diag.txt" in tree
    assert "Diag_x_CoM" in tree["s1.txt"][1].columns


def _cell_by_cell(rows, records):
    """The per-cell loop ``write_scalars_into_rows`` replaced: the reference."""
    for record in records:
        mask = rows["Shotnumber"] == record["Shotnumber"]
        for key, value in record.items():
            if key != "Shotnumber":
                rows.loc[mask, key] = value


@pytest.mark.parametrize(
    "records",
    [
        # overwrite an existing column on some shots; add a new one
        [{"Shotnumber": n, "old": 10.0 * n, "new": n + 0.5} for n in (2, 4, 5)],
        # a later record for the same shot wins
        [{"Shotnumber": 3, "new": 1.0}, {"Shotnumber": 3, "new": 2.0}],
        # a record lacking a key leaves that shot's existing cell alone
        [{"Shotnumber": 1, "old": 7.0, "new": 1.0}, {"Shotnumber": 2, "new": 2.0}],
        # NaN overwrites an existing value, as the loop did
        [{"Shotnumber": 4, "old": float("nan")}],
        # a shot absent from the s-file still creates the column
        [{"Shotnumber": 99, "only_missing": 3.0}],
        # an integer scalar into an existing float column and a new column
        [{"Shotnumber": n, "old": n, "count": n} for n in (1, 6)],
    ],
    ids=["overwrite+new", "last-wins", "missing-key", "nan", "no-match", "ints"],
)
def test_write_scalars_into_rows_matches_the_cell_loop(records):
    rows = pd.DataFrame(
        {"Shotnumber": [1, 2, 3, 4, 5, 6], "Bin #": 1, "old": np.arange(6.0)}
    )
    expected = rows.copy()
    _cell_by_cell(expected, records)
    write_scalars_into_rows(rows, records)
    pd.testing.assert_frame_equal(rows, expected)
