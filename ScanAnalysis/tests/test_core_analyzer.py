"""The core route reproduces the legacy wrappers' products, scalars and contract."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisDiagnostic

import scan_analysis.base as base
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)
from scan_analysis.base import DataUnavailableWarning
from scan_analysis.config import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer, core_supports
from geecs_analysis.compat.convert import to_v3
from geecs_analysis.recipe import is_line
from scan_analysis.route_compare import compare_snapshots, snapshot_analysis_tree

TAG = ScanTag(year=2026, month=1, day=1, number=1, experiment="Test")
PARAM_COLUMN = "U_Motor Position Alias:motor"
SORT_COLUMN = "U_Charge Value"
SHOTS = 6


def document(kind="beam", *, mode="per_shot", renderer=None, **scan):
    line = kind in {"line", "trace"}
    image = (
        {
            "type": "line",
            "data_loading": {"data_type": "npy"},
            "storage_dtype": "float32",
        }
        if line
        else {"type": "camera", "pipeline": []}
    )
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Spec" if line else "Camera",
            "output_name": "Diag",
            "analyzer": {"kind": kind},
            "image": image,
            "scan": {
                "mode": mode,
                "file_tail": ".npy",
                "renderer": renderer or {"dpi": 30},
                **scan,
            },
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
    monkeypatch, tmp_path, route, doc, *, noscan, device_files=True, single_bin=False
):
    base_dir = tmp_path / route
    line = is_line(doc)
    scan = build_scan(
        base_dir,
        line=line,
        noscan=noscan,
        device_files=device_files,
        single_bin=single_bin,
    )
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    if route == "legacy":
        # The oracle is the legacy wrapper, forced explicitly: without
        # route="legacy" the factory would hand back the core for every
        # supported recipe and this comparison would test core against core.
        analyzer = create_scan_analyzer(doc, id="Diag", priority=1, route="legacy")
        assert isinstance(analyzer, SingleDeviceScanAnalyzer)
    else:
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
        {"waterfall_sort_key": "U_Charge", "dpi": 30},
        id="line-noscan-sorted",
    ),
]


@pytest.mark.parametrize("kind,mode,noscan,renderer", CASES)
def test_core_route_matches_legacy_outputs(
    tmp_path, monkeypatch, kind, mode, noscan, renderer
):
    doc = document(kind, mode=mode, renderer=renderer)
    legacy_scan, legacy_display, _ = run(
        monkeypatch, tmp_path, "legacy", doc, noscan=noscan
    )
    core_scan, core_display, plan = run(
        monkeypatch, tmp_path, "core", doc, noscan=noscan
    )
    legacy, core = snapshot(legacy_scan), snapshot(core_scan)
    assert sorted(core) == sorted(legacy)
    assert any(name.endswith(".png") for name in core)
    if not noscan:
        # Figures name the scan by the cleaned ScanInfo string, as legacy did.
        assert plan.position_label == "U_Motor Position"
    assert any(name.endswith(".h5") for name in core)
    # Exact everywhere except noscan averages: the legacy wrapper sums shots
    # in directory-listing order, the core in scalar-row order, so those
    # differ by summation rounding only (see compare_snapshots).
    assert compare_snapshots(legacy, core, average_ulps=4) == []
    assert relative_display(core_scan, core_display) == relative_display(
        legacy_scan, legacy_display
    )
    assert core_display


def test_single_bin_scan_keeps_the_cleaned_parameter_label(tmp_path, monkeypatch):
    _, _, plan = run(
        monkeypatch, tmp_path, "core", document(), noscan=False, single_bin=True
    )
    assert [p.identifier for p in plan.singles] == [1] and not plan.summary
    assert plan.position_label == "U_Motor Position"


def test_scalars_persist_without_products_when_save_is_off(tmp_path, monkeypatch):
    doc = document(save=False)
    legacy_scan, legacy_display, _ = run(
        monkeypatch, tmp_path, "legacy", doc, noscan=False
    )
    core_scan, core_display, _ = run(monkeypatch, tmp_path, "core", doc, noscan=False)
    legacy, core = snapshot(legacy_scan), snapshot(core_scan)
    assert sorted(core) == sorted(legacy) == ["Scan001/Scan001_Diag.txt", "s1.txt"]
    assert compare_snapshots(legacy, core) == []
    assert "Diag_x_CoM" in core["s1.txt"][1].columns
    assert core_display == legacy_display == []


@pytest.mark.parametrize("route", ["legacy", "core"])
def test_missing_device_folder_is_no_data(tmp_path, monkeypatch, route):
    with pytest.raises(DataUnavailableWarning):
        run(monkeypatch, tmp_path, route, document(), noscan=False, device_files=False)
    assert not list((tmp_path / route).rglob("*.h5"))


@pytest.mark.parametrize("route", ["legacy", "core"])
def test_missing_sfile_returns_none(tmp_path, monkeypatch, route):
    base_dir = tmp_path / route
    scan = build_scan(base_dir, line=False, noscan=False)
    (scan.parent.parent / "analysis" / "s1.txt").unlink()
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    doc = document()
    analyzer = (
        create_scan_analyzer(doc, id="Diag", priority=1, route="legacy")
        if route == "legacy"
        else CoreScanAnalyzer(doc, id="Diag", priority=1)
    )
    assert isinstance(analyzer, SingleDeviceScanAnalyzer) is (route == "legacy")
    assert analyzer.run_analysis(TAG) is None
    assert not list((tmp_path / route).rglob("*.h5"))


# Sort values by shot: 30, 10, 50, 20, 60, 40 (mean 35, population std 17.08).
SORT_CASES = [
    pytest.param({}, [2, 4, 1, 6, 3, 5], id="sigma-default-3-keeps-all"),
    pytest.param(
        {"waterfall_sort_sigma": 1.0}, [4, 1, 6, 3], id="sigma-1-drops-10-and-60"
    ),
    pytest.param(
        {"waterfall_sort_sigma": 1.0, "waterfall_sort_bounds": [15, 45]},
        [4, 1, 6],
        id="bounds-override-sigma",
    ),
]


@pytest.mark.parametrize("options,expected_shots", SORT_CASES)
def test_sorted_waterfall_rows_follow_legacy_sigma_and_bounds_rules(
    tmp_path, monkeypatch, options, expected_shots
):
    doc = document(
        "line", renderer={"waterfall_sort_key": "U_Charge", "dpi": 30, **options}
    )
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
    doc = document("line", renderer={"waterfall_sort_key": "Diag_CoM", "dpi": 30})
    scan, _, plan = run(monkeypatch, tmp_path, "core", doc, noscan=True)
    assert plan.position_label == "Diag_CoM"
    positions = [p.position for p in plan.summary]
    assert len(positions) == SHOTS and positions == sorted(positions)
    sidecar = scan.parent.parent / "analysis" / "Scan001" / "Scan001_Diag.txt"
    assert "Diag_CoM" in pd.read_csv(sidecar, sep="\t").columns
    sfile = pd.read_csv(scan.parent.parent / "analysis" / "s1.txt", sep="\t")
    assert ("Diag_CoM" in sfile.columns) is not merge_refused


def test_contract_attributes_and_cleanup():
    analyzer = CoreScanAnalyzer(document(), id="MyDiag", priority=7)
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


def test_core_supports_only_recipes_the_core_can_run():
    assert core_supports(document())
    assert core_supports(document("line"))
    assert not core_supports(document(background_source={"scan_number": 5}))
    assert not core_supports(
        AnalysisDiagnostic.model_validate(
            {
                "name": "ICT",
                "analyzer": {"kind": "ict"},
                "image": {"type": "line", "data_loading": {"data_type": "npy"}},
            }
        )
    )


@pytest.mark.parametrize("kind,mode,noscan,renderer", CASES)
def test_converted_recipe_matches_its_v2_source_on_the_core(
    tmp_path, monkeypatch, kind, mode, noscan, renderer
):
    """A v3 recipe converted from a v2 diagnostic writes the identical tree."""
    doc = document(kind, mode=mode, renderer=renderer)
    v2_scan, v2_display, _ = run(monkeypatch, tmp_path, "v2", doc, noscan=noscan)
    v3_scan, v3_display, _ = run(
        monkeypatch, tmp_path, "v3", to_v3(doc).recipe, noscan=noscan
    )
    v2, v3 = snapshot(v2_scan), snapshot(v3_scan)
    assert sorted(v2) == sorted(v3)
    # One evaluator, one shot order: exact, the noscan average included.
    assert compare_snapshots(v2, v3, average_ulps=0) == []
    assert relative_display(v2_scan, v2_display) == relative_display(
        v3_scan, v3_display
    )
