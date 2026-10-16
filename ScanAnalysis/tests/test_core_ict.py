"""ICT on the core route: the golden (legacy) products, and the legacy charges to a stated tolerance."""

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
from scan_analysis.config import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer, core_supports
from scan_analysis.route_compare import compare_snapshots, snapshot_analysis_tree

TAG = ScanTag(year=2026, month=1, day=1, number=1, experiment="Test")
SHOTS = 6
DT = 4e-9
SCALARS = ("U_BCaveICT_charge_pC", "U_BCaveICT_ICT Signal Peak_us")


def document(dt: float | None = DT) -> AnalysisDiagnostic:
    analyzer = {"kind": "ict", "calibration_factor": 0.2}
    if dt is not None:
        analyzer["dt"] = dt
    return AnalysisDiagnostic.model_validate(
        {
            "name": "U_BCaveICT",
            "analyzer": analyzer,
            "image": {
                "type": "line",
                "data_loading": {"data_type": "npy"},
                "storage_dtype": "float32",
                "pipeline": [],
            },
            "scan": {"mode": "per_shot", "file_tail": ".npy", "renderer": {"dpi": 30}},
        }
    )


def build_scan(base_dir: Path) -> Path:
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    device = scan / "U_BCaveICT"
    device.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan001.ini").write_text(
        '[Scan Info]\nScan No = "1"\nScan Parameter = "U_Motor:Position"\n'
        'Start = "1"\nEnd = "3"\nStep size = "1"\nShots per step = "2"\n'
    )
    rng = np.random.default_rng(5)
    i = np.arange(2000)
    for shot in range(1, SHOTS + 1):
        at = 400 + 150 * shot
        volts = (
            -0.04 * (1 + shot / 5) * np.exp(-(((i - at) / 20.0) ** 2))
            + 0.003 * np.sin(2 * np.pi * i / 250)
            + rng.normal(0, 0.001, i.size)
        )
        np.save(
            device / f"Scan001_U_BCaveICT_{shot:03d}.npy",
            np.column_stack([i * DT, volts]),
        )
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    pd.DataFrame(
        {
            "Shotnumber": range(1, SHOTS + 1),
            "Bin #": [1 + k // 2 for k in range(SHOTS)],
            "U_Motor Position Alias:motor": [1.0 + k // 2 for k in range(SHOTS)],
        }
    ).to_csv(analysis / "s1.txt", sep="\t", index=False)
    return scan


def run(monkeypatch, base_dir: Path, doc, route: str) -> Path:
    scan = build_scan(base_dir)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    if route == "legacy":
        analyzer = create_scan_analyzer(doc, id="ICT", priority=1, route="legacy")
        assert isinstance(analyzer, SingleDeviceScanAnalyzer)
    else:
        analyzer = CoreScanAnalyzer(doc, id="ICT", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    return scan


def test_an_ict_diagnostic_runs_on_the_core_route():
    assert core_supports(document())


@pytest.mark.parametrize("dt", [DT, None], ids=["dt", "no-dt"])
def test_the_core_route_writes_the_golden_tree(
    tmp_path, monkeypatch, golden, request, dt
):
    """The legacy wrapper's files, its products at the declared float32, exact."""
    core = run(monkeypatch, tmp_path / "core", document(dt), "core")
    new = snapshot_analysis_tree(core.parent.parent / "analysis")
    expected = golden(f"ict/{request.node.callspec.id}", new)
    assert sorted(new) == sorted(expected.files)
    assert any(name.endswith(".h5") for name in new)
    assert compare_snapshots(expected.files, new, average_ulps=0) == []
    rows = new["s1.txt"][1]
    assert (rows["U_BCaveICT_charge_pC"] > 0).all()


@pytest.mark.parametrize("dt", [DT, None], ids=["dt", "no-dt"])
def test_the_legacy_ict_wrapper_wrote_the_golden_tree(
    tmp_path, monkeypatch, golden, request, dt
):
    """The oracle, to its stated tolerances."""
    legacy = run(monkeypatch, tmp_path / "legacy", document(dt), "legacy")
    old = snapshot_analysis_tree(legacy.parent.parent / "analysis")
    expected = golden(f"ict/{request.node.callspec.id}")
    # The same files. Legacy ICT saved its raw input trace at float64,
    # ignoring the config's storage_dtype; the golden holds the processed
    # trace at the declared float32 — equal to float32 precision.
    products = {name for name in old if name.endswith(".h5")}
    assert products
    assert sorted(old) == sorted(expected.files)
    for name in products:
        _, key, legacy_data, legacy_dtype = old[name]
        _, key2, golden_data, golden_dtype = expected.files[name]
        assert (key, legacy_dtype, golden_dtype) == (key2, "float64", "float32"), name
        np.testing.assert_allclose(golden_data, legacy_data, rtol=1e-6, atol=1e-9)
    # The charge is filtered at float64 on the core (the legacy low-pass ran
    # in float32): a stated tolerance, not equality.
    rows_old, rows_golden = old["s1.txt"][1], expected.files["s1.txt"][1]
    for column in SCALARS:
        np.testing.assert_allclose(rows_golden[column], rows_old[column], rtol=1e-6)
