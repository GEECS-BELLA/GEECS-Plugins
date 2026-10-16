"""ICT on the core route: the golden products and charges."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisDiagnostic

import scan_analysis.base as base
from scan_analysis.config import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer
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


def run(monkeypatch, base_dir: Path, doc) -> Path:
    scan = build_scan(base_dir)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    analyzer = CoreScanAnalyzer(doc, id="ICT", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    return scan


def test_the_diagnostic_builds_on_the_core():
    assert isinstance(create_scan_analyzer(document()), CoreScanAnalyzer)


@pytest.mark.parametrize("dt", [DT, None], ids=["dt", "no-dt"])
def test_the_core_route_writes_the_golden_tree(
    tmp_path, monkeypatch, golden, request, dt
):
    """The retired legacy wrapper's files and charges, the products at the declared float32, exact.

    The wrapper saved its raw input trace at float64, ignoring the config's
    ``storage_dtype``, and low-passed the charge in float32; the core stores
    the processed trace at the declared float32 and filters at float64 — the
    golden holds the core's, matched to the wrapper's at 1e-6 relative when
    the two were compared.
    """
    core = run(monkeypatch, tmp_path / "core", document(dt))
    new = snapshot_analysis_tree(core.parent.parent / "analysis")
    expected = golden(f"ict/{request.node.callspec.id}", new)
    assert sorted(new) == sorted(expected.files)
    assert any(name.endswith(".h5") for name in new)
    assert compare_snapshots(expected.files, new, average_ulps=0) == []
    rows = new["s1.txt"][1]
    assert (rows["U_BCaveICT_charge_pC"] > 0).all()
