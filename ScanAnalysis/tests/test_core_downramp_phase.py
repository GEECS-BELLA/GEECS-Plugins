"""Downramp phase on the core route: the legacy wrapper's tree, its PDF no longer in the scan folder."""

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
DEVICE = "U_HasoLift"
#: The shock column moves with the bin, so the scalars have to follow it.
SHOCK_COLUMNS = (45, 60, 75)
SCALARS = (
    "Plasma downramp shock_angle",
    "Plasma downramp shock slope (phase/pixel)",
    "Plasma downramp shock location (pixel)",
    "Plasma downramp plateau avg (phase)",
    "Plasma downramp peak to plateau (phase)",
    "image_com_slope_x",
)


def document(mode: str) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": DEVICE,
            "output_name": "Diag",
            "analyzer": {"kind": "downramp_phase"},
            "image": {"type": "camera", "pipeline": []},
            "scan": {"mode": mode, "file_tail": ".npy", "renderer": {"dpi": 30}},
        }
    )


def phase_map(shot: int) -> np.ndarray:
    """A plasma channel (a phase dip) whose phase steps at the bin's shock column."""
    rng = np.random.default_rng(shot)
    y, x = np.mgrid[0:64, 0:96]
    edge = SHOCK_COLUMNS[(shot - 1) // 2] + 0.2 * (y - 32)
    along = 1.0 + 1.5 / (1 + np.exp(-(x - edge) / 2.0))
    return (
        3.0
        - along * np.exp(-0.5 * ((y - 32) / 8) ** 2)
        + 0.02 * rng.normal(size=x.shape)
    )


def build_scan(base_dir: Path) -> Path:
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    device = scan / DEVICE
    device.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan001.ini").write_text(
        '[Scan Info]\nScan No = "1"\nScan Parameter = "U_Motor:Position"\n'
        'Start = "1"\nEnd = "3"\nStep size = "1"\nShots per step = "2"\n'
    )
    for shot in range(1, SHOTS + 1):
        np.save(device / f"Scan001_{DEVICE}_{shot:03d}.npy", phase_map(shot))
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
        analyzer = create_scan_analyzer(doc, id="Diag", priority=1, route="legacy")
        assert isinstance(analyzer, SingleDeviceScanAnalyzer)
    else:
        analyzer = CoreScanAnalyzer(doc, id="Diag", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    return scan


def test_a_downramp_phase_diagnostic_runs_on_the_core_route():
    assert core_supports(document("per_shot"))


@pytest.mark.parametrize("mode", ["per_shot", "per_bin"])
def test_the_core_route_matches_the_legacy_wrapper(tmp_path, monkeypatch, mode):
    legacy = run(monkeypatch, tmp_path / "legacy", document(mode), "legacy")
    core = run(monkeypatch, tmp_path / "core", document(mode), "core")
    old = snapshot_analysis_tree(legacy.parent.parent / "analysis")
    new = snapshot_analysis_tree(core.parent.parent / "analysis")
    assert sorted(old) == sorted(new)
    products = [name for name in old if name.endswith(".h5")]
    assert products
    for name in products:
        _, key, legacy_data, legacy_dtype = old[name]
        _, key2, core_data, core_dtype = new[name]
        # The wrapper stored the analyzer's uint16 phase product; the core
        # stores its frame, the same values as float64.
        assert (key, core_dtype) == (key2, np.float64), name
        np.testing.assert_array_equal(core_data, legacy_data, err_msg=name)
    rest_old = {name: old[name] for name in old if not name.endswith(".h5")}
    rest_new = {name: new[name] for name in new if not name.endswith(".h5")}
    assert compare_snapshots(rest_old, rest_new) == []
    rows = pd.read_csv(
        core.parent.parent / "analysis" / "s1.txt",
        sep="\t",
        float_precision="round_trip",
    )
    for key in SCALARS:
        assert rows[f"Diag_{key}"].notna().all(), key
    location = rows["Diag_Plasma downramp shock location (pixel)"]
    expected = [SHOCK_COLUMNS[(shot - 1) // 2] for shot in rows["Shotnumber"]]
    np.testing.assert_allclose(location, expected, atol=3)
    # The wrapper's analyzer drew its PDF into the raw device folder (per
    # shot, where it had a file path); the core route writes nothing there.
    pdfs = list((legacy / DEVICE).glob("*_combined_shock_analysis.pdf"))
    assert bool(pdfs) == (mode == "per_shot")
    assert not list((core / DEVICE).glob("*.pdf"))
