"""Downramp phase on the core route: the golden tree, no PDF in the scan folder."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisRecipe

import scan_analysis.base as base
from scan_analysis.config import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer
from scan_analysis.route_compare import (
    GOLDEN_RTOL,
    compare_snapshots,
    snapshot_analysis_tree,
)

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


def document(mode: str) -> AnalysisRecipe:
    return AnalysisRecipe.model_validate(
        {
            "device": DEVICE,
            "output_name": "Diag",
            "input": {"kind": "camera", "file_tail": ".npy"},
            "measure": {"kind": "downramp_phase"},
            "scan": {"average_frames_first": mode == "per_bin"},
            "figure": {"imshow": {"vmin": 0}, "fig": {"dpi": 30}},
            "summaries": [
                {"kind": "image_grid", "panel_size": [6.0, 6.0]},
                {"kind": "average"},
            ],
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


def run(monkeypatch, base_dir: Path, doc) -> Path:
    scan = build_scan(base_dir)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    analyzer = CoreScanAnalyzer(doc, id="Diag", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    return scan


def test_the_diagnostic_builds_on_the_core():
    assert isinstance(create_scan_analyzer(document("per_shot")), CoreScanAnalyzer)


@pytest.mark.parametrize("mode", ["per_shot", "per_bin"])
def test_the_core_route_writes_the_golden_tree(tmp_path, monkeypatch, golden, mode):
    """The retired legacy wrapper's tree (its phase product as the core's float64), exact."""
    core = run(monkeypatch, tmp_path / "core", document(mode))
    new = snapshot_analysis_tree(core.parent.parent / "analysis")
    expected = golden(f"downramp_phase/{mode}", new)
    assert sorted(new) == sorted(expected.files)
    assert any(name.endswith(".h5") for name in new)
    assert (
        compare_snapshots(expected.files, new, average_ulps=0, rtol=GOLDEN_RTOL) == []
    )
    rows = pd.read_csv(
        core.parent.parent / "analysis" / "s1.txt",
        sep="\t",
        float_precision="round_trip",
    )
    for key in SCALARS:
        assert rows[f"Diag_{key}"].notna().all(), key
    location = rows["Diag_Plasma downramp shock location (pixel)"]
    shock = [SHOCK_COLUMNS[(shot - 1) // 2] for shot in rows["Shotnumber"]]
    np.testing.assert_allclose(location, shock, atol=3)
    # The legacy analyzer drew its PDF into the raw device folder (per
    # shot, where it had a file path); the core route writes nothing there.
    assert not list((core / DEVICE).glob("*.pdf"))
