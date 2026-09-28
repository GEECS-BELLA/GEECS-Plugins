"""HiResMagCam on the core route: the legacy wrapper's products and scalars, per shot and per bin."""

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
BOWTIE = [
    "Diag_emittance_proxy",
    "Diag_total_counts",
    "Diag_bowtie_x0",
    "Diag_bowtie_w0",
    "Diag_bowtie_theta",
    "Diag_bowtie_r_squared",
]


def document(mode: str) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": "UC_HiResMagCam",
            "output_name": "Diag",
            "analyzer": {"kind": "hi_res_mag_cam", "min_total_counts": 1500},
            "image": {"type": "camera", "pipeline": []},
            "scan": {"mode": mode, "file_tail": ".npy", "renderer": {"dpi": 30}},
        }
    )


def build_scan(base_dir: Path) -> Path:
    """A three-bin scan of bow-ties whose waist column moves with the bin."""
    from image_analysis.tools.synthetic_generators import generate_bowtie_image

    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    device = scan / "UC_HiResMagCam"
    device.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan001.ini").write_text(
        '[Scan Info]\nScan No = "1"\nScan Parameter = "U_Motor:Position"\n'
        'Start = "1"\nEnd = "3"\nStep size = "1"\nShots per step = "2"\n'
    )
    for shot in range(1, SHOTS + 1):
        frame = generate_bowtie_image(
            shape=(64, 128),
            total_charge=1.0,
            noise_level=10.0,
            background_level=0,
            energy_center=50 + 10 * ((shot - 1) // 2),
            vertical_offset=shot - 3,
            seed=shot,
        )
        np.save(device / f"Scan001_UC_HiResMagCam_{shot:03d}.npy", frame)
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


def test_a_hi_res_mag_cam_diagnostic_runs_on_the_core_route():
    assert core_supports(document("per_shot"))


@pytest.mark.parametrize("mode", ["per_shot", "per_bin"])
def test_the_core_route_matches_the_legacy_wrapper(tmp_path, monkeypatch, mode):
    legacy = run(monkeypatch, tmp_path / "legacy", document(mode), "legacy")
    core = run(monkeypatch, tmp_path / "core", document(mode), "core")
    old = snapshot_analysis_tree(legacy.parent.parent / "analysis")
    new = snapshot_analysis_tree(core.parent.parent / "analysis")
    assert sorted(old) == sorted(new)
    assert any(name.endswith(".h5") for name in new)
    assert compare_snapshots(old, new) == []
    rows = pd.read_csv(core.parent.parent / "analysis" / "s1.txt", sep="\t")
    assert set(BOWTIE) <= set(rows.columns)
    accepted = rows["Diag_emittance_proxy"] != 1e6
    assert accepted.all(), "the fixture's bow-ties must all fit"
    # Per bin, the waist column follows the fixture's energy centre; per
    # shot, every shot of a bin reads the same fit input, so equal within it.
    x0 = rows["Diag_bowtie_x0"]
    assert x0.notna().all() and (x0.diff().dropna() >= -1.0).all()
    if mode == "per_bin":
        assert (rows.groupby("Bin #")["Diag_bowtie_x0"].nunique() == 1).all()
