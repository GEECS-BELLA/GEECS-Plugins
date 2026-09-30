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
#: The generator puts the waist at column 64 of 128; each bin's frames are
#: shifted so the waist really moves: 54, 64, then 104 — four columns past
#: the crop below, where the fit still accepts it and must report it there.
WAIST = 64
SHIFTS = (-10, 0, 40)
ROI = {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 64}
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
            "image": {"type": "camera", "pipeline": ["roi"], "roi": ROI},
            "scan": {"mode": mode, "file_tail": ".npy", "renderer": {"dpi": 30}},
        }
    )


def shift_columns(frame: np.ndarray, k: int) -> np.ndarray:
    """Move every column by ``k`` (positive = rightwards), zero-filling; no wrap."""
    out = np.zeros_like(frame)
    if k >= 0:
        out[:, k:] = frame[:, : frame.shape[1] - k]
    else:
        out[:, :k] = frame[:, -k:]
    return out


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
            noise_level=10.0,
            background_level=0,
            vertical_offset=shot - 3,
            seed=shot,
        )
        frame = shift_columns(frame, SHIFTS[(shot - 1) // 2])
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
    assert (rows["Diag_emittance_proxy"] != 1e6).all(), "every bow-tie must fit"
    # The waist column tracks where each bin's frames put it, in sensor
    # pixels. The third bin's waist sits four columns past the crop: an
    # extrapolated fit, so looser — but past the edge, never clamped to it.
    x0 = rows["Diag_bowtie_x0"]
    expected = np.array(
        [WAIST + SHIFTS[(shot - 1) // 2] for shot in rows["Shotnumber"]]
    )
    inside = (rows["Bin #"] != 3).to_numpy()
    np.testing.assert_allclose(x0[inside], expected[inside], atol=1.5)
    np.testing.assert_allclose(x0[~inside], expected[~inside], atol=5)
    # A clamped value would sit exactly on the last ROI column (x_max - 1).
    assert (x0[~inside] > ROI["x_max"] - 1).all()
    if mode == "per_bin":
        assert (rows.groupby("Bin #")["Diag_bowtie_x0"].nunique() == 1).all()
