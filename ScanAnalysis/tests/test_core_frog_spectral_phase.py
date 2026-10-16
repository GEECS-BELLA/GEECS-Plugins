"""FROG spectral phase on the core route: the golden (legacy) tree, the weights as a channel."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisDiagnostic

import scan_analysis.base as base
from geecs_analysis.compat.v2 import analyze_v2, compile_v2
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)
from scan_analysis.config import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer, core_supports
from scan_analysis.route_compare import compare_snapshots, snapshot_analysis_tree

TAG = ScanTag(year=2026, month=1, day=1, number=1, experiment="Test")
SHOTS = 6
DEVICE = "U_FROG_Grenouille-Temporal"
COLUMNS = [
    "time_fs",
    "temporal_intensity",
    "temporal_phase",
    "wavelength_nm",
    "spectral_intensity",
    "spectral_phase",
]
SCALARS = ("phi0_rad", "gd_fs", "gdd_fs2", "tod_fs3", "flipped")


def document(storage: str = "float64", mode: str = "per_shot") -> AnalysisDiagnostic:
    """The acceptance document's shape: the frog step's TSV, weights at column 4."""
    return AnalysisDiagnostic.model_validate(
        {
            "name": DEVICE,
            "analyzer": {"kind": "frog_spectral_phase", "fit_order": 3},
            "image": {
                "type": "line",
                "storage_dtype": storage,
                "data_loading": {
                    "data_type": "tsv",
                    "delimiter": "\t",
                    "x_column": 3,
                    "y_column": 5,
                    "auxiliary_columns": {"weights": 4},
                },
                "roi": {"x_min": 750.0, "x_max": 850.0},
                "pipeline": ["roi"],
            },
            "scan": {
                "mode": mode,
                "file_tail": "_retrieved_lineouts.tsv",
                "renderer": {"dpi": 30},
            },
        }
    )


def lineout(shot: int) -> pd.DataFrame:
    """One shot's retrieved lineouts, as the ``frog`` measure writes them."""
    rng = np.random.default_rng(shot)
    wavelength = np.linspace(700.0, 900.0, 200)
    omega = 2 * np.pi * 299_792_458.0 / (wavelength * 1e-9) * 1e-15
    detuning = omega - 2 * np.pi * 299_792_458.0 / 800e-9 * 1e-15
    gdd = 200.0 + 40.0 * shot * (-1) ** shot
    phase = (
        0.1 * shot
        + 5.0 * detuning
        + 0.5 * gdd * detuning**2
        + rng.normal(0, 0.05, wavelength.size)
    )
    intensity = np.exp(-(((wavelength - 800.0 + shot) / 15.0) ** 2))
    time = np.full(wavelength.size, np.nan)
    time[:50] = np.linspace(-100.0, 100.0, 50)
    return pd.DataFrame(
        dict(
            zip(
                COLUMNS,
                [time, time * 0 + 1, time * 0, wavelength, intensity, phase],
                strict=True,
            )
        )
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
        lineout(shot).to_csv(
            device / f"Scan001_{DEVICE}_{shot:03d}_retrieved_lineouts.tsv",
            sep="\t",
            index=False,
            float_format="%.17g",
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
        analyzer = create_scan_analyzer(doc, id="FSP", priority=1, route="legacy")
        assert isinstance(analyzer, SingleDeviceScanAnalyzer)
    else:
        analyzer = CoreScanAnalyzer(doc, id="FSP", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    return scan


def test_a_frog_spectral_phase_diagnostic_runs_on_the_core_route():
    assert core_supports(document())


@pytest.mark.parametrize("storage", ["float64", "float32"])
def test_the_core_route_writes_the_golden_tree(tmp_path, monkeypatch, golden, storage):
    """The legacy wrapper's tree at float64 storage, exact; at float32 the
    same tree with the declared storage dtype (see the oracle test)."""
    core = run(monkeypatch, tmp_path / "core", document(storage), "core")
    new = snapshot_analysis_tree(core.parent.parent / "analysis")
    expected = golden(f"frog_spectral_phase/{storage}", new)
    assert sorted(new) == sorted(expected.files)
    assert any(name.endswith(".h5") for name in new)
    assert compare_snapshots(expected.files, new, average_ulps=0) == []


def test_the_legacy_wrapper_wrote_the_golden_tree(tmp_path, monkeypatch, golden):
    """The oracle at float64 storage: products, bin averages and scalars exact."""
    legacy = run(monkeypatch, tmp_path / "legacy", document(), "legacy")
    old = snapshot_analysis_tree(legacy.parent.parent / "analysis")
    expected = golden("frog_spectral_phase/float64")
    assert sorted(old) == sorted(expected.files)
    assert compare_snapshots(expected.files, old, average_ulps=0) == []


def test_the_weights_reach_the_fit(tmp_path, monkeypatch):
    core = run(monkeypatch, tmp_path / "core", document(), "core")
    rows = pd.read_csv(
        core.parent.parent / "analysis" / "s1.txt",
        sep="\t",
        float_precision="round_trip",
    )
    for key in SCALARS:
        assert rows[f"{DEVICE}_{key}"].notna().all(), key
    # The weights reached the fit: the shots' GDDs alternate as built.
    assert rows[f"{DEVICE}_gdd_fs2"].diff().abs().min() > 10.0


def test_the_legacy_wrapper_at_float32_differs_only_by_the_stored_dtype(
    tmp_path, monkeypatch, golden
):
    legacy = run(monkeypatch, tmp_path / "legacy", document("float32"), "legacy")
    old = snapshot_analysis_tree(legacy.parent.parent / "analysis")
    expected = golden("frog_spectral_phase/float32")
    assert sorted(old) == sorted(expected.files)
    # The legacy wrapper saved the float64 fit ignoring storage_dtype (as it
    # did for ICT); the golden stores the declared float32. The fit itself is
    # float64 on both routes. Every product of this scanned run is a bin
    # average, which the core takes at the storage dtype (float32
    # accumulation, the GEECS-Analysis convention) and the wrapper in
    # float64: a few float32 ulps (measured: 2.1 eps).
    rtol = 4 * float(np.finfo(np.float32).eps)
    products = [name for name in old if name.endswith(".h5")]
    assert products
    for name in products:
        _, key, legacy_data, legacy_dtype = old[name]
        _, key2, golden_data, golden_dtype = expected.files[name]
        assert (key, legacy_dtype, golden_dtype) == (key2, "float64", "float32"), name
        np.testing.assert_allclose(
            golden_data, legacy_data, rtol=rtol, atol=0, err_msg=name
        )
    # The scalars are computed from identically rounded inputs: exact.
    tables = [name for name in old if old[name][0] == "table"]
    rest_old = {name: old[name] for name in tables}
    rest = {name: expected.files[name] for name in tables}
    assert compare_snapshots(rest, rest_old) == []


def test_a_raw_bin_run_fits_the_bin_mean_with_the_bin_mean_weights(
    tmp_path, monkeypatch
):
    """Per bin, the channel is averaged with the trace it belongs to.

    The legacy per-bin path loaded traces without their auxiliary columns,
    so its bin fits were unweighted (and masked on |phase|); the core
    averages the weights row for row with the trace.
    """
    doc = document(mode="per_bin")
    core = run(monkeypatch, tmp_path, doc, "core")
    rows = pd.read_csv(
        core.parent.parent / "analysis" / "s1.txt",
        sep="\t",
        float_precision="round_trip",
    )
    recipe = compile_v2(doc)
    for first in (1, 3, 5):
        raw = [
            lineout(shot).iloc[:, [3, 5, 4]].to_numpy() for shot in (first, first + 1)
        ]
        expected = analyze_v2(np.mean(np.stack(raw), axis=0), recipe).scalars
        for shot in (first, first + 1):
            got = rows.loc[rows["Shotnumber"] == shot, f"{DEVICE}_gdd_fs2"].item()
            assert got == expected["gdd_fs2"]
