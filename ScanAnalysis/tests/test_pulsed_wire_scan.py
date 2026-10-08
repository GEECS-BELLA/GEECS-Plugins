"""A pulsed-wire magnet scan end to end on the public ScanAnalysis route.

One TDMS scope trace per shot (wire deflection versus time, the time axis
built from the channel's ``wf_start_offset`` / ``wf_increment``), a hexapod
scanning the magnet assembly across seven positions. Each element's kick is
linear in the position and crosses zero at the element's magnetic centre;
the ``pulsed_wire`` measure reads the kicks off the drift plateaus and the
``scalar_fit`` summary recovers every slope and centre into its JSON sidecar.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisRecipe

import scan_analysis.base as base
from scan_analysis.core_analyzer import CoreScanAnalyzer

nptdms = pytest.importorskip("nptdms")

TAG = ScanTag(year=2026, month=1, day=1, number=1, experiment="Test")
DEVICE = "U_PulsedWire"
PARAM_COLUMN = "U_Hexapod X Alias:hexapod"
POSITIONS = np.linspace(-1.5, 1.5, 7)
SHOTS_PER_BIN = 3
SAMPLES = 4000
T0, DT = 0.0, 1.5e-6  # the trace spans 0 to 6 ms
#: Per element: slope a_i and magnetic centre c_i of kick_i(x) = a_i (x - c_i).
SLOPES = (2.0, -3.0, 1.5)
CENTRES = (0.3, -0.2, 0.5)
#: Element ramps and the drift windows around them, in seconds.
ELEMENTS = ((2.3e-3, 2.8e-3), (3.4e-3, 3.9e-3), (4.5e-3, 5.0e-3))
WINDOWS = (
    (1.8e-3, 2.2e-3),
    (2.9e-3, 3.3e-3),
    (4.0e-3, 4.4e-3),
    (5.1e-3, 5.5e-3),
)


def wire_trace(x: float, rng: np.random.Generator) -> np.ndarray:
    """First field integral: zero before 2.2 ms, a ramp per element, flat drifts."""
    t = T0 + np.arange(SAMPLES) * DT
    y = np.zeros(SAMPLES)
    for (start, end), a, c in zip(ELEMENTS, SLOPES, CENTRES):
        y += a * (x - c) * np.clip((t - start) / (end - start), 0.0, 1.0)
    return y + rng.normal(0.0, 2e-3, SAMPLES)


def write_tdms(path: Path, wire: np.ndarray, other: np.ndarray) -> None:
    """Two scope channels with the waveform properties the reader builds time from."""
    from nptdms import ChannelObject, TdmsWriter

    props = {"wf_start_offset": T0, "wf_increment": DT}
    with TdmsWriter(str(path)) as writer:
        writer.write_segment(
            [
                ChannelObject("Scope", "Wire", wire, properties=props),
                ChannelObject("Scope", "Trigger", other, properties=props),
            ]
        )


def build_scan(base_dir: Path) -> Path:
    """A completed hexapod scan in the GEECS layout: TDMS shots, s-file, ScanInfo."""
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    device = scan / DEVICE
    device.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan001.ini").write_text(
        '[Scan Info]\nScan No = "1"\nScan Parameter = "U_Hexapod:X"\n'
        f'Start = "{POSITIONS[0]}"\nEnd = "{POSITIONS[-1]}"\n'
        f'Step size = "0.5"\nShots per step = "{SHOTS_PER_BIN}"\n'
    )
    rng = np.random.default_rng(11)
    shots, bins, xs = [], [], []
    for b, x in enumerate(POSITIONS, start=1):
        for _ in range(SHOTS_PER_BIN):
            shot = len(shots) + 1
            write_tdms(
                device / f"Scan001_{DEVICE}_{shot:03d}.tdms",
                wire_trace(x, rng),
                np.where(np.arange(SAMPLES) < 100, 1.0, 0.0),
            )
            shots.append(shot)
            bins.append(b)
            xs.append(float(x))
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    pd.DataFrame({"Shotnumber": shots, "Bin #": bins, PARAM_COLUMN: xs}).to_csv(
        analysis / "s1.txt", sep="\t", index=False
    )
    return scan


def recipe(steps, measure, summaries) -> AnalysisRecipe:
    return AnalysisRecipe.model_validate(
        {
            "schema_version": 3,
            "device": DEVICE,
            "input": {
                "kind": "line",
                "file_tail": ".tdms",
                "loading": {"data_type": "tdms_scope", "trace_index": 0},
                "x_unit": "s",
                "y_unit": "V",
            },
            "steps": steps,
            "measure": measure,
            "scan": {"average_frames_first": True},
            "figure": {"fig": {"dpi": 30}},
            "summaries": summaries,
        }
    )


def run(monkeypatch, tmp_path, doc) -> tuple[Path, list]:
    """Run the recipe on the public route; the scan folder is never added to."""
    scan = build_scan(tmp_path)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    scans_before = sorted(p.relative_to(tmp_path) for p in scan.parent.rglob("*"))
    analyzer = CoreScanAnalyzer(doc, id="PulsedWire", priority=1)
    try:
        display = analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    # The scan-folder invariant: nothing under scans/ was created or removed.
    assert sorted(p.relative_to(tmp_path) for p in scan.parent.rglob("*")) == (
        scans_before
    )
    return scan, display or []


def output_file(scan: Path, name: str) -> Path:
    found = list((scan.parent.parent / "analysis").rglob(name))
    assert len(found) == 1, (name, found)
    return found[0]


def test_kicks_fit_to_the_magnet_centres(tmp_path, monkeypatch):
    windows = [{"start": s, "end": e} for s, e in WINDOWS]
    doc = recipe(
        steps=[{"step": "lowpass", "order": 2, "critical_frequency": 0.2}],
        measure={
            "kind": "pulsed_wire",
            "windows": windows,
            "elements": [{"name": "Q1"}, {"name": "Q2"}, {"name": "Q3"}],
        },
        summaries=[
            {"kind": "waterfall"},
            {"kind": "scalar_fit", "scalars": ["kick_1", "kick_2", "kick_3"]},
        ],
    )
    scan, _ = run(monkeypatch, tmp_path, doc)

    rows = pd.read_csv(scan.parent.parent / "analysis" / "s1.txt", sep="\t")
    for i in (1, 2, 3):
        column = f"{DEVICE}_kick_{i}"
        assert column in rows.columns
        expected = SLOPES[i - 1] * (rows[PARAM_COLUMN] - CENTRES[i - 1])
        np.testing.assert_allclose(rows[column], expected, atol=2e-3)
    assert f"{DEVICE}_plateau_0" in rows.columns
    # no lengths were given, so no per-length scalars
    assert not any("kick_per_length" in c for c in rows.columns)

    png = output_file(scan, f"{DEVICE}_summary_scalar_fit.png")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    sidecar = json.loads(
        output_file(scan, f"{DEVICE}_summary_scalar_fit.json").read_text()
    )
    assert sidecar["kind"] == "scalar_fit"
    fitted = sidecar["scalars"]
    for i, (a, c) in enumerate(zip(SLOPES, CENTRES), start=1):
        assert fitted[f"kick_{i}_points"] == len(POSITIONS)
        assert fitted[f"kick_{i}_slope"] == pytest.approx(a, rel=1e-3)
        assert fitted[f"kick_{i}_zero_crossing"] == pytest.approx(c, abs=1e-3)
        assert fitted[f"kick_{i}_r2"] > 0.9999
    assert output_file(scan, f"{DEVICE}_summary_waterfall.png").stat().st_size > 0


def test_the_field_view_renders_a_waterfall(tmp_path, monkeypatch):
    doc = recipe(
        steps=[
            {"step": "lowpass", "order": 2, "critical_frequency": 0.2},
            {"step": "derivative"},
        ],
        measure={"kind": "none"},
        summaries=[{"kind": "waterfall"}],
    )
    scan, _ = run(monkeypatch, tmp_path, doc)
    png = output_file(scan, f"{DEVICE}_summary_waterfall.png")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list((scan.parent.parent / "analysis").rglob("*_scalar_fit.json"))
