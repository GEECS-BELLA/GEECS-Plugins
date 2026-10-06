"""The line stitcher on the core route: the source joins sibling traces, as the legacy stitcher did."""

from __future__ import annotations

import logging
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe

import scan_analysis.base as base
from geecs_analysis.compat.convert import to_v3
from scan_analysis import core_workers
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)
from scan_analysis.config import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer, core_supports
from scan_analysis.core_source import prepare_source
from scan_analysis.route_compare import compare_snapshots, snapshot_analysis_tree

TAG = ScanTag(year=2026, month=1, day=1, number=1, experiment="Test")
SHOTS = 6
SUFFIX = "-interpSpec"
DEVICES = ("MagSpec1", "MagSpec2", "MagSpec3")
#: Each camera covers its own stretch of the energy axis; 2 and 3 overlap.
RANGES = {"MagSpec1": (0.0, 10.0), "MagSpec2": (10.0, 22.0), "MagSpec3": (18.0, 30.0)}


def trace(device: str, shot: int) -> np.ndarray:
    lo, hi = RANGES[device]
    x = np.linspace(lo, hi, 25)
    y = (1 + shot / 10) * np.exp(-((x - 12 - shot) ** 2) / 30) + 0.1 * (
        DEVICES.index(device) + 1
    )
    return np.column_stack([x, y])


def document(**scan) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": f"MagSpec1{SUFFIX}",
            "analyzer": {
                "kind": "line_stitcher",
                "sibling_devices": [f"MagSpec2{SUFFIX}", f"MagSpec3{SUFFIX}"],
                "output_label": "Stitched",
            },
            "image": {
                "type": "line",
                "data_loading": {"data_type": "tsv"},
                "label": "Charge density vs Energy",
                "x_units": "MeV",
                "storage_dtype": "float32",
                "roi": {"x_min": 4.0, "x_max": 26.0},
                "filtering": {"method": "median", "kernel_size": 3},
                "thresholding": {
                    "method": "absolute",
                    "threshold_value": 0.0,
                    "clip_below": True,
                },
                "pipeline": ["roi", "filtering", "thresholding"],
            },
            "scan": {
                "mode": "per_shot",
                "file_tail": ".tsv",
                "renderer": {"dpi": 30},
                **scan,
            },
        }
    )


def build_scan(base_dir: Path, *, native: bool = False, skip=()) -> Path:
    """A completed scan: one folder per camera, legacy or native file names.

    ``skip`` lists ``(device, shot)`` files that were never written.
    """
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    scan.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan001.ini").write_text(
        '[Scan Info]\nScan No = "1"\nScan Parameter = "U_Motor:Position"\n'
        'Start = "1"\nEnd = "3"\nStep size = "1"\nShots per step = "2"\n'
    )
    rows = {
        "Shotnumber": list(range(1, SHOTS + 1)),
        "Bin #": [1 + i // 2 for i in range(SHOTS)],
        "U_Motor Position Alias:motor": [1.0 + i // 2 for i in range(SHOTS)],
    }
    for k, device in enumerate(DEVICES):
        folder = scan / f"{device}{SUFFIX}"
        folder.mkdir()
        stamps = [3800000000.0 + 10 * shot + 0.25 * k for shot in range(1, SHOTS + 1)]
        if native:
            rows[f"{device} acq_timestamp"] = stamps
        for shot, stamp in zip(range(1, SHOTS + 1), stamps, strict=True):
            if (device, shot) in skip:
                continue
            # A native file carries its folder's name (the device's variant)
            # and the device's own timestamp; a legacy one the shot number.
            name = (
                f"{device}{SUFFIX}_{stamp:.3f}.tsv"
                if native
                else f"Scan001_{device}_{shot:03d}.tsv"
            )
            np.savetxt(folder / name, trace(device, shot), delimiter="\t")
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    pd.DataFrame(rows).to_csv(analysis / "s1.txt", sep="\t", index=False)
    return scan


def run(monkeypatch, base_dir: Path, doc, *, route="core", **build) -> Path:
    scan = build_scan(base_dir, **build)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    if route == "legacy":
        analyzer = create_scan_analyzer(doc, id="Stitch", priority=1, route="legacy")
        assert isinstance(analyzer, SingleDeviceScanAnalyzer)
    else:
        analyzer = CoreScanAnalyzer(doc, id="Stitch", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    return scan


def snapshot(scan: Path) -> dict:
    return snapshot_analysis_tree(scan.parent.parent / "analysis")


def stitched(scan: Path, shot: int, doc=None) -> np.ndarray:
    """The source's joined trace for one shot, before any processing."""
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s1.txt", sep="\t")
    return prepare_source(doc or document(), scan, rows).load(shot)


def test_a_stitcher_diagnostic_runs_on_the_core_route():
    assert core_supports(document())


def test_the_core_route_matches_the_legacy_stitcher(tmp_path, monkeypatch):
    legacy = run(monkeypatch, tmp_path / "legacy", document(), route="legacy")
    core = run(monkeypatch, tmp_path / "core", document())
    old, new = snapshot(legacy), snapshot(core)
    assert sorted(old) == sorted(new)
    assert any(name.endswith(".h5") for name in new)
    old_rows, new_rows = old["s1.txt"][1], new["s1.txt"][1]
    columns = new_rows.columns
    assert any(c.startswith(f"MagSpec1{SUFFIX}_") for c in columns)
    # The one deliberate difference: a joined trace is unevenly spaced, and
    # the core measures its widths in MeV (GEECS-Analysis 0.26.0, #1029)
    # where the legacy analyzer counted samples times the spacing at the
    # centroid. Every other file, column and sample is identical — the two
    # width columns are dropped from every table (the s-file and the
    # per-scan copy) before the comparison.
    widths = [c for c in columns if c.endswith(("_rms", "_fwhm"))]
    assert len(widths) == 2
    for column in widths:
        assert not np.allclose(old_rows[column], new_rows[column])
    for name, (kind, *payload) in list(new.items()):
        if kind == "table":
            old[name] = ("table", old[name][1].drop(columns=widths))
            new[name] = ("table", payload[0].drop(columns=widths))
    assert compare_snapshots(old, new) == []


def test_the_joined_trace_is_every_segment_sorted_by_x(tmp_path):
    scan = build_scan(tmp_path)
    joined = stitched(scan, 3)
    expected = np.concatenate([trace(d, 3) for d in DEVICES])
    expected = expected[expected[:, 0].argsort()]
    np.testing.assert_array_equal(joined, expected)


def test_siblings_join_by_their_own_timestamps_on_native_file_names(tmp_path):
    """Every camera stamps its own files: the legacy filename swap finds none.

    The device-named shape (``name`` the device, ``scan.device`` its folder)
    joins the input by its timestamp column; each sibling's device is its
    folder less the same suffix, so its own column joins it.
    """
    scan = build_scan(tmp_path, native=True)
    doc = document(device=f"MagSpec1{SUFFIX}")
    doc = doc.model_copy(update={"name": "MagSpec1"})
    joined = stitched(scan, 2, doc)
    assert len(joined) == 3 * 25
    np.testing.assert_array_equal(np.sort(joined[:, 0]), joined[:, 0])


def test_a_shot_a_sibling_lacks_is_stitched_without_it(tmp_path, caplog):
    scan = build_scan(tmp_path, skip={("MagSpec3", 4)})
    with caplog.at_level(logging.WARNING, logger="scan_analysis.core_source"):
        joined = stitched(scan, 4)
    assert len(joined) == 2 * 25
    assert any(f"MagSpec3{SUFFIX}" in r.getMessage() for r in caplog.records)
    assert len(stitched(scan, 5)) == 3 * 25


def test_a_missing_sibling_folder_stitches_the_rest(tmp_path, caplog):
    scan = build_scan(tmp_path)
    for f in (scan / f"MagSpec3{SUFFIX}").iterdir():
        f.unlink()
    (scan / f"MagSpec3{SUFFIX}").rmdir()
    with caplog.at_level(logging.WARNING, logger="scan_analysis.core_source"):
        joined = stitched(scan, 1)
    assert len(joined) == 2 * 25
    assert not (scan / f"MagSpec3{SUFFIX}").exists()


def test_the_source_pickles_with_its_siblings(tmp_path):
    scan = build_scan(tmp_path)
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s1.txt", sep="\t")
    source = prepare_source(document(), scan, rows)
    copy = pickle.loads(pickle.dumps(source))
    np.testing.assert_array_equal(copy.load(2), source.load(2))


def test_the_converted_recipe_stitches_and_writes_what_the_v2_route_writes(
    tmp_path, monkeypatch
):
    conversion = to_v3(document())
    recipe = conversion.recipe
    assert recipe.input.siblings == [f"MagSpec2{SUFFIX}", f"MagSpec3{SUFFIX}"]
    assert any("output_label dropped" in note for note in conversion.notes)
    v2 = run(monkeypatch, tmp_path / "v2", document())
    v3 = run(monkeypatch, tmp_path / "v3", recipe)
    assert compare_snapshots(snapshot(v2), snapshot(v3), average_ulps=0) == []


def test_a_pooled_stitch_writes_the_serial_tree(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(core_workers, "MIN_UNITS_FOR_POOL", 1)
    monkeypatch.setattr(CoreScanAnalyzer, "worker_cap", 2)
    recipe = to_v3(document()).recipe
    pooled = recipe.model_copy(
        update={"scan": recipe.scan.model_copy(update={"workers": 2})}
    )
    serial_scan = run(monkeypatch, tmp_path / "serial", recipe)
    with caplog.at_level(logging.INFO, logger="scan_analysis.core_analyzer"):
        pooled_scan = run(monkeypatch, tmp_path / "pooled", pooled)
    assert any(r.getMessage().endswith("units, 2 workers") for r in caplog.records)
    assert (
        compare_snapshots(snapshot(serial_scan), snapshot(pooled_scan), average_ulps=0)
        == []
    )


def test_the_core_route_writes_nothing_into_the_scan(tmp_path, monkeypatch):
    """The legacy stitched TSVs are dropped (owner ruling 2026-09-27)."""
    scan = build_scan(tmp_path / "pre")
    before = sorted(p.relative_to(scan) for p in scan.rglob("*"))
    scan = run(monkeypatch, tmp_path / "run", document())
    after = sorted(p.relative_to(scan) for p in scan.rglob("*"))
    assert after == before


@pytest.mark.parametrize(
    "siblings,message",
    [
        (["B", "B"], "repeats"),
        (["A"], "own folder"),
        (["x/y"], "one folder"),
    ],
)
def test_a_recipe_refuses_bad_siblings(siblings, message):
    with pytest.raises(ValueError, match=message):
        AnalysisRecipe.model_validate(
            {
                "schema_version": 3,
                "device": "A",
                "input": {
                    "kind": "line",
                    "loading": {"data_type": "tsv"},
                    "siblings": siblings,
                },
            }
        )


def test_a_stack_only_sibling_without_a_stack_is_stitched_around(tmp_path, caplog):
    """A pva_stack input whose sibling folder holds no stack keeps running."""
    from scan_analysis import core_source

    scan = build_scan(tmp_path)
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s1.txt", sep="\t")
    from dataclasses import replace

    spec = replace(core_source._resolved(document()), prefer_stack=True)
    with caplog.at_level(logging.WARNING, logger="scan_analysis.core_source"):
        refs = core_source._sibling_references(
            spec, f"MagSpec2{SUFFIX}", scan, rows, stacks_only=True
        )
    assert refs == {}
    assert any("stitching without it" in r.getMessage() for r in caplog.records)
