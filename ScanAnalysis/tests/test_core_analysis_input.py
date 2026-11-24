"""A recipe whose input is an earlier analysis's per-shot outputs (``analysis:<folder>``)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.io.analysis_outputs import OUTPUT_PREFIX
from geecs_schemas.analysis import ANALYSIS_OUTPUT_PREFIX, AnalysisRecipe

import scan_analysis.base as base
from scan_analysis.core_analyzer import CoreScanAnalyzer
from scan_analysis.core_source import (
    OutputUnavailable,
    prepare_source,
    source_directory,
)

TAG = ScanTag(year=2026, month=1, day=2, number=3, experiment="Test")
FOLDER = "py-magspec-interpSpec"


def line_recipe(folder: str = f"analysis:{FOLDER}", **extra) -> AnalysisRecipe:
    return AnalysisRecipe.model_validate(
        {
            "device": "MagSpecLineout",
            "input": {
                "kind": "line",
                "folder": folder,
                "loading": {"data_type": "tsv"},
            },
            "measure": {"kind": "line"},
            "figure": {"fig": {"dpi": 20}},
            **extra,
        }
    )


def build(base_dir: Path, shots=(1, 2, 3, 4), written=(1, 2, 4), rows=None) -> Path:
    """A noscan whose earlier analysis wrote a spectrum for *written* shots only."""
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    scan.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan003.ini").write_text(
        '[Scan Info]\nScan No = "3"\nScan Parameter = "noscan"\n'
        'Start = "1"\nEnd = "1"\nStep size = "1"\nShots per step = "1"\n'
    )
    day = scan.parent.parent
    output = day / "analysis" / scan.name / FOLDER
    output.mkdir(parents=True)  # fixture: the earlier analysis's outputs
    energy = np.linspace(50.0, 150.0, 21)
    for shot in written:
        charge = np.exp(-((energy - 90 - shot) ** 2) / 50.0)
        table = "Energy_MeV\tCharge_pCperMeV\n" + "\n".join(
            f"{e:.6g}\t{c:.6g}" for e, c in zip(energy, charge)
        )
        (output / f"Scan003_{FOLDER}_{shot:03d}.txt").write_text(table)
    frame = (
        rows
        if rows is not None
        else pd.DataFrame({"Shotnumber": list(shots), "Bin #": 1})
    )
    frame.to_csv(day / "analysis" / "s3.txt", sep="\t", index=False)
    return scan


def test_the_prefix_is_the_readers():
    """The schema's name for an analysis output is the one data-utils resolves."""
    assert ANALYSIS_OUTPUT_PREFIX == OUTPUT_PREFIX


def test_each_shot_maps_to_its_own_file_and_a_missing_one_to_none(tmp_path, caplog):
    scan = build(tmp_path)
    rows = pd.DataFrame({"Shotnumber": [1, 2, 3, 4]})
    source = prepare_source(line_recipe(), scan, rows)
    assert source.data_dir == scan.parent.parent / "analysis" / scan.name / FOLDER
    assert sorted(source.references) == [1, 2, 4]  # shot 3 was never written
    assert "1 shot(s) of this scan have no file" in caplog.text  # said, not silent
    assert source.references[4].name == f"Scan003_{FOLDER}_004.txt"
    trace = source.load(2)
    assert trace.shape == (21, 2) and trace[0, 0] == pytest.approx(50.0)


def test_a_scan_mode_run_maps_by_the_announced_shot(tmp_path):
    """The earlier run named its files by the shot each row announced."""
    scan = build(tmp_path, written=(7, 8))
    rows = pd.DataFrame({"Shotnumber": [1, 2], "Announced Shot": [7, 8]})
    source = prepare_source(line_recipe(), scan, rows)
    assert {s: p.name for s, p in source.references.items()} == {
        1: f"Scan003_{FOLDER}_007.txt",
        2: f"Scan003_{FOLDER}_008.txt",
    }


def test_an_output_this_scan_lacks_is_unavailable_and_nothing_is_created(tmp_path):
    scan = build(tmp_path)
    before = sorted(p for p in tmp_path.rglob("*"))
    with pytest.raises(OutputUnavailable):
        prepare_source(
            line_recipe("analysis:not-there"), scan, pd.DataFrame({"Shotnumber": [1]})
        )
    with pytest.raises(OutputUnavailable):
        prepare_source(
            line_recipe(f"analysis:{FOLDER}/nope"),
            scan,
            pd.DataFrame({"Shotnumber": [1]}),
        )
    assert sorted(p for p in tmp_path.rglob("*")) == before


def test_the_folder_resolves_under_the_analysis_tree(tmp_path):
    scan = build(tmp_path)
    assert source_directory(line_recipe(), scan) == (
        scan.parent.parent / "analysis" / scan.name / FOLDER
    )


def test_a_run_reads_the_outputs_and_writes_its_own_columns(tmp_path, monkeypatch):
    """The whole route: three spectra measured, shot 3 left alone, outputs under its own name."""
    scan = build(tmp_path)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    analyzer = CoreScanAnalyzer(line_recipe(), id="MagSpecLineout", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    sfile = pd.read_csv(scan.parent.parent / "analysis" / "s3.txt", sep="\t")
    centroid = [
        c
        for c in sfile.columns
        if c.startswith("MagSpecLineout_") and c.endswith("_CoM")
    ]
    assert centroid, sfile.columns
    values = sfile.set_index("Shotnumber")[centroid[0]]
    assert np.isfinite(values[[1, 2, 4]]).all()
    assert pd.isna(values[3])
    assert values[4] > values[1]  # shot 4's peak sits higher
    # its own output folder, beside (never inside) the one it read
    assert (scan.parent.parent / "analysis" / scan.name / "MagSpecLineout").is_dir()


def test_a_run_without_the_output_folder_is_no_data(tmp_path, monkeypatch):
    scan = build(tmp_path)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    analyzer = CoreScanAnalyzer(line_recipe("analysis:absent"), id="X", priority=1)
    try:
        with pytest.raises(base.DataUnavailableWarning):
            analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    assert not (scan.parent.parent / "analysis" / scan.name / "absent").exists()


def test_a_camera_reads_an_output_image_series(tmp_path):
    """A folder of several series: the recipe names the image one."""
    from PIL import Image

    scan = build(tmp_path, written=())
    output = scan.parent.parent / "analysis" / scan.name / "py-ebeam"
    output.mkdir()
    for shot in (1, 2):
        Image.fromarray(np.full((4, 5), 10 * shot, dtype=np.uint16)).save(
            output / f"Scan003_py-ebeam_{shot:03d}.png"
        )
        (output / f"Scan003_py-ebeamX_{shot:03d}.txt").write_text("x\ty\n0\t1\n1\t2\n")
    recipe = AnalysisRecipe.model_validate(
        {
            "device": "Ebeam",
            "input": {"kind": "camera", "folder": "analysis:py-ebeam/py-ebeam"},
        }
    )
    source = prepare_source(recipe, scan, pd.DataFrame({"Shotnumber": [1, 2]}))
    assert source.load(2).shape == (4, 5)
    assert int(source.load(2)[0, 0]) == 20
