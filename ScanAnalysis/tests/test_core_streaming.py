"""A long camera run streams: memory is products, not shots; the stack opens once."""

from __future__ import annotations

import gc
import weakref
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.io import scan_stack
from geecs_data_utils.io.scan_stack import (
    FRAMES_DATASET,
    LABVIEW_EPOCH_OFFSET,
    TIMESTAMPS_DATASET,
)
from geecs_schemas.analysis import AnalysisRecipe

import scan_analysis.base as base
from scan_analysis import core_scan, core_source
from scan_analysis.core_analyzer import CoreScanAnalyzer
from scan_analysis.core_scan import prepare_scan

TAG = ScanTag(year=2026, month=1, day=2, number=3, experiment="Test")
SHOTS = 300


def recipe(**scan):
    return AnalysisRecipe.model_validate(
        {
            "device": "Camera",
            "input": {"kind": "camera", "file_tail": ".npy"},
            "measure": {"kind": "beam"},
            "scan": scan,
            "figure": {"fig": {"dpi": 20}},
            "summaries": [{"kind": "average"}],
        }
    )


def build_noscan(base_dir: Path, shots: int = SHOTS) -> Path:
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    device = scan / "Camera"
    device.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan003.ini").write_text(
        '[Scan Info]\nScan No = "3"\nScan Parameter = "noscan"\n'
        'Start = "1"\nEnd = "1"\nStep size = "1"\nShots per step = "1"\n'
    )
    yy, xx = np.mgrid[:12, :12]
    for shot in range(1, shots + 1):
        image = 100 * np.exp(-((xx - 6 - shot % 3) ** 2 + (yy - 6) ** 2) / 4) + 2
        np.save(device / f"Scan003_Camera_{shot:03d}.npy", image.astype(np.uint16))
    rows = pd.DataFrame({"Shotnumber": range(1, shots + 1), "Bin #": 1})
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    rows.to_csv(analysis / "s3.txt", sep="\t", index=False)
    return scan


def test_a_long_camera_run_keeps_a_bounded_number_of_frames_alive(
    tmp_path, monkeypatch
):
    """The consumer folds each frame and drops it; the old list held all 300.

    Frames are weakly tracked as the run yields them; before each yield the
    count of earlier frames still alive is recorded. Streaming keeps the
    running average's template plus the outcome the analyzer is still
    holding; collecting the outcomes into a list keeps every one.
    """
    scan = build_noscan(tmp_path)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    real = core_scan.run_units
    refs: list[weakref.ref] = []
    alive: list[int] = []

    def tracking(*args, **kwargs):
        for outcome in real(*args, **kwargs):
            gc.collect()
            alive.append(sum(ref() is not None for ref in refs))
            if outcome.measurement is not None:
                refs.append(weakref.ref(outcome.measurement.frame))
            yield outcome

    monkeypatch.setattr(core_scan, "run_units", tracking)
    analyzer = CoreScanAnalyzer(recipe(), id="Camera", priority=1)
    try:
        display = analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    assert len(alive) == SHOTS and len(refs) == SHOTS
    assert max(alive) <= 3, f"up to {max(alive)} frames alive of {SHOTS}"
    assert display and display[0].endswith("Camera_average_processed_visual.png")
    averaged = (
        scan.parent.parent
        / "analysis"
        / scan.name
        / "Camera"
        / "Array2DScanAnalyzer"
        / "Camera_average_processed.h5"
    )
    with h5py.File(averaged) as f:
        data = f[list(f)[0]][...]
    frames = [
        np.load(scan / "Camera" / f"Scan003_Camera_{n:03d}.npy")
        for n in range(1, SHOTS + 1)
    ]
    np.testing.assert_array_equal(data, np.mean(frames, axis=0).astype(data.dtype))


def _stack(device: Path, frames: np.ndarray) -> Path:
    device.mkdir(parents=True)
    path = device / f"{device.name}.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            FRAMES_DATASET, data=frames, chunks=(1, *frames.shape[1:])
        )
        handle.create_dataset(
            TIMESTAMPS_DATASET, data=[100.0 + n for n in range(len(frames))]
        )
    return path


def test_the_stack_is_opened_once_per_run(tmp_path, monkeypatch):
    frames = np.arange(5 * 4 * 6, dtype=np.uint16).reshape(5, 4, 6)
    _stack(tmp_path / "Camera", frames)
    rows = pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3, 4, 5],
            "Camera:acq_timestamp": [
                100.0 + n + LABVIEW_EPOCH_OFFSET for n in range(5)
            ],
        }
    )
    doc = AnalysisRecipe.model_validate(
        {
            "device": "Camera",
            "input": {"kind": "camera", "format": "device_hdf5"},
            "measure": {"kind": "none"},
        }
    )
    opens = []
    real = core_source.open_stack

    def counted(path, mode="r"):
        opens.append(Path(path))
        return real(path, mode)

    monkeypatch.setattr(core_source, "open_stack", counted)
    monkeypatch.setattr(scan_stack, "open_stack", counted)
    prepared = prepare_scan(doc, tmp_path, rows)
    # Mapping the rows to frames reads the timestamps (its own opens); the
    # run's reads are what one handle covers.
    opens.clear()
    outcomes = list(prepared.run())
    assert [o.group.key for o in outcomes] == [1, 2, 3, 4, 5]
    for outcome, frame in zip(outcomes, frames, strict=True):
        np.testing.assert_array_equal(outcome.measurement.frame.data, frame)
    assert len(opens) == 1, "one handle for the whole run"
    assert not prepared.source.open_stacks, "closed when the run completed"
    # A read outside the run's context opens and closes by itself, as before.
    np.testing.assert_array_equal(prepared.source.load(2), frames[1])
    assert len(opens) == 2 and not prepared.source.open_stacks


def test_a_run_closed_early_releases_its_handle(tmp_path):
    frames = np.ones((4, 3, 3), dtype=np.uint16)
    _stack(tmp_path / "Camera", frames)
    rows = pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3, 4],
            "Camera:acq_timestamp": [
                100.0 + n + LABVIEW_EPOCH_OFFSET for n in range(4)
            ],
        }
    )
    doc = AnalysisRecipe.model_validate(
        {
            "device": "Camera",
            "input": {"kind": "camera", "format": "device_hdf5"},
            "measure": {"kind": "none"},
        }
    )
    prepared = prepare_scan(doc, tmp_path, rows)
    outcomes = prepared.run()
    next(outcomes)
    assert prepared.source.open_stacks == (tmp_path / "Camera" / "Camera.h5",)
    outcomes.close()
    assert not prepared.source.open_stacks


def test_the_source_pickles_closed_and_reads_after_a_round_trip(tmp_path):
    import pickle

    frames = np.arange(2 * 2 * 2, dtype=np.uint16).reshape(2, 2, 2)
    _stack(tmp_path / "Camera", frames)
    rows = pd.DataFrame(
        {
            "Shotnumber": [7, 8],
            "Camera:acq_timestamp": [
                100.0 + LABVIEW_EPOCH_OFFSET,
                101.0 + LABVIEW_EPOCH_OFFSET,
            ],
        }
    )
    doc = AnalysisRecipe.model_validate(
        {
            "device": "Camera",
            "input": {"kind": "camera", "format": "device_hdf5"},
            "measure": {"kind": "none"},
        }
    )
    prepared = prepare_scan(doc, tmp_path, rows)
    with prepared.source as source:
        source.load(7)
        assert source.open_stacks
        copy = pickle.loads(pickle.dumps(source))
    assert not copy.open_stacks and copy.references == prepared.source.references
    with pytest.raises(TypeError):
        copy.references[9] = tmp_path
    with copy:
        np.testing.assert_array_equal(copy.load(8), frames[1])
    assert not copy.open_stacks
    # The prepared recipe travels too (its bound inputs as a read-only view).
    recipe_copy = pickle.loads(pickle.dumps(prepared.prepared))
    assert recipe_copy.recipe == prepared.prepared.recipe
    with pytest.raises(TypeError):
        recipe_copy.inputs["x"] = None
