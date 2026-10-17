"""Scan backgrounds on the core route: exact strip statistics, the golden trees, no stray writes."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisRecipe

import scan_analysis.base as base
import scan_analysis.core_backgrounds as core_backgrounds
from geecs_analysis.engine.compiled import ScanBackground
from scan_analysis.core_analyzer import CoreScanAnalyzer
from scan_analysis.core_backgrounds import resolve_scan_background, scan_statistic
from scan_analysis.core_inputs import ScanContextRequired
from scan_analysis.core_preview import prepare_document
from scan_analysis.route_compare import (
    GOLDEN_RTOL,
    compare_snapshots,
    snapshot_analysis_tree,
)

DEVICE = "Camera"
SHOTS = 6


def stack(n: int = 7, shape=(13, 9), seed: int = 2) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 4000, size=shape).astype(np.uint16) for _ in range(n)]


def whole_stack_statistic(frames, statistic: str, percentile=None) -> np.ndarray:
    """The retired legacy aggregation, verbatim: one float64 stack, one numpy call.

    What ImageAnalysis' ``_aggregate_image_stack`` computed (mean over the
    raw stack; median / percentile over the stack cast to float64) — the
    strip statistic must reproduce it bit for bit.
    """
    if statistic == "mean":
        return np.mean(np.stack(frames), axis=0)
    cast = np.stack([f.astype(np.float64) for f in frames], axis=0)
    if statistic == "median":
        return np.median(cast, axis=0)
    return np.percentile(cast, percentile, axis=0)


@pytest.mark.parametrize(
    "statistic,percentile",
    [("mean", None), ("median", None), ("percentile", 10.0), ("percentile", 37.5)],
)
@pytest.mark.parametrize("budget", [1, 9 * 8 * 7 * 2, 1 << 30])
def test_the_strip_statistic_is_the_legacy_whole_stack_one_bit_for_bit(
    statistic, percentile, budget, tmp_path
):
    """Tiny budgets force one-row strips over the on-disk copy; a big one stays in memory."""
    frames = stack()
    expected = whole_stack_statistic(frames, statistic, percentile)
    loaders = [lambda f=f: f for f in frames]
    actual = scan_statistic(
        loaders, statistic, percentile, budget_bytes=budget, scratch_dir=tmp_path
    )
    np.testing.assert_array_equal(actual, expected)
    assert list(tmp_path.iterdir()) == []  # the scratch copy is gone


def test_an_unreadable_frame_is_skipped_and_a_mismatched_one_refused():
    frames = stack(5)

    def broken():
        raise OSError("truncated file")

    loaders = (
        [lambda f=f: f for f in frames[:2]]
        + [broken]
        + [lambda f=f: f for f in frames[2:]]
    )
    np.testing.assert_array_equal(
        scan_statistic(loaders, "percentile", 20.0, budget_bytes=1),
        whole_stack_statistic(frames, "percentile", 20.0),
    )
    odd = [lambda: frames[0], lambda: frames[1][:4]]
    with pytest.raises(ValueError, match="differ in shape"):
        scan_statistic(odd, "median")


def test_a_capture_stack_is_read_frame_by_frame(tmp_path):
    h5py = pytest.importorskip("h5py")
    from geecs_data_utils.io.scan_stack import FRAMES_DATASET, TIMESTAMPS_DATASET

    frames = np.stack(stack(4))
    device = tmp_path / "day" / "scans" / "Scan003" / DEVICE
    device.mkdir(parents=True)  # fixture acquisition
    with h5py.File(device / f"{DEVICE}.h5", "w") as f:
        f.create_dataset(FRAMES_DATASET, data=frames, chunks=(1, *frames.shape[1:]))
        f.create_dataset(TIMESTAMPS_DATASET, data=np.arange(4) + 1.0)
    (tmp_path / "day" / "analysis").mkdir()
    request = ScanBackground("camera_background", None, "median")
    result = resolve_scan_background(
        request, data_dir=device, device=DEVICE, file_tail=".png", prefer_stack=True
    )
    np.testing.assert_array_equal(result, np.median(frames.astype(float), axis=0))


# --------------------------------------------------------------- scan route


def document(**from_scan) -> AnalysisRecipe:
    return AnalysisRecipe.model_validate(
        {
            "device": DEVICE,
            "output_name": "Diag",
            "input": {"kind": "camera", "file_tail": ".npy"},
            "inputs": {"camera_background": {"from_scan": from_scan}},
            "steps": [
                {
                    "step": "background_frame",
                    "source": "camera_background",
                    "alignment": "samples",
                }
            ],
            "measure": {"kind": "beam"},
            "figure": {"imshow": {"vmin": 0}, "fig": {"dpi": 30}},
            "summaries": [
                {"kind": "image_grid", "panel_size": [6.0, 6.0]},
                {"kind": "average"},
            ],
        }
    )


def tag(number: int) -> ScanTag:
    return ScanTag(year=2026, month=1, day=1, number=number, experiment="Test")


def build_day(base_dir: Path, *, dark: bool = True) -> Path:
    """Scan 1 is a dark scan, scan 2 the data; returns scan 2's folder."""
    rng = np.random.default_rng(9)
    if dark:
        scan1 = ScanPaths.get_scan_folder_path(tag=tag(1), base_directory=base_dir)
        (scan1 / DEVICE).mkdir(parents=True)  # fixture acquisition
        for shot in range(1, 5):
            np.save(
                scan1 / DEVICE / f"Scan001_{DEVICE}_{shot:03d}.npy",
                rng.integers(0, 20, size=(24, 24)).astype(np.uint16),
            )
    scan2 = ScanPaths.get_scan_folder_path(tag=tag(2), base_directory=base_dir)
    (scan2 / DEVICE).mkdir(parents=True)  # fixture acquisition
    (scan2 / "ScanInfoScan002.ini").write_text(
        '[Scan Info]\nScan No = "2"\nScan Parameter = "U_Motor:Position"\n'
        'Start = "1"\nEnd = "3"\nStep size = "1"\nShots per step = "2"\n'
    )
    yy, xx = np.mgrid[:24, :24]
    for shot in range(1, SHOTS + 1):
        image = 300 * np.exp(-((xx - 6 - shot) ** 2 + (yy - 12) ** 2) / 8)
        noise = rng.integers(0, 20, size=(24, 24))
        np.save(
            scan2 / DEVICE / f"Scan002_{DEVICE}_{shot:03d}.npy",
            (image + noise).astype(np.uint16),
        )
    analysis = scan2.parent.parent / "analysis"
    analysis.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "Shotnumber": range(1, SHOTS + 1),
            "Bin #": [1 + i // 2 for i in range(SHOTS)],
            "U_Motor Position Alias:motor": [1.0 + i // 2 for i in range(SHOTS)],
        }
    ).to_csv(analysis / "s2.txt", sep="\t", index=False)
    return scan2


def run(monkeypatch, base_dir: Path, doc) -> Path:
    scan = build_day(base_dir)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    analyzer = CoreScanAnalyzer(doc, id="Diag", priority=1)
    try:
        analyzer.run_analysis(tag(2))
    finally:
        analyzer.cleanup()
    return scan


def outputs(scan: Path) -> dict:
    """The analysis tree, less the cached backgrounds."""
    tree = snapshot_analysis_tree(scan.parent.parent / "analysis")
    return {k: v for k, v in tree.items() if "background" not in Path(k).name}


SOURCES = [
    pytest.param({"scan": 1, "statistic": "mean"}, id="dark-scan-mean"),
    pytest.param(
        {"statistic": "percentile", "percentile": 20.0},
        id="this-scan-p20",
    ),
    pytest.param({"statistic": "median"}, id="this-scan-median"),
]


@pytest.mark.parametrize("source", SOURCES)
def test_the_core_route_writes_the_golden_tree(
    tmp_path, monkeypatch, golden, request, source
):
    """The tree the retired legacy wrapper wrote with this background, exact (#1039)."""
    core = run(monkeypatch, tmp_path / "core", document(**source))
    new = outputs(core)
    expected = golden(f"backgrounds/{request.node.callspec.id}", new)
    assert any(name.endswith(".h5") for name in new)
    assert sorted(new) == sorted(expected.files)
    assert (
        compare_snapshots(expected.files, new, average_ulps=0, rtol=GOLDEN_RTOL) == []
    )


def test_a_dark_scan_is_averaged_once_and_shared(tmp_path, monkeypatch):
    doc = document(scan=1, statistic="mean")
    scan = run(monkeypatch, tmp_path, doc)
    cache = (
        scan.parent.parent
        / "analysis"
        / "Scan001"
        / DEVICE
        / f"{DEVICE}_background_avg.npy"
    )
    assert cache.is_file()

    def must_not_recompute(*args, **kwargs):
        raise AssertionError("recomputed a cached background")

    monkeypatch.setattr(core_backgrounds, "scan_statistic", must_not_recompute)
    analyzer = CoreScanAnalyzer(doc, id="Diag", priority=1)
    analyzer.run_analysis(tag(2))
    analyzer.cleanup()


def test_a_missing_dark_scan_fails_and_creates_nothing(tmp_path, monkeypatch):
    doc = document(scan=7, statistic="mean")
    scan = build_day(tmp_path)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    analyzer = CoreScanAnalyzer(doc, id="Diag", priority=1)
    with pytest.raises(FileNotFoundError, match="Scan007"):
        analyzer.run_analysis(tag(2))
    assert not (scan.parent / "Scan007").exists()
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert [p for p in after if "scans" in p.parts] == [
        p for p in before if "scans" in p.parts
    ]


def test_a_preview_uses_a_computed_background_and_never_computes_one(
    tmp_path, monkeypatch
):
    doc = document(statistic="percentile", percentile=20.0)
    scan = build_day(tmp_path)
    with pytest.raises(ScanContextRequired):
        prepare_document(doc, scan_folder=scan)
    with pytest.raises(ScanContextRequired):
        prepare_document(doc)
    assert not any((scan.parent.parent / "analysis").rglob("*.npy"))
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    analyzer = CoreScanAnalyzer(doc, id="Diag", priority=1)
    analyzer.run_analysis(tag(2))
    analyzer.cleanup()
    before = sorted((scan.parent.parent / "analysis").rglob("*"))
    prepared = prepare_document(doc, scan_folder=scan)
    assert "camera_background" in prepared.inputs
    assert sorted((scan.parent.parent / "analysis").rglob("*")) == before


def test_the_frames_are_the_devices_own_folder_as_the_legacy_wrapper_took_them(
    tmp_path,
):
    """A suffixed input folder (derived traces) still takes the raw camera's background."""
    from scan_analysis.core_backgrounds import background_cache_path

    data_dir = tmp_path / "day" / "scans" / "Scan002" / f"{DEVICE}-interp"
    request = ScanBackground("camera_background", 1, "mean")
    frames_dir, cache = background_cache_path(request, data_dir, DEVICE)
    assert frames_dir == tmp_path / "day" / "scans" / "Scan001" / DEVICE
    assert cache == (
        tmp_path
        / "day"
        / "analysis"
        / "Scan001"
        / DEVICE
        / f"{DEVICE}_background_avg.npy"
    )


def test_concurrent_computations_each_publish_a_whole_cache(tmp_path):
    """Two runs sharing a dark scan: both succeed, one complete file, no leftovers."""
    from concurrent.futures import ThreadPoolExecutor

    scan = build_day(tmp_path)
    request = ScanBackground("camera_background", 1, "mean")

    def resolve(_):
        return resolve_scan_background(
            request,
            data_dir=scan / DEVICE,
            device=DEVICE,
            file_tail=".npy",
            prefer_stack=False,
        )

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(resolve, range(4)))
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    folder = scan.parent.parent / "analysis" / "Scan001" / DEVICE
    assert [p.name for p in folder.iterdir()] == [f"{DEVICE}_background_avg.npy"]
    np.testing.assert_array_equal(np.load(next(folder.iterdir())), results[0])
