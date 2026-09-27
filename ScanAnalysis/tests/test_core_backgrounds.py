"""Scan backgrounds on the core route: exact strip statistics, the legacy wrapper's outputs, no stray writes."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisDiagnostic

import scan_analysis.base as base
import scan_analysis.core_backgrounds as core_backgrounds
from geecs_analysis.compat.v2 import ScanBackground
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)
from scan_analysis.config import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer, core_supports
from scan_analysis.core_backgrounds import resolve_scan_background, scan_statistic
from scan_analysis.core_inputs import ScanContextRequired
from scan_analysis.core_preview import prepare_document
from scan_analysis.route_compare import compare_snapshots, snapshot_analysis_tree

DEVICE = "Camera"
SHOTS = 6


def stack(n: int = 7, shape=(13, 9), seed: int = 2) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 4000, size=shape).astype(np.uint16) for _ in range(n)]


@pytest.mark.parametrize(
    "statistic,percentile",
    [("mean", None), ("median", None), ("percentile", 10.0), ("percentile", 37.5)],
)
@pytest.mark.parametrize("budget", [1, 9 * 8 * 7 * 2, 1 << 30])
def test_the_strip_statistic_is_the_legacy_whole_stack_one_bit_for_bit(
    statistic, percentile, budget, tmp_path
):
    """Tiny budgets force one-row strips over the on-disk copy; a big one stays in memory."""
    from image_analysis.processing.array2d.background import _aggregate_image_stack

    frames = stack()
    expected = _aggregate_image_stack(frames, method=statistic, percentile=percentile)
    loaders = [lambda f=f: f for f in frames]
    actual = scan_statistic(
        loaders, statistic, percentile, budget_bytes=budget, scratch_dir=tmp_path
    )
    np.testing.assert_array_equal(actual, expected)
    assert list(tmp_path.iterdir()) == []  # the scratch copy is gone


def test_an_unreadable_frame_is_skipped_and_a_mismatched_one_refused():
    from image_analysis.processing.array2d.background import _aggregate_image_stack

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
        _aggregate_image_stack(frames, method="percentile", percentile=20.0),
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


def document(**source) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": DEVICE,
            "output_name": "Diag",
            "analyzer": {"kind": "beam"},
            "image": {
                "type": "camera",
                "background": {"method": "constant", "constant_level": 2.0},
                "pipeline": ["background"],
            },
            "scan": {
                "mode": "per_shot",
                "file_tail": ".npy",
                "renderer": {"dpi": 30},
                "background_source": source,
            },
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


def run(monkeypatch, base_dir: Path, doc, route: str = "core") -> Path:
    scan = build_day(base_dir)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    if route == "legacy":
        analyzer = create_scan_analyzer(doc, id="Diag", priority=1, route="legacy")
        assert isinstance(analyzer, SingleDeviceScanAnalyzer)
        # The legacy wrapper resolves the dark scan through ScanPaths itself.
        import scan_analysis.analyzers.common.single_device_scan_analyzer as sdsa
        import geecs_data_utils

        monkeypatch.setattr(
            geecs_data_utils, "ScanPaths", partial(ScanPaths, base_directory=base_dir)
        )
        assert sdsa  # imported for its lazy ScanPaths lookup
    else:
        analyzer = CoreScanAnalyzer(doc, id="Diag", priority=1)
    try:
        analyzer.run_analysis(tag(2))
    finally:
        analyzer.cleanup()
    return scan


def outputs(scan: Path) -> dict:
    """The analysis tree, less the cached backgrounds (named per route)."""
    tree = snapshot_analysis_tree(scan.parent.parent / "analysis")
    return {k: v for k, v in tree.items() if "background" not in Path(k).name}


SOURCES = [
    pytest.param({"scan_number": 1}, id="dark-scan-mean"),
    pytest.param(
        {"from_current_scan": {"method": "percentile", "percentile": 20}},
        id="this-scan-p20",
    ),
    pytest.param({"from_current_scan": {"method": "median"}}, id="this-scan-median"),
]


@pytest.mark.parametrize("source", SOURCES)
def test_the_core_route_matches_the_legacy_wrapper(tmp_path, monkeypatch, source):
    doc = document(**source)
    assert core_supports(doc)
    # Each route gets its own document: the legacy wrapper rewrites the
    # background section of the one it is given (to a from_file background
    # pointing at its cache), which would feed the core a pre-resolved file.
    legacy = run(monkeypatch, tmp_path / "legacy", document(**source), "legacy")
    core = run(monkeypatch, tmp_path / "core", doc)
    old, new = outputs(legacy), outputs(core)
    assert any(name.endswith(".h5") for name in new)
    assert sorted(old) == sorted(new)
    assert compare_snapshots(old, new) == []


def test_a_dark_scan_is_averaged_once_and_shared(tmp_path, monkeypatch):
    doc = document(scan_number=1)
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
    doc = document(scan_number=7)
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
    doc = document(from_current_scan={"method": "percentile", "percentile": 20})
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
