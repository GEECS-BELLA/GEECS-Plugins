"""The ``himg_to_stack`` kind: factory dispatch and the converter run."""

from __future__ import annotations

import importlib
import struct
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.io.himg import himg_bytes
from geecs_data_utils.io.scan_stack import find_stack_file, read_stack_timestamps
from geecs_data_utils.io.himg_stack import verify_himg_stack
from geecs_schemas.analysis import ANALYZER_SPECS, AnalysisDiagnostic

from scan_analysis import base
from scan_analysis.analyzers.common.himg_to_stack import HimgToStackAnalyzer
from scan_analysis.base import DataUnavailableWarning
from scan_analysis.config.diagnostic_factory import (
    SCAN_SCOPED_CLASS_PATHS,
    create_scan_analyzer,
)
from scan_analysis.core_analyzer import core_supports

TAG = ScanTag(year=2026, month=3, day=10, number=12, experiment="Test")
DEVICE = "U_HasoLift"
STAMPS = [3873135602.613, 3873135603.611, 3873135604.615]


def document(**scan) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": DEVICE,
            "analyzer": {"kind": "himg_to_stack"},
            "scan": {"priority": 5, **scan},
        }
    )


def _himg(seed: int) -> bytes:
    header = b"\x00" + struct.pack("<4I", 2, 6, 4, 8) + bytes([seed]) * 8
    rng = np.random.default_rng(seed)
    return himg_bytes(header, rng.integers(0, 16, (4, 6), dtype=np.uint16))


def build_scan(base_dir: Path, *, device_files=True, device_folder=True) -> Path:
    """A completed legacy-named HASO scan with its s-file and ScanInfo."""
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    scan.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan012.ini").write_text(
        '[Scan Info]\nScan No = "12"\nScan Parameter = "noscan"\n'
    )
    device = scan / DEVICE
    if device_folder:
        device.mkdir()
    if device_files:
        for shot in range(1, 4):
            (device / f"Scan012_{DEVICE}_{shot:03d}.himg").write_bytes(_himg(shot))
            (device / f"Scan012_{DEVICE}_{shot:03d}_raw.has").write_bytes(b"has")
    rows = pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3],
            "Bin #": [1, 1, 1],
            f"{DEVICE} acq_timestamp": STAMPS,
        }
    )
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    rows.to_csv(analysis / "s12.txt", sep="\t", index=False)
    return scan


def run(monkeypatch, base_dir: Path, doc=None):
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    analyzer = create_scan_analyzer(doc or document(), id="HasoStack", priority=1)
    try:
        return analyzer, analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()


class TestFactory:
    def test_the_kind_routes_to_the_converter(self):
        analyzer = create_scan_analyzer(document(), id="HasoStack")
        assert isinstance(analyzer, HimgToStackAnalyzer)
        assert analyzer.id == "HasoStack" and analyzer.priority == 5
        assert analyzer.device_name == DEVICE == analyzer.data_device_name
        assert analyzer.background_source is None
        assert analyzer.spec.kind == "himg_to_stack"

    def test_scan_device_names_the_data_folder(self):
        analyzer = create_scan_analyzer(document(device="U_HasoLift-Raw"))
        assert analyzer.device_name == DEVICE
        assert analyzer.data_device_name == "U_HasoLift-Raw"

    def test_the_kind_has_no_core_or_injected_route(self):
        assert core_supports(document()) is False
        with pytest.raises(ValueError, match="scan-scoped"):
            create_scan_analyzer(document(), route="core")
        with pytest.raises(ValueError, match="injected-data"):
            create_scan_analyzer(document(), use_injected_data=True)

    def test_every_scan_scoped_kind_has_a_scan_analysis_class(self):
        scan_scoped = {k for k, m in ANALYZER_SPECS.items() if m.scope == "scan"}
        assert scan_scoped == set(SCAN_SCOPED_CLASS_PATHS)
        for class_path in SCAN_SCOPED_CLASS_PATHS.values():
            module_path, class_name = class_path.rsplit(".", 1)
            cls = getattr(importlib.import_module(module_path), class_name)
            assert issubclass(cls, base.ScanAnalyzer)


class TestRun:
    def test_run_writes_the_stack_and_only_the_stack(self, tmp_path, monkeypatch):
        scan = build_scan(tmp_path)
        before = {p: p.read_bytes() for p in scan.rglob("*") if p.is_file()}
        analyzer, labels = run(monkeypatch, tmp_path)
        stack = find_stack_file(scan / DEVICE)
        assert stack == scan / DEVICE / f"{DEVICE}.h5"
        assert len(labels) == 1
        assert (
            labels[0].startswith(f"{DEVICE}.h5: 3 frames") and "verified" in labels[0]
        )
        after = {p: p.read_bytes() for p in scan.rglob("*") if p.is_file()}
        assert set(after) - set(before) == {stack}
        assert all(after[p] == data for p, data in before.items())
        np.testing.assert_array_equal(
            read_stack_timestamps(stack, labview_epoch=True), STAMPS
        )
        assert verify_himg_stack(stack, against_files=True).ok
        assert analyzer.last_report is None  # cleanup() ran

    def test_a_second_run_verifies_instead_of_rewriting(self, tmp_path, monkeypatch):
        scan = build_scan(tmp_path)
        run(monkeypatch, tmp_path)
        stack = scan / DEVICE / f"{DEVICE}.h5"
        written = stack.read_bytes()
        _, labels = run(monkeypatch, tmp_path)
        assert labels == [f"{DEVICE}.h5: already converted — 3 frames verified"]
        assert stack.read_bytes() == written

    def test_a_stack_short_of_the_folder_is_reported(self, tmp_path, monkeypatch):
        scan = build_scan(tmp_path)
        run(monkeypatch, tmp_path)
        (scan / DEVICE / f"Scan012_{DEVICE}_004.himg").write_bytes(_himg(4))
        with pytest.raises(RuntimeError, match="holds 3 frames but U_HasoLift holds 4"):
            run(monkeypatch, tmp_path)

    def test_missing_device_folder_is_no_data_and_stays_missing(
        self, tmp_path, monkeypatch
    ):
        scan = build_scan(tmp_path, device_files=False, device_folder=False)
        with pytest.raises(DataUnavailableWarning, match="No data directory"):
            run(monkeypatch, tmp_path)
        assert not (scan / DEVICE).exists()

    def test_empty_device_folder_is_no_data(self, tmp_path, monkeypatch):
        scan = build_scan(tmp_path, device_files=False)
        with pytest.raises(DataUnavailableWarning, match="no .himg files"):
            run(monkeypatch, tmp_path)
        assert list((scan / DEVICE).iterdir()) == []

    def test_missing_sfile_returns_none(self, tmp_path, monkeypatch):
        scan = build_scan(tmp_path)
        (scan.parent.parent / "analysis" / "s12.txt").unlink()
        _, result = run(monkeypatch, tmp_path)
        assert result is None
        assert find_stack_file(scan / DEVICE) is None
