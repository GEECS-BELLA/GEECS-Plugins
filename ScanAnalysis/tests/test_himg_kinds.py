"""The scan-scoped ``.himg`` kinds: factory dispatch, the converter, compaction and restore runs."""

from __future__ import annotations

import importlib
import os
import struct
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.io.himg import himg_bytes
from geecs_data_utils.io.himg_compact import (
    MANIFEST_NAME,
    HimgFolderActive,
    manifest_path_for,
)
from geecs_data_utils.io.himg_stack import list_himg_files, verify_himg_stack
from geecs_data_utils.io.scan_stack import find_stack_file, read_stack_timestamps
from geecs_schemas.analysis import ANALYZER_SPECS, AnalysisDiagnostic

from scan_analysis import base
from scan_analysis.analyzers.common.himg_kinds import (
    HimgCompactAnalyzer,
    HimgRestoreAnalyzer,
    HimgToStackAnalyzer,
)
from scan_analysis.base import DataUnavailableWarning
from scan_analysis.config.diagnostic_factory import (
    SCAN_SCOPED_CLASS_PATHS,
    DestructiveKindRefused,
    create_scan_analyzer,
)
from scan_analysis.task_queue import load_analyzers_from_config
from scan_analysis.core_analyzer import core_supports

TAG = ScanTag(year=2026, month=3, day=10, number=12, experiment="Test")
DEVICE = "U_HasoLift"
STAMPS = [3873135602.613, 3873135603.611, 3873135604.615]
OLD = time.time() - 3600  # past the compaction age guard


def document(kind: str = "himg_to_stack", **scan) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": DEVICE,
            "analyzer": {"kind": kind},
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
            path = device / f"Scan012_{DEVICE}_{shot:03d}.himg"
            path.write_bytes(_himg(shot))
            os.utime(path, (OLD, OLD))
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


def run(monkeypatch, base_dir: Path, doc=None, *, progress=None):
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    # The host role: a destructive kind is built only with the opt-in the
    # portal passes after its typed-scan-number check.
    analyzer = create_scan_analyzer(
        doc or document(), id="HasoStack", priority=1, allow_destructive=True
    )
    analyzer.progress = progress
    try:
        return analyzer, analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()


def sources(device_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in list_himg_files(device_dir)}


class TestFactory:
    def test_the_kinds_route_to_their_classes(self):
        analyzer = create_scan_analyzer(document(), id="HasoStack")
        assert isinstance(analyzer, HimgToStackAnalyzer)
        assert analyzer.id == "HasoStack" and analyzer.priority == 5
        assert analyzer.device_name == DEVICE == analyzer.data_device_name
        assert analyzer.background_source is None
        assert analyzer.spec.kind == "himg_to_stack"
        assert isinstance(
            create_scan_analyzer(document("himg_compact"), allow_destructive=True),
            HimgCompactAnalyzer,
        )
        assert isinstance(
            create_scan_analyzer(document("himg_restore")), HimgRestoreAnalyzer
        )

    def test_scan_device_names_the_data_folder(self):
        analyzer = create_scan_analyzer(document(device="U_HasoLift-Raw"))
        assert analyzer.device_name == DEVICE
        assert analyzer.data_device_name == "U_HasoLift-Raw"

    @pytest.mark.parametrize("kind", ["himg_to_stack", "himg_compact", "himg_restore"])
    def test_the_kinds_have_no_core_or_injected_route(self, kind):
        assert core_supports(document(kind)) is False
        with pytest.raises(ValueError, match="scan-scoped"):
            create_scan_analyzer(document(kind), route="core", allow_destructive=True)
        with pytest.raises(ValueError, match="injected-data"):
            create_scan_analyzer(
                document(kind), use_injected_data=True, allow_destructive=True
            )

    def test_every_scan_scoped_kind_has_a_scan_analysis_class(self):
        scan_scoped = {k for k, m in ANALYZER_SPECS.items() if m.scope == "scan"}
        assert scan_scoped == set(SCAN_SCOPED_CLASS_PATHS)
        for class_path in SCAN_SCOPED_CLASS_PATHS.values():
            module_path, class_name = class_path.rsplit(".", 1)
            cls = getattr(importlib.import_module(module_path), class_name)
            assert issubclass(cls, base.ScanAnalyzer)

    def test_only_the_compaction_is_destructive(self):
        assert document("himg_compact").destructive is True
        assert document("himg_to_stack").destructive is False
        assert document("himg_restore").destructive is False


class TestConvert:
    def test_run_writes_the_stack_and_only_the_stack(self, tmp_path, monkeypatch):
        scan = build_scan(tmp_path)
        before = {p: p.read_bytes() for p in scan.rglob("*") if p.is_file()}
        phases: list[tuple[int, int, str]] = []
        analyzer, labels = run(
            monkeypatch, tmp_path, progress=lambda *p: phases.append(p)
        )
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
        # The child's frames done/total reached the host, phase by phase.
        assert (3, 3, "writing") in phases and (3, 3, "verifying") in phases

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


class TestCompactAndRestore:
    def test_compact_deletes_after_verifying_and_restore_brings_the_bytes_back(
        self, tmp_path, monkeypatch
    ):
        scan = build_scan(tmp_path)
        device_dir = scan / DEVICE
        run(monkeypatch, tmp_path)  # convert
        originals = sources(device_dir)
        outside_before = {
            p: p.read_bytes()
            for p in scan.parent.parent.rglob("*")
            if p.is_file() and device_dir not in p.parents
        }
        stack = device_dir / f"{DEVICE}.h5"
        stack_bytes = stack.read_bytes()
        phases: list[tuple[int, int, str]] = []

        _, labels = run(
            monkeypatch,
            tmp_path,
            document("himg_compact"),
            progress=lambda *p: phases.append(p),
        )

        assert len(labels) == 1
        assert "3 .himg files verified against U_HasoLift.h5 and deleted" in labels[0]
        assert "4 files /" in labels[0] and "-> 1 file /" in labels[0]
        assert list_himg_files(device_dir) == []
        assert manifest_path_for(device_dir).is_file()
        assert stack.read_bytes() == stack_bytes  # headers kept
        assert [p[2] for p in phases] == ["verifying"] * 3 + ["deleting"] * 3
        # The sidecars and everything outside the device folder are untouched.
        assert sorted(p.name for p in device_dir.iterdir()) == sorted(
            [f"{DEVICE}.h5", MANIFEST_NAME]
            + [f"Scan012_{DEVICE}_{s:03d}_raw.has" for s in (1, 2, 3)]
        )
        outside_after = {
            p: p.read_bytes()
            for p in scan.parent.parent.rglob("*")
            if p.is_file() and device_dir not in p.parents
        }
        assert outside_after == outside_before

        _, labels = run(monkeypatch, tmp_path, document("himg_restore"))
        assert "3 .himg files restored from U_HasoLift.h5, verified" in labels[0]
        assert sources(device_dir) == originals
        assert not manifest_path_for(device_dir).exists()

    def test_compact_refuses_a_scan_that_may_still_be_running(
        self, tmp_path, monkeypatch
    ):
        scan = build_scan(tmp_path)
        run(monkeypatch, tmp_path)
        young = list_himg_files(scan / DEVICE)[0]
        now = time.time()
        os.utime(young, (now, now))
        with pytest.raises(HimgFolderActive, match="may still be running"):
            run(monkeypatch, tmp_path, document("himg_compact"))
        assert len(list_himg_files(scan / DEVICE)) == 3

    def test_compact_without_a_stack_names_the_converter(self, tmp_path, monkeypatch):
        scan = build_scan(tmp_path)
        with pytest.raises(RuntimeError, match="run the himg_to_stack kind first"):
            run(monkeypatch, tmp_path, document("himg_compact"))
        assert len(list_himg_files(scan / DEVICE)) == 3

    def test_compact_twice_is_a_label_not_an_error(self, tmp_path, monkeypatch):
        build_scan(tmp_path)
        run(monkeypatch, tmp_path)
        run(monkeypatch, tmp_path, document("himg_compact"))
        _, labels = run(monkeypatch, tmp_path, document("himg_compact"))
        assert labels[0].startswith(f"{DEVICE}: already compacted — 3 frames")

    def test_restore_without_a_stack_fails_plainly(self, tmp_path, monkeypatch):
        build_scan(tmp_path)
        with pytest.raises(RuntimeError, match="nothing to restore from"):
            run(monkeypatch, tmp_path, document("himg_restore"))

    def test_missing_device_folder_is_no_data_for_both(self, tmp_path, monkeypatch):
        scan = build_scan(tmp_path, device_files=False, device_folder=False)
        for kind in ("himg_compact", "himg_restore"):
            with pytest.raises(DataUnavailableWarning, match="No data directory"):
                run(monkeypatch, tmp_path, document(kind))
        assert not (scan / DEVICE).exists()


class TestDestructiveGate:
    """A kind that deletes data files is built only for a host that confirmed it."""

    def test_the_factory_refuses_without_the_opt_in(self):
        with pytest.raises(DestructiveKindRefused, match="deletes data files"):
            create_scan_analyzer(document("himg_compact"))
        assert isinstance(
            create_scan_analyzer(document("himg_compact"), allow_destructive=True),
            HimgCompactAnalyzer,
        )
        # The non-destructive kinds never needed the opt-in.
        assert isinstance(
            create_scan_analyzer(document("himg_restore")), HimgRestoreAnalyzer
        )
        assert isinstance(
            create_scan_analyzer(document("himg_to_stack")), HimgToStackAnalyzer
        )

    def test_the_group_loader_skips_it_with_a_reason(self, tmp_path, caplog):
        """The post-scan queue runs unasked: a group naming the compaction gets everything else."""
        analyzers = tmp_path / "analyzers" / "HTU"
        analyzers.mkdir(parents=True)
        for name, kind in (
            ("HasoLift_compact", "himg_compact"),
            ("HasoLift_stack", "himg_to_stack"),
        ):
            (analyzers / f"{name}.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 2,
                        "name": DEVICE,
                        "output_name": name,
                        "analyzer": {"kind": kind},
                        "scan": {"priority": 5},
                    }
                )
            )
        groups = tmp_path / "groups" / "HTU"
        groups.mkdir(parents=True)
        (groups / "haso.yaml").write_text(
            yaml.safe_dump(
                {"name": "haso", "analyzers": ["HasoLift_compact", "HasoLift_stack"]}
            )
        )
        caplog.set_level("WARNING")
        loaded = load_analyzers_from_config("haso", config_dir=tmp_path)
        assert [type(a) for a in loaded] == [HimgToStackAnalyzer]
        assert "skipping HasoLift_compact" in caplog.text
        assert "destructive kind" in caplog.text
