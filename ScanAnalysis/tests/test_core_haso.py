"""HASO on the core route: the stack is the input, the host builds WaveKit, the store holds every shot."""

from __future__ import annotations

import importlib
import logging
import struct
import textwrap
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.io.himg import write_himg
from geecs_data_utils.io.himg_stack import convert_himg_folder
from geecs_data_utils.shot_files import StackMappingUnavailable
from geecs_schemas.analysis import AnalysisRecipe

import scan_analysis.base as base
from scan_analysis import core_services, core_workers
from scan_analysis.base import DataUnavailableWarning
from scan_analysis.core_analyzer import CoreScanAnalyzer
from scan_analysis.core_source import prepare_source, stack_required

TAG = ScanTag(year=2026, month=3, day=10, number=12, experiment="Test")
DEVICE = "U_HasoLift"
SENSOR = "WFS_HASO4_LIFT_680_8244_gain_enabled.dat"
SHOTS = 5
HEIGHT, WIDTH = 6, 8
STAMPS = [3952000000.0 + n for n in range(1, SHOTS + 1)]  # LabVIEW s

# The engine travels to spawned pool workers, which import its module by
# name, so it lives in a real module on sys.path (see test_core_frog).
HELPER = textwrap.dedent(
    '''
    """A picklable stand-in for HasoWaveKit."""
    import numpy as np


    class Result:
        def __init__(self, pixels, mask):
            total = float(pixels.astype(np.float64).sum())
            grid = np.arange(12, dtype=np.float32).reshape(3, 4)
            pupil = np.zeros((3, 4), dtype=bool)
            top, bottom, left, right = mask
            pupil[top:bottom, left:right] = True
            self.processed_phase = np.where(pupil, grid + total, np.nan).astype(np.float32)
            self.raw_phase = grid + total
            self.intensity = grid * 2 + total
            self.slopes_x = grid * 3
            self.slopes_y = grid * 4
            self.pupil = pupil


    class Engine:
        def __init__(self, header):
            self.header = header
            self.threads = None
            self.shared = []
            self.references = []

        def share_cores(self, workers):
            self.shared.append(workers)
            self.threads = max(1, 4 // workers)

        def compute(self, pixels, **parameters):
            self.references.append(parameters.get("reference"))
            return Result(pixels, parameters["mask"])


    class OddEngine(Engine):
        """Shot 3 (pixels all 3) comes back on a smaller grid."""

        def compute(self, pixels, **parameters):
            result = Result(pixels, parameters["mask"])
            if int(pixels.max()) == 3:
                for key in ("processed_phase", "raw_phase", "intensity", "slopes_x", "slopes_y"):
                    setattr(result, key, getattr(result, key)[:2])
                result.pupil = result.pupil[:2]
            return result
    '''
)


def _header() -> bytes:
    blob = b"\x05" * 10
    return b"\x00" + struct.pack("<4I", 2, WIDTH, HEIGHT, len(blob)) + blob


def _pixels(shot: int) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), shot, dtype=np.uint16)


def recipe(**overrides) -> AnalysisRecipe:
    return AnalysisRecipe.model_validate(
        {
            "device": DEVICE,
            "input": {"kind": "camera", "file_tail": ".himg", "format": "device_hdf5"},
            "measure": {
                "kind": "haso",
                "sensor_config": SENSOR,
                "mask": {"top": 1, "bottom": 3, "left": 0, "right": 4},
            },
            "scan": {"save": True},
            **overrides,
        }
    )


def build_scan(base_dir: Path, *, convert: bool = True) -> Path:
    """A scan of natively named .himg files, with (or without) its stack."""
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    device = scan / DEVICE
    device.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan012.ini").write_text(
        '[Scan Info]\nScan No = "12"\nScan Parameter = "noscan"\n'
    )
    for shot, stamp in enumerate(STAMPS, start=1):
        write_himg(device / f"{DEVICE}_{stamp:.3f}.himg", _header(), _pixels(shot))
    if convert:
        convert_himg_folder(device)
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    pd.DataFrame(
        {
            "Shotnumber": range(1, SHOTS + 1),
            "Bin #": [1] * SHOTS,
            f"{DEVICE} acq_timestamp": STAMPS,
        }
    ).to_csv(analysis / "s12.txt", sep="\t", index=False)
    return scan


@pytest.fixture
def helper(tmp_path, monkeypatch):
    (tmp_path / "haso_host_helper.py").write_text(HELPER)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    module = importlib.import_module("haso_host_helper")
    built = []

    def factory(data_dir):
        engine = module.Engine(core_services.stack_header(stack_required(data_dir)))
        built.append(engine)
        return engine

    monkeypatch.setitem(core_services.SERVICE_FACTORIES, "haso", factory)
    module.built = built
    yield module
    monkeypatch.delitem(importlib.sys.modules, "haso_host_helper", raising=False)


def run(monkeypatch, base_dir: Path, doc, *, convert: bool = True) -> Path:
    scan = build_scan(base_dir, convert=convert)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    analyzer = CoreScanAnalyzer(doc, id=DEVICE, priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    return scan


def test_scalars_persist_and_every_shot_lands_in_the_wavefront_store(
    tmp_path, monkeypatch, helper
):
    scan = run(monkeypatch, tmp_path, recipe())
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s12.txt", sep="\t")
    assert (
        f"{DEVICE}_phase_rms" in rows.columns and f"{DEVICE}_phase_pv" in rows.columns
    )
    # The engine was built from the stack's own header, once, and told the
    # pool width (serial: one worker).
    (engine,) = helper.built
    assert engine.header == _header()
    assert engine.shared == [1]
    store = (
        scan.parent.parent
        / "analysis"
        / "Scan012"
        / DEVICE
        / "Array2DScanAnalyzer"
        / f"{DEVICE}_wavefront.h5"
    )
    assert store.is_file() and not store.with_name(store.name + ".part").exists()
    with h5py.File(store) as f:
        assert f["shots"][:].tolist() == list(range(1, SHOTS + 1))
        assert f["frame"].shape == (SHOTS, 3, 4) and f["frame"].dtype == np.float32
        assert f["frame"].chunks == (1, 3, 4) and f["frame"].compression == "gzip"
        assert sorted(f["extras"]) == [
            "intensity",
            "pupil",
            "raw_phase",
            "slopes_x",
            "slopes_y",
        ]
        for shot in range(1, SHOTS + 1):
            expected = helper.Result(_pixels(shot), (1, 3, 0, 4))
            np.testing.assert_array_equal(
                f["frame"][shot - 1], expected.processed_phase
            )
            np.testing.assert_array_equal(
                f["extras/intensity"][shot - 1], expected.intensity
            )
            np.testing.assert_array_equal(f["extras/pupil"][shot - 1], expected.pupil)
    # No file was written beside the raw data: the stack and the .himg only.
    assert (
        sorted(p.suffix for p in (scan / DEVICE).iterdir())
        == [".h5"] + [".himg"] * SHOTS
    )


def test_a_reference_scan_is_averaged_from_its_stack_and_handed_to_every_shot(
    tmp_path, monkeypatch, helper
):
    # The scan is its own reference here: the mean of shots 1..5 is 3.
    doc = recipe(
        inputs={"probe": {"from_scan": {"scan": 12, "statistic": "mean"}}},
        measure={
            "kind": "haso",
            "sensor_config": SENSOR,
            "mask": {"top": 1, "bottom": 3, "left": 0, "right": 4},
            "reference": "probe",
        },
    )
    run(monkeypatch, tmp_path, doc)
    (engine,) = helper.built
    assert len(engine.references) == SHOTS
    for reference in engine.references:
        assert reference.dtype == np.uint16
        np.testing.assert_array_equal(reference, np.full((HEIGHT, WIDTH), 3))


def test_an_unconverted_scan_is_refused_naming_the_converter(
    tmp_path, monkeypatch, helper
):
    scan = build_scan(tmp_path, convert=False)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    analyzer = CoreScanAnalyzer(recipe(), id=DEVICE, priority=1)
    with pytest.raises(DataUnavailableWarning, match="HasoLift_stack") as refused:
        analyzer.run_analysis(TAG)
    assert "himg_to_stack" in str(refused.value)
    assert helper.built == []
    assert not (scan.parent.parent / "analysis" / "Scan012" / DEVICE).exists()
    assert sorted(p.suffix for p in (scan / DEVICE).iterdir()) == [".himg"] * SHOTS


def test_the_source_reads_himg_only_through_the_stack(tmp_path):
    """Any camera recipe over .himg files: the per-shot files are never mapped."""
    scan = build_scan(tmp_path, convert=False)
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s12.txt", sep="\t")
    plain = recipe(
        measure={"kind": "none"}, input={"kind": "camera", "file_tail": ".himg"}
    )
    with pytest.raises(StackMappingUnavailable, match="stack only"):
        prepare_source(plain, scan, rows)
    convert_himg_folder(scan / DEVICE)
    source = prepare_source(plain, scan, rows)
    assert sorted(source.references) == list(range(1, SHOTS + 1))
    np.testing.assert_array_equal(source.load(3), _pixels(3))


def _store_of(scan: Path) -> Path:
    return (
        scan.parent.parent
        / "analysis"
        / "Scan012"
        / DEVICE
        / "Array2DScanAnalyzer"
        / f"{DEVICE}_wavefront.h5"
    )


def test_a_disagreeing_shot_discards_the_store_and_the_run_goes_on(
    tmp_path, monkeypatch, helper, caplog
):
    """A store failure costs the store, never the run: scalars for every shot, no file."""
    monkeypatch.setitem(
        core_services.SERVICE_FACTORIES,
        "haso",
        lambda data_dir: helper.OddEngine(
            core_services.stack_header(stack_required(data_dir))
        ),
    )
    with caplog.at_level(logging.WARNING, logger="scan_analysis.core_analyzer"):
        scan = run(monkeypatch, tmp_path, recipe())
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s12.txt", sep="\t")
    assert rows[f"{DEVICE}_phase_rms"].notna().sum() == SHOTS
    store = _store_of(scan)
    assert not store.exists() and not store.with_name(store.name + ".part").exists()
    assert any("shot store not written" in r.getMessage() for r in caplog.records)


def test_a_store_that_cannot_be_renamed_keeps_the_scalars(
    tmp_path, monkeypatch, helper, caplog
):
    """The final flush/rename is a product write: an OSError there loses only the store."""
    import os

    from scan_analysis import core_sink

    real_replace = os.replace

    def refuse(src, dst):
        if str(dst).endswith("_wavefront.h5"):
            raise PermissionError(f"{dst}: sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(core_sink.os, "replace", refuse)
    with caplog.at_level(logging.WARNING, logger="scan_analysis.core_analyzer"):
        scan = run(monkeypatch, tmp_path, recipe())
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s12.txt", sep="\t")
    assert rows[f"{DEVICE}_phase_rms"].notna().sum() == SHOTS
    store = _store_of(scan)
    assert not store.exists() and not store.with_name(store.name + ".part").exists()
    assert any("not kept" in r.getMessage() for r in caplog.records)


def test_a_leftover_part_file_refuses_the_store_not_the_run(
    tmp_path, monkeypatch, helper, caplog
):
    scan = build_scan(tmp_path)
    part = _store_of(scan).with_name(f"{DEVICE}_wavefront.h5.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"a run died here")
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    analyzer = CoreScanAnalyzer(recipe(), id=DEVICE, priority=1)
    with caplog.at_level(logging.WARNING, logger="scan_analysis.core_analyzer"):
        try:
            analyzer.run_analysis(TAG)
        finally:
            analyzer.cleanup()
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s12.txt", sep="\t")
    assert rows[f"{DEVICE}_phase_rms"].notna().sum() == SHOTS
    # The stale part is left for a human; nothing else was written; the
    # operator's one signal is an ERROR record naming the file.
    assert part.read_bytes() == b"a run died here" and not _store_of(scan).exists()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and str(part) in errors[0].getMessage()


def test_save_false_stores_nothing_but_keeps_the_scalars(tmp_path, monkeypatch, helper):
    scan = run(monkeypatch, tmp_path, recipe(scan={"save": False}))
    analysis = scan.parent.parent / "analysis"
    rows = pd.read_csv(analysis / "s12.txt", sep="\t")
    assert f"{DEVICE}_phase_rms" in rows.columns
    written = sorted(p.name for p in analysis.rglob("*") if p.is_file())
    assert written == ["Scan012_U_HasoLift.txt", "s12.txt"]


def test_a_pooled_run_stores_what_the_serial_run_stores(
    tmp_path, monkeypatch, helper, caplog
):
    monkeypatch.setattr(core_workers, "MIN_UNITS_FOR_POOL", 1)
    monkeypatch.setattr(CoreScanAnalyzer, "worker_cap", 2)
    serial_scan = run(monkeypatch, tmp_path / "serial", recipe())
    with caplog.at_level(logging.INFO, logger="scan_analysis.core_analyzer"):
        pooled_scan = run(monkeypatch, tmp_path / "pooled", recipe(scan={"workers": 2}))
    assert any(r.getMessage().endswith("units, 2 workers") for r in caplog.records)
    assert helper.built[-1].shared == [2]

    def store_of(scan: Path) -> dict[str, np.ndarray]:
        path = (
            scan.parent.parent
            / "analysis"
            / "Scan012"
            / DEVICE
            / "Array2DScanAnalyzer"
            / f"{DEVICE}_wavefront.h5"
        )
        with h5py.File(path) as f:
            return {
                "shots": f["shots"][:],
                "frame": f["frame"][:],
                **{f"extras/{k}": f[f"extras/{k}"][:] for k in f["extras"]},
            }

    serial, pooled = store_of(serial_scan), store_of(pooled_scan)
    assert serial.keys() == pooled.keys()
    for key in serial:
        np.testing.assert_array_equal(serial[key], pooled[key])


def test_a_context_free_preview_cannot_build_the_engine():
    """The real factory: no scan folder, no header, no engine — before any config is read."""
    from geecs_analysis.measures.haso import HasoSpec

    with pytest.raises(LookupError, match="no scan folder"):
        core_services.services_for(HasoSpec(sensor_config=SENSOR))
