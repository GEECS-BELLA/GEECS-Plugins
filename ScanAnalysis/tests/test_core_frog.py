"""FROG on the core route: the host builds the retriever, scalars persist, lineouts sit beside each shot."""

from __future__ import annotations

import importlib
import textwrap
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.io.scan_stack import ShotRef
from geecs_schemas.analysis import AnalysisDiagnostic

import scan_analysis.base as base
from geecs_analysis.compat.convert import to_v3
from geecs_analysis.measures.beam import BeamSpec
from geecs_analysis.measures.frog import FrogSpec
from scan_analysis import core_services, core_workers
from scan_analysis.core_analyzer import CoreScanAnalyzer, core_supports
from scan_analysis.core_sink import shot_table_path

TAG = ScanTag(year=2026, month=1, day=1, number=1, experiment="Test")
SHOTS = 6
LEGACY_COLUMNS = [
    "time_fs",
    "temporal_intensity",
    "temporal_phase",
    "wavelength_nm",
    "spectral_intensity",
    "spectral_phase",
]

# The retriever travels to spawned pool workers, which import its module by
# name, so it lives in a real module on sys.path (see GEECS-Analysis's
# test_v2_pool). Its time and wavelength grids differ in length, as the
# DLL's do, so the table's padding is exercised.
HELPER = textwrap.dedent(
    '''
    """A picklable stand-in for FrogDllRetrieval."""
    import numpy as np


    class Result:
        def __init__(self, trace):
            total = float(trace.sum())
            self.temporal_fwhm = total % 97.0
            self.spectral_fwhm = float(trace.max())
            self.frog_error = 1.0 / (1.0 + total)
            self.num_iterations = 11
            self.retrieved_trace = trace[:6, :6] * 2.0
            self.time = np.linspace(-10.0, 10.0, 5)
            self.temporal_intensity = np.linspace(0.0, 1.0, 5) + total
            self.temporal_phase = np.zeros(5)
            self.wavelength = np.linspace(399.0, 401.0, 8)
            self.spectral_intensity = np.ones(8)
            self.spectral_phase = np.linspace(-1.0, 1.0, 8)
            self.tw_per_joule = 2.5


    class Retriever:
        def retrieve_pulse(self, trace, **parameters):
            return Result(trace)
    '''
)


@pytest.fixture
def helper(tmp_path, monkeypatch):
    (tmp_path / "frog_host_helper.py").write_text(HELPER)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    module = importlib.import_module("frog_host_helper")
    monkeypatch.setitem(core_services.SERVICE_FACTORIES, "frog", module.Retriever)
    yield module
    monkeypatch.delitem(importlib.sys.modules, "frog_host_helper", raising=False)


def document(**scan) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Camera",
            "analyzer": {"kind": "frog_retrieval", "max_iterations": 300},
            "image": {"type": "camera", "pipeline": []},
            "scan": {
                "mode": "per_shot",
                "file_tail": ".npy",
                "save": False,
                **scan,
            },
        }
    )


def build_scan(base_dir: Path) -> Path:
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    device = scan / "Camera"
    device.mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan001.ini").write_text(
        '[Scan Info]\nScan No = "1"\nScan Parameter = "noscan"\n'
    )
    for shot in range(1, SHOTS + 1):
        np.save(
            device / f"Scan001_Camera_{shot:03d}.npy",
            np.full((10, 12), shot, dtype=np.uint16),
        )
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    pd.DataFrame({"Shotnumber": range(1, SHOTS + 1), "Bin #": [1] * SHOTS}).to_csv(
        analysis / "s1.txt", sep="\t", index=False
    )
    return scan


def run(monkeypatch, base_dir: Path, doc) -> Path:
    scan = build_scan(base_dir)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=base_dir))
    analyzer = CoreScanAnalyzer(doc, id="Camera", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    return scan


def test_a_frog_diagnostic_runs_on_the_core_route():
    assert core_supports(document())


def test_scalars_persist_and_each_shot_gets_its_legacy_lineout_table(
    tmp_path, monkeypatch, helper
):
    scan = run(monkeypatch, tmp_path, document())
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s1.txt", sep="\t")
    for key in (
        "temporal_fwhm",
        "spectral_fwhm",
        "frog_error",
        "frog_iterations",
        "tw_per_joule",
    ):
        assert f"Camera_{key}" in rows.columns
    assert rows["Camera_frog_iterations"].tolist() == [11.0] * SHOTS
    np.testing.assert_allclose(
        rows["Camera_spectral_fwhm"], np.arange(1, SHOTS + 1, dtype=float)
    )
    device = scan / "Camera"
    for shot in range(1, SHOTS + 1):
        table = pd.read_csv(
            device / f"Scan001_Camera_{shot:03d}_retrieved_lineouts.tsv",
            sep="\t",
            float_precision="round_trip",
        )
        assert list(table.columns) == LEGACY_COLUMNS
        assert len(table) == 8
        expected = helper.Result(np.full((10, 12), shot, dtype=np.float64))
        np.testing.assert_array_equal(table["time_fs"][:5], expected.time)
        assert table["time_fs"][5:].isna().all()
        np.testing.assert_array_equal(
            table["temporal_intensity"][:5], expected.temporal_intensity
        )
        np.testing.assert_array_equal(table["wavelength_nm"], expected.wavelength)
    # save: false writes no products; the only analysis files are the scalars.
    written = sorted(
        p.relative_to(scan.parent.parent / "analysis").as_posix()
        for p in (scan.parent.parent / "analysis").rglob("*")
        if p.is_file()
    )
    assert written == ["Scan001/Scan001_Camera.txt", "s1.txt"]


def test_a_pooled_frog_run_writes_what_the_serial_run_writes(
    tmp_path, monkeypatch, helper, caplog
):
    import logging

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

    def outputs(scan: Path) -> dict[str, str]:
        root = scan.parent.parent
        return {
            p.relative_to(root).as_posix(): p.read_text()
            for p in sorted(root.rglob("*.t*"))
            if p.suffix in {".tsv", ".txt"}
        }

    assert outputs(serial_scan) == outputs(pooled_scan)
    assert len([k for k in outputs(pooled_scan) if k.endswith(".tsv")]) == SHOTS


def test_services_are_built_only_for_a_measure_that_names_one(monkeypatch):
    built = []
    monkeypatch.setitem(
        core_services.SERVICE_FACTORIES, "frog", lambda: built.append(1) or "retriever"
    )
    assert core_services.services_for(BeamSpec()) == {}
    assert core_services.services_for(FrogSpec()) == {"frog": "retriever"}
    assert built == [1]
    monkeypatch.delitem(core_services.SERVICE_FACTORIES, "frog")
    with pytest.raises(LookupError, match="does not provide"):
        core_services.services_for(FrogSpec())


def test_a_host_without_the_dll_fails_when_the_run_is_prepared(tmp_path, monkeypatch):
    def unconfigured():
        raise FileNotFoundError("frog_dll_path not found in config.ini")

    monkeypatch.setitem(core_services.SERVICE_FACTORIES, "frog", unconfigured)
    scan = build_scan(tmp_path)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    analyzer = CoreScanAnalyzer(document(), id="Camera", priority=1)
    with pytest.raises(FileNotFoundError, match="frog_dll_path"):
        analyzer.run_analysis(TAG)
    assert not list((scan / "Camera").glob("*.tsv"))


def test_shot_table_paths_sit_beside_the_file_or_the_stack():
    assert shot_table_path(Path("/d/Dev/Scan001_Dev_004.png"), 4, "t") == Path(
        "/d/Dev/Scan001_Dev_004_t.tsv"
    )
    assert shot_table_path(ShotRef("/d/Dev/Dev.h5", 3), 4, "t") == Path(
        "/d/Dev/Dev_004_t.tsv"
    )
    with pytest.raises(ValueError):
        shot_table_path(Path("/d/x.png"), 1, "../escape")


def test_a_per_request_view_never_starts_the_dll_but_the_explicit_preview_does(
    monkeypatch, helper
):
    from geecs_analysis.compat.v2 import UnsupportedRecipe

    from scan_analysis.core_inputs import ServicesNotRequested
    from scan_analysis.core_preview import prepare_document, preview_frame

    built = []

    def factory():
        built.append(1)
        return helper.Retriever()

    monkeypatch.setitem(core_services.SERVICE_FACTORIES, "frog", factory)
    # The portal's shot browser prepares without services: refused as an
    # unported recipe, so it keeps its old route (which refuses FROG too).
    with pytest.raises(ServicesNotRequested) as refused:
        prepare_document(document())
    assert isinstance(refused.value, UnsupportedRecipe)
    assert built == []
    # The editor's preview is an explicit request: the retrieval runs.
    figure = preview_frame(document(), np.full((10, 12), 3, dtype=np.uint16))
    assert built == [1]
    assert figure.axes
