"""magspec on the core route: the legacy analyzer's scalars, products and -interp files, byte for byte."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_analysis.compat.convert import to_v3
from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.config_roots import scan_analysis_config
from geecs_data_utils.io.images import write_png16
from geecs_schemas.analysis import AnalysisDiagnostic

import geecs_data_utils
import scan_analysis.base as base
from scan_analysis.config import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer, core_supports
from scan_analysis.core_inputs import load_table
from scan_analysis.core_scan import shot_metadata
from scan_analysis.core_sink import write_sibling_outputs
from scan_analysis.route_compare import (
    compare_snapshots,
    snapshot_analysis_tree,
    snapshot_scan_products,
)

#: The column the legacy analyzer hardcoded; the converted recipe names it.
COLUMN = "HTT-MagTeslameter-DTM141 Field"
DEVICE = "TestCam"
TAG = ScanTag(year=2026, month=1, day=1, number=2, experiment="Test")
FIELDS = (0.29, 0.30, 0.31, 0.30, 0.295, 0.305)
SHOTS = len(FIELDS)
IMAGE = {
    "type": "camera",
    "background": {"additional_constant": 1.0},
    "vignette": {
        "method": "radial_polynomial",
        "full_width": 160,
        "full_height": 64,
        "x_offset": 0,
        "y_offset": 3,
        "vgnt4": -4.0e-9,
        "vgnt2": -6.0e-5,
        "vgnt0": 1.0,
        "min_model_value": 1.0e-9,
    },
    "roi": {"x_min": 10, "x_max": 138, "y_min": 4, "y_max": 60},
    "thresholding": {"method": "constant", "value": 0.0, "mode": "to_zero"},
    "filtering": {"median_kernel_size": 3},
    "pipeline": ["background", "vignette", "roi", "thresholding", "filtering"],
}


def write_tsv(path: Path, columns: list[str], rows: list[list]) -> Path:
    lines = ["\t".join(columns), *("\t".join(str(c) for c in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n")
    return path


def calibrations(root: Path) -> dict[str, Path]:
    """Camera 1 (128 px, back screen) in the real tables' vocabulary, under calibrations/HTT."""
    folder = root / "calibrations" / "HTT"
    folder.mkdir(parents=True)
    cam_columns = [
        "cam number", "FOV [mm]", "Left edge [mm]", "ROI Y offset", "ROI height",
        "ROI X offset", "ROI width", "Y center pixel", "Y Start", "Y End",
        "X Start", "X End", "rot [deg]", "set", "screen", "name", "sensitivity",
    ]  # fmt: skip
    cam = [1, 100.0, 150.0, 0, 64, 0, 128, 32, 1, 64, 1, 128, 0, 1, "back", "Cam1", 0.5]
    front = np.linspace(0.170, 0.030, 24)
    side = np.linspace(0.150, 0.260, 20)
    trj_columns = [
        "momentum [MeV/c]",
        "side logic",
        "front screen [m]",
        "side screen [m]",
    ]
    lanex_columns = [
        "setting number", "FOV slope", "FOV offset",
        "sensitivity 2", "sensitivity 1", "sensitivity 0",
    ]  # fmt: skip
    return {
        "camera": write_tsv(folder / "cam.txt", cam_columns, [cam]),
        "trajectory": write_tsv(
            folder / "trj.txt",
            trj_columns,
            [[8.0 / s, 0, s, s] for s in front] + [[3.0 / s, 1, s, s] for s in side],
        ),
        "lanex": write_tsv(
            folder / "lanex.txt",
            lanex_columns,
            [[1, 0.7, 20.0, 1.5e-5, 1.2e-3, 0.015]],
        ),
    }


def diagnostic(
    tables: dict[str, Path] | None, *, mode="per_shot", save=True, dark=True
) -> AnalysisDiagnostic:
    """The HTT-MagCam document's shape: dark scan, vignette, roi, threshold, median."""
    if tables is None:
        calibration = {"kind": "polynomial", "coeffs": [20.0, 0.5, 2e-3]}
    else:
        calibration = {
            "kind": "dnn_axis",
            "camera_calibration_file": str(tables["camera"]),
            "trajectory_calibration_file": str(tables["trajectory"]),
            "lanex_calibration_file": str(tables["lanex"]),
            "camera_number": 1,
            "magnetic_field_t": 0.3,
        }
    return AnalysisDiagnostic.model_validate(
        {
            "schema_version": 2,
            "name": DEVICE,
            "analyzer": {
                "kind": "magspec",
                "calibration": calibration,
                "energy_range": [1.2, 1501.3],
                "num_energy_points": 300,
            },
            "image": IMAGE,
            "scan": {
                "mode": mode,
                "save": save,
                "renderer": {"dpi": 30},
                **({"background_source": {"scan_number": 1}} if dark else {}),
            },
        }
    )


def recipe(document: AnalysisDiagnostic):
    """The converted v3 recipe, its tables under calibrations/HTT and the field column named."""
    column = COLUMN if document.analyzer.calibration.kind == "dnn_axis" else None
    return to_v3(document, table_namespace="HTT", magnetic_field_column=column).recipe


def frame(seed: int) -> np.ndarray:
    """A dispersed beam: two vertical stripes on noise, 64 x 160 uint16."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:160]
    img = 3000 * np.exp(-0.5 * (((x - 70) / 9) ** 2 + ((y - 30) / 7) ** 2))
    img += 1200 * np.exp(-0.5 * (((x - 110) / 5) ** 2 + ((y - 34) / 9) ** 2))
    img += 30 + rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def build_scan(base_dir: Path, *, fields=FIELDS, bins=None) -> Path:
    """Dark Scan001 and Scan002: shot-number-named PNGs and an s-file with the field."""
    dark = ScanPaths.get_scan_folder_path(
        tag=TAG.model_copy(update={"number": 1}), base_directory=base_dir
    )
    (dark / DEVICE).mkdir(parents=True)  # fixture acquisition
    (dark / "ScanInfoScan001.ini").write_text('[Scan Info]\nScan No = "1"\n')
    for shot in range(1, 4):
        noise = np.random.default_rng(100 + shot).normal(30, 4, (64, 160))
        write_png16(
            dark / DEVICE / f"Scan001_{DEVICE}_{shot:03d}.png", noise.astype(np.uint16)
        )
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=base_dir)
    (scan / DEVICE).mkdir(parents=True)  # fixture acquisition
    (scan / "ScanInfoScan002.ini").write_text(
        '[Scan Info]\nScan No = "2"\nScan Parameter = "Shotnumber"\n'
    )
    for shot in range(1, SHOTS + 1):
        write_png16(scan / DEVICE / f"Scan002_{DEVICE}_{shot:03d}.png", frame(shot))
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    pd.DataFrame(
        {
            "Shotnumber": range(1, SHOTS + 1),
            "Bin #": bins or [1] * SHOTS,
            COLUMN: list(fields),
        }
    ).to_csv(analysis / "s2.txt", sep="\t", index=False)
    return scan


@pytest.fixture
def configs(tmp_path, monkeypatch) -> dict[str, Path]:
    """The analysis-configs root holding the tables, as the host resolves it."""
    root = tmp_path / "configs"
    tables = calibrations(root)
    monkeypatch.setattr(scan_analysis_config, "_base_dir", root)
    return tables


def private_scan_paths(monkeypatch, base_dir: Path) -> None:
    """Resolve every scan tag under ``base_dir`` for the rest of the test."""

    class PrivateScanPaths(ScanPaths):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("base_directory", base_dir)
            super().__init__(*args, **kwargs)

    # The legacy wrapper looks the dark scan up through a call-time import.
    monkeypatch.setattr(base, "ScanPaths", PrivateScanPaths)
    monkeypatch.setattr(geecs_data_utils, "ScanPaths", PrivateScanPaths)


def run(
    monkeypatch, base_dir: Path, document, route: str, *, workers=1, **scan
) -> Path:
    """One route over a fresh scan under ``base_dir``; the legacy one shot at a time."""
    scan_folder = build_scan(base_dir, **scan)
    private_scan_paths(monkeypatch, base_dir)
    if route == "legacy":
        analyzer = create_scan_analyzer(document, id="Diag", priority=1, route="legacy")
        # The legacy analyzer writes each shot's field into one shared
        # calibration; its thread pool could measure a shot with another's.
        analyzer.max_workers = 1
    else:
        converted = recipe(document)
        converted = converted.model_copy(
            update={"scan": converted.scan.model_copy(update={"workers": workers})}
        )
        analyzer = CoreScanAnalyzer(converted, id="Diag", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()
    return scan_folder


def snapshot(scan: Path) -> dict[str, tuple]:
    tree = snapshot_analysis_tree(scan.parent.parent / "analysis")
    tree.update(snapshot_scan_products(scan, {DEVICE, f"ScanInfo{scan.name}.ini"}))
    return tree


def test_a_v2_magspec_document_keeps_its_legacy_route(configs):
    """Only the converted recipe opts in; the v2 document never reaches the core."""
    assert not core_supports(diagnostic(configs))
    assert core_supports(recipe(diagnostic(configs)))


@pytest.mark.parametrize("save", [True, False])
def test_the_core_route_matches_the_legacy_analyzer_byte_for_byte(
    tmp_path, monkeypatch, configs, save
):
    document = diagnostic(configs, save=save)
    legacy = run(monkeypatch, tmp_path / "legacy", document, "legacy")
    core = run(monkeypatch, tmp_path / "core", document, "core")
    old, new = snapshot(legacy), snapshot(core)
    assert compare_snapshots(old, new) == []
    for suffix, tail in (("-interp", ".png"), ("-interpSpec", ".tsv")):
        written = sorted(n for n in new if f"/{DEVICE}{suffix}/" in n)
        assert written == [
            f"Scan002/{DEVICE}{suffix}/Scan002_{DEVICE}_{shot:03d}{tail}"
            for shot in range(1, SHOTS + 1)
        ], "the sibling outputs are written whatever save says"
    assert any(name.endswith(".h5") for name in new) is save
    rows = pd.read_csv(core.parent.parent / "analysis" / "s2.txt", sep="\t")
    peaks = rows[f"{DEVICE}_peak_energy_MeV"]
    assert peaks.notna().all() and peaks.nunique() > 1, "the field moves each shot"


def test_a_pooled_run_writes_the_identical_tree(tmp_path, monkeypatch, configs, caplog):
    """The field travels to the workers on each shot's metadata; every byte is the serial run's."""
    from scan_analysis import core_workers

    monkeypatch.setattr(core_workers, "MIN_UNITS_FOR_POOL", 1)
    serial = run(monkeypatch, tmp_path / "serial", diagnostic(configs), "core")
    with caplog.at_level(logging.INFO, logger="scan_analysis.core_analyzer"):
        pooled = run(
            monkeypatch, tmp_path / "pooled", diagnostic(configs), "core", workers=2
        )
    assert any(r.getMessage().endswith("units, 2 workers") for r in caplog.records)
    assert compare_snapshots(snapshot(serial), snapshot(pooled), average_ulps=0) == []


def test_per_bin_runs_write_no_sibling_outputs(tmp_path, monkeypatch, configs):
    """The legacy per-bin path had no file to name them by; neither route writes them."""
    document = diagnostic(None, mode="per_bin")
    bins = dict(bins=[1, 1, 2, 2, 3, 4])  # bin 4: one shot, still a bin
    legacy = run(monkeypatch, tmp_path / "legacy", document, "legacy", **bins)
    core = run(monkeypatch, tmp_path / "core", document, "core", **bins)
    old, new = snapshot(legacy), snapshot(core)
    assert compare_snapshots(old, new) == []
    assert not [name for name in new if "-interp" in name]
    assert any(name.endswith(".h5") for name in new)


def test_a_missing_field_reading_takes_the_documents_field(
    tmp_path, monkeypatch, configs
):
    """A NaN cell is left out of the shot's metadata: that shot uses magnetic_field_t."""
    fields = (0.29, float("nan"), 0.31, 0.30, 0.295, 0.305)
    core = run(
        monkeypatch, tmp_path / "core", diagnostic(configs), "core", fields=fields
    )
    steady = (0.29, 0.3, 0.31, 0.30, 0.295, 0.305)
    legacy = run(
        monkeypatch, tmp_path / "legacy", diagnostic(configs), "legacy", fields=steady
    )
    old, new = snapshot(legacy), snapshot(core)
    for tree in (old, new):  # the recorded column itself differs by construction
        tree["s2.txt"] = ("table", tree["s2.txt"][1].drop(columns=[COLUMN]))
    assert compare_snapshots(old, new) == []


def test_shot_metadata_carries_finite_cells_only():
    rows = pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3, 4],
            COLUMN: [0.3, np.nan, "bad", 0.31],
            "Other": [1.0, 2.0, 3.0, 4.0],
        }
    )
    metadata = shot_metadata(rows, DEVICE, [COLUMN])
    assert sorted(metadata) == [1, 4]
    assert dict(metadata[4].scalars) == {COLUMN: 0.31}
    assert (metadata[4].device, metadata[4].shot_number) == (DEVICE, 4)
    assert metadata[4].acq_timestamp is None, "the runner's identity, nothing more"
    assert shot_metadata(rows, DEVICE, ["Absent"]) == {}
    assert shot_metadata(rows, DEVICE, []) == {}


def test_a_stitcher_reads_the_core_written_spectra_on_a_timestamp_named_scan(
    tmp_path, monkeypatch, configs, caplog
):
    """The chain: raw-stem ``-interpSpec`` files join a stitcher by timestamp, every shot."""
    from geecs_data_utils.native_files import native_file_name

    root = tmp_path / "share"
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=root)
    rows = {"Shotnumber": list(range(1, SHOTS + 1)), COLUMN: list(FIELDS)}
    for camera in (DEVICE, "TestCam2"):
        (scan / camera).mkdir(parents=True)  # fixture acquisition
        stamps = [
            3_900_000_000.0 + shot + (camera == "TestCam2") * 0.25
            for shot in rows["Shotnumber"]
        ]
        for shot, stamp in zip(rows["Shotnumber"], stamps):
            write_png16(
                scan / camera / native_file_name(camera, stamp, ".png"), frame(shot)
            )
        rows[f"{camera} acq_timestamp"] = stamps
    (scan / "ScanInfoScan002.ini").write_text(
        '[Scan Info]\nScan No = "2"\nScan Parameter = "Shotnumber"\n'
    )
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    pd.DataFrame(rows).to_csv(analysis / "s2.txt", sep="\t", index=False)
    private_scan_paths(monkeypatch, root)
    for camera in (DEVICE, "TestCam2"):
        document = recipe(diagnostic(configs, dark=False))
        _run_core(document.model_copy(update={"device": camera}))
    stitcher = AnalysisDiagnostic.model_validate(
        {
            "schema_version": 2,
            "name": DEVICE,
            "analyzer": {
                "kind": "line_stitcher",
                "sibling_devices": ["TestCam2-interpSpec"],
            },
            "image": {
                "type": "line",
                "data_loading": {"data_type": "tsv"},
                "pipeline": [],
            },
            "scan": {
                "device": f"{DEVICE}-interpSpec",
                "file_tail": ".tsv",
                "save": False,
            },
        }
    )
    with caplog.at_level(logging.WARNING):
        _run_core(stitcher)
    assert "No file found" not in caplog.text, "both cameras' spectra join every shot"
    assert "stitching without" not in caplog.text
    stitched = pd.read_csv(analysis / "s2.txt", sep="\t")
    assert stitched[f"{DEVICE}_integrated_intensity"].notna().all()
    assert stitched[f"{DEVICE}_integrated_intensity"].gt(0).all()


def test_stack_frames_name_their_outputs_by_the_announced_shot(
    tmp_path, monkeypatch, configs
):
    """A scan-mode stack's -interpSpec files map back to their own rows.

    The rows' announced shots run one ahead of ``Shotnumber`` (a rewind
    replayed a shot under the next number); data-utils maps shot-number
    names through that column, so the files carry it.
    """
    import h5py
    from geecs_data_utils.io.scan_stack import (
        FRAMES_DATASET,
        LABVIEW_EPOCH_OFFSET,
        TIMESTAMPS_DATASET,
    )
    from geecs_data_utils.shot_files import map_shot_files

    root = tmp_path / "share"
    scan = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=root)
    (scan / DEVICE).mkdir(parents=True)  # fixture acquisition
    shots = [1, 2, 3]
    stamps = [100.0 + shot for shot in shots]
    with h5py.File(scan / DEVICE / f"{DEVICE}.h5", "w") as handle:
        handle.create_dataset(FRAMES_DATASET, data=np.stack([frame(s) for s in shots]))
        handle.create_dataset(TIMESTAMPS_DATASET, data=stamps)
    (scan / "ScanInfoScan002.ini").write_text(
        '[Scan Info]\nScan No = "2"\nScan Parameter = "Shotnumber"\n'
    )
    rows = pd.DataFrame(
        {
            "Shotnumber": shots,
            "Announced Shot": [shot + 1 for shot in shots],
            COLUMN: list(FIELDS[:3]),
            f"{DEVICE} acq_timestamp": [t + LABVIEW_EPOCH_OFFSET for t in stamps],
        }
    )
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    rows.to_csv(analysis / "s2.txt", sep="\t", index=False)
    private_scan_paths(monkeypatch, root)
    document = diagnostic(configs, dark=False)
    document = document.model_copy(
        update={"scan": document.scan.model_copy(update={"data_format": "device_hdf5"})}
    )
    _run_core(recipe(document))
    spectra = scan / f"{DEVICE}-interpSpec"
    mapped = map_shot_files(spectra, rows, device=DEVICE, file_tail=".tsv")
    assert {shot: path.name for shot, path in mapped.items()} == {
        shot: f"Scan002_{DEVICE}_{shot + 1:03d}.tsv" for shot in shots
    }


def _run_core(document) -> None:
    analyzer = CoreScanAnalyzer(document, id="Diag", priority=1)
    try:
        analyzer.run_analysis(TAG)
    finally:
        analyzer.cleanup()


def test_the_writer_never_creates_a_scan_folder(tmp_path, configs):
    from geecs_analysis.measures.magspec import SIBLING_OUTPUTS

    from geecs_analysis.measurement import Measurement
    from geecs_data_utils.frames import Axis, Frame

    energy = Axis(np.linspace(10.0, 20.0, 5), unit="MeV", label="Energy")
    measurement = Measurement(
        scalars={},
        frame=Frame(np.ones((3, 5)), (Axis(np.arange(3.0), unit="px"), energy)),
        extras={
            "spectrum": Frame(
                np.ones(5), (energy,), unit="pC/MeV", label="Charge Density"
            )
        },
    )
    missing = tmp_path / "scans" / "Scan009" / DEVICE / "Scan009_TestCam_001.png"
    with pytest.raises(FileNotFoundError, match="refusing to create"):
        write_sibling_outputs(measurement, SIBLING_OUTPUTS, missing, 1, DEVICE)
    assert not (tmp_path / "scans").exists()
    scan = tmp_path / "scans" / "Scan008"
    scan.mkdir(parents=True)  # the scan exists; its device folder need not
    written = write_sibling_outputs(
        measurement,
        SIBLING_OUTPUTS,
        scan / DEVICE / "Scan008_TestCam_001.png",
        1,
        DEVICE,
    )
    assert [p.relative_to(scan).as_posix() for p in written] == [
        f"{DEVICE}-interp/Scan008_TestCam_001.png",
        f"{DEVICE}-interpSpec/Scan008_TestCam_001.tsv",
    ]
    assert (
        written[1].read_text().splitlines()[0]
        == "Energy [MeV]\tCharge Density [pC/MeV]"
    )
    assert sorted(p.name for p in scan.iterdir()) == [
        f"{DEVICE}-interp",
        f"{DEVICE}-interpSpec",
    ]


def test_a_sibling_write_failure_keeps_the_runs_scalars(
    tmp_path, monkeypatch, configs, caplog
):
    """The legacy analyzer logged a failed save and went on; so does the core route."""
    original = build_scan

    def blocked(base_dir, **kwargs):
        scan = original(base_dir, **kwargs)
        (scan / f"{DEVICE}-interp").write_text("a file where the folder goes")
        return scan

    monkeypatch.setattr(sys.modules[__name__], "build_scan", blocked)
    with caplog.at_level(logging.WARNING):
        scan = run(monkeypatch, tmp_path, diagnostic(configs), "core")
    rows = pd.read_csv(scan.parent.parent / "analysis" / "s2.txt", sep="\t")
    assert rows[f"{DEVICE}_peak_energy_MeV"].notna().all()
    assert caplog.text.count("sibling outputs not written") == SHOTS


def test_table_inputs_resolve_against_the_configs_root(tmp_path, monkeypatch, configs):
    assert load_table("calibrations/HTT/cam.txt").columns[0] == "cam number"
    assert load_table(str(configs["lanex"])).columns[0] == "setting number"
    with pytest.raises(FileNotFoundError):
        load_table("calibrations/HTT/absent.txt")
    monkeypatch.setattr(scan_analysis_config, "_base_dir", None)
    with pytest.raises(FileNotFoundError, match="no analysis-configs root"):
        load_table("calibrations/HTT/cam.txt")
