"""Completed-scan sources retain native precision, identity and stack indices."""

import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
from geecs_data_utils.io.scan_stack import (
    FRAMES_DATASET,
    LABVIEW_EPOCH_OFFSET,
    TIMESTAMPS_DATASET,
    ShotRef,
)
from geecs_data_utils.shot_files import StackMappingUnavailable
from geecs_schemas.analysis import AnalysisDiagnostic

from scan_analysis.core_source import V2ShotSource, prepare_source


def document(*, line=False, data_type="csv", **scan):
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Detector",
            "analyzer": {"kind": "line" if line else "beam"},
            "image": (
                {"type": "line", "data_loading": {"data_type": data_type}}
                if line
                else {"type": "camera"}
            ),
            "scan": scan,
        }
    )


@pytest.mark.parametrize("line", [False, True])
def test_native_arrays_keep_reader_precision_and_folder_override(tmp_path, line):
    folder = tmp_path / "Scan001"
    device = folder / "SavedDetector"
    device.mkdir(parents=True)
    path = device / "Scan001_SavedDetector_002.npy"
    raw = np.arange(12, dtype=np.float32).reshape(6, 2)
    np.save(path, raw)
    doc = document(line=line, data_type="npy", device=device.name, file_tail=".npy")
    rows = pd.DataFrame({"Shotnumber": [1, 2]})
    source = prepare_source(doc, folder, rows)
    assert source.data_dir == device
    assert dict(source.references) == {2: path}
    before = path.read_bytes()
    loaded = source.load(2)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, raw)
    loaded[:] = -1
    np.testing.assert_array_equal(source.load(2), raw)
    assert path.read_bytes() == before
    with pytest.raises(KeyError):
        source.load(1)
    with pytest.raises(TypeError):
        source.references[1] = path


def test_trace_settings_are_snapshotted_before_document_edits(tmp_path):
    device = tmp_path / "Detector"
    device.mkdir()
    path = device / "Scan001_Detector_001.csv"
    path.write_text("1,10,100\n2,20,200\n")
    doc = document(line=True)
    source = prepare_source(doc, tmp_path, pd.DataFrame({"Shotnumber": [1]}))
    doc.image.data_loading.y_column = 2
    doc.scan.device = "Other"
    np.testing.assert_array_equal(source.load(1), [[1, 10], [2, 20]])


def test_reference_map_is_owned_but_retains_shotref_identity(tmp_path):
    ref = ShotRef(tmp_path / "stack.h5", 7)
    refs = {2: ref}
    source = V2ShotSource(tmp_path, refs)
    refs.clear()
    assert source.references[2] is ref


@pytest.mark.parametrize("line", [False, True])
def test_stack_join_uses_diagnostic_identity_and_reads_exact_frame(tmp_path, line):
    device = tmp_path / "SavedDetector"
    device.mkdir()
    path = device / "SavedDetector.h5"
    raw = np.arange(24, dtype=np.uint16).reshape(2, 6, 2)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(FRAMES_DATASET, data=raw)
        handle.create_dataset(TIMESTAMPS_DATASET, data=[100.0, 200.0])
        if line:
            prefix = "/entry/instrument/NDAttributes/detector-hdf-interpspec"
            for name in ("wave_x0", "wave_dx", "wave_samples"):
                handle.create_dataset(f"{prefix}-{name}", data=[np.nan, np.nan])
    doc = document(
        line=line, data_type="pva_stack", device=device.name, data_format="device_hdf5"
    )
    rows = pd.DataFrame(
        {
            "Shotnumber": [3],
            "Detector:acq_timestamp": [200.0 + LABVIEW_EPOCH_OFFSET],
            "SavedDetector:acq_timestamp": [100.0 + LABVIEW_EPOCH_OFFSET],
        }
    )
    source = prepare_source(doc, tmp_path, rows)
    assert isinstance(source.references[3], ShotRef)
    assert source.references[3].shot_index == 1
    np.testing.assert_array_equal(source.load(3), raw[1])


def test_stack_only_trace_does_not_fall_back_to_native_file(tmp_path):
    device = tmp_path / "Detector"
    device.mkdir()
    (device / "Scan001_Detector_001.csv").write_text("1,2\n")
    doc = document(line=True, data_type="pva_stack", data_format="device_hdf5")
    with pytest.raises(StackMappingUnavailable):
        prepare_source(doc, tmp_path, pd.DataFrame({"Shotnumber": [1]}))


def test_camera_stack_preference_can_fall_back_to_native_default_suffix(tmp_path):
    device = tmp_path / "Detector"
    device.mkdir()
    path = device / "Scan001_Detector_001.png"
    path.write_bytes(b"bad image")
    doc = document(data_format="device_hdf5")
    source = prepare_source(doc, tmp_path, pd.DataFrame({"Shotnumber": [1]}))
    assert source.references == {1: path}
    # Mapping does not load or silently discard a corrupt file.
    with pytest.raises(Exception):
        source.load(1)


def test_missing_scan_raises_without_creating_any_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="Scan folder"):
        prepare_source(document(), tmp_path / "scans" / "Scan999", pd.DataFrame())
    assert not list(tmp_path.iterdir())


def test_missing_device_is_empty_and_does_not_create_outputs(tmp_path):
    source = prepare_source(document(), tmp_path, pd.DataFrame({"Shotnumber": [1]}))
    assert not source.references
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("device", ["..", ".", "../Elsewhere", "a/b", "a\\b"])
def test_device_override_stays_within_one_scan_subfolder(tmp_path, device):
    with pytest.raises(ValueError, match="one scan subfolder"):
        prepare_source(document(device=device), tmp_path, pd.DataFrame())


def test_source_imports_no_legacy_analyzers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from scan_analysis.core_source import prepare_source; "
            "assert not any(n.startswith('image_analysis') for n in sys.modules)",
        ],
        check=True,
    )
