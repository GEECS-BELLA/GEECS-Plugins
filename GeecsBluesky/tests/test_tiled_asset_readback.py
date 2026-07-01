"""Tests for Tiled-backed local external asset readback."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import png
import pytest

from geecs_bluesky.assets import tiled_readback
from geecs_bluesky.assets.readback import EXTERNAL_ASSET_DOCUMENT_SCHEMA
from geecs_bluesky.assets.specs import (
    MAGSPEC_CAMERA_DEVICE_TYPE,
    POINTGREY_CAMERA_DEVICE_TYPE,
)
from geecs_bluesky.assets.tiled_readback import (
    event_by_scan_event_index,
    find_geecs_run,
    load_asset_from_tiled,
    load_asset_from_tiled_run,
    load_camera_image_from_tiled_run,
    resolve_asset_from_event,
    resolve_camera_asset_from_event,
)


class _FakePrimary:
    """Minimal Tiled primary stream shim."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    def read(self) -> pd.DataFrame:
        """Return the fake primary event stream."""
        return self._dataframe


class _FakeRun:
    """Minimal Tiled run shim."""

    def __init__(self, start_doc: dict, dataframe: pd.DataFrame) -> None:
        self.metadata = {"start": start_doc}
        self._primary = _FakePrimary(dataframe)

    def __getitem__(self, key: str) -> _FakePrimary:
        """Return the requested stream."""
        if key != "primary":
            raise KeyError(key)
        return self._primary


class _FakeCatalog:
    """Minimal searchable Tiled catalog shim."""

    def __init__(self, runs: list[_FakeRun]) -> None:
        self._runs = runs

    def search(self, _query: object) -> "_FakeCatalog":
        """Accept Tiled query objects; filtering is tested in ``find_geecs_run``."""
        return self

    def values(self) -> list[_FakeRun]:
        """Return fake runs."""
        return self._runs


def test_event_by_scan_event_index_returns_matching_row() -> None:
    """Event lookup should use the schema-level scan event index."""
    table = pd.DataFrame(
        [
            {"scan_event_index": 1, "value": "first"},
            {"scan_event_index": 2, "value": "second"},
        ]
    )

    assert event_by_scan_event_index(table, 2)["value"] == "second"


def test_event_by_scan_event_index_reports_available_range() -> None:
    """Missing event lookup should report available contiguous event indices."""
    table = pd.DataFrame(
        [
            {"scan_event_index": 1, "value": "first"},
            {"scan_event_index": 2, "value": "second"},
            {"scan_event_index": 3, "value": "third"},
        ]
    )

    with pytest.raises(
        LookupError,
        match=(
            "No event row found for scan_event_index=5. "
            "Available scan_event_index values: 1-3 \\(3 events\\)."
        ),
    ):
        event_by_scan_event_index(table, 5)


def test_event_by_scan_event_index_reports_sparse_values() -> None:
    """Missing event lookup should report sparse event indices explicitly."""
    table = pd.DataFrame(
        [
            {"scan_event_index": 1, "value": "first"},
            {"scan_event_index": 3, "value": "third"},
            {"scan_event_index": 8, "value": "eighth"},
        ]
    )

    with pytest.raises(
        LookupError,
        match=(
            "No event row found for scan_event_index=5. "
            "Available scan_event_index values: 1, 3, 8 \\(3 events\\)."
        ),
    ):
        event_by_scan_event_index(table, 5)


def test_resolve_camera_asset_from_tiled_event_fills_native_file(
    tmp_path: Path,
) -> None:
    """Tiled event metadata should deterministically resolve a native image."""
    scan_folder = tmp_path / "scans" / "Scan014"
    image_path = scan_folder / "UC_Amp2_IR_input" / "UC_Amp2_IR_input_1000.500.png"
    image_path.parent.mkdir(parents=True)
    expected = np.array([[1, 3], [5, 7]], dtype=np.uint8)
    with image_path.open("wb") as stream:
        png.Writer(width=2, height=2, greyscale=True, bitdepth=8).write(
            stream, expected.tolist()
        )

    asset = resolve_camera_asset_from_event(
        start_doc={
            "uid": "run-uid",
            "time": datetime(
                2026, 6, 23, tzinfo=ZoneInfo("America/Los_Angeles")
            ).timestamp(),
            "scan_number": 14,
            "scan_folder": str(scan_folder),
            "experiment": "Undulator",
        },
        event={
            "scan_event_index": 1,
            "uc_amp2_ir_input-acq_timestamp": 1000.5,
            "uc_amp2_ir_input-nonscalar_save_path": str(
                scan_folder / "UC_Amp2_IR_input"
            ),
            "uc_amp2_ir_input-image": "resource-uid/0",
        },
        device_name="UC_Amp2_IR_input",
        device_type=POINTGREY_CAMERA_DEVICE_TYPE,
    )

    assert asset.data_key == "uc_amp2_ir_input-image"
    assert asset.datum_id == "resource-uid/0"
    assert asset.file_path == image_path
    assert asset.resource_root == str(scan_folder)
    assert asset.resource_path == "UC_Amp2_IR_input/UC_Amp2_IR_input_1000.500.png"


def test_load_camera_image_from_tiled_run_uses_event_timestamp(
    tmp_path: Path,
) -> None:
    """Camera readback should fill the shot selected from the Tiled event table."""
    scan_folder = tmp_path / "scans" / "Scan014"
    device_folder = scan_folder / "UC_Amp2_IR_input"
    device_folder.mkdir(parents=True)
    first = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    second = np.array([[2, 2], [2, 2]], dtype=np.uint8)
    for image_path, image in (
        (device_folder / "UC_Amp2_IR_input_1000.500.png", first),
        (device_folder / "UC_Amp2_IR_input_1001.500.png", second),
    ):
        with image_path.open("wb") as stream:
            png.Writer(width=2, height=2, greyscale=True, bitdepth=8).write(
                stream, image.tolist()
            )

    start_doc = {
        "uid": "run-uid",
        "time": datetime(
            2026, 6, 23, tzinfo=ZoneInfo("America/Los_Angeles")
        ).timestamp(),
        "scan_number": 14,
        "scan_folder": str(scan_folder),
        "experiment": "Undulator",
    }
    table = pd.DataFrame(
        [
            {
                "scan_event_index": 1,
                "uc_amp2_ir_input-acq_timestamp": 1000.5,
                "uc_amp2_ir_input-nonscalar_save_path": str(device_folder),
                "uc_amp2_ir_input-image": "resource-first/0",
            },
            {
                "scan_event_index": 2,
                "uc_amp2_ir_input-acq_timestamp": 1001.5,
                "uc_amp2_ir_input-nonscalar_save_path": str(device_folder),
                "uc_amp2_ir_input-image": "resource-second/0",
            },
        ]
    )

    loaded = load_camera_image_from_tiled_run(
        _FakeRun(start_doc, table),
        device_name="UC_Amp2_IR_input",
        shot_number=2,
        device_type=POINTGREY_CAMERA_DEVICE_TYPE,
        retry_intervals=[],
    )

    assert loaded.asset.file_path == device_folder / "UC_Amp2_IR_input_1001.500.png"
    np.testing.assert_array_equal(loaded.image, second)


def test_load_camera_image_from_tiled_run_maps_windows_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows scan metadata should fill from the configured local root map."""
    local_data_root = tmp_path / "hdna2" / "data"
    local_scan_folder = (
        local_data_root
        / "Undulator"
        / "Y2026"
        / "06-Jun"
        / "26_0625"
        / "scans"
        / "Scan001"
    )
    device_folder = local_scan_folder / "UC_Amp4_IR_input"
    device_folder.mkdir(parents=True)
    expected = np.array([[4, 4], [8, 8]], dtype=np.uint8)
    image_path = device_folder / "UC_Amp4_IR_input_3865254648.364.png"
    with image_path.open("wb") as stream:
        png.Writer(width=2, height=2, greyscale=True, bitdepth=8).write(
            stream, expected.tolist()
        )

    start_doc = {
        "uid": "run-uid",
        "time": datetime(
            2026, 6, 25, tzinfo=ZoneInfo("America/Los_Angeles")
        ).timestamp(),
        "scan_number": 1,
        "scan_folder": (r"Z:\data\Undulator\Y2026\06-Jun\26_0625\scans\Scan001"),
        "experiment": "Undulator",
    }
    table = pd.DataFrame(
        [
            {
                "scan_event_index": 5,
                "uc_amp4_ir_input-acq_timestamp": 3865254648.364,
                "uc_amp4_ir_input-nonscalar_save_path": (
                    r"Z:\data\Undulator\Y2026\06-Jun\26_0625\scans"
                    r"\Scan001\UC_Amp4_IR_input"
                ),
                "uc_amp4_ir_input-image": "resource-uid/0",
            },
        ]
    )
    monkeypatch.setattr(
        tiled_readback,
        "read_geecs_root_map",
        lambda: {"Z:/data": str(local_data_root)},
    )

    loaded = load_camera_image_from_tiled_run(
        _FakeRun(start_doc, table),
        device_name="UC_Amp4_IR_input",
        shot_number=5,
        device_type=POINTGREY_CAMERA_DEVICE_TYPE,
        retry_intervals=[],
    )

    assert loaded.asset.file_path == image_path
    assert loaded.asset.resource_root == "Z:/data"
    assert loaded.asset.resource_path == (
        "Undulator/Y2026/06-Jun/26_0625/scans/Scan001/"
        "UC_Amp4_IR_input/UC_Amp4_IR_input_3865254648.364.png"
    )
    np.testing.assert_array_equal(loaded.image, expected)


def test_load_asset_from_tiled_run_handles_text_array_asset(tmp_path: Path) -> None:
    """Generic Tiled readback should fill registered non-camera assets."""
    scan_folder = tmp_path / "scans" / "Scan042"
    device_folder = scan_folder / "U_BCaveMagSpec"
    text_folder = scan_folder / "U_BCaveMagSpec-interpSpec"
    text_folder.mkdir(parents=True)
    expected = np.array([[400.0, 1.0], [401.0, 2.0]])
    np.savetxt(text_folder / "U_BCaveMagSpec-interpSpec_1000.500.txt", expected)

    start_doc = {
        "uid": "run-uid",
        "time": datetime(
            2026, 6, 23, tzinfo=ZoneInfo("America/Los_Angeles")
        ).timestamp(),
        "scan_number": 42,
        "scan_folder": str(scan_folder),
        "experiment": "Undulator",
    }
    table = pd.DataFrame(
        [
            {
                "scan_event_index": 7,
                "u_bcavemagspec-acq_timestamp": 1000.5,
                "u_bcavemagspec-nonscalar_save_path": str(device_folder),
                "u_bcavemagspec-interpspec": "text-resource/0",
            },
        ]
    )

    loaded = load_asset_from_tiled_run(
        _FakeRun(start_doc, table),
        device_name="U_BCaveMagSpec",
        device_type=MAGSPEC_CAMERA_DEVICE_TYPE,
        event_field="interpSpec",
        shot_number=7,
        retry_intervals=[],
    )

    assert loaded.asset.data_key == "u_bcavemagspec-interpspec"
    start = next(doc for name, doc in loaded.asset.documents if name == "start")
    assert (
        start["geecs_external_asset_document_schema"] == EXTERNAL_ASSET_DOCUMENT_SCHEMA
    )
    assert loaded.asset.resource_path == (
        "U_BCaveMagSpec-interpSpec/U_BCaveMagSpec-interpSpec_1000.500.txt"
    )
    np.testing.assert_array_equal(loaded.data, expected)


def test_load_asset_from_tiled_finds_run_and_loads_text_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Date/scan generic readback should mirror the run-level helper."""
    scan_folder = tmp_path / "scans" / "Scan042"
    device_folder = scan_folder / "U_BCaveMagSpec"
    text_folder = scan_folder / "U_BCaveMagSpec-interpSpec"
    text_folder.mkdir(parents=True)
    expected = np.array([[400.0, 1.0], [401.0, 2.0]])
    np.savetxt(text_folder / "U_BCaveMagSpec-interpSpec_1000.500.txt", expected)

    start_doc = {
        "uid": "run-uid",
        "time": datetime(
            2026, 6, 23, tzinfo=ZoneInfo("America/Los_Angeles")
        ).timestamp(),
        "scan_number": 42,
        "scan_folder": str(scan_folder),
        "experiment": "Undulator",
    }
    table = pd.DataFrame(
        [
            {
                "scan_event_index": 7,
                "u_bcavemagspec-acq_timestamp": 1000.5,
                "u_bcavemagspec-nonscalar_save_path": str(device_folder),
                "u_bcavemagspec-interpspec": "text-resource/0",
            },
        ]
    )
    monkeypatch.setattr(
        tiled_readback,
        "load_tiled_client",
        lambda *, tiled_uri=None, tiled_api_key=None: _FakeCatalog(
            [_FakeRun(start_doc, table)]
        ),
    )

    loaded = load_asset_from_tiled(
        year=2026,
        month=6,
        day=23,
        scan_number=42,
        experiment="Undulator",
        device_name="U_BCaveMagSpec",
        device_type=MAGSPEC_CAMERA_DEVICE_TYPE,
        event_field="interpSpec",
        shot_number=7,
        retry_intervals=[],
    )

    assert loaded.asset.data_key == "u_bcavemagspec-interpspec"
    np.testing.assert_array_equal(loaded.data, expected)


def test_resolve_asset_from_event_requires_field_for_multi_asset_device() -> None:
    """Multi-asset devices should require an event field selection."""
    with pytest.raises(ValueError, match="Pass event_field for multi-asset devices"):
        resolve_asset_from_event(
            start_doc={
                "uid": "run-uid",
                "time": datetime(
                    2026, 6, 23, tzinfo=ZoneInfo("America/Los_Angeles")
                ).timestamp(),
                "scan_number": 42,
                "scan_folder": "/tmp/scans/Scan042",
                "experiment": "Undulator",
            },
            event={
                "scan_event_index": 1,
                "u_bcavemagspec-acq_timestamp": 1000.5,
                "u_bcavemagspec-interpspec": "text-resource/0",
            },
            device_name="U_BCaveMagSpec",
            device_type=MAGSPEC_CAMERA_DEVICE_TYPE,
        )


def test_find_geecs_run_matches_scan_identity_by_date() -> None:
    """Run lookup should qualify the day-scoped scan number with date."""
    target_time = datetime(
        2026, 6, 23, 12, 0, tzinfo=ZoneInfo("America/Los_Angeles")
    ).timestamp()
    other_time = datetime(
        2026, 6, 22, 12, 0, tzinfo=ZoneInfo("America/Los_Angeles")
    ).timestamp()
    target = _FakeRun(
        {
            "uid": "target",
            "time": target_time,
            "scan_number": 14,
            "experiment": "Undulator",
        },
        pd.DataFrame(),
    )
    other_day = _FakeRun(
        {
            "uid": "other-day",
            "time": other_time,
            "scan_number": 14,
            "experiment": "Undulator",
        },
        pd.DataFrame(),
    )

    found = find_geecs_run(
        _FakeCatalog([other_day, target]),
        year=2026,
        month=6,
        day=23,
        scan_number=14,
        experiment="Undulator",
    )

    assert found is target


def test_find_geecs_run_ignores_derived_analysis_runs() -> None:
    """Raw run lookup should not collide with derived analysis records."""
    target_time = datetime(
        2026, 6, 29, 21, 0, tzinfo=ZoneInfo("America/Los_Angeles")
    ).timestamp()
    raw_run = _FakeRun(
        {
            "uid": "44b60a76-c8e2-425c-b3cf-5ef8c92c864c",
            "time": target_time,
            "scan_number": 1,
            "experiment": "Undulator",
        },
        pd.DataFrame(),
    )
    analysis_run = _FakeRun(
        {
            "uid": "22241641-679a-44de-8670-5c6891421641",
            "time": target_time + 60,
            "scan_number": 1,
            "experiment": "Undulator",
            "purpose": "geecs_bluesky_analysis",
            "analysis_of": "44b60a76-c8e2-425c-b3cf-5ef8c92c864c",
        },
        pd.DataFrame(),
    )

    found = find_geecs_run(
        _FakeCatalog([raw_run, analysis_run]),
        year=2026,
        month=6,
        day=29,
        scan_number=1,
        experiment="Undulator",
    )

    assert found is raw_run
