"""Tests for Tiled-backed local external asset readback."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import png

from geecs_bluesky.assets.specs import POINTGREY_CAMERA_DEVICE_TYPE
from geecs_bluesky.assets.tiled_readback import (
    event_by_scan_event_index,
    find_geecs_run,
    load_camera_image_from_tiled_run,
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
