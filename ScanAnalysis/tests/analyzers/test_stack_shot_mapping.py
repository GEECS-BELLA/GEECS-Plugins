"""Tests for the capture-stack mapping strategy in SingleDeviceScanAnalyzer.

``data_format="device_hdf5"`` opts a diagnostic into the per-device capture
frame stack (``<device>/<device>.h5``, written by the capture daemon —
contract in ``GeecsBluesky/geecs_bluesky/capture/FORMAT.md``). The join
mirrors the acq_timestamp file join (canonical-millisecond keys), producing
``ShotRef`` values that travel the existing per-shot pipeline. Every failure
shape (no stack, wrong schema, zero joins, unset flag) must fall back to the
per-shot-file strategies so the old basis keeps working unconditionally.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET, ShotRef
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)

DEVICE = "UC_Amp4_IR_input"


def _make_analyzer(
    tmp_path: Path,
    aux: pd.DataFrame,
    file_tail: str = ".png",
    data_format: str | None = "device_hdf5",
):
    """Cheap instance: _build_data_file_map only touches these attributes."""
    sa = SingleDeviceScanAnalyzer.__new__(SingleDeviceScanAnalyzer)
    sa.device_name = DEVICE
    sa.file_tail = file_tail
    sa.path_dict = {"data": tmp_path}
    sa.auxiliary_data = aux
    sa.data_format = data_format
    sa._data_file_map = {}
    return sa


def _write_stack(device_dir: Path, lv_timestamps, schema="geecs-capture/1") -> Path:
    """Write a contract-shaped stack whose frames' values equal their index."""
    device_dir.mkdir(parents=True, exist_ok=True)
    path = device_dir / f"{device_dir.name}.h5"
    n = len(lv_timestamps)
    with h5py.File(path, "w", libver="latest") as f:
        f.attrs["schema"] = schema
        f.create_dataset(
            "frames",
            data=np.stack([np.full((3, 3), i, dtype=np.uint16) for i in range(n)]),
            chunks=(1, 3, 3),
        )
        f.create_dataset(
            "acq_timestamp",
            data=np.asarray(lv_timestamps, dtype=float) - LABVIEW_EPOCH_OFFSET,
        )
    return path


def _aux(lv_timestamps) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Shotnumber": list(range(1, len(lv_timestamps) + 1)),
            "Bin #": [1] * len(lv_timestamps),
            f"{DEVICE}:acq_timestamp": lv_timestamps,
        }
    )


class TestStackJoin:
    def test_maps_shots_to_shotrefs(self, tmp_path):
        ts = [3866137959.524, 3866137960.525, 3866137961.526]
        device_dir = tmp_path / DEVICE
        stack = _write_stack(device_dir, ts)
        sa = _make_analyzer(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert set(sa._data_file_map) == {1, 2, 3}
        for shot, ref in sa._data_file_map.items():
            assert isinstance(ref, ShotRef)
            assert Path(ref) == stack
            assert ref.shot_index == shot - 1  # stack order matches here

    def test_extra_stack_frames_do_not_join(self, tmp_path):
        # The daemon can capture pre-save-window frames the LV set lacks;
        # rows only join frames whose timestamps the aux frame carries.
        ts = [3866137959.524, 3866137960.525]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, [3866137952.111, *ts])  # leading extra
        sa = _make_analyzer(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert {r.shot_index for r in sa._data_file_map.values()} == {1, 2}

    def test_no_stack_falls_back_to_files(self, tmp_path):
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        device_dir.mkdir()
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_analyzer(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert sa._data_file_map == {1: png}

    def test_wrong_schema_falls_back(self, tmp_path):
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, ts, schema="not-a-capture-stack/0")
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_analyzer(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert sa._data_file_map == {1: png}

    def test_zero_joins_falls_back(self, tmp_path):
        # Stack exists but its timestamps match nothing in the aux frame.
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, [3866000000.0])
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_analyzer(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert sa._data_file_map == {1: png}

    def test_default_data_format_ignores_stack(self, tmp_path):
        # No opt-in => per-shot files even when a stack is present.
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, ts)
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_analyzer(device_dir, _aux(ts), data_format=None)
        sa._build_data_file_map()
        assert sa._data_file_map == {1: png}

    def test_corrupt_stack_missing_timestamps_falls_back(self, tmp_path):
        # Valid schema + frames but no /acq_timestamp dataset: the read
        # raises inside the strategy, which must fall back — never fail
        # the task (review finding 1).
        ts = [3866137959.524]
        device_dir = tmp_path / DEVICE
        device_dir.mkdir(parents=True)
        with h5py.File(device_dir / f"{DEVICE}.h5", "w") as f:
            f.attrs["schema"] = "geecs-capture/1"
            f.create_dataset("frames", data=np.zeros((1, 3, 3), dtype=np.uint16))
        png = device_dir / f"{DEVICE}_3866137959.524.png"
        png.write_bytes(b"")
        sa = _make_analyzer(device_dir, _aux(ts))
        sa._build_data_file_map()
        assert sa._data_file_map == {1: png}

    def test_no_timestamp_column_falls_back(self, tmp_path):
        # Stack present but the aux frame has no acq_timestamp column for
        # this device: fall back to legacy shot-number mapping.
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, [3866137959.524])
        png = device_dir / f"Scan001_{DEVICE}_001.png"
        png.write_bytes(b"")
        aux = pd.DataFrame({"Shotnumber": [1], "Bin #": [1]})
        sa = _make_analyzer(device_dir, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {1: png}

    def test_valid_column_false_skips_row(self, tmp_path):
        ts = [3866137959.524, 3866137960.525]
        device_dir = tmp_path / DEVICE
        _write_stack(device_dir, ts)
        aux = _aux(ts)
        aux[f"{DEVICE}:valid"] = [True, False]
        sa = _make_analyzer(device_dir, aux)
        sa._build_data_file_map()
        assert set(sa._data_file_map) == {1}
