"""Tests for the acq_timestamp file-mapping strategy in SingleDeviceScanAnalyzer.

Bluesky-produced scans save native files named by the device's own
``acq_timestamp`` rather than MC-convention shot numbers; the analyzer joins
shots to files through the device's per-shot timestamp column when the
auxiliary frame carries one, and falls back to the legacy shot-number
filename parsing otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)

DEVICE = "UC_Amp2_IR_input"


def _make_analyzer(tmp_path: Path, aux: pd.DataFrame, file_tail: str = ".png"):
    """Cheap instance: _build_data_file_map only touches these attributes."""
    sa = SingleDeviceScanAnalyzer.__new__(SingleDeviceScanAnalyzer)
    sa.device_name = DEVICE
    sa.file_tail = file_tail
    sa.path_dict = {"data": tmp_path}
    sa.auxiliary_data = aux
    sa._data_file_map = {}
    return sa


def _touch(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"")
    return p


class TestTimestampJoin:
    def test_joins_by_device_own_timestamp(self, tmp_path):
        f1 = _touch(tmp_path, f"{DEVICE}_3866137959.524.png")
        f2 = _touch(tmp_path, f"{DEVICE}_3866137960.525.png")
        aux = pd.DataFrame(
            {
                "Shotnumber": [1, 2],
                "Bin #": [1, 1],
                f"{DEVICE}:acq_timestamp": [3866137959.524, 3866137960.525],
            }
        )
        sa = _make_analyzer(tmp_path, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {1: f1, 2: f2}

    @pytest.mark.parametrize(
        "column",
        [
            f"{DEVICE} acq_timestamp",  # s-file header form
            f"{DEVICE}:acq_timestamp",  # in-memory frame form
            "uc_amp2_ir_input-acq_timestamp",  # raw event-key form
        ],
    )
    def test_recognises_all_column_spellings(self, tmp_path, column):
        f1 = _touch(tmp_path, f"{DEVICE}_3866137959.524.png")
        aux = pd.DataFrame({"Shotnumber": [1], "Bin #": [1], column: [3866137959.524]})
        sa = _make_analyzer(tmp_path, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {1: f1}

    def test_float_repr_of_row_value_still_joins(self, tmp_path):
        # The row carries the raw double; the filename is its %.3f rendering.
        f1 = _touch(tmp_path, f"{DEVICE}_3866137959.524.png")
        aux = pd.DataFrame(
            {
                "Shotnumber": [1],
                "Bin #": [1],
                f"{DEVICE}:acq_timestamp": [3866137959.5239997],
            }
        )
        sa = _make_analyzer(tmp_path, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {1: f1}

    def test_invalid_rows_get_no_file(self, tmp_path):
        # An invalid row's timestamp points at a different physical shot's
        # file — mapping it would attach the wrong image. Skip it.
        _touch(tmp_path, f"{DEVICE}_3866137959.524.png")
        aux = pd.DataFrame(
            {
                "Shotnumber": [1],
                "Bin #": [1],
                f"{DEVICE}:acq_timestamp": [3866137959.524],
                f"{DEVICE}:valid": [False],
            }
        )
        sa = _make_analyzer(tmp_path, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {}

    def test_other_devices_columns_are_ignored(self, tmp_path):
        # Per-device join: another device's timestamp column must not be used.
        _touch(tmp_path, f"{DEVICE}_3866137959.524.png")
        aux = pd.DataFrame(
            {
                "Shotnumber": [1],
                "Bin #": [1],
                "U_BCaveICT:acq_timestamp": [3866137959.612],
            }
        )
        sa = _make_analyzer(tmp_path, aux)
        # No column for THIS device → falls back to legacy mapping → no match
        # for timestamp-named files.
        sa._build_data_file_map()
        assert sa._data_file_map == {}

    def test_nonpositive_timestamp_skipped(self, tmp_path):
        _touch(tmp_path, f"{DEVICE}_3866137959.524.png")
        aux = pd.DataFrame(
            {
                "Shotnumber": [1],
                "Bin #": [1],
                f"{DEVICE}:acq_timestamp": [0.0],  # never-acquired placeholder
            }
        )
        sa = _make_analyzer(tmp_path, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {}


class TestLegacyFallback:
    def test_mc_filenames_still_map_by_shot_number(self, tmp_path):
        f5 = _touch(tmp_path, f"Scan012_{DEVICE}_005.png")
        aux = pd.DataFrame({"Shotnumber": [5], "Bin #": [1]})
        sa = _make_analyzer(tmp_path, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {5: f5}

    def test_timestamp_column_wins_when_present(self, tmp_path):
        # Frame has the device timestamp column → timestamp strategy is used
        # even if an MC-style file also exists.
        _touch(tmp_path, f"Scan012_{DEVICE}_001.png")
        f_ts = _touch(tmp_path, f"{DEVICE}_3866137959.524.png")
        aux = pd.DataFrame(
            {
                "Shotnumber": [1],
                "Bin #": [1],
                f"{DEVICE}:acq_timestamp": [3866137959.524],
            }
        )
        sa = _make_analyzer(tmp_path, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {1: f_ts}
