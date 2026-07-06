"""Tests for the acq_timestamp file-mapping strategy in SingleDeviceScanAnalyzer.

Bluesky-produced scans save native files named by the device's own
``acq_timestamp`` rather than MC-convention shot numbers; the analyzer joins
shots to files through the device's per-shot timestamp column when the
auxiliary frame carries one, and falls back to the legacy shot-number
filename parsing otherwise.

Crucially, the *column alone* does not discriminate the two worlds: the
legacy scanner force-appends ``acq_timestamp`` to every synchronous device,
so MC/GUI-produced s-files carry the column while the data files are
shot-number-named. ``TestLegacyScanWithTimestampColumn`` pins the
zero-join fallback that keeps those scans re-analyzable.
"""

from __future__ import annotations

import logging
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


class TestLegacyScanWithTimestampColumn:
    """Legacy MC/GUI scans carry BOTH the acq_timestamp column and
    shot-number-named files — the zero-join fallback must kick in."""

    @staticmethod
    def _legacy_aux(n_shots: int) -> pd.DataFrame:
        """Realistic legacy s-file column set, space-separated header form.

        The legacy device_manager force-appends acq_timestamp to every
        synchronous device, so the column is present even though the
        file_mover renamed the data files to Scan{NNN}_{device}_{shot:03d}.
        """
        shots = list(range(1, n_shots + 1))
        return pd.DataFrame(
            {
                "Shotnumber": shots,
                "Bin #": [1] * n_shots,
                "Elapsed Time": [0.5 * s for s in shots],
                "U_ESP_JetXYZ Position.Axis 1": [7.0 + 0.1 * s for s in shots],
                # Epoch-style timestamps that do NOT appear in any filename.
                f"{DEVICE} acq_timestamp": [3866137959.0 + s for s in shots],
                f"{DEVICE} MeanCounts": [1000.0 + s for s in shots],
            }
        )

    def test_legacy_scan_falls_back_to_shot_number_mapping(self, tmp_path):
        files = {
            s: _touch(tmp_path, f"Scan012_{DEVICE}_{s:03d}.png") for s in (1, 2, 3)
        }
        sa = _make_analyzer(tmp_path, self._legacy_aux(3))
        sa._build_data_file_map()
        assert sa._data_file_map == files

    def test_fallback_is_logged_at_info(self, tmp_path, caplog):
        _touch(tmp_path, f"Scan012_{DEVICE}_001.png")
        sa = _make_analyzer(tmp_path, self._legacy_aux(1))
        with caplog.at_level(
            logging.INFO,
            logger="scan_analysis.analyzers.common.single_device_scan_analyzer",
        ):
            sa._build_data_file_map()
        assert any("falling back" in rec.getMessage() for rec in caplog.records)

    def test_partial_timestamp_map_does_not_fall_back(self, tmp_path):
        # Bluesky scan with one invalid row: the timestamp join maps the
        # valid shot only. A partial map is legitimate — it must NOT trigger
        # the legacy fallback, even if a legacy-named file happens to exist
        # for the unmapped shot.
        f_ts = _touch(tmp_path, f"{DEVICE}_3866137959.524.png")
        _touch(tmp_path, f"Scan012_{DEVICE}_002.png")
        aux = pd.DataFrame(
            {
                "Shotnumber": [1, 2],
                "Bin #": [1, 1],
                f"{DEVICE}:acq_timestamp": [3866137959.524, 3866137960.525],
                f"{DEVICE}:valid": [True, False],
            }
        )
        sa = _make_analyzer(tmp_path, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {1: f_ts}

    def test_bluesky_scan_with_sfile_column_spelling_uses_timestamp_join(
        self, tmp_path
    ):
        # Bluesky world with the s-file (space-separated) column spelling:
        # timestamp-named files matching the rows → join maps them all, no
        # fallback needed even though a stray legacy-named file exists.
        f1 = _touch(tmp_path, f"{DEVICE}_3866137959.524.png")
        f2 = _touch(tmp_path, f"{DEVICE}_3866137960.525.png")
        _touch(tmp_path, f"Scan012_{DEVICE}_001.png")
        aux = pd.DataFrame(
            {
                "Shotnumber": [1, 2],
                "Bin #": [1, 1],
                f"{DEVICE} acq_timestamp": [3866137959.524, 3866137960.525],
            }
        )
        sa = _make_analyzer(tmp_path, aux)
        sa._build_data_file_map()
        assert sa._data_file_map == {1: f1, 2: f2}
