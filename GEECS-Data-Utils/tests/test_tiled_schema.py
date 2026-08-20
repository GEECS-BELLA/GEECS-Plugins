"""Unit tests for the one event-schema knowledge module (tiled_schema)."""

from __future__ import annotations

from geecs_data_utils import tiled_schema

COLUMNS = [
    "scan_event_index",
    "bin_number",
    "shot_index_in_bin",
    "mono",
    "cam-counts",
    "cam-acq_timestamp",
    "cam-t0_acq_timestamp",
    "cam-shot_id",
    "cam-shot_offset",
    "cam-valid",
    "cam-nonscalar_save_path",
    "aux-acq_timestamp",
    "aux-signal",
    "telemetry_mag-current",
    "telemetry_mag-acq_timestamp",
    "telemetry_mode-label",
]


class TestColumnFamilies:
    def test_companion_columns_detected(self):
        for suffix in tiled_schema.COMPANION_SUFFIXES:
            assert tiled_schema.is_companion_column(f"cam{suffix}")
        assert not tiled_schema.is_companion_column("cam-counts")

    def test_data_columns_exclude_id_and_companions(self):
        data = tiled_schema.data_columns(COLUMNS)
        assert "cam-counts" in data
        assert "mono" in data
        assert "telemetry_mag-current" in data
        assert "scan_event_index" not in data
        assert "cam-shot_id" not in data
        assert "cam-valid" not in data
        assert "telemetry_mag-acq_timestamp" not in data

    def test_telemetry_columns(self):
        telemetry = tiled_schema.telemetry_columns(COLUMNS)
        assert telemetry == ["telemetry_mag-current", "telemetry_mode-label"]


class TestDisplayName:
    def test_legacy_header_wins(self):
        headers = {"cam-counts": "UC_Cam Counts"}
        assert tiled_schema.display_name("cam-counts", headers) == "UC_Cam Counts"

    def test_fallback_splits_device_and_variable(self):
        assert tiled_schema.display_name("cam-counts", None) == "cam : counts"

    def test_telemetry_prefix_stripped_and_marked(self):
        name = tiled_schema.display_name("telemetry_mag-current", None)
        assert name == "mag : current [t]"

    def test_plain_column_passes_through(self):
        assert tiled_schema.display_name("mono", None) == "mono"


class TestPinnedColumns:
    def test_reference_device_timestamp_pinned(self):
        start = {"reference_device": "aux"}
        pinned = tiled_schema.pinned_columns(COLUMNS, start)
        assert pinned == ["scan_event_index", "aux-acq_timestamp"]

    def test_first_acq_timestamp_without_reference(self):
        pinned = tiled_schema.pinned_columns(COLUMNS, {})
        assert pinned == ["scan_event_index", "cam-acq_timestamp"]

    def test_no_sync_device(self):
        pinned = tiled_schema.pinned_columns(
            ["scan_event_index", "bin_number", "snap-value"], {}
        )
        assert pinned == ["scan_event_index"]


class TestScanClassification:
    def test_noscan(self):
        assert tiled_schema.scan_mode({"motor": None}) == "NOSCAN"
        assert not tiled_schema.is_stepped_scan({"motor": None})

    def test_1d(self):
        start = {"motor": "mono", "plan_name": "geecs_step_scan"}
        assert tiled_schema.scan_mode(start) == "1D"
        assert tiled_schema.is_stepped_scan(start)

    def test_grid_from_motor_list(self):
        assert tiled_schema.scan_mode({"motor": ["m1", "m2"]}) == "GRID"

    def test_grid_from_grid_shape(self):
        assert tiled_schema.scan_mode({"motor": "m1", "grid_shape": [3, 4]}) == "GRID"

    def test_optimization(self):
        assert tiled_schema.scan_mode({"plan_name": "geecs_adaptive_scan"}) == "OPT"

    def test_scan_variable_columns(self):
        start = {"motor": "mono"}
        assert tiled_schema.scan_variable_columns(COLUMNS, start) == ["mono"]
        assert tiled_schema.scan_variable_columns(COLUMNS, {"motor": None}) == []

    def test_scan_variable_columns_prefixed_readback(self):
        columns = ["scan_event_index", "hex-ypos", "cam-counts"]
        assert tiled_schema.scan_variable_columns(columns, {"motor": "hex"}) == [
            "hex-ypos"
        ]

    def test_total_shots(self):
        assert tiled_schema.total_shots({"num_points": 5, "shots_per_step": 4}) == 20
        assert tiled_schema.total_shots({}) is None
