"""Unit tests for the one event-schema knowledge module (tiled_schema)."""

from __future__ import annotations

import pytest

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


class TestPlottableColumns:
    """The shared front-end pick-list rule (console B4 + data portal)."""

    def _frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "scan_event_index": [0, 1],  # id machinery
                "cam-MaxCounts": [1.0, 2.0],
                "cam-acq_timestamp": [1.0, 2.0],  # companion machinery
                "cam-label": ["a", "b"],  # non-numeric
                "telemetry_dev-val": ["1.5", "2.5"],  # numeric strings
                "cam-dead": [float("nan"), float("nan")],  # all-NaN
            }
        )

    def test_machinery_and_unplottable_excluded(self):
        from geecs_data_utils.tiled_schema import plottable_columns

        assert plottable_columns(self._frame()) == [
            "cam-MaxCounts",
            "telemetry_dev-val",
        ]

    def test_numeric_series_coerces_strings(self):
        from geecs_data_utils.tiled_schema import numeric_series

        series = numeric_series(self._frame(), "telemetry_dev-val")
        assert series is not None
        assert list(series) == [1.5, 2.5]

    def test_numeric_series_none_for_absent_all_nan_and_strings(self):
        from geecs_data_utils.tiled_schema import numeric_series

        frame = self._frame()
        assert numeric_series(frame, "missing") is None
        assert numeric_series(frame, "cam-dead") is None
        assert numeric_series(frame, "cam-label") is None

    def test_numeric_series_handles_nullable_dtypes_without_raising(self):
        # pd.to_numeric keeps Int64/Float64 (pd.NA) — a per-value float()
        # loop crashes on NA; the vectorized path must not.
        import pandas as pd

        from geecs_data_utils.tiled_schema import numeric_series

        frame = pd.DataFrame({"n": pd.array([1, None], dtype="Int64")})
        series = numeric_series(frame, "n")
        assert series is not None
        assert series.dtype == float and series.iloc[0] == 1.0
        all_na = pd.DataFrame({"n": pd.array([None, None], dtype="Int64")})
        assert numeric_series(all_na, "n") is None

    def test_numeric_series_rejects_datetimes_and_duplicate_labels(self):
        # datetime64 would coerce to "plottable" ~1e18 ns ints; a
        # duplicated label makes frame[column] a DataFrame.
        import pandas as pd

        from geecs_data_utils.tiled_schema import numeric_series

        frame = pd.DataFrame({"t": pd.to_datetime(["2026-08-29", "2026-08-30"])})
        assert numeric_series(frame, "t") is None
        dup = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "a"])
        assert numeric_series(dup, "a") is None


class TestDeviceAcqTimestampColumn:
    """Schema-safe device→column matching (the ScanAnalysis rule)."""

    COLUMNS = [
        "uc_amp4_ir_input-acq_timestamp",
        "ts_uc_amp4_ir_input-acq_timestamp",
        "telemetry_uc_other-acq_timestamp",
        "u_bcavemagspec_interpspec-acq_timestamp",
        "uc_amp4_ir_input-maxcounts",
    ]

    def test_folder_stem_matches_schema_safe_column(self):
        from geecs_data_utils.tiled_schema import device_acq_timestamp_column

        assert (
            device_acq_timestamp_column(self.COLUMNS, "UC_Amp4_IR_input")
            == "uc_amp4_ir_input-acq_timestamp"
        )

    def test_hyphenated_diagnostic_stem_matches(self):
        from geecs_data_utils.tiled_schema import device_acq_timestamp_column

        assert (
            device_acq_timestamp_column(self.COLUMNS, "U_BCaveMagSpec-interpSpec")
            == "u_bcavemagspec_interpspec-acq_timestamp"
        )

    def test_companion_and_telemetry_prefixes_never_match(self):
        from geecs_data_utils.tiled_schema import device_acq_timestamp_column

        assert device_acq_timestamp_column(self.COLUMNS, "uc_other") is None

    def test_unknown_device_is_none(self):
        from geecs_data_utils.tiled_schema import device_acq_timestamp_column

        assert device_acq_timestamp_column(self.COLUMNS, "nope") is None


class TestTimestampColumns:
    """ts_ event-recording times + the two-epoch convention (W1e)."""

    def test_is_key_timestamp_column(self):
        from geecs_data_utils.tiled_schema import is_key_timestamp_column

        assert is_key_timestamp_column("ts_cam-MaxCounts")
        assert is_key_timestamp_column("ts_telemetry_dev-val")
        assert not is_key_timestamp_column("cam-MaxCounts")
        assert not is_key_timestamp_column("telemetry_dev-val")

    def test_timestamp_epoch_two_conventions(self):
        from geecs_data_utils.tiled_schema import timestamp_epoch

        # Reader-side ts_ columns: Unix epoch (Bluesky event times).
        assert timestamp_epoch("ts_cam-MaxCounts") == "unix"
        # acq_timestamp in ANY spelling: LabVIEW epoch (the wire
        # convention) — event companions and s-file headers alike.
        assert timestamp_epoch("cam-acq_timestamp") == "labview"
        assert timestamp_epoch("UC_Amp4_IR_input acq_timestamp") == "labview"
        assert timestamp_epoch("cam-MaxCounts") is None


class TestSteppedDevicesFromEitherBackend:
    """The stepped-device key differs between the two GEECS scan backends.

    Stock ``bluesky.plans`` verbs — which the native-Bluesky scanner registers
    — write ``motors`` (plural, a list). The retired GEECS funnel wrote
    ``motor`` (singular). These readers looked only for the singular key, so
    every 1D scan taken on the native path classified as ``NOSCAN``,
    contributed no scan-variable column, and reported ``is_stepped_scan() is
    False`` — while its own ``ScanInfo`` ini correctly said
    ``ScanMode = "standard"``. Found in the data portal on 26_0912's Scan018.
    """

    # Exactly what Scan018 (a native rel_scan over one axis) carries.
    NATIVE_1D = {
        "plan_name": "rel_scan",
        "motors": ["u_compaerotech-position_axis1"],
        "num_points": 5,
        "shots_per_step": 2,
        "acquisition": "gated",
    }
    FUNNEL_1D = {
        "plan_name": "geecs_step_scan",
        "motor": "u_compaerotech-position_axis1",
    }
    NATIVE_GRID = {
        "plan_name": "grid_scan",
        "motors": ["u_s1h-current", "u_s1v-current"],
    }
    NATIVE_COUNT = {"plan_name": "count", "num_points": 5}

    def test_a_native_1d_scan_is_not_a_noscan(self):
        assert tiled_schema.scan_mode(self.NATIVE_1D) == "1D"

    def test_a_funnel_1d_scan_still_works(self):
        assert tiled_schema.scan_mode(self.FUNNEL_1D) == "1D"

    def test_a_native_grid_is_a_grid(self):
        assert tiled_schema.scan_mode(self.NATIVE_GRID) == "GRID"

    def test_a_motorless_run_is_still_a_noscan(self):
        assert tiled_schema.scan_mode(self.NATIVE_COUNT) == "NOSCAN"

    def test_is_stepped_scan_sees_a_native_scan(self):
        assert tiled_schema.is_stepped_scan(self.NATIVE_1D) is True
        assert tiled_schema.is_stepped_scan(self.FUNNEL_1D) is True
        assert tiled_schema.is_stepped_scan(self.NATIVE_COUNT) is False

    def test_the_scanned_axis_column_is_found_for_a_native_scan(self):
        """Not just the chip: the portal could not identify the X axis either."""
        columns = [
            "u_compaerotech-position_axis1",
            "uc_amp4_ir_input-meancounts",
            "shot_index",
        ]
        assert tiled_schema.scan_variable_columns(columns, self.NATIVE_1D) == [
            "u_compaerotech-position_axis1"
        ]
        assert tiled_schema.scan_variable_columns(columns, self.NATIVE_COUNT) == []

    def test_scan_motors_normalises_both_shapes(self):
        assert tiled_schema.scan_motors(self.NATIVE_1D) == [
            "u_compaerotech-position_axis1"
        ]
        assert tiled_schema.scan_motors(self.FUNNEL_1D) == [
            "u_compaerotech-position_axis1"
        ]
        assert tiled_schema.scan_motors({}) == []
        assert tiled_schema.scan_motors({"motors": []}) == []


class TestStockPlanPatternDecidesGrid:
    """A list of motors is not a grid on the native path.

    Stock ``scan`` / ``rel_scan`` / ``list_scan`` move N motors along ONE
    correlated trajectory (``plan_pattern`` ``inner_product`` /
    ``inner_list_product``) and are 1D however many motors they name; only
    ``grid_scan``'s ``outer_product`` is a grid. The funnel had no
    ``plan_pattern`` and only ever wrote a list for a grid, so its documents
    keep the old reading — which is why the discriminator has to be the
    pattern, not the motor count.
    """

    def test_a_two_motor_inner_product_scan_is_1d_not_a_grid(self):
        assert (
            tiled_schema.scan_mode(
                {
                    "plan_name": "scan",
                    "motors": ["u_s1h-current", "u_s2h-current"],
                    "plan_pattern": "inner_product",
                }
            )
            == "1D"
        )

    def test_a_two_motor_list_scan_is_1d(self):
        assert (
            tiled_schema.scan_mode(
                {
                    "plan_name": "list_scan",
                    "motors": ["u_s1h-current", "u_s2h-current"],
                    "plan_pattern": "inner_list_product",
                }
            )
            == "1D"
        )

    def test_an_outer_product_grid_scan_is_a_grid(self):
        assert (
            tiled_schema.scan_mode(
                {
                    "plan_name": "grid_scan",
                    "motors": ("u_s1h-current", "u_s2h-current"),
                    "plan_pattern": "outer_product",
                }
            )
            == "GRID"
        )

    def test_a_funnel_motor_list_is_still_a_grid(self):
        """No plan_pattern: the funnel only ever wrote a list for a grid."""
        assert (
            tiled_schema.scan_mode(
                {
                    "plan_name": "geecs_step_scan",
                    "motor": ["u_s1h-current", "u_s2h-current"],
                }
            )
            == "GRID"
        )


class TestScanMotorsShapes:
    """Every shape either backend can actually put in a start document."""

    def test_a_tuple_is_accepted(self):
        """grid_scan's in-memory shape is a tuple; only JSON makes it a list."""
        assert tiled_schema.scan_motors({"motors": ("a", "b")}) == ["a", "b"]

    def test_an_empty_plural_does_not_shadow_a_populated_singular(self):
        """A merged or patched start doc can carry both; the real one wins."""
        assert tiled_schema.scan_motors({"motors": [], "motor": "m1"}) == ["m1"]
        assert tiled_schema.scan_motors({"motors": None, "motor": "m1"}) == ["m1"]

    def test_a_populated_plural_wins_over_a_singular(self):
        assert tiled_schema.scan_motors({"motors": ["a"], "motor": "b"}) == ["a"]

    def test_a_non_iterable_does_not_raise(self):
        """The old code raised on this; the helper is the tolerant reader."""
        assert tiled_schema.scan_motors({"motor": 3}) == ["3"]


@pytest.mark.parametrize(
    "trajectory, expected",
    [
        ({"kind": "axes", "combine": "zip"}, "1D"),
        ({"kind": "axes", "combine": "product"}, "GRID"),
        ({"kind": "spiral"}, "1D"),
        ({"kind": "x2x"}, "1D"),
    ],
)
def test_sweep_classification_uses_combination_not_motor_count(trajectory, expected):
    start = {
        "plan_name": "sweep",
        "motors": ["a", "b", "c", "d", "e"],
        "sweep": {"trajectory": trajectory},
    }
    assert tiled_schema.scan_mode(start) == expected
