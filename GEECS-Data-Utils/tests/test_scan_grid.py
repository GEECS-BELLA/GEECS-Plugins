"""Grid geometry must survive filters, snake traversal and partial scans."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from geecs_data_utils.data.row_filters import RowFilters
from geecs_data_utils.scan_grid import GridConfig, grid_scan


def grid_start():
    return {
        "motors": ["slow", "fast"],
        "plan_name": "list_grid_scan",
        "plan_pattern": "outer_list_product",
        "plan_pattern_args": {
            "args": ["slow", [1, 10], "fast", [3, 1, 2]],
            "snake_axes": "True",
        },
    }


def grid_frame():
    # Bin 4 revisits x=2 at y=10 as the fast axis reverses.
    return pd.DataFrame(
        {
            "bin_number": np.repeat([1, 2, 3, 4, 5], 3),
            "scan_event_index": np.arange(1, 16),
            "slow": np.repeat([1, 1, 1, 10, 10], 3),
            "fast": np.repeat([3, 1, 2, 2, 1], 3),
            "signal": [1, 2, 9, 2, 4, 6, 3, 6, 9, 4, 8, 12, 5, 10, 15],
        }
    )


def test_snake_positions_not_bin_reshape():
    r = grid_scan(grid_frame(), grid_start(), GridConfig(value="signal"))
    c = r.cells.set_index("bin")
    assert r.kind == "grid"
    assert tuple(c.loc[4, ["x", "y"]]) == (2, 10)
    assert tuple(c.loc[6, ["x", "y"]]) == (3, 10)
    assert c.loc[6, "status"] == "unacquired"
    assert c.loc[1, "center"] == 2
    assert c.loc[1, "error"] == 4  # Q75 - Q25 = 5.5 - 1.5


def test_filter_removes_values_not_geometry():
    f = RowFilters.model_validate(
        {
            "groups": [
                {"conditions": [{"column": "scan_event_index", "low": 4, "high": 5}]}
            ]
        }
    )
    r = grid_scan(grid_frame(), grid_start(), GridConfig(value="signal"), f)
    assert len(r.cells) == 6
    c = r.cells.set_index("bin")
    assert c.loc[1, "status"] == "filtered"
    assert c.loc[2, "count"] == 2 and c.loc[2, "status"] == "low_count"
    assert c.loc[2, "shots"] == [4, 5]
    assert np.isnan(c.loc[2, "error"])
    assert r.passing == 2


@pytest.mark.parametrize("average", ["mean", "median"])
@pytest.mark.parametrize("error", ["std", "stderr", "mad", "iqr", "percentile"])
def test_independent_statistics(average, error):
    r = grid_scan(
        grid_frame(),
        grid_start(),
        GridConfig(value="signal", average=average, error=error),
    )
    c = r.cells.iloc[0]
    assert c.center == (4 if average == "mean" else 2)
    if error == "std":
        assert c.error == pytest.approx(np.std([1, 2, 9], ddof=1))
    if error == "stderr":
        assert c.error == pytest.approx(np.std([1, 2, 9], ddof=1) / np.sqrt(3))
    if error == "mad":
        assert c.error == pytest.approx(1.4826)


def test_interval_width_is_true_quantile_width_when_mean_outside_interval():
    f = grid_frame()
    f["signal"] = f.signal.astype(float)
    f.loc[:2, "signal"] = [0, 0, 100]
    cfg = GridConfig(
        value="signal", average="mean", error="percentile", lower=0.1, upper=0.2
    )
    assert grid_scan(f, grid_start(), cfg).cells.iloc[0].error == 0


def test_readback_jitter_and_filters_do_not_move_planned_cells():
    f = grid_frame()
    f.fast += np.arange(len(f)) / 1000
    r = grid_scan(f, grid_start(), GridConfig(value="signal"))
    assert r.x_values == [1, 2, 3]
    assert r.cells.iloc[0].x_readback == pytest.approx(3.001)


def test_repeated_coordinates_are_separate_visits():
    start = grid_start()
    start["plan_pattern_args"]["args"][3] = [3, 3, 2]
    r = grid_scan(grid_frame(), start, GridConfig(value="signal"))
    assert r.visits == 2
    assert r.cells.iloc[0].visit == 1 and r.cells.iloc[1].visit == 2
    assert r.cells.iloc[0].center != r.cells.iloc[1].center


def test_log_sweep_positions_and_swapped_axes():
    start = {
        "motors": ["slow", "fast"],
        "plan_name": "sweep",
        "sweep": {
            "trajectory": {
                "kind": "axes",
                "combine": "product",
                "snake": True,
                "axes": [
                    {"kind": "log", "start_exp": 0, "stop_exp": 1, "num": 2},
                    {"kind": "list", "positions": [3, 1, 2]},
                ],
            }
        },
    }
    r = grid_scan(
        grid_frame(),
        start,
        GridConfig(x="slow", y="fast", value="signal", xscale="log"),
    )
    assert r.x_values == [1, 10] and r.y_values == [1, 2, 3]
    assert tuple(r.cells.iloc[3][["x", "y"]]) == (10, 2)


def test_relative_grid_uses_readback_anchor():
    start = grid_start()
    start["plan_name"] = "rel_list_grid_scan"
    f = grid_frame()
    f.slow += 100
    f.fast += 200
    r = grid_scan(f, start, GridConfig(value="signal"))
    assert r.x_values == [201, 202, 203] and r.y_values == [101, 110]


def test_missing_signal_counts_selected_scalar_only():
    f = grid_frame()
    f["signal"] = f.signal.astype(float)
    f.loc[:2, "signal"] = [np.nan, np.inf, np.nan]
    f["unrelated"] = np.nan
    r = grid_scan(f, grid_start(), GridConfig(value="signal"))
    assert r.cells.iloc[0].status == "missing"
    assert r.cells.iloc[1]["count"] == 3


def test_spiral_and_together_are_points_not_fabricated_raster():
    start = {
        "motors": ["slow", "fast"],
        "plan_name": "spiral",
        "plan_pattern": "spiral",
    }
    r = grid_scan(grid_frame(), start, GridConfig(value="signal"))
    assert r.kind == "points" and len(r.cells) == 5


def test_legacy_sfile_bin_identity():
    f = grid_frame().rename(
        columns={"bin_number": "Bin #", "scan_event_index": "Shotnumber"}
    )
    assert grid_scan(f, grid_start(), GridConfig(value="signal")).bin_column == "Bin #"


@pytest.mark.parametrize(
    "changes",
    [
        {"x": "slow", "y": "slow"},
        {"lower": 0.9, "upper": 0.1},
        {"xscale": "log", "x": "signal", "y": "slow"},
    ],
)
def test_invalid_choices(changes):
    f = grid_frame()
    if changes.get("xscale") == "log":
        f.signal = -1
    with pytest.raises(ValueError):
        grid_scan(f, grid_start(), GridConfig(value="signal", **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"min_count": True},
        {"visit": 0},
        {"average": []},
        {"xscale": "foo"},
        {"lower": "0.1"},
    ],
)
def test_config_types(changes):
    with pytest.raises(ValidationError):
        GridConfig(**changes)


def test_oversized_grid_refused_before_product_allocation():
    start = {
        "motors": ["slow", "fast"],
        "plan_pattern": "outer_product",
        "shape": [20000, 20000],
        "extents": [[0, 1], [0, 1]],
    }
    with pytest.raises(ValueError, match="exceeds"):
        grid_scan(grid_frame(), start, GridConfig(value="signal"))


def test_more_than_two_motors_is_not_silently_projected():
    start = grid_start()
    start["motors"].append("third")
    with pytest.raises(ValueError, match="higher-dimensional"):
        grid_scan(grid_frame(), start, GridConfig(value="signal"))


def test_bin_namespaces_are_not_reconciled():
    f = grid_frame()
    f["Bin #"] = f.bin_number
    f.loc[:2, "bin_number"] = np.nan
    r = grid_scan(f, grid_start(), GridConfig(value="signal"))
    assert r.bin_column == "bin_number"
    assert r.cells.iloc[0].status == "unacquired"
    assert r.cells.iloc[0].shots == []
    f["bin_number"] = np.nan
    assert grid_scan(f, grid_start(), GridConfig(value="signal")).bin_column == "Bin #"


def test_iqr_uses_quartiles_even_after_custom_percentiles():
    cfg = GridConfig(value="signal", error="iqr", lower=0.1, upper=0.9)
    result = grid_scan(grid_frame(), grid_start(), cfg)
    assert result.config.lower == 0.25 and result.config.upper == 0.75
    assert result.cells.iloc[0].error == 4
