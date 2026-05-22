"""Tests for geecs_data_utils.data.columns."""

import pandas as pd
import pytest

from geecs_data_utils.data.columns import (
    find_cols,
    resolve_col,
    resolve_col_detailed,
)


def test_resolve_col_exact_ci_single():
    df = pd.DataFrame({"Charge_nC": [1.0], "other": [2.0]})
    assert resolve_col(df, "charge_nc") == "Charge_nC"


def test_resolve_col_contains_tie_break_shortest_then_lex():
    df = pd.DataFrame(
        {
            "device_laser_readback": [1.0],
            "laser_readback": [2.0],
            "x": [0.0],
        }
    )
    # Both contain "readback"; no column equals spec case-insensitively as a whole
    assert resolve_col(df, "readback", mode="contains") == "laser_readback"


def test_resolve_col_detailed_marks_ambiguous_with_candidates():
    df = pd.DataFrame(
        {
            "ab_laser": [1.0],
            "cd_laser": [2.0],
        }
    )
    r = resolve_col_detailed(df, "laser", mode="contains")
    assert r.ambiguous is True
    assert r.column == "ab_laser"  # same length as cd_laser; lexicographic tie-break
    assert r.candidates is not None
    assert set(r.candidates) == {"ab_laser", "cd_laser"}


def test_resolve_col_startswith_falls_back_to_contains():
    df = pd.DataFrame({"foo_bar": [1.0], "zzz": [0.0]})
    # startswith("bar") matches nothing; fallback contains finds foo_bar
    assert resolve_col(df, "bar", mode="startswith") == "foo_bar"


def test_resolve_col_no_match_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="No columns match"):
        resolve_col(df, "missing")


def test_find_cols_regex():
    df = pd.DataFrame({"device1:readback": [1], "device2:readback": [2]})
    m = find_cols(df, r"device\d", mode="regex")
    assert len(m) == 2


def test_data_package_exports_resolve_col():
    from geecs_data_utils.data import resolve_col as resolve_from_pkg

    df = pd.DataFrame({"charge": [1.0]})
    assert resolve_from_pkg(df, "charge") == "charge"
