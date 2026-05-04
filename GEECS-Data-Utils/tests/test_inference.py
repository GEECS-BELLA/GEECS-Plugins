"""Tests for ml.inference column resolution (Task C: shared find_cols)."""

import pandas as pd

from geecs_data_utils.ml.inference import _resolve_feature_map


def test_resolve_feature_map_unions_find_cols_matches():
    df = pd.DataFrame({"laser_a": [1.0], "laser_b": [2.0], "charge": [3.0]})
    # Each spec contributes matches; expected names must appear in that pool
    m = _resolve_feature_map(df, ["laser", "charge"], ["laser_a", "charge"])
    assert m == {"laser_a": "laser_a", "charge": "charge"}


def test_resolve_feature_map_exact_column_when_no_substring_match():
    df = pd.DataFrame({"exact_name": [1.0]})
    m = _resolve_feature_map(df, ["exact_name"], ["exact_name"])
    assert m == {"exact_name": "exact_name"}
