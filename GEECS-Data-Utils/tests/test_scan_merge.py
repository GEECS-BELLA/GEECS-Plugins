"""Hermetic tests for merging several scans' frames into one."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from geecs_data_utils.data.row_filters import (
    FilterCondition,
    FilterGroup,
    RowFilters,
    apply_filters,
)
from geecs_data_utils.data.scan_merge import merge_scan_frames

D1 = date(2026, 10, 26)
D2 = date(2026, 10, 27)


def _scan(shots: int, **columns) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Shotnumber": list(range(1, shots + 1)),
            "Bin #": [1] * shots,
            **columns,
        }
    )


def test_rows_are_stacked_in_order_and_tagged():
    merged = merge_scan_frames(
        {
            (D1, 6): _scan(2, charge=[1.0, 2.0]),
            (D2, 3): _scan(3, charge=[3.0, 4.0, 5.0]),
        }
    )
    assert merged["charge"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert merged["scan"].tolist() == [6, 6, 3, 3, 3]
    assert merged["day"].tolist() == ["2026-10-26"] * 2 + ["2026-10-27"] * 3
    assert list(merged.columns[:2]) == ["scan", "day"]


def test_each_shot_keeps_its_own_number():
    merged = merge_scan_frames({(D1, 1): _scan(2), (D1, 2): _scan(2)})
    assert merged["Shotnumber"].tolist() == [1, 2, 1, 2]
    assert merged["Bin #"].tolist() == [1, 1, 1, 1]


def test_the_index_is_fresh_and_unique():
    # Two s-files both indexed 0..n-1: the merged labels must not repeat,
    # or a filtered frame's index would name two rows at once.
    merged = merge_scan_frames({(D1, 1): _scan(2), (D1, 2): _scan(3)})
    assert merged.index.tolist() == [0, 1, 2, 3, 4]


def test_columns_are_the_union_missing_is_nan():
    merged = merge_scan_frames(
        {
            (D1, 1): _scan(1, a=[1.0], b=[2.0]),
            (D1, 2): _scan(1, b=[3.0], c=[4.0]),
        }
    )
    assert list(merged.columns) == ["scan", "day", "Shotnumber", "Bin #", "a", "b", "c"]
    assert np.isnan(merged["a"].iloc[1])
    assert np.isnan(merged["c"].iloc[0])
    assert merged["b"].tolist() == [2.0, 3.0]


def test_one_scan_is_that_scan_tagged():
    frame = _scan(3, charge=[1.0, 2.0, 3.0])
    merged = merge_scan_frames({(D1, 7): frame})
    pd.testing.assert_frame_equal(merged.drop(columns=["scan", "day"]), frame)


def test_the_inputs_are_not_changed():
    frame = _scan(2, charge=[1.0, 2.0])
    before = frame.copy()
    merge_scan_frames({(D1, 1): frame})
    pd.testing.assert_frame_equal(frame, before)


def test_shot_filters_can_name_the_scan():
    merged = merge_scan_frames(
        {(D1, 1): _scan(2, x=[1.0, 2.0]), (D1, 2): _scan(2, x=[3.0, 4.0])}
    )
    only_two = RowFilters(
        groups=[FilterGroup(conditions=[FilterCondition(column="scan", low=2, high=2)])]
    )
    assert apply_filters(merged, only_two)["x"].tolist() == [3.0, 4.0]


def test_nothing_to_merge_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        merge_scan_frames({})


@pytest.mark.parametrize("name", ["scan", "day"])
def test_a_frame_with_a_tag_column_is_refused(name):
    frame = _scan(1)
    frame[name] = [5]
    with pytest.raises(ValueError, match=name):
        merge_scan_frames({(D1, 1): frame})
