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
from geecs_data_utils.data.scan_merge import (
    MERGED_BIN_COLUMN,
    default_bin_tolerance,
    merge_scan_frames,
    merged_bin_numbers,
    with_merged_bins,
)

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


def test_an_own_scan_column_naming_the_scan_is_the_tag():
    # The Bluesky s-files carry ``scan``: the scan number on every row.
    frame = _scan(2)
    frame["scan"] = [6, 6]
    merged = merge_scan_frames({(D1, 6): frame, (D2, 3): _scan(1)})
    assert merged["scan"].tolist() == [6, 6, 3]
    assert list(merged.columns).count("scan") == 1


def test_an_own_day_column_naming_the_day_is_the_tag():
    frame = _scan(1)
    frame["day"] = ["2026-10-26"]
    assert merge_scan_frames({(D1, 1): frame})["day"].tolist() == ["2026-10-26"]


@pytest.mark.parametrize(
    "name, values",
    [("scan", [5, 6]), ("scan", ["a", "a"]), ("day", ["2026-10-27"] * 2)],
)
def test_an_own_column_naming_another_scan_is_refused(name, values):
    frame = _scan(2)
    frame[name] = values
    with pytest.raises(ValueError, match=name):
        merge_scan_frames({(D1, 6): frame})


def test_shot_numbers_in_a_named_index_survive():
    # ScanData's TDMS frame is indexed by Shotnumber.
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([1, 2], name="Shotnumber"))
    merged = merge_scan_frames({(D1, 1): frame, (D1, 2): frame})
    assert merged["Shotnumber"].tolist() == [1, 2, 1, 2]
    assert merged.index.tolist() == [0, 1, 2, 3]


def test_multi_level_columns_are_refused():
    frame = pd.DataFrame([[1.0]], columns=pd.MultiIndex.from_tuples([("dev", "x")]))
    with pytest.raises(ValueError, match="multi-level"):
        merge_scan_frames({(D1, 1): frame})


# --- one bin number across merged scans -------------------------------------


def _binned_scan(xs, bins):
    import pandas as pd

    return pd.DataFrame({"Bin #": bins, "pos": xs, "Shotnumber": range(1, len(xs) + 1)})


def _two_scans():
    from datetime import date

    # Scan 1 steps 1, 2, 3 (a readback that wobbles); scan 2 steps 2, 4.
    a = _binned_scan([1.0, 1.01, 2.0, 1.98, 3.0, 3.02], [1, 1, 2, 2, 3, 3])
    b = _binned_scan([2.03, 1.99, 4.0, 4.01], [1, 1, 2, 2])
    return merge_scan_frames({(date(2026, 11, 3), 5): a, (date(2026, 11, 4), 2): b})


def test_the_same_setting_in_two_scans_is_one_bin():
    found = merged_bin_numbers(_two_scans(), "pos")
    # Bins at 1, 2 (scan 1's bin 2 and scan 2's bin 1, pooled), 3, 4.
    assert found.numbers.tolist() == [1, 1, 2, 2, 3, 3, 2, 2, 4, 4]
    assert found.tolerance == pytest.approx(0.985 / 2)


def test_the_default_tolerance_is_half_the_smallest_step():
    assert default_bin_tolerance(_two_scans(), "pos") == pytest.approx(0.985 / 2)


def test_a_tolerance_of_zero_keeps_the_scans_apart():
    found = merged_bin_numbers(_two_scans(), "pos", tolerance=0.0)
    # 2.0 (scan 1) and 2.01 (scan 2) now differ: five bins, still by X.
    assert found.numbers.tolist() == [1, 1, 2, 2, 4, 4, 3, 3, 5, 5]


def test_each_bin_with_no_x_value_is_a_bin_of_its_own():
    """A scan without the X column keeps its bins apart, after the placed ones.

    One shared NaN would make binning average all of them as one setting.
    """
    import numpy as np

    from geecs_data_utils.data.binning import BinningConfig, bin_rows

    frame = _two_scans()
    frame.loc[frame["scan"] == 2, "pos"] = np.nan
    pooled, _ = with_merged_bins(frame, "pos")
    assert pooled[MERGED_BIN_COLUMN].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    groups = bin_rows(pooled, BinningConfig(bin_col=MERGED_BIN_COLUMN))
    assert [len(rows) for _, rows in groups] == [2, 2, 2, 2, 2]


def test_without_an_x_the_bins_follow_scan_and_bin_order():
    found = merged_bin_numbers(_two_scans(), None)
    assert found.numbers.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_a_scan_without_bin_numbers_is_one_group():
    from datetime import date

    import pandas as pd

    a = _binned_scan([1.0, 2.0], [1, 2])
    b = pd.DataFrame({"pos": [5.0, 5.1], "Shotnumber": [1, 2]})
    merged = merge_scan_frames({(date(2026, 1, 1), 1): a, (date(2026, 1, 1), 2): b})
    assert merged_bin_numbers(merged, "pos").numbers.tolist() == [1, 2, 3, 3]


def test_a_negative_tolerance_is_refused():
    with pytest.raises(ValueError, match="0 or more"):
        merged_bin_numbers(_two_scans(), "pos", tolerance=-1)


def test_with_merged_bins_adds_a_column_and_keeps_bin():
    frame = _two_scans()
    out, tolerance = with_merged_bins(frame, "pos")
    assert MERGED_BIN_COLUMN in out and MERGED_BIN_COLUMN not in frame
    assert out["Bin #"].tolist() == frame["Bin #"].tolist()
    assert tolerance == pytest.approx(0.985 / 2)
