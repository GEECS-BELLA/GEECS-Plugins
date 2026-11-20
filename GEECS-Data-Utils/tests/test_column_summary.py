"""summarize_columns: each column over all rows, as bin_frame reduces one bin."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from geecs_data_utils.data.binning import summarize_columns

FRAME = pd.DataFrame(
    {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, np.nan, 30.0, "x"], "c": [np.nan] * 4}
)


def test_mean_and_std_per_column():
    out = summarize_columns(FRAME, ["a", "b"])
    assert out["a"].center == pytest.approx(2.5)
    assert (
        out["a"].err_low
        == out["a"].err_high
        == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    )
    assert out["a"].count == 4
    # b's NaN and its text cell are left out, not a's rows
    assert out["b"].center == pytest.approx(20.0) and out["b"].count == 2


def test_median_and_quartiles():
    s = summarize_columns(FRAME, ["a"], agg="median", err="iqr")["a"]
    assert s.center == pytest.approx(2.5)
    assert s.center - s.err_low == pytest.approx(1.75)
    assert s.center + s.err_high == pytest.approx(3.25)


def test_standard_error():
    s = summarize_columns(FRAME, ["a"], err="stderr")["a"]
    assert s.err_low == pytest.approx(np.std([1, 2, 3, 4], ddof=1) / 2)


def test_empty_and_absent_columns():
    out = summarize_columns(FRAME, ["c", "nope"])
    assert "nope" not in out
    assert out["c"].count == 0 and math.isnan(out["c"].center)
