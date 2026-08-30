"""Hermetic tests for the OR-of-AND row filters (W1b)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from geecs_data_utils.data.row_filters import (
    FilterCondition,
    FilterGroup,
    RowFilters,
    apply_filters,
    filter_mask,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3, 4, 5],
            "charge": [10.0, 25.0, 40.0, np.nan, 55.0],
            "fwhm": [100.0, 150.0, 200.0, 250.0, 300.0],
            "label": ["a", "b", "c", "d", "e"],  # dtype-tolerant: coerces NaN
        }
    )


def _cond(column, low, high, mode="within") -> FilterCondition:
    return FilterCondition(column=column, low=low, high=high, mode=mode)


class TestConditions:
    def test_within_is_inclusive(self):
        filters = RowFilters(groups=[FilterGroup(conditions=[_cond("charge", 10, 40)])])
        assert filter_mask(_frame(), filters).tolist() == [
            True,
            True,
            True,
            False,
            False,
        ]

    def test_outside_is_the_complement_minus_nan(self):
        filters = RowFilters(
            groups=[FilterGroup(conditions=[_cond("charge", 10, 40, mode="outside")])]
        )
        # NaN fails under 'exclude' for BOTH modes — 'outside' must not
        # silently pass NaN rows through the complement.
        assert filter_mask(_frame(), filters).tolist() == [
            False,
            False,
            False,
            False,
            True,
        ]

    def test_nan_policy_keep_passes_nan_rows(self):
        filters = RowFilters(
            groups=[FilterGroup(conditions=[_cond("charge", 10, 40)])],
            nan_policy="keep",
        )
        assert filter_mask(_frame(), filters).tolist() == [
            True,
            True,
            True,
            True,
            False,
        ]

    def test_string_column_coerces_like_the_pick_list(self):
        # dtype-tolerant contract: a string column coerces (all-NaN here),
        # so under 'exclude' nothing passes — never a crash.
        filters = RowFilters(groups=[FilterGroup(conditions=[_cond("label", 0, 1)])])
        assert not filter_mask(_frame(), filters).any()

    def test_unknown_column_raises_with_a_clear_message(self):
        filters = RowFilters(groups=[FilterGroup(conditions=[_cond("nope", 0, 1)])])
        with pytest.raises(ValueError, match="nope"):
            filter_mask(_frame(), filters)

    def test_inverted_bounds_refused_at_validation(self):
        with pytest.raises(ValidationError):
            _cond("charge", 40, 10)


class TestAlgebra:
    def test_conditions_and_within_a_group(self):
        group = FilterGroup(
            conditions=[_cond("charge", 10, 40), _cond("fwhm", 140, 210)]
        )
        assert filter_mask(_frame(), RowFilters(groups=[group])).tolist() == [
            False,
            True,
            True,
            False,
            False,
        ]

    def test_groups_or_together(self):
        filters = RowFilters(
            groups=[
                FilterGroup(conditions=[_cond("charge", 10, 10)]),  # shot 1
                FilterGroup(conditions=[_cond("fwhm", 300, 300)]),  # shot 5
            ]
        )
        assert filter_mask(_frame(), filters).tolist() == [
            True,
            False,
            False,
            False,
            True,
        ]

    def test_disabled_and_empty_groups_are_ignored(self):
        filters = RowFilters(
            groups=[
                FilterGroup(enabled=False, conditions=[_cond("charge", 0, 0)]),
                FilterGroup(conditions=[]),  # empty: vacuous, must not pass-all
                FilterGroup(conditions=[_cond("fwhm", 300, 300)]),
            ]
        )
        assert filter_mask(_frame(), filters).tolist() == [
            False,
            False,
            False,
            False,
            True,
        ]

    def test_no_active_groups_is_the_identity(self):
        for filters in (
            RowFilters(),
            RowFilters(
                groups=[FilterGroup(enabled=False, conditions=[_cond("charge", 0, 1)])]
            ),
            RowFilters(groups=[FilterGroup(conditions=[])]),
        ):
            assert filter_mask(_frame(), filters).all()

    def test_apply_filters_matches_the_mask(self):
        filters = RowFilters(groups=[FilterGroup(conditions=[_cond("charge", 10, 40)])])
        frame = _frame()
        filtered = apply_filters(frame, filters)
        assert filtered["Shotnumber"].tolist() == [1, 2, 3]
        assert len(filtered) == int(filter_mask(frame, filters).sum())


class TestSerialization:
    def test_round_trips_through_json(self):
        filters = RowFilters(
            groups=[
                FilterGroup(
                    name="beam charge",
                    conditions=[
                        _cond("charge", 18, 60),
                        _cond("fwhm", 120, 240),
                    ],
                ),
                FilterGroup(
                    name="filter 2",
                    conditions=[_cond("charge", 0, 0.5, mode="outside")],
                ),
            ],
            nan_policy="keep",
        )
        # The endpoint/URL/config form: one JSON blob, byte-stable.
        blob = filters.model_dump_json()
        assert RowFilters.model_validate_json(blob) == filters

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            RowFilters.model_validate({"groups": [], "surprise": 1})
