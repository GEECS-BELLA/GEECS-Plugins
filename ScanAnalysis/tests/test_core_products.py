"""Scan product plans preserve bin/scalar membership and waterfall ordering."""

import numpy as np
import pandas as pd
import pytest
from geecs_analysis.compat.v2 import compile_v2
from geecs_analysis.compat.v2_run import ShotGroup, UnitResult
from geecs_analysis.measurement import Measurement
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import AnalysisDiagnostic

from scan_analysis.core_products import plan_products


def recipe(line=False):
    image = (
        {"type": "line", "data_loading": {"data_type": "npy"}}
        if line
        else {"type": "camera"}
    )
    return compile_v2(
        AnalysisDiagnostic.model_validate(
            {
                "name": "Device",
                "analyzer": {"kind": "line" if line else "beam"},
                "image": image,
            }
        )
    )


def outcome(key, value, *, line=False, members=None, shape=None):
    if shape is not None:
        frame = Frame.from_array(np.full(shape, value))
    elif line:
        frame = Frame.from_trace([[1, value], [2, value + 1], [3, value + 2]])
    else:
        frame = Frame.from_array(np.full((2, 2), value))
    return UnitResult(
        ShotGroup(key, members or (key,)), (key,), Measurement({"signal": value}, frame)
    )


def rows():
    return pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3, 4, 5, 6],
            "Bin #": [1, 1, 2, 2, 3, 3],
            "motor": [10, 12, 20, 22, 30, 32],
        }
    )


@pytest.mark.parametrize("line", [False, True])
def test_scanned_per_shot_bins_use_all_scalar_rows_for_positions(line):
    outcomes = [outcome(n, n, line=line) for n in (5, 1, 3)]
    frame = rows()
    original = frame.copy(deep=True)
    plan = plan_products(
        recipe(line),
        outcomes,
        frame,
        average_before_analysis=False,
        noscan=False,
        parameter_column="motor",
    )
    assert [p.identifier for p in plan.singles] == [1, 2, 3]
    assert [p.position for p in plan.singles] == [11, 21, 31]
    assert [p.measurement.scalars["signal"] for p in plan.singles] == [1, 3, 5]
    assert plan.position_label == "motor"
    pd.testing.assert_frame_equal(frame, original)


def test_preanalyzed_raw_bins_are_reused_without_second_average():
    outcomes = [outcome(n, n * 10, members=(2 * n - 1, 2 * n)) for n in (3, 1, 2)]
    plan = plan_products(
        recipe(),
        outcomes,
        rows(),
        average_before_analysis=True,
        noscan=False,
        parameter_column="motor",
    )
    assert [p.measurement.scalars["signal"] for p in plan.singles] == [10, 20, 30]
    assert plan.singles[0].measurement is outcomes[1].measurement
    assert [p.position for p in plan.singles] == [11, 21, 31]


@pytest.mark.parametrize("line", [False, True])
def test_noscan_average_uses_all_units_while_waterfall_orders_by_key(line):
    outcomes = [outcome(n, n, line=line) for n in (3, 1, 2)]
    plan = plan_products(
        recipe(line), outcomes, rows(), average_before_analysis=False, noscan=True
    )
    assert plan.singles[0].identifier == "average"
    assert plan.singles[0].measurement.scalars["signal"] == 2
    assert not plan.singles[0].measurement.overlays
    if line:
        assert [p.identifier for p in plan.summary] == [1, 2, 3]
        assert [p.position for p in plan.summary] == [1, 2, 3]
        assert plan.position_label == "Shot Number"
    else:
        assert not plan.summary


def test_sort_key_bypasses_scan_bins_and_filters_only_the_waterfall():
    outcomes = [outcome(n, n, line=True) for n in range(1, 7)]
    frame = rows().assign(charge=[3, np.nan, 1, 100, 2, np.inf])
    plan = plan_products(
        recipe(True),
        outcomes,
        frame,
        average_before_analysis=False,
        noscan=False,
        parameter_column="motor",
        sort_requested=True,
        sort_column="charge",
        sort_bounds=(1, 3),
        sort_sigma=0,
    )
    assert [p.identifier for p in plan.summary] == [3, 5, 1]
    assert [p.position for p in plan.summary] == [1, 2, 3]
    assert plan.singles[0].identifier == "average"
    assert plan.singles[0].measurement.scalars["signal"] == 3.5
    assert plan.notes == ("Waterfall sort excluded 3 units",)


def test_sigma_filter_uses_finite_values_and_population_std():
    outcomes = [outcome(n, n, line=True) for n in range(1, 7)]
    frame = rows().assign(charge=[1, 2, 3, 4, 100, np.nan])
    plan = plan_products(
        recipe(True),
        outcomes,
        frame,
        average_before_analysis=False,
        noscan=True,
        sort_column="charge",
        sort_sigma=1,
    )
    assert [p.identifier for p in plan.summary] == [1, 2, 3, 4]
    assert plan.notes == ("Waterfall sort excluded 2 units",)


def test_unresolved_sort_request_still_bypasses_bins_and_uses_shot_order():
    outcomes = [outcome(n, n, line=True) for n in (3, 1, 2)]
    plan = plan_products(
        recipe(True),
        outcomes,
        rows(),
        average_before_analysis=False,
        noscan=False,
        parameter_column="motor",
        sort_requested=True,
    )
    assert plan.singles[0].identifier == "average"
    assert [p.position for p in plan.summary] == [1, 2, 3]


def test_legacy_minimum_counts_successful_units_not_requested_shots():
    outcomes = [
        outcome(1, 1),
        outcome(2, 2),
        UnitResult(ShotGroup(3, (3,)), (), None, error="missing"),
    ]
    plan = plan_products(
        recipe(), outcomes, rows(), average_before_analysis=False, noscan=True
    )
    assert not plan.singles and not plan.summary
    assert "more than two" in plan.notes[0]


def test_mixed_shapes_skip_average_without_discarding_line_summary_members():
    outcomes = [
        outcome(1, 1, line=True),
        outcome(2, 2, line=True),
        outcome(3, 3, shape=(4,)),
    ]
    plan = plan_products(
        recipe(True), outcomes, rows(), average_before_analysis=False, noscan=True
    )
    assert not plan.singles
    assert len(plan.summary) == 3
    assert "incompatible result shapes" in plan.notes[0]


def test_missing_scan_parameter_omits_products_explicitly():
    outcomes = [outcome(n, n) for n in (1, 2, 3)]
    plan = plan_products(
        recipe(), outcomes, rows(), average_before_analysis=False, noscan=False
    )
    assert not plan.singles and not plan.summary
    assert "missing bin or parameter" in plan.notes[0]


def test_duplicate_unit_keys_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="repeat"):
        plan_products(
            recipe(),
            [outcome(1, 1), outcome(1, 2), outcome(2, 3)],
            rows(),
            average_before_analysis=False,
            noscan=True,
        )


@pytest.mark.parametrize("averaged", [False, True])
def test_scanned_products_match_legacy_bin_adapter(averaged):
    from image_analysis.config import create_image_analyzer
    from image_analysis.types import ImageAnalyzerResult
    from scan_analysis.analyzers.common.array2D_scan_analysis import (
        Array2DScanAnalyzer,
    )

    doc = AnalysisDiagnostic.model_validate(
        {
            "name": "Device",
            "analyzer": {"kind": "beam"},
            "image": {"type": "camera", "pipeline": []},
            "scan": {"mode": "per_bin" if averaged else "per_shot"},
        }
    )
    outcomes = [outcome(n, n * 3) for n in ((3, 1, 2) if averaged else (5, 1, 3))]
    old = Array2DScanAnalyzer(
        device_name="Device",
        image_analyzer=create_image_analyzer(doc),
        analysis_mode=doc.scan.mode,
    )
    old.auxiliary_data = rows()
    old.noscan = False
    old.scan_parameter = "motor"
    old.results = {
        o.group.key: ImageAnalyzerResult(
            data_type="2d",
            processed_image=o.measurement.frame.data,
            scalars=dict(o.measurement.scalars),
        )
        for o in outcomes
    }
    expected = old.get_binned_data()
    plan = plan_products(
        compile_v2(doc),
        outcomes,
        rows(),
        average_before_analysis=averaged,
        noscan=False,
        parameter_column="motor",
    )
    assert [p.identifier for p in plan.singles] == list(expected)
    for product in plan.singles:
        entry = expected[product.identifier]
        assert product.position == entry["value"]
        assert dict(product.measurement.scalars) == entry["result"].scalars
        np.testing.assert_array_equal(
            product.measurement.frame.data, entry["result"].processed_image
        )
