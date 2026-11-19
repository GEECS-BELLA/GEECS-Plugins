"""The pure verdict: every condition kind, hysteresis, and every fail-safe."""

import math

import pytest
from geecs_schemas import AlignmentCondition, StaleCondition, ThresholdCondition

from geecs_interlocks.conditions import Inputs, describe, evaluate, number, watched


def thr(op=">", limit=10.0, hysteresis=0.0):
    return ThresholdCondition(
        kind="threshold",
        device="G",
        variable="P",
        operator=op,
        limit=limit,
        hysteresis=hysteresis,
    )


def align(tolerance=5.0, hysteresis=0.0, target_device=None):
    return AlignmentCondition(
        kind="alignment",
        device="Cam",
        variable="X",
        target_device=target_device,
        target_variable="X0",
        tolerance=tolerance,
        hysteresis=hysteresis,
    )


class TestThreshold:
    @pytest.mark.parametrize(
        "op,value,unsafe",
        [
            (">", 11, True),
            (">", 10, False),
            (">=", 10, True),
            ("<", 9, True),
            ("<", 10, False),
            ("<=", 10, True),
            (">", 9.99, False),
        ],
    )
    def test_operators(self, op, value, unsafe):
        assert evaluate(thr(op), Inputs(value=value), tripped=False).unsafe is unsafe

    def test_diagnostic_names_value_and_limit(self):
        verdict = evaluate(thr(), Inputs(value=12.5), tripped=False)
        assert verdict.diagnostic == "P=12.5 (limit: >10)"

    @pytest.mark.parametrize("op,value", [(">", 9.5), ("<", 10.5)])
    def test_hysteresis_holds_a_trip(self, op, value):
        condition = thr(op, hysteresis=1.0)
        assert not evaluate(condition, Inputs(value=value), tripped=False).unsafe
        assert evaluate(condition, Inputs(value=value), tripped=True).unsafe

    def test_hysteresis_releases_past_the_band(self):
        assert not evaluate(thr(hysteresis=1.0), Inputs(value=8.9), tripped=True).unsafe


class TestAlignment:
    def test_inside_and_outside(self):
        assert not evaluate(align(), Inputs(value=3, target=0), tripped=False).unsafe
        assert evaluate(align(), Inputs(value=-5, target=0), tripped=False).unsafe

    def test_hysteresis(self):
        condition = align(hysteresis=2.0)
        assert not evaluate(condition, Inputs(value=4, target=0), tripped=False).unsafe
        assert evaluate(condition, Inputs(value=4, target=0), tripped=True).unsafe

    def test_missing_target_is_unsafe(self):
        verdict = evaluate(align(), Inputs(value=0), tripped=False)
        assert verdict.unsafe and verdict.waiting


class TestStale:
    def stale(self, max_age=2.0):
        return StaleCondition(kind="stale", device="G", max_age_s=max_age)

    def test_fresh_and_old(self):
        assert not evaluate(self.stale(), Inputs(age_s=1.0), tripped=False).unsafe
        assert evaluate(self.stale(), Inputs(age_s=2.5), tripped=False).unsafe

    def test_disconnect_only_ignores_age(self):
        assert not evaluate(self.stale(None), Inputs(age_s=1e6), tripped=False).unsafe

    def test_nothing_yet_is_waiting(self):
        verdict = evaluate(self.stale(), Inputs(), tripped=False)
        assert verdict.unsafe and verdict.waiting


class TestFailSafe:
    @pytest.mark.parametrize("value", [None, math.nan, math.inf, "abc", True])
    def test_bad_values_are_unsafe(self, value):
        assert evaluate(thr(), Inputs(value=value), tripped=False).unsafe

    def test_only_a_missing_value_is_waiting(self):
        assert evaluate(thr(), Inputs(), tripped=False).waiting
        assert not evaluate(thr(), Inputs(value=math.nan), tripped=False).waiting

    def test_disconnect_wins_over_a_safe_value(self):
        verdict = evaluate(thr(), Inputs(value=0, disconnected=["G"]), tripped=False)
        assert verdict.unsafe and verdict.diagnostic == "G disconnected"

    def test_numeric_strings_are_numbers(self):
        assert number("3.5") == 3.5
        assert not evaluate(thr(), Inputs(value="3.5"), tripped=False).unsafe


class TestHelpers:
    def test_watched(self):
        assert watched(thr()) == [("G", "P")]
        assert watched(align(target_device="Ref")) == [("Cam", "X"), ("Ref", "X0")]
        assert watched(StaleCondition(kind="stale", device="G")) == [("G", None)]

    def test_describe(self):
        assert describe(thr()) == "G P > 10"
        assert describe(StaleCondition(kind="stale", device="G")) == "G connected"
