"""The pure interlock verdict: kinds, fail-safe inputs, hysteresis."""

from __future__ import annotations

import pytest
from geecs_schemas import AlignmentCondition, StaleCondition, ThresholdCondition

from geecs_bluesky.interlocks.conditions import Inputs, Sample, evaluate


def threshold(operator=">", limit=10.0, hysteresis=0.0):
    return ThresholdCondition(
        kind="threshold",
        device="U_D",
        variable="X",
        operator=operator,
        limit=limit,
        hysteresis=hysteresis,
    )


def value(v, **kw):
    return Inputs(value=Sample(v, **kw))


@pytest.mark.parametrize(
    "operator,reading,unsafe",
    [
        (">", 11, True),
        (">", 10, False),
        (">=", 10, True),
        ("<", 9, True),
        ("<", 10, False),
        ("<=", 10, True),
    ],
)
def test_threshold_operators(operator, reading, unsafe) -> None:
    assert evaluate(threshold(operator), value(reading), tripped=False).unsafe is unsafe


def test_threshold_diagnostic() -> None:
    verdict = evaluate(threshold(), value(12.5), tripped=False)
    assert verdict.diagnostic == "X=12.5 (limit: >10)"


@pytest.mark.parametrize(
    "operator,reading,still_unsafe",
    [(">", 9, True), (">", 7.9, False), ("<", 11, True), ("<", 12.1, False)],
)
def test_hysteresis_holds_only_while_tripped(operator, reading, still_unsafe) -> None:
    condition = threshold(operator, hysteresis=2.0)
    assert evaluate(condition, value(reading), tripped=True).unsafe is still_unsafe
    assert not evaluate(condition, value(reading), tripped=False).unsafe


@pytest.mark.parametrize(
    "inputs,reason",
    [
        (value(None), "X: no data"),
        (value(1.0, invalid=True), "X INVALID (device down)"),
        (value("on"), "X='on' is not a number"),
        (value(float("nan")), "X: NaN"),
        (Inputs(value=Sample(1.0), disconnected=("U_D",)), "U_D disconnected"),
    ],
)
def test_fail_safe(inputs, reason) -> None:
    verdict = evaluate(threshold(), inputs, tripped=False)
    assert verdict.unsafe and verdict.diagnostic == reason


def test_alignment() -> None:
    condition = AlignmentCondition(
        kind="alignment",
        device="U_D",
        variable="X",
        target_variable="T",
        tolerance=1.0,
        hysteresis=0.5,
    )
    near = Inputs(value=Sample(10.4), target=Sample(10.0))
    far = Inputs(value=Sample(11.0), target=Sample(10.0))
    between = Inputs(value=Sample(10.7), target=Sample(10.0))
    assert not evaluate(condition, near, tripped=False).unsafe
    assert evaluate(condition, far, tripped=False).diagnostic == (
        "X=11 is 1 from T=10 (tolerance: 1)"
    )
    assert evaluate(condition, between, tripped=True).unsafe
    assert not evaluate(condition, between, tripped=False).unsafe
    no_target = Inputs(value=Sample(10.0))
    assert evaluate(condition, no_target, tripped=False).diagnostic == "T: no data"


def test_stale() -> None:
    condition = StaleCondition(kind="stale", device="U_D", max_age_s=2.0)
    assert not evaluate(condition, Inputs(stamp_age_s=1.0), tripped=False).unsafe
    verdict = evaluate(condition, Inputs(stamp_age_s=3.0), tripped=False)
    assert verdict.diagnostic == "U_D: no new data for 3.0 s (max 2 s)"
    liveness_only = StaleCondition(kind="stale", device="U_D")
    assert not evaluate(liveness_only, Inputs(), tripped=False).unsafe
    assert evaluate(liveness_only, Inputs(disconnected=("U_D",)), tripped=False).unsafe
