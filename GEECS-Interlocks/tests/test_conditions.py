"""The pure verdict: every condition kind, hysteresis, and every fail-safe."""

import math

import pytest
from geecs_schemas import (
    AlignmentCondition,
    ConfigChangeCondition,
    ExternalCondition,
    StaleCondition,
    ThresholdCondition,
)

from geecs_interlocks.conditions import (
    Inputs,
    describe,
    evaluate,
    number,
    same_setting,
    watched,
)
from geecs_interlocks.external import NO_SIGNAL, SignalReading


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
            ("==", 10, True),
            ("==", 3, False),
            ("!=", 3, True),
            ("!=", 10, False),
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

    def test_new_kinds(self):
        change = ConfigChangeCondition(
            kind="config_change", device="W", variables=["Pos", "Speed"]
        )
        assert watched(change) == [("W", "Pos"), ("W", "Speed")]
        assert describe(change) == "W Pos, Speed unchanged"
        external = ExternalCondition(kind="external", port=5050, source="stage")
        assert watched(external) == []
        assert describe(external) == "stage on UDP 5050"


class TestExternal:
    def ext(self, max_age=None):
        return ExternalCondition(
            kind="external", port=5050, source="stage", max_age_s=max_age
        )

    def check(self, state, raw="", age=1.0, max_age=None):
        signal = SignalReading(state, 0.0, raw)
        return evaluate(
            self.ext(max_age), Inputs(signal=signal, age_s=age), tripped=False
        )

    def test_nothing_yet_is_waiting(self):
        verdict = evaluate(self.ext(), Inputs(signal=NO_SIGNAL), tripped=False)
        assert verdict.unsafe and verdict.waiting
        assert "no packet yet on UDP port 5050" in verdict.diagnostic

    def test_safe_and_not_safe(self):
        assert not self.check(False, "safe").unsafe
        verdict = self.check(True, "warning: moving")
        assert verdict.unsafe and not verdict.waiting
        assert "alarm ('warning: moving')" in verdict.diagnostic

    def test_silence_is_normal_without_max_age(self):
        assert not self.check(False, "safe", age=1e6).unsafe

    def test_max_age(self):
        assert self.check(False, "safe", age=3.0, max_age=2.0).unsafe


class TestConfigChange:
    def change(self):
        return ConfigChangeCondition(
            kind="config_change", device="W", variables=["Pos", "Mode"]
        )

    def check(self, settings, baseline):
        return evaluate(
            self.change(),
            Inputs(settings=settings, baseline=baseline),
            tripped=False,
        )

    def test_unarmed_is_unsafe_but_not_waiting(self):
        verdict = self.check({"Pos": 1, "Mode": "a"}, None)
        assert verdict.unsafe and not verdict.waiting
        assert "not armed" in verdict.diagnostic

    def test_unchanged(self):
        assert not self.check(
            {"Pos": "1.0", "Mode": "a"}, {"Pos": 1, "Mode": "a"}
        ).unsafe

    def test_a_change_names_the_setting(self):
        verdict = self.check({"Pos": 2, "Mode": "a"}, {"Pos": 1, "Mode": "a"})
        assert verdict.unsafe and "Pos changed from 1 to 2" in verdict.diagnostic

    def test_missing_setting_waits(self):
        verdict = self.check({"Pos": 1}, {"Pos": 1, "Mode": "a"})
        assert verdict.unsafe and verdict.waiting

    @pytest.mark.parametrize(
        "a,b,same",
        [(1, "1.0", True), ("on", "on", True), ("on", "off", False), (1, 1.5, False)],
    )
    def test_same_setting(self, a, b, same):
        assert same_setting(a, b) is same


class TestStaleness:
    def test_a_silent_device_is_unsafe(self):
        verdict = evaluate(thr(), Inputs(value=0, stale=[("G", 7.25)]), tripped=False)
        assert verdict.unsafe and verdict.diagnostic == "G: no new data for 7.2 s"


class TestExternalSender:
    def check(self, sender, expected):
        condition = ExternalCondition(kind="external", port=5050, sender=expected)
        signal = SignalReading(False, 0.0, "safe", sender)
        return evaluate(condition, Inputs(signal=signal, age_s=1.0), tripped=False)

    def test_the_named_sender_clears_it(self):
        assert not self.check("10.0.0.5", "10.0.0.5").unsafe

    def test_any_other_host_cannot(self):
        verdict = self.check("10.0.0.9", "10.0.0.5")
        assert verdict.unsafe and "expected 10.0.0.5" in verdict.diagnostic

    def test_no_sender_set_accepts_any(self):
        assert not self.check("10.0.0.9", "").unsafe


def test_an_unarmed_check_names_no_missing_control():
    change = ConfigChangeCondition(kind="config_change", device="W", variables=["P"])
    verdict = evaluate(change, Inputs(settings={"P": 1}, baseline=None), tripped=False)
    assert verdict.diagnostic == "W: not armed"
