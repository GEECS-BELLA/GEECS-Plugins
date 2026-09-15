"""Tests for wire-value coercion."""

from __future__ import annotations


import random

import pytest

from geecs_core.transport._coerce import coerce_scalar, format_float


class TestCoerceScalar:
    """Test coerce_scalar behavior across numeric and non-numeric inputs."""

    def test_integer_string(self) -> None:
        """Integer strings coerce to int."""
        assert coerce_scalar("5") == 5
        assert coerce_scalar("0") == 0
        assert coerce_scalar("-42") == -42

    def test_leading_zeros(self) -> None:
        """Leading zeros are dropped (lossy)."""
        assert coerce_scalar("007") == 7
        assert coerce_scalar("0005") == 5

    def test_float_string_with_decimal(self) -> None:
        """Floats with decimal points stay as floats."""
        assert coerce_scalar("5.0") == 5.0
        assert coerce_scalar("3.14") == 3.14
        assert coerce_scalar("-2.5") == -2.5

    def test_scientific_notation_whole(self) -> None:
        """Scientific notation that resolves to a whole number returns int."""
        # "1e5" = 100000.0, which equals int(100000), and has no "."
        result = coerce_scalar("1e5")
        assert result == 100000
        assert isinstance(result, int)

    def test_scientific_notation_fractional(self) -> None:
        """Scientific notation with fractional exponent returns float."""
        result = coerce_scalar("1.5e2")
        assert result == 150.0
        assert isinstance(result, float)

    def test_non_numeric_string(self) -> None:
        """Non-numeric strings pass through unchanged."""
        assert coerce_scalar("abc") == "abc"
        assert coerce_scalar("hello") == "hello"
        assert coerce_scalar("") == ""

    def test_positive_infinity(self) -> None:
        """Positive infinity passes through as raw string."""
        assert coerce_scalar("inf") == "inf"
        assert coerce_scalar("Infinity") == "Infinity"

    def test_negative_infinity(self) -> None:
        """Negative infinity passes through as raw string."""
        assert coerce_scalar("-inf") == "-inf"
        assert coerce_scalar("-Infinity") == "-Infinity"

    def test_large_exponent_infinity(self) -> None:
        """Sufficiently large exponent that overflows to infinity passes through."""
        assert coerce_scalar("1e400") == "1e400"

    def test_nan(self) -> None:
        """NaN passes through as raw string (pre-existing behavior)."""
        assert coerce_scalar("nan") == "nan"
        assert coerce_scalar("NaN") == "NaN"

    def test_mixed_case_infinity(self) -> None:
        """Case variations of infinity are handled."""
        # Python's float() accepts "inf" and "Infinity" (case-insensitive on some platforms)
        result_1 = coerce_scalar("inf")
        result_2 = coerce_scalar("INF")
        # Both should pass through as strings (non-finite)
        assert isinstance(result_1, str)
        assert isinstance(result_2, str)


class TestFormatFloat:
    """Outbound float formatting for set commands (issue #819).

    The wire string must carry exactly the digits the caller asked for — no
    binary-representation tail, no truncation, no exponent notation.
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # The issue's table: %.12f sent 40854.246249999997 etc.
            (40854.24625, "40854.24625"),
            (40865.8875, "40865.8875"),
            (40966.0, "40966.0"),
            # Small magnitudes a fixed-decimals format would zero (the DB's
            # tightest tolerance and minimum found so far, and smaller).
            (0.001, "0.001"),
            (1e-05, "0.00001"),
            (1e-07, "0.0000001"),
            (2.5e-05, "0.000025"),
            # Large magnitudes where repr switches to exponent form.
            (1e16, "10000000000000000.0"),
            (1.5e17, "150000000000000000.0"),
            (123456789.123456789, "123456789.12345679"),
            # Integers-as-floats keep a decimal point, as %.12f did.
            (1.0, "1.0"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            # Negatives, both ranges.
            (-40854.24625, "-40854.24625"),
            (-1e-07, "-0.0000001"),
            (-0.001, "-0.001"),
        ],
    )
    def test_shortest_round_trip_plain_decimal(
        self, value: float, expected: str
    ) -> None:
        text = format_float(value)
        assert text == expected
        assert float(text) == value  # exact round trip, not approx

    def test_non_finite_pass_through(self) -> None:
        """inf/nan render as before (%.12f gave the same tokens)."""
        assert format_float(float("inf")) == "inf"
        assert format_float(float("-inf")) == "-inf"
        assert format_float(float("nan")) == "nan"

    def test_five_decimal_working_range_has_no_representation_tail(self) -> None:
        """The issue's measurement: 86% of five-decimal values in the
        Aerotech's working range grew a garbage tail under %.12f. The
        shortest form never does — at most the five decimals asked for, and
        an exact round trip — over a seeded sample of that range.
        """
        rng = random.Random(819)
        for _ in range(20_000):
            value = round(rng.uniform(40_000, 41_000), 5)
            text = format_float(value)
            assert "e" not in text
            assert float(text) == value
            decimals = text.split(".")[1]
            assert len(decimals) <= 5, (value, text)

    def test_never_exponent_notation_across_magnitudes(self) -> None:
        for exp in range(-12, 20):
            for mantissa in (1.0, 1.5, 7.25, -3.125):
                value = mantissa * 10.0**exp
                text = format_float(value)
                assert "e" not in text and "E" not in text, (value, text)
                assert "." in text
                assert float(text) == value
