"""Tests for wire-value coercion."""

from __future__ import annotations


from geecs_ca_gateway.transport._coerce import coerce_scalar


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
