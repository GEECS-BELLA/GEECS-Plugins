"""Unit tests for value coercion (no network)."""

from __future__ import annotations

import pytest

from geecs_ca_gateway.channels import cast_value


def test_cast_scalar_values() -> None:
    """Plain scalars from the GEECS stream coerce by dtype."""
    assert cast_value("float", "3.5") == pytest.approx(3.5)
    assert cast_value("int", 4.0) == 4
    assert cast_value("string", 12) == "12"


def test_cast_unwraps_ca_array_values() -> None:
    """CA puts arrive as length-1 arrays/lists — unwrap to a scalar.

    Regression: a CA ``caput`` delivers ``[8.8]``; ``float([8.8])`` raises, so
    the setpoint path must extract the leading element first.
    """
    assert cast_value("float", [8.8]) == pytest.approx(8.8)
    assert cast_value("int", [4.0]) == 4


def test_cast_string_not_unwrapped() -> None:
    """Strings have ``__len__`` but must not be indexed into."""
    assert cast_value("string", "hello") == "hello"
