"""Unit tests for value coercion (no network)."""

from __future__ import annotations

import pytest

from geecs_ca_gateway.channels import cast_value, enum_geecs_value, enum_index


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


def test_enum_index_maps_label_to_index() -> None:
    """Readback: a GEECS option string resolves to its enum index."""
    assert enum_index(["on", "off"], "on") == 0
    assert enum_index(["on", "off"], "off") == 1


def test_enum_index_tolerant_and_none() -> None:
    """A numeric index passes through; an unknown label yields None."""
    assert enum_index(["on", "off"], 1) == 1
    assert enum_index(["on", "off"], "bogus") is None


def test_enum_geecs_value_index_and_label() -> None:
    """Setpoint: a CA index (or label) maps to the GEECS option string."""
    assert enum_geecs_value(["on", "off"], 1) == "off"  # caput by index
    assert enum_geecs_value(["on", "off"], "on") == "on"  # caput by label
