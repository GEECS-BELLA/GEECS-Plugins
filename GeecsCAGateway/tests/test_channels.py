"""Unit tests for value coercion and path (long-string) channels (no network)."""

from __future__ import annotations

import pytest

from geecs_ca_gateway.channels import (
    cast_value,
    enum_geecs_value,
    enum_index,
    make_readback_channel,
    make_setpoint_channel,
)
from geecs_ca_gateway.config import VariableSpec

# A realistic GEECS device-server save path — 68 chars, well past the 40-char
# EPICS DBR_STRING cap that motivated the char-array representation.
_LONG_PATH = r"Z:\data\Undulator\Y2026\07-Jul\26_0703\scans\Scan012\UC_Amp2_IR_input"


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


def test_enum_index_numeric_labels_match_by_value_not_index() -> None:
    """Numeric labels resolve by VALUE — "2.000000" is the label "2", not index 2.

    Regression (strict shot control): DG645-style configs have numeric option
    labels (["1", "2", "5"]).  A device streaming "2.000000" missed the exact
    label match and was interpreted as INDEX 2, silently selecting "5".
    """
    choices = ["1", "2", "5"]
    assert enum_index(choices, "2.0") == 1
    assert enum_index(choices, "2.000000") == 1
    assert enum_index(choices, "2") == 1  # exact label match still first
    assert enum_index(choices, "5.0") == 2
    assert enum_index(choices, "1.000000") == 0


def test_enum_index_numeric_labels_no_value_match_is_none() -> None:
    """Numeric labels with no value match yield None — never an index guess."""
    assert enum_index(["1", "2", "5"], "0") is None
    assert enum_index(["1", "2", "5"], "3.0") is None


def test_enum_index_index_fallback_only_without_numeric_labels() -> None:
    """Non-numeric labels keep the index-interpretation fallback."""
    assert enum_index(["Off", "On"], "1") == 1
    assert enum_index(["Off", "On"], "1.0") == 1
    assert enum_index(["Off", "On"], "On") == 1  # exact label match still first
    assert enum_index(["Off", "On"], "2") is None  # out of range
    assert enum_index(["Off", "On"], "garbage") is None


def test_enum_geecs_value_index_and_label() -> None:
    """Setpoint: a CA index (or label) maps to the GEECS option string."""
    assert enum_geecs_value(["on", "off"], 1) == "off"  # caput by index
    assert enum_geecs_value(["on", "off"], "on") == "on"  # caput by label


def test_cast_path_decodes_char_arrays() -> None:
    """Path values arrive as text, bytes, or integer char codes — all decode."""
    assert cast_value("path", _LONG_PATH) == _LONG_PATH
    assert cast_value("path", _LONG_PATH.encode()) == _LONG_PATH
    codes = [ord(c) for c in _LONG_PATH] + [0, 0]  # NUL-padded CA char array
    assert cast_value("path", codes) == _LONG_PATH


async def test_path_readback_holds_long_string() -> None:
    """A path readback channel stores strings far beyond the 40-char cap.

    Regression: path variables served as DBR_STRING silently truncated their
    readback at 40 characters.
    """
    channel = make_readback_channel(VariableSpec(geecs_var="p", dtype="path"))
    await channel.write(_LONG_PATH)
    assert cast_value("path", channel.value) == _LONG_PATH


async def test_path_setpoint_forwards_full_text() -> None:
    """A path setpoint forwards the full decoded text to the GEECS setter.

    Regression: DBR_STRING setpoints rejected >40-char paths outright
    (CAException 186), so native image saving could not be configured over CA.
    """
    sent: list[str] = []

    async def setter(value: str) -> None:
        sent.append(value)

    channel = make_setpoint_channel(
        VariableSpec(geecs_var="p", dtype="path", settable=True), setter
    )
    await channel.write(_LONG_PATH)
    assert sent == [_LONG_PATH]
    # A char-array put (integer codes) decodes to the same text.
    await channel.write([ord(c) for c in _LONG_PATH])
    assert sent[-1] == _LONG_PATH
