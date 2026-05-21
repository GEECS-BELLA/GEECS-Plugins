"""Unit tests for ScientificDoubleSpinBox.

Requires a QApplication (provided by the session-scoped ``qapp`` fixture
in conftest.py).  Tests run headlessly via QT_QPA_PLATFORM=offscreen.
Marked ``gui`` so they are skipped in headless CI environments without Qt.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed — skipping gui tests")

from PyQt5.QtGui import QValidator  # noqa: E402

from ConfigFileGUI.field_widgets import ScientificDoubleSpinBox  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def spinbox(qapp):
    box = ScientificDoubleSpinBox()
    box.setDecimals(15)
    box.setRange(-1e300, 1e300)
    return box


# ---------------------------------------------------------------------------
# textFromValue
# ---------------------------------------------------------------------------


class TestTextFromValue:
    def test_zero(self, spinbox):
        assert spinbox.textFromValue(0.0) == "0"

    def test_normal_float(self, spinbox):
        text = spinbox.textFromValue(0.2008)
        assert float(text) == pytest.approx(0.2008, rel=1e-9)

    def test_normal_float_no_trailing_zeros(self, spinbox):
        text = spinbox.textFromValue(0.5)
        assert "000" not in text
        assert float(text) == pytest.approx(0.5)

    def test_small_value_uses_sci_notation(self, spinbox):
        text = spinbox.textFromValue(1e-13)
        assert "e" in text.lower()
        assert float(text) == pytest.approx(1e-13, rel=1e-6)

    def test_large_value_uses_sci_notation(self, spinbox):
        text = spinbox.textFromValue(1e8)
        assert "e" in text.lower()
        assert float(text) == pytest.approx(1e8, rel=1e-6)

    def test_negative_small_value(self, spinbox):
        text = spinbox.textFromValue(-1e-13)
        assert text.startswith("-")
        assert float(text) == pytest.approx(-1e-13, rel=1e-6)

    def test_boundary_lower(self, spinbox):
        # 1e-4 is at the boundary — should use decimal notation
        text = spinbox.textFromValue(1e-4)
        assert float(text) == pytest.approx(1e-4, rel=1e-9)

    def test_boundary_upper(self, spinbox):
        # 1e6 is above the boundary — should use scientific notation
        text = spinbox.textFromValue(1e6)
        assert float(text) == pytest.approx(1e6, rel=1e-6)


# ---------------------------------------------------------------------------
# valueFromText
# ---------------------------------------------------------------------------


class TestValueFromText:
    def test_plain_integer(self, spinbox):
        assert spinbox.valueFromText("42") == pytest.approx(42.0)

    def test_decimal(self, spinbox):
        assert spinbox.valueFromText("0.2008") == pytest.approx(0.2008, rel=1e-9)

    def test_scientific_small(self, spinbox):
        assert spinbox.valueFromText("1e-13") == pytest.approx(1e-13, rel=1e-9)

    def test_scientific_large(self, spinbox):
        assert spinbox.valueFromText("3.5e8") == pytest.approx(3.5e8, rel=1e-9)

    def test_negative_scientific(self, spinbox):
        assert spinbox.valueFromText("-2.5e-7") == pytest.approx(-2.5e-7, rel=1e-9)

    def test_whitespace_stripped(self, spinbox):
        assert spinbox.valueFromText("  1e-13  ") == pytest.approx(1e-13, rel=1e-9)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def _state(self, spinbox, text):
        state, _, _ = spinbox.validate(text, 0)
        return state

    def test_empty_is_intermediate(self, spinbox):
        assert self._state(spinbox, "") == QValidator.Intermediate

    def test_sign_only_is_intermediate(self, spinbox):
        assert self._state(spinbox, "-") == QValidator.Intermediate
        assert self._state(spinbox, "+") == QValidator.Intermediate

    def test_plain_integer_acceptable(self, spinbox):
        assert self._state(spinbox, "42") == QValidator.Acceptable

    def test_decimal_acceptable(self, spinbox):
        assert self._state(spinbox, "0.2008") == QValidator.Acceptable

    def test_scientific_acceptable(self, spinbox):
        assert self._state(spinbox, "1e-13") == QValidator.Acceptable
        assert self._state(spinbox, "3.5e8") == QValidator.Acceptable
        assert self._state(spinbox, "-2.5e-7") == QValidator.Acceptable

    def test_partial_exponent_intermediate(self, spinbox):
        assert self._state(spinbox, "1e") == QValidator.Intermediate
        assert self._state(spinbox, "1e-") == QValidator.Intermediate
        assert self._state(spinbox, "1e+") == QValidator.Intermediate
        assert self._state(spinbox, "1.5e-") == QValidator.Intermediate

    def test_trailing_decimal_acceptable(self, spinbox):
        # Python's float("1.") == 1.0 is valid, so the widget correctly accepts it.
        assert self._state(spinbox, "1.") == QValidator.Acceptable

    def test_garbage_invalid(self, spinbox):
        assert self._state(spinbox, "abc") == QValidator.Invalid
        assert self._state(spinbox, "1e-13x") == QValidator.Invalid
        assert self._state(spinbox, "1.2.3") == QValidator.Invalid


# ---------------------------------------------------------------------------
# Round-trip: setValue → textFromValue → valueFromText
# ---------------------------------------------------------------------------


class TestRoundtrip:
    @pytest.mark.parametrize(
        "value",
        [0.0, 1.0, -1.0, 0.2008, 1e-13, -1e-13, 3.5e8, 1e-4, 1e6],
    )
    def test_value_roundtrip(self, spinbox, value):
        text = spinbox.textFromValue(value)
        recovered = spinbox.valueFromText(text)
        if value == 0.0:
            assert recovered == 0.0
        else:
            assert math.isclose(recovered, value, rel_tol=1e-6)
