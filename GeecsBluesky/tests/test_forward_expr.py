"""forward_expr: the AST-whitelist compiler for pseudo forward formulas.

Pins two things: (1) every expression shape in the legacy
``composite_variables.yaml`` corpus evaluates to the value the legacy
numexpr path produced, and (2) anything outside the whitelist — unknown
names, attribute access, calls to non-whitelisted functions, non-numeric
literals — is refused at compile time with a clear error, never evaluated.
"""

from __future__ import annotations

import math

import pytest

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.forward_expr import (
    _CONSTANTS,
    _FUNCTIONS,
    SCAN_VALUE_NAMES,
    compile_forward,
)

# ---------------------------------------------------------------------------
# The full production corpus (Undulator + Thomson), expression → (input, expected)
# ---------------------------------------------------------------------------

CORPUS = [
    ("composite_var * 1", 2.5, 2.5),
    ("composite_var * -2", 1.5, -3.0),
    ("composite_var * -1", 0.7, -0.7),
    ("composite_var", 41000.0, 41000.0),
    ("(composite_var-41000) * 14/1000 - 20", 42000.0, -6.0),
    ("composite_var * 1.0", 3.0, 3.0),
    ("composite_var * -1.5", 2.0, -3.0),
    ("(composite_var)*8", 0.5, 4.0),
    ("-(composite_var + 0.2411) * 9", -0.2411, 0.0),
    ("-(composite_var)*9.0", 1.0, -9.0),
    ("8.5 + (composite_var-10)*2.5", 10.0, 8.5),
    ("composite_var * 1.5", 2.0, 3.0),
    ("sqrt(100 ** 2 * composite_var / 560968.636)", 560968.636, 100.0),
    ("100 - composite_var", 30.0, 70.0),
]


@pytest.mark.parametrize(("expression", "value", "expected"), CORPUS)
def test_corpus_expression_evaluates(expression, value, expected) -> None:
    compiled = compile_forward(expression)
    assert compiled(value) == pytest.approx(expected)
    assert compiled.source == expression


def test_x_alias_matches_composite_var() -> None:
    """``x`` and ``composite_var`` are the same scanned value."""
    assert compile_forward("x * -2")(1.5) == compile_forward("composite_var * -2")(1.5)


def test_constants_and_functions() -> None:
    assert compile_forward("sin(pi / 2)")(0.0) == pytest.approx(1.0)
    assert compile_forward("log(e)")(0.0) == pytest.approx(1.0)
    assert compile_forward("abs(x)")(-3.0) == 3.0


@pytest.mark.parametrize(
    "expression",
    [
        "unknown_name * 2",  # name outside the whitelist
        "__import__('os')",  # smuggled call
        "x.__class__",  # attribute access
        "x[0]",  # subscript
        "'text' + x",  # non-numeric literal
        "lambda v: v",  # disallowed syntax
        "open('f')",  # non-whitelisted function
        "sqrt(x, foo=1)",  # keyword arguments
        "x +",  # syntax error
    ],
)
def test_disallowed_expression_refused_at_compile(expression) -> None:
    with pytest.raises(GeecsConfigurationError, match="forward expression"):
        compile_forward(expression)


def test_runtime_domain_error_is_configuration_error() -> None:
    """A math-domain failure at set time surfaces clearly, not as ValueError."""
    compiled = compile_forward("sqrt(x)")
    assert compiled(4.0) == 2.0
    with pytest.raises(GeecsConfigurationError, match="sqrt"):
        compiled(-1.0)


def test_complex_and_overflow_results_are_configuration_errors() -> None:
    """Every runtime failure class is wrapped — none escape as raw errors.

    ``x ** 0.5`` at negative x returns *complex* (the natural typo for
    ``sqrt(x)`` scanned into negative values) and an int-constant power can
    overflow the float conversion; both must surface as the same clear
    configuration error the sqrt case does (review finding, PR #594).
    """
    with pytest.raises(GeecsConfigurationError, match="failed at"):
        compile_forward("x ** 0.5")(-1.0)
    with pytest.raises(GeecsConfigurationError, match="failed at"):
        compile_forward("10 ** 400")(0.0)


def test_scan_value_names_disjoint_from_whitelist_names() -> None:
    """The scanned-value tokens must not collide with functions/constants.

    Under the shared core, symbols shadow same-named whitelist entries at
    evaluate time — disjointness is what keeps that direction unobservable
    here.  Adding e.g. a constant ``x`` would silently change semantics;
    this makes it a test failure instead.
    """
    assert not set(SCAN_VALUE_NAMES) & (set(_FUNCTIONS) | set(_CONSTANTS))


def test_result_is_float() -> None:
    result = compile_forward("x // 2")(5.0)
    assert isinstance(result, float)
    assert result == 2.0
    assert math.isfinite(result)
