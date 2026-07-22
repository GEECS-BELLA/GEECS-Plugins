"""restricted_expr: the shared AST-whitelist expression core.

Pins the security contract both consumers (the gateway's derived channels
and the engine's forward formulas) stand on: everything outside the
supplied whitelist — unknown names, attribute access, subscripts, calls
to non-whitelisted functions, non-numeric literals, operators the
whitelist omits — is refused at compile time with
``ExpressionWhitelistError``, never evaluated; evaluation runs with empty
builtins; and the parameterization points (operator sets, bool/compare
gating, symbol shadowing, ``is_boolean``) behave as the consumers assume.
"""

from __future__ import annotations

import ast
import math

import pytest

from geecs_schemas.restricted_expr import (
    ExpressionWhitelist,
    ExpressionWhitelistError,
    compile_expression,
)

ARITHMETIC = ExpressionWhitelist(
    functions={"sqrt": math.sqrt, "abs": abs},
    constants={"pi": math.pi, "e": math.e},
    binary_ops=(ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv),
    unary_ops=(ast.UAdd, ast.USub),
)

WITH_LOGIC = ExpressionWhitelist(
    functions={"isfinite": math.isfinite},
    constants={"pi": math.pi},
    binary_ops=(ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod),
    unary_ops=(ast.UAdd, ast.USub, ast.Not),
    bool_ops=(ast.And, ast.Or),
    compare_ops=(ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE),
)


def test_arithmetic_evaluates() -> None:
    compiled = compile_expression("(x - 41000) * 14 / 1000 - 20", {"x"}, ARITHMETIC)
    assert compiled.evaluate({"x": 42000.0}) == pytest.approx(-6.0)
    assert compiled.source == "(x - 41000) * 14 / 1000 - 20"
    assert compiled.is_boolean is False


def test_functions_and_constants() -> None:
    compiled = compile_expression("sqrt(abs(x)) + pi - pi", {"x"}, ARITHMETIC)
    assert compiled.evaluate({"x": -4.0}) == pytest.approx(2.0)


def test_multiple_symbols() -> None:
    compiled = compile_expression("a * b", {"a", "b"}, ARITHMETIC)
    assert compiled.evaluate({"a": 3.0, "b": 4.0}) == 12.0


def test_symbol_shadows_constant_and_function() -> None:
    """Values are bound last, so a symbol wins over a same-named constant."""
    compiled = compile_expression("pi", {"pi"}, ARITHMETIC)
    assert compiled.evaluate({"pi": 1.5}) == 1.5


def test_operator_outside_whitelist_refused() -> None:
    """The same expression compiles or not purely per the operator set."""
    assert compile_expression("x // 2", {"x"}, ARITHMETIC).evaluate({"x": 5.0}) == 2.0
    with pytest.raises(ExpressionWhitelistError):
        compile_expression("x // 2", {"x"}, WITH_LOGIC)


def test_bool_and_compare_gated_by_whitelist() -> None:
    with pytest.raises(ExpressionWhitelistError):
        compile_expression("x < 1", {"x"}, ARITHMETIC)
    with pytest.raises(ExpressionWhitelistError):
        compile_expression("x and 1", {"x"}, ARITHMETIC)
    compiled = compile_expression("x < 1e-5 and isfinite(x)", {"x"}, WITH_LOGIC)
    assert compiled.evaluate({"x": 1e-6}) is True


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("x < 1", True), ("x and 1", True), ("not x", True), ("x + 1", False)],
)
def test_is_boolean_reflects_top_level_intent(expression, expected) -> None:
    assert compile_expression(expression, {"x"}, WITH_LOGIC).is_boolean is expected


@pytest.mark.parametrize(
    "expression",
    [
        "unknown_name * 2",  # name outside symbols and constants
        "__import__('os')",  # smuggled call
        "x.__class__",  # attribute access
        "x[0]",  # subscript
        "'text' + x",  # non-numeric literal
        "lambda v: v",  # disallowed syntax
        "open('f')",  # non-whitelisted function
        "sqrt(x, foo=1)",  # keyword arguments
        "[x for x in (1,)]",  # comprehension
        "f'{x}'",  # f-string
        "x +",  # syntax error
    ],
)
def test_disallowed_expression_refused_at_compile(expression) -> None:
    with pytest.raises(ExpressionWhitelistError):
        compile_expression(expression, {"x"}, ARITHMETIC)


def test_evaluation_namespace_has_no_builtins() -> None:
    """Names resolve only from the whitelist and *values* — no builtins.

    A declared-but-unbound symbol is a NameError at evaluate time, not a
    silent fallback to a builtin; consumers guard missing inputs upstream.
    """
    compiled = compile_expression("x", {"x"}, ARITHMETIC)
    with pytest.raises(NameError):
        compiled.evaluate({})


def test_runtime_errors_propagate_unwrapped() -> None:
    """The core imposes no error contract — consumers wrap."""
    compiled = compile_expression("sqrt(x)", {"x"}, ARITHMETIC)
    with pytest.raises(ValueError):
        compiled.evaluate({"x": -1.0})
    with pytest.raises(ZeroDivisionError):
        compile_expression("1 / x", {"x"}, ARITHMETIC).evaluate({"x": 0.0})


def test_extra_symbols_permit_bare_function_names() -> None:
    """A consumer may pass function names as symbols (gateway legacy)."""
    symbols = {"v"} | set(ARITHMETIC.functions)
    compiled = compile_expression("sqrt", symbols, ARITHMETIC)
    assert compiled.evaluate({"v": 0.0}) is math.sqrt


def test_syntax_error_message_names_the_problem() -> None:
    with pytest.raises(ExpressionWhitelistError, match="syntax error"):
        compile_expression("x +", {"x"}, ARITHMETIC)
