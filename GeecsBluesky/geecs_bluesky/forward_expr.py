"""Compile a pseudo variable's ``forward`` formula into a callable.

A :class:`~geecs_schemas.scan_variables.PseudoComponent` carries its device's
setting as a math expression of the single scanned number (the schema writes
the scanned value as ``composite_var``; the shorter alias ``x`` is also
accepted).  This module turns that string into a plain ``float -> float``
callable, safely: the expression is parsed with :mod:`ast` and validated
against an explicit whitelist of node types, operators, functions, and names
**before** anything is evaluated, so a config cannot smuggle attribute
access, imports, subscripts, or arbitrary names into the engine.

The whitelist covers the full legacy ``composite_variables.yaml`` corpus
(arithmetic, parentheses, ``sqrt``) with ordinary math headroom (trig,
``exp``/``log``, ``abs``, the constants ``pi``/``e``).  Compilation failures
raise :class:`~geecs_bluesky.exceptions.GeecsConfigurationError` naming the
offending construct — the runner compiles every formula fail-fast pre-claim,
so a bad expression can never burn a scan number.

The compile-then-restricted-eval skeleton is the shared
:mod:`geecs_schemas.restricted_expr` core (also behind the gateway's
derived-channel ``ExpressionEvaluator``) — a hardening or semantics fix
lands there once.  This module supplies the forward-formula whitelist
(arithmetic incl. ``//``, ``abs``, no comparisons/bool-ops), the scanned-value
symbols, and the engine's error contract.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable

from geecs_schemas.restricted_expr import (
    CompiledExpression,
    ExpressionWhitelist,
    ExpressionWhitelistError,
    compile_expression,
)

from geecs_bluesky.exceptions import GeecsConfigurationError

#: Names bound to the scanned value: the schema's token and its short alias.
#: Must stay disjoint from ``_FUNCTIONS``/``_CONSTANTS`` — under the shared
#: core, symbols shadow same-named functions/constants at evaluate time.
SCAN_VALUE_NAMES = ("composite_var", "x")

#: Callables an expression may invoke, by name.
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
}

#: Bare names an expression may reference besides the scanned value.
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

_WHITELIST = ExpressionWhitelist(
    functions=_FUNCTIONS,
    constants=_CONSTANTS,
    binary_ops=(ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv),
    unary_ops=(ast.UAdd, ast.USub),
)


def _reject(expression: str, detail: str) -> GeecsConfigurationError:
    return GeecsConfigurationError(
        f"invalid forward expression {expression!r}: {detail}. Allowed: "
        f"numbers, + - * / ** % //, parentheses, the scanned value as "
        f"{' or '.join(SCAN_VALUE_NAMES)}, constants {sorted(_CONSTANTS)}, "
        f"and functions {sorted(_FUNCTIONS)}"
    )


@dataclass(frozen=True)
class CompiledForward:
    """A validated ``forward`` formula, callable as ``float -> float``.

    Attributes
    ----------
    source : str
        The original expression text (recorded in run metadata).
    """

    source: str
    _compiled: CompiledExpression

    def __call__(self, value: float) -> float:
        """Evaluate the formula at scanned value *value*."""
        values = {token: float(value) for token in SCAN_VALUE_NAMES}
        try:
            # Inside the try: float() itself can raise — e.g. `x ** 0.5` at
            # a negative x returns complex (TypeError), and an int-constant
            # power can overflow the float conversion.
            return float(self._compiled.evaluate(values))
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise GeecsConfigurationError(
                f"forward expression {self.source!r} failed at "
                f"{SCAN_VALUE_NAMES[0]}={value}: {exc}"
            ) from exc


def compile_forward(expression: str) -> CompiledForward:
    """Parse and whitelist-validate *expression*; return the callable form.

    Raises
    ------
    GeecsConfigurationError
        Syntax errors, or any construct outside the whitelist (unknown
        names, attribute access, subscripts, non-numeric literals, ...).
    """
    try:
        compiled = compile_expression(
            expression, frozenset(SCAN_VALUE_NAMES), _WHITELIST, filename="<forward>"
        )
    except ExpressionWhitelistError as exc:
        raise _reject(expression, str(exc)) from exc
    return CompiledForward(source=expression, _compiled=compiled)
