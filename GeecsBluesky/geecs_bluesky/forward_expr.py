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

Known sibling: the gateway's derived-channel ``ExpressionEvaluator``
(``geecs_ca_gateway/derived.py``) is the same compile-then-restricted-eval
skeleton with a different whitelist (comparisons/bool-ops, ``isfinite``,
no ``abs``).  They are deliberately separate today (different languages,
different error contracts); if either eval site is ever hardened or its
semantics fixed, apply the change to both — a shared stdlib-only core in
GEECS-Schemas is the sketched consolidation home.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable

from geecs_bluesky.exceptions import GeecsConfigurationError

#: Names bound to the scanned value: the schema's token and its short alias.
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

_BINARY_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_UNARY_OPS = (ast.UAdd, ast.USub)


def _reject(expression: str, detail: str) -> GeecsConfigurationError:
    return GeecsConfigurationError(
        f"invalid forward expression {expression!r}: {detail}. Allowed: "
        f"numbers, + - * / ** % //, parentheses, the scanned value as "
        f"{' or '.join(SCAN_VALUE_NAMES)}, constants {sorted(_CONSTANTS)}, "
        f"and functions {sorted(_FUNCTIONS)}"
    )


def _validate(node: ast.AST, expression: str) -> None:
    """Recursively whitelist-check one AST node (raises on anything else)."""
    if isinstance(node, ast.Expression):
        _validate(node.body, expression)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, _BINARY_OPS):
        _validate(node.left, expression)
        _validate(node.right, expression)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY_OPS):
        _validate(node.operand, expression)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise _reject(expression, f"literal {node.value!r} is not a number")
    elif isinstance(node, ast.Name):
        if node.id not in SCAN_VALUE_NAMES and node.id not in _CONSTANTS:
            raise _reject(expression, f"unknown name {node.id!r}")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise _reject(expression, "only whitelisted function calls are allowed")
        if node.keywords:
            raise _reject(expression, "keyword arguments are not allowed")
        for arg in node.args:
            _validate(arg, expression)
    else:
        raise _reject(expression, f"disallowed syntax ({type(node).__name__})")


@dataclass(frozen=True)
class CompiledForward:
    """A validated ``forward`` formula, callable as ``float -> float``.

    Attributes
    ----------
    source : str
        The original expression text (recorded in run metadata).
    """

    source: str
    _code: object

    def __call__(self, value: float) -> float:
        """Evaluate the formula at scanned value *value*."""
        names: dict[str, float] = {token: float(value) for token in SCAN_VALUE_NAMES}
        names.update(_CONSTANTS)
        names.update(_FUNCTIONS)  # type: ignore[arg-type]
        try:
            result = eval(  # noqa: S307 — AST-whitelisted at compile time
                self._code, {"__builtins__": {}}, names
            )
            # Inside the try: float() itself can raise — e.g. `x ** 0.5` at
            # a negative x returns complex (TypeError), and an int-constant
            # power can overflow the float conversion.
            return float(result)
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
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise _reject(expression, f"syntax error ({exc.msg})") from exc
    _validate(tree, expression)
    return CompiledForward(source=expression, _code=compile(tree, "<forward>", "eval"))
