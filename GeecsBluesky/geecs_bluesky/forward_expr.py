"""Compile a pseudo variable's ``forward`` and ``inverse`` formulas.

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
offending construct — compile every formula fail-fast before a scan number
is claimed, so a bad expression can never burn one.

The compile-then-restricted-eval skeleton is the shared
:mod:`geecs_schemas.restricted_expr` core (also behind the gateway's
derived-channel ``ExpressionEvaluator``) — a hardening or semantics fix
lands there once.  This module supplies the forward-formula whitelist
(arithmetic incl. ``//``, ``abs``, no comparisons/bool-ops), the scanned-value
symbols, and the engine's error contract.

Two more readers of the same AST serve the pseudo positioners
(:mod:`geecs_bluesky.devices.ca.pseudo`):

- :func:`affine_coefficients` recognises a formula that is affine in the
  scanned value — ``a*x + b`` in any spelling — and returns ``(a, b)``
  exactly (a symbolic walk, not a numeric fit), so the software can invert
  it itself; 24 of the 26 corpus formulas are.
- :func:`compile_inverse` compiles a physicist-supplied ``inverse`` — the
  scanned value as an expression of the components' readbacks, named by
  the symbols the caller passes — for the formulas that are not.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import AbstractSet, Callable

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


# ---------------------------------------------------------------- affine


class _NotAffine(Exception):
    """Internal: the walk met a construct that is not affine in the scanned value."""


def _affine(node: ast.AST) -> tuple[float, float]:
    """``(a, b)`` such that *node* == ``a*x + b`` for every x; else :class:`_NotAffine`.

    A symbolic walk over the whitelisted AST: constants and ``x``-free
    subtrees are ``(0, value)``; ``x`` is ``(1, 0)``; ``+``/``-`` add;
    ``*`` needs a constant side, ``/`` a constant divisor.  Anything else
    involving ``x`` (a power, a function call, ``%``, ``//``) is not affine.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _NotAffine
        return 0.0, float(node.value)
    if isinstance(node, ast.Name):
        if node.id in SCAN_VALUE_NAMES:
            return 1.0, 0.0
        if node.id in _CONSTANTS:
            return 0.0, _CONSTANTS[node.id]
        raise _NotAffine
    if isinstance(node, ast.UnaryOp):
        a, b = _affine(node.operand)
        if isinstance(node.op, ast.UAdd):
            return a, b
        if isinstance(node.op, ast.USub):
            return -a, -b
        raise _NotAffine
    if isinstance(node, ast.BinOp):
        la, lb = _affine(node.left)
        ra, rb = _affine(node.right)
        if isinstance(node.op, ast.Add):
            return la + ra, lb + rb
        if isinstance(node.op, ast.Sub):
            return la - ra, lb - rb
        if isinstance(node.op, ast.Mult):
            if la == 0.0:
                return lb * ra, lb * rb
            if ra == 0.0:
                return la * rb, lb * rb
            raise _NotAffine
        if isinstance(node.op, ast.Div):
            if ra == 0.0 and rb != 0.0:
                return la / rb, lb / rb
            raise _NotAffine
        if la == 0.0 and ra == 0.0:
            # An x-free subtree under any other operator (``2 ** 3``,
            # ``7 % 3``) is a constant: evaluate it once.
            return 0.0, float(_evaluate_constant(node))
        raise _NotAffine
    if isinstance(node, ast.Call):
        # ``sqrt(2)`` is a constant; ``sqrt(x)`` is not affine.
        if any(_uses_scan_value(arg) for arg in node.args):
            raise _NotAffine
        return 0.0, float(_evaluate_constant(node))
    raise _NotAffine


def _uses_scan_value(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Name) and n.id in SCAN_VALUE_NAMES for n in ast.walk(node)
    )


def _evaluate_constant(node: ast.AST) -> float:
    """Evaluate an x-free whitelisted subtree (it compiled, so it is safe)."""
    source = ast.unparse(node)
    return compile_forward(source)(0.0)


def affine_coefficients(expression: str) -> tuple[float, float] | None:
    """``(a, b)`` when *expression* is ``a*x + b`` for every scanned value; else ``None``.

    Exact, from the AST — ``composite_var * -2`` is ``(-2, 0)``,
    ``(composite_var-41000) * 14/1000 - 20`` is ``(0.014, -594)``,
    ``-(composite_var + 0.2411) * 9`` is ``(-9, -2.1699)``.  A formula
    with the scanned value under a power, a function or ``%``/``//`` (the
    R56 square root) is not affine and needs a catalog ``inverse``.

    Raises
    ------
    GeecsConfigurationError
        The expression does not compile (same contract as
        :func:`compile_forward`).
    """
    compile_forward(expression)  # the whitelist verdict comes first
    try:
        a, b = _affine(ast.parse(expression, mode="eval").body)
    except _NotAffine:
        return None
    return a + 0.0, b + 0.0  # never -0.0 in a message


# --------------------------------------------------------------- inverse


@dataclass(frozen=True)
class CompiledInverse:
    """A validated ``inverse`` formula: the scanned value from the components' readbacks.

    Attributes
    ----------
    source : str
        The original expression text.
    symbols : frozenset of str
        The component names the expression may reference.
    """

    source: str
    symbols: frozenset[str]
    _compiled: CompiledExpression

    def __call__(self, readbacks: Mapping[str, float]) -> float:
        """Evaluate at the components' readbacks (symbol → value)."""
        values = {name: float(readbacks[name]) for name in self.symbols}
        try:
            return float(self._compiled.evaluate(values))
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise GeecsConfigurationError(
                f"inverse expression {self.source!r} failed at {dict(values)}: {exc}"
            ) from exc


def compile_inverse(expression: str, symbols: AbstractSet[str]) -> CompiledInverse:
    """Parse and whitelist-validate an ``inverse``; *symbols* are the component names.

    The same whitelist as the forward formulas, with the components in
    place of the scanned value: ``560968.636 * U_ChicaneInner**2 / 100**2``.

    Raises
    ------
    GeecsConfigurationError
        Syntax errors, or any construct outside the whitelist — an
        unknown name is the usual one (a component spelled other than
        the caller's symbols).
    """
    try:
        compiled = compile_expression(
            expression, frozenset(symbols), _WHITELIST, filename="<inverse>"
        )
    except ExpressionWhitelistError as exc:
        raise GeecsConfigurationError(
            f"invalid inverse expression {expression!r}: {exc}. Allowed: numbers, "
            f"+ - * / ** % //, parentheses, the components as "
            f"{sorted(symbols)}, constants {sorted(_CONSTANTS)}, and functions "
            f"{sorted(_FUNCTIONS)}"
        ) from exc
    return CompiledInverse(
        source=expression, symbols=frozenset(symbols), _compiled=compiled
    )
