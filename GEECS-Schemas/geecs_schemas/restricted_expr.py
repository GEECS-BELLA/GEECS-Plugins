"""Shared AST-whitelist core for restricted math expressions.

Two GEECS eval sites compile small operator-authored math expressions —
the gateway's derived channels (``geecs_ca_gateway.derived``) and the
engine's pseudo-variable forward formulas (``geecs_bluesky.forward_expr``).
Both follow the same security-sensitive skeleton: parse with :mod:`ast`,
validate every node against an explicit whitelist **before** anything is
evaluated (so a config cannot smuggle attribute access, imports,
subscripts, or arbitrary names), then evaluate the compiled code with an
empty ``__builtins__``.  This module is that skeleton, factored out once
so a hardening or semantics fix lands in one place.

Each consumer supplies its own :class:`ExpressionWhitelist` (functions,
constants, operator sets) and the set of symbol names its expressions may
reference, and wraps :class:`ExpressionWhitelistError` into its own error
contract.  The core stays stdlib-only (``ast`` + ``dataclasses``) and
imposes no result coercion: :meth:`CompiledExpression.evaluate` returns
the raw value, and runtime failures (domain errors, overflow, division by
zero) propagate for the consumer to translate.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import CodeType
from typing import Any


class ExpressionWhitelistError(ValueError):
    """Raised when an expression falls outside the supported subset.

    Covers both syntax errors and whitelist violations; the message names
    the offending construct.  Consumers wrap this into their own error
    type (``DerivedExpressionError``, ``GeecsConfigurationError``, ...).
    """


@dataclass(frozen=True)
class ExpressionWhitelist:
    """The constructs one eval site permits in its expressions.

    Attributes
    ----------
    functions : Mapping[str, Callable]
        Callables an expression may invoke, by name.
    constants : Mapping[str, float]
        Bare names an expression may reference besides the symbols.
    binary_ops, unary_ops : tuple of ast operator types
        Permitted arithmetic operators.
    bool_ops, compare_ops : tuple of ast operator types
        Permitted boolean/comparison operators; empty (the default)
        refuses those node shapes entirely.
    """

    functions: Mapping[str, Callable[..., float]]
    constants: Mapping[str, float]
    binary_ops: tuple[type[ast.operator], ...]
    unary_ops: tuple[type[ast.unaryop], ...]
    bool_ops: tuple[type[ast.boolop], ...] = ()
    compare_ops: tuple[type[ast.cmpop], ...] = ()


def _is_boolean_expression(node: ast.AST) -> bool:
    """Return whether the expression's top-level result is boolean intent."""
    return isinstance(node, (ast.BoolOp, ast.Compare)) or (
        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
    )


def _validate(
    node: ast.AST, symbols: AbstractSet[str], whitelist: ExpressionWhitelist
) -> None:
    """Recursively whitelist-check one AST node (raises on anything else)."""
    if isinstance(node, ast.Expression):
        _validate(node.body, symbols, whitelist)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, whitelist.binary_ops):
        _validate(node.left, symbols, whitelist)
        _validate(node.right, symbols, whitelist)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, whitelist.unary_ops):
        _validate(node.operand, symbols, whitelist)
    elif isinstance(node, ast.BoolOp) and isinstance(node.op, whitelist.bool_ops):
        for value in node.values:
            _validate(value, symbols, whitelist)
    elif isinstance(node, ast.Compare) and all(
        isinstance(op, whitelist.compare_ops) for op in node.ops
    ):
        _validate(node.left, symbols, whitelist)
        for comparator in node.comparators:
            _validate(comparator, symbols, whitelist)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ExpressionWhitelistError(f"literal {node.value!r} is not a number")
    elif isinstance(node, ast.Name):
        if node.id not in symbols and node.id not in whitelist.constants:
            raise ExpressionWhitelistError(f"unknown name {node.id!r}")
    elif isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in whitelist.functions
        ):
            raise ExpressionWhitelistError(
                "only whitelisted function calls are allowed"
            )
        if node.keywords:
            raise ExpressionWhitelistError("keyword arguments are not allowed")
        for arg in node.args:
            _validate(arg, symbols, whitelist)
    else:
        raise ExpressionWhitelistError(f"disallowed syntax ({type(node).__name__})")


@dataclass(frozen=True)
class CompiledExpression:
    """A validated, compiled expression ready for restricted evaluation.

    Attributes
    ----------
    source : str
        The original expression text.
    is_boolean : bool
        Whether the top-level node is boolean intent (a comparison,
        ``and``/``or``, or ``not``) — consumers that publish floats use
        this to coerce ``True``/``False`` to ``1.0``/``0.0``.
    """

    source: str
    is_boolean: bool
    _code: CodeType = field(repr=False)
    _whitelist: ExpressionWhitelist = field(repr=False)

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        """Evaluate with *values* bound to the symbol names; return the raw result.

        The namespace is functions, then constants, then *values* — so a
        symbol deliberately shadows a same-named constant or function.
        Runtime failures propagate unwrapped.
        """
        namespace: dict[str, Any] = {}
        namespace.update(self._whitelist.functions)
        namespace.update(self._whitelist.constants)
        namespace.update(values)
        return eval(  # noqa: S307 — AST-whitelisted at compile time
            self._code, {"__builtins__": {}}, namespace
        )


def compile_expression(
    expression: str,
    symbols: AbstractSet[str],
    whitelist: ExpressionWhitelist,
    *,
    filename: str = "<restricted-expr>",
) -> CompiledExpression:
    """Parse, whitelist-validate, and compile *expression*.

    Parameters
    ----------
    expression : str
        The expression text (``eval`` mode: a single expression).
    symbols : set of str
        Names the expression may reference as input values, in addition
        to the whitelist's constants.
    whitelist : ExpressionWhitelist
        The permitted functions, constants, and operators.
    filename : str, optional
        The pseudo-filename recorded in the compiled code (shows up in
        runtime tracebacks).

    Raises
    ------
    ExpressionWhitelistError
        Syntax errors, or any construct outside the whitelist (unknown
        names, attribute access, subscripts, non-numeric literals, ...).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionWhitelistError(f"syntax error ({exc.msg})") from exc
    _validate(tree, symbols, whitelist)
    return CompiledExpression(
        source=expression,
        is_boolean=_is_boolean_expression(tree.body),
        _code=compile(tree, filename, "eval"),
        _whitelist=whitelist,
    )
