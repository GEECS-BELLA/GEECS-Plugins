"""Derived numeric channel loading and expression evaluation."""

from __future__ import annotations

import ast
import math
from pathlib import Path

from geecs_schemas import (
    DerivedChannel as DerivedChannelSpec,
    DerivedChannels,
    DerivedInput as DerivedInputSpec,
)
from geecs_schemas.restricted_expr import (
    CompiledExpression,
    ExpressionWhitelist,
    ExpressionWhitelistError,
    compile_expression,
)

from geecs_core.configs_repo import experiment_config_path
from geecs_core.configs_repo import scanner_configs_base as _scanner_configs_base
from geecs_core.pv_naming import pv_name

_ALLOWED_FUNCS = {
    name: getattr(math, name)
    for name in (
        "acos",
        "asin",
        "atan",
        "cos",
        "exp",
        "isfinite",
        "log",
        "log10",
        "sin",
        "sqrt",
        "tan",
    )
}
_ALLOWED_CONSTS = {"e": math.e, "pi": math.pi, "tau": math.tau}
_WHITELIST = ExpressionWhitelist(
    functions=_ALLOWED_FUNCS,
    constants=_ALLOWED_CONSTS,
    binary_ops=(ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod),
    unary_ops=(ast.UAdd, ast.USub, ast.Not),
    bool_ops=(ast.And, ast.Or),
    compare_ops=(ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE),
)
GATEWAY_CONFIG_FOLDER = "gateway"
DERIVED_CHANNELS_FILENAME = "derived_channels.yaml"

__all__ = [
    "DerivedChannelSpec",
    "DerivedChannels",
    "DerivedExpressionError",
    "DerivedInputSpec",
    "ExpressionEvaluator",
    "default_derived_channels_path",
    "derived_pv_name",
    "load_derived_channels",
    "scanner_configs_base",
]


class DerivedExpressionError(ValueError):
    """Raised when a derived-channel expression is outside the supported subset."""


def derived_pv_name(
    spec: DerivedChannelSpec, default_experiment: str | None = None
) -> str:
    """Return the full output PV name for a derived-channel declaration.

    The components come from the schema (``DerivedChannel.pv_parts``: the
    ``experiment`` / ``pv`` overrides resolved) and the join from
    ``geecs_core.pv_naming`` — the archiver mints its archive requests from
    the same two, so the served and the requested name cannot differ.
    """
    return pv_name(*spec.pv_parts(default_experiment))


def load_derived_channels(path: str | Path) -> list[DerivedChannelSpec]:
    """Load a YAML or JSON derived-channel document from *path*."""
    config_path = Path(path)
    try:
        document = DerivedChannels.from_path(config_path)
    except Exception as exc:
        raise ValueError(f"failed to load derived channels from {config_path}") from exc
    return list(document.derived_channels)


def scanner_configs_base() -> Path | None:
    """Resolve the configs repo ``scanner_configs/experiments`` base if known.

    Delegates to ``geecs_core.configs_repo.scanner_configs_base`` — the one
    resolver every consumer of the configs repository shares.
    """
    return _scanner_configs_base()


def default_derived_channels_path(experiment: str) -> Path | None:
    """Return the conventional configs-repo derived-channel file, if present."""
    return experiment_config_path(
        experiment, GATEWAY_CONFIG_FOLDER, DERIVED_CHANNELS_FILENAME
    )


class ExpressionEvaluator:
    """Compile and evaluate a restricted numeric/status expression.

    The compile-then-restricted-eval skeleton is the shared
    :mod:`geecs_schemas.restricted_expr` core; this class supplies the
    derived-channel whitelist (comparisons/bool-ops, ``isfinite``,
    ``tau``), the per-expression symbol set, and the gateway's error and
    result contracts (``DerivedExpressionError``; booleans published as
    ``1.0``/``0.0``).
    """

    def __init__(self, expression: str, symbols: set[str]) -> None:
        self.expression = expression
        self.symbols = symbols
        try:
            # Function names double as extra symbols: a bare function name
            # used as a value is compile-legal (it fails at evaluate time
            # in float()) — longstanding behavior, kept.
            self._compiled: CompiledExpression = compile_expression(
                expression,
                symbols | set(_ALLOWED_FUNCS),
                _WHITELIST,
                filename="<derived-channel>",
            )
        except ExpressionWhitelistError as exc:
            raise DerivedExpressionError(str(exc)) from exc
        self._coerce_bool_result = self._compiled.is_boolean

    def evaluate(self, values: dict[str, float]) -> float:
        """Evaluate the expression with numeric input values.

        Boolean/status expressions are accepted and stored as ``1.0``/``0.0``
        on the derived float PV.
        """
        missing = self.symbols - values.keys()
        if missing:
            raise KeyError(f"missing derived-channel input(s): {sorted(missing)}")
        result = self._compiled.evaluate(values)
        if self._coerce_bool_result:
            return float(bool(result))
        return float(result)
