"""Derived numeric channel loading and expression evaluation."""

from __future__ import annotations

import ast
import configparser
import json
import math
import os
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

from .naming import normalize_pv_component

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
    """Return the full output PV name for a derived-channel declaration."""
    parts: list[str] = []
    experiment = spec.experiment or default_experiment
    if experiment:
        parts.append(normalize_pv_component(experiment))
    parts.append(normalize_pv_component(spec.device))
    parts.append(normalize_pv_component(spec.pv or spec.variable))
    return ":".join(parts)


def load_derived_channels(path: str | Path) -> list[DerivedChannelSpec]:
    """Load a YAML or JSON derived-channel document from *path*."""
    config_path = Path(path)
    try:
        if config_path.suffix.lower() == ".json":
            data = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load derived-channel YAML files. "
                    "Install the GeecsCAGateway package dependencies."
                ) from exc
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        document = DerivedChannels.model_validate(data or {})
    except Exception as exc:
        raise ValueError(f"failed to load derived channels from {config_path}") from exc
    return list(document.derived_channels)


def scanner_configs_base() -> Path | None:
    """Resolve the configs repo ``scanner_configs/experiments`` base if known.

    Resolution mirrors the scanner/Bluesky production path without importing
    either package:

    1. ``GEECS_SCANNER_CONFIG_DIR`` points directly at
       ``scanner_configs/experiments``.
    2. ``GEECS_PLUGINS_CONFIGS`` points at the configs repo root.
    3. ``~/.config/geecs_python_api/config.ini`` has
       ``[Paths] scanner_config_root_path`` pointing at the configs repo root.
    """
    env = os.environ.get("GEECS_SCANNER_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    repo_env = os.environ.get("GEECS_PLUGINS_CONFIGS")
    if repo_env:
        return Path(repo_env).expanduser().resolve() / "scanner_configs" / "experiments"
    config_ini = Path("~/.config/geecs_python_api/config.ini").expanduser()
    if config_ini.exists():
        parser = configparser.ConfigParser()
        parser.read(config_ini)
        root = parser.get("Paths", "scanner_config_root_path", fallback=None)
        if root:
            return Path(root).expanduser().resolve() / "scanner_configs" / "experiments"
    return None


def default_derived_channels_path(experiment: str) -> Path | None:
    """Return the conventional configs-repo derived-channel file, if present."""
    base = scanner_configs_base()
    if base is None:
        return None
    path = base / experiment / GATEWAY_CONFIG_FOLDER / DERIVED_CHANNELS_FILENAME
    return path if path.exists() else None


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
