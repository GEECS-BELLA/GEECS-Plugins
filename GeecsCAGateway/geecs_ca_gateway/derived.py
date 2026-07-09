"""Derived numeric channel loading and expression evaluation."""

from __future__ import annotations

import ast
import configparser
import json
import math
import os
from pathlib import Path
from typing import Any

from geecs_schemas import (
    DerivedChannel as DerivedChannelSpec,
    DerivedChannels,
    DerivedInput as DerivedInputSpec,
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
_RESERVED_NAMES = set(_ALLOWED_FUNCS) | set(_ALLOWED_CONSTS)
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not)
_ALLOWED_BOOLOPS = (ast.And, ast.Or)
_ALLOWED_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
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
    """Compile and evaluate a restricted numeric/status expression."""

    def __init__(self, expression: str, symbols: set[str]) -> None:
        self.expression = expression
        self.symbols = symbols
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise DerivedExpressionError(str(exc)) from exc
        self._validate_node(tree)
        self._code = compile(tree, "<derived-channel>", "eval")

    def evaluate(self, values: dict[str, float]) -> float:
        """Evaluate the expression with numeric input values.

        Boolean/status expressions are accepted and stored as ``1.0``/``0.0``
        on the derived float PV.
        """
        missing = self.symbols - values.keys()
        if missing:
            raise KeyError(f"missing derived-channel input(s): {sorted(missing)}")
        namespace: dict[str, Any] = {}
        namespace.update(_ALLOWED_FUNCS)
        namespace.update(_ALLOWED_CONSTS)
        namespace.update(values)
        result = eval(self._code, {"__builtins__": {}}, namespace)  # noqa: S307
        return float(result)

    def _validate_node(self, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            self._validate_node(node.body)
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise DerivedExpressionError("only numeric constants are allowed")
            return
        if isinstance(node, ast.Name):
            if node.id not in self.symbols and node.id not in _RESERVED_NAMES:
                raise DerivedExpressionError(f"unknown name in expression: {node.id}")
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise DerivedExpressionError("unsupported binary operator")
            self._validate_node(node.left)
            self._validate_node(node.right)
            return
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, _ALLOWED_BOOLOPS):
                raise DerivedExpressionError("unsupported boolean operator")
            for value in node.values:
                self._validate_node(value)
            return
        if isinstance(node, ast.Compare):
            self._validate_node(node.left)
            for op in node.ops:
                if not isinstance(op, _ALLOWED_CMPOPS):
                    raise DerivedExpressionError("unsupported comparison operator")
            for comparator in node.comparators:
                self._validate_node(comparator)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise DerivedExpressionError("unsupported unary operator")
            self._validate_node(node.operand)
            return
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _ALLOWED_FUNCS
            ):
                raise DerivedExpressionError("only whitelisted math calls are allowed")
            if node.keywords:
                raise DerivedExpressionError("keyword arguments are not allowed")
            for arg in node.args:
                self._validate_node(arg)
            return
        raise DerivedExpressionError(
            f"unsupported expression element: {type(node).__name__}"
        )
