"""Declarative measurements and a GEST search space for native optimization."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Annotated, ClassVar, Literal

from gest_api.vocs import ContinuousVariable, VOCS
from pydantic import Field, JsonValue, WithJsonSchema, field_validator, model_validator

from ._base import SchemaModel, VersionedSchemaModel
from .restricted_expr import ExpressionWhitelist, compile_expression
from .scan_variables import split_device_variable

GENERATOR_NAMES = (
    "random",
    "bayes_default",
    "bayes_ucb",
    "bayes_ucb_explore",
    "bayes_turbo_standard",
    "bayes_turbo_ucb",
    "bayes_turbo_HTU_e_beam_brightness",
    "multipoint_bax_alignment",
    "multipoint_bax_alignment_l2",
    "multipoint_bax_alignment_simulated",
)

MEASUREMENT_MATH = ExpressionWhitelist(
    functions={
        **{
            name: getattr(math, name)
            for name in (
                "sqrt",
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "exp",
                "log",
                "log10",
            )
        },
        "abs": abs,
        "min": min,
        "max": max,
    },
    constants={"pi": math.pi, "e": math.e},
    binary_ops=(ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv),
    unary_ops=(ast.UAdd, ast.USub),
)

# GEST 0.1 uses arbitrary validated dict types without JSON-schema hooks.
# Describe its compact authoring spelling here; VOCS remains the runtime validator.
_VOCS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["variables"],
    "properties": {
        "variables": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
                "oneOf": [
                    {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    {
                        "type": "object",
                        "required": ["type", "domain"],
                        "properties": {
                            "type": {"const": "ContinuousVariable"},
                            "domain": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "dtype": {},
                            "default_value": {"type": ["number", "null"]},
                        },
                        "additionalProperties": False,
                    },
                ]
            },
        },
        "objectives": {
            "type": "object",
            "additionalProperties": {
                "oneOf": [{"enum": ["MINIMIZE", "MAXIMIZE"]}, {"type": "object"}]
            },
        },
        "observables": {
            "oneOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "object"},
            ]
        },
        "constraints": {
            "type": "object",
            "additionalProperties": {
                "oneOf": [
                    {"type": "array", "minItems": 2, "maxItems": 2},
                    {"type": "object"},
                ]
            },
        },
        "constants": {"type": "object", "additionalProperties": {"type": "number"}},
    },
}


class MeasurementOptions(SchemaModel):
    """How valid shots are reduced into one iteration's measurement."""

    reduce: Literal["mean", "median", "min", "max", "sum", "std"] = Field(
        "mean", description="Reduction of valid per-shot scalar values."
    )
    min_shots: int = Field(
        1, ge=1, description="Minimum valid shots needed for a finite measurement."
    )


class SignalMeasurement(MeasurementOptions):
    """A subscribed scalar in the shot reading."""

    signal: str = Field(description="Device:Variable to read on every shot.")

    @field_validator("signal")
    @classmethod
    def _signal(cls, value: str) -> str:
        split_device_variable(value)
        return value


class DiagnosticMeasurement(MeasurementOptions):
    """A diagnostic document applied to live frames."""

    diagnostic: str = Field(
        min_length=1, description="Diagnostic document ID (file stem)."
    )
    frames: Literal["per_bin", "per_shot"] = Field(
        "per_bin",
        description="Analyze the mean frame, or analyze each frame before reducing scalars.",
    )
    overrides: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Processing/analyzer overrides deep-merged into the diagnostic.",
    )

    @field_validator("diagnostic")
    @classmethod
    def _stem(cls, value: str) -> str:
        if value in (".", "..") or any(c in value for c in ("/", "\\")):
            raise ValueError("diagnostic must be a document ID, not a path")
        return value

    @field_validator("overrides")
    @classmethod
    def _overrides(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if set(value) - {"analyzer", "image"}:
            raise ValueError(
                "diagnostic overrides may change analyzer or image settings only"
            )
        return value


class PythonDerived(SchemaModel):
    """Worker callable over the reduced scalars, for non-expressible objectives."""

    python: str = Field(
        pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$",
        description="Importable module:function accepting a scalar mapping and returning a float.",
    )


class OptimizerGenerator(SchemaModel):
    """Generator recipe and its algorithm-specific settings."""

    name: str = Field(description="Worker generator recipe name.")
    options: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Recipe-specific options validated by the worker before opening a run.",
    )

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if value not in GENERATOR_NAMES:
            raise ValueError(
                f"unknown generator {value!r}; choose from {GENERATOR_NAMES}"
            )
        return value


class OptimizationRun(SchemaModel):
    """Run defaults; queue-item arguments may override the iteration and shot budgets."""

    on_finish: Literal["best", "hold"] = Field(
        "best",
        description="Move to the best feasible observation, or hold; no best restores initial positions. Relative pseudos always restore on unstage.",
    )
    seed_dumps: list[str] = Field(
        default_factory=list,
        description="Previous xopt_dump.yaml files; relative paths resolve beside this config.",
    )
    shots_per_step: int = Field(
        5, ge=1, description="Successful strict acquisitions per iteration."
    )
    max_iterations: int | None = Field(
        None,
        ge=1,
        description="Iteration limit; required here or in the submitted plan arguments.",
    )


class OptimizerConfig(VersionedSchemaModel):
    """An optimization: search space, live measurements, derived outputs and generator."""

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1
    vocs: Annotated[VOCS, WithJsonSchema(_VOCS_SCHEMA)] = Field(
        description="GEST variables, objectives, constraints, constants and observables; compact input and typed canonical output are accepted."
    )
    measurements: dict[str, SignalMeasurement | DiagnosticMeasurement] = Field(
        min_length=1, description="Named live scalar or diagnostic measurements."
    )
    derived: dict[str, str | PythonDerived] = Field(
        default_factory=dict,
        description="Expressions or worker callables, in dependency order.",
    )
    generator: OptimizerGenerator = Field(
        description="The generator recipe used to propose positions."
    )
    run: OptimizationRun = Field(
        default_factory=OptimizationRun, description="Run defaults and finish policy."
    )

    @model_validator(mode="before")
    @classmethod
    def _shape(cls, data: object) -> object:
        if isinstance(data, Mapping):
            if "evaluator" in data or "device_requirements" in data:
                raise ValueError(
                    "legacy optimizer config is not loadable: the "
                    "'evaluator'/'device_requirements' shape was replaced by "
                    "schema_version 1 (vocs / measurements / derived / "
                    "generator / run) — re-author the document, or take the "
                    "regenerated one from the configs repo"
                )
            if data.get("schema_version", 1) != 1:
                raise ValueError("unsupported optimizer schema_version (expected 1)")
            # GEST's typed-dict parser pops type tags; do not mutate caller data.
            return deepcopy(dict(data))
        return data

    @model_validator(mode="after")
    def _references(self) -> OptimizerConfig:
        if not self.vocs.variables:
            raise ValueError("vocs.variables must not be empty")
        for name, variable in self.vocs.variables.items():
            if not isinstance(variable, ContinuousVariable) or not all(
                math.isfinite(v) for v in variable.domain
            ):
                raise ValueError(
                    f"{name}: native optimization requires finite continuous bounds"
                )
        if not self.vocs.objectives and not self.vocs.observables:
            raise ValueError("vocs needs an objective or observable")
        names = set(self.measurements)
        for name in [*self.measurements, *self.derived]:
            if (
                not re.fullmatch(r"[A-Za-z_]\w*", name)
                or name in MEASUREMENT_MATH.functions
                or name in MEASUREMENT_MATH.constants
            ):
                raise ValueError(
                    f"invalid or reserved measurement/derived name {name!r}"
                )
        if names & self.derived.keys():
            raise ValueError("measurement and derived names must be distinct")
        known = {
            name
            for name, m in self.measurements.items()
            if isinstance(m, SignalMeasurement)
        }
        for name, expression in self.derived.items():
            if isinstance(expression, str):
                refs = expression_references(expression)
                for ref in refs:
                    self._check_reference(ref, known)
                compile_expression(expression, known | refs, MEASUREMENT_MATH)
            known.add(name)
        outputs = [
            *self.vocs.objectives,
            *self.vocs.observables,
            *self.vocs.constraints,
        ]
        if len(outputs) != len(set(outputs)):
            raise ValueError(
                "objective, observable and constraint names must be distinct"
            )
        for ref in outputs:
            self._check_reference(ref, known)
        if set(self.vocs.variables) & (
            set(outputs) | set(self.measurements) | set(self.derived)
        ):
            raise ValueError("variable and output names must be distinct")
        return self

    def _check_reference(self, ref: str, known: set[str]) -> None:
        if ref in known:
            return
        owner, dot, scalar = ref.partition(".")
        if (
            dot
            and scalar
            and isinstance(self.measurements.get(owner), DiagnosticMeasurement)
        ):
            return  # Emitted keys belong to the diagnostic resolved by the worker.
        raise ValueError(f"unknown or forward measurement reference {ref!r}")


def expression_references(expression: str) -> set[str]:
    """Extract scalar symbols without treating math functions as measurements."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid derived expression: {exc.msg}") from exc
    refs: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Attribute):
            refs.add(ast.unparse(node))
        elif isinstance(node, ast.Name):
            if (
                node.id not in MEASUREMENT_MATH.functions
                and node.id not in MEASUREMENT_MATH.constants
            ):
                refs.add(node.id)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child)

    visit(tree)
    return refs


def optimizer_required_devices(
    cfg: OptimizerConfig, diagnostic_devices: Mapping[str, str]
) -> frozenset[str]:
    """Return required run devices using the consumer's resolved diagnostic names."""
    return frozenset(
        split_device_variable(m.signal)[0]
        if isinstance(m, SignalMeasurement)
        else diagnostic_devices[m.diagnostic]
        for m in cfg.measurements.values()
    )
