"""The scalar-columns job: new s-file columns computed from existing ones.

A ``scalar_columns`` job sits in the analyzers tree beside the recipes and
the ``.himg`` jobs (groups and the portal's Analysis tab list it by file
stem).  It reads no device folder: it takes columns already in the scan's
s-file (``analysis/sNNN.txt`` — the scanner's own columns plus any an
earlier analysis appended), evaluates one formula per new column, shot by
shot, and merges the results back into the s-file::

    kind: scalar_columns
    columns:
      - name: UC_TopView x (mm)
        expr: x * 0.0244
        inputs: {x: UC_TopView centroid_x}
      - name: charge ratio
        expr: a * b / c
        inputs: {a: ..., b: ..., c: ...}

Columns are evaluated in order, so a later formula may use an earlier
column's ``name`` as an input.  The formula language is the restricted one
the CA gateway's derived channels use (:mod:`geecs_schemas.restricted_expr`):
numbers, the symbols named in ``inputs``, ``+ - * / ** %``, and the
functions in :data:`SCALAR_FUNCTIONS` — never attribute access, imports or
any other Python.  The syntax is checked here, when the file is loaded, so
a typo fails in the editor and the catalog rather than at run time.
"""

from __future__ import annotations

import ast
from typing import Literal, Optional

from pydantic import Field, PrivateAttr, field_validator, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.restricted_expr import (
    ExpressionWhitelist,
    ExpressionWhitelistError,
    compile_expression,
)

#: The functions a formula may call (each applied shot by shot).  The
#: runner binds them to their numpy namesakes; ``abs``/``min``/``max`` are
#: element-wise (``minimum``/``maximum`` for two inputs).
SCALAR_FUNCTIONS: tuple[str, ...] = (
    "abs",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "hypot",
    "minimum",
    "maximum",
)

#: The bare constants a formula may name.
SCALAR_CONSTANTS: tuple[str, ...] = ("pi", "e")

#: The operators a formula may use.
SCALAR_BINARY_OPS: tuple[type[ast.operator], ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
)
SCALAR_UNARY_OPS: tuple[type[ast.unaryop], ...] = (ast.UAdd, ast.USub)

_RESERVED = frozenset(SCALAR_FUNCTIONS) | frozenset(SCALAR_CONSTANTS)


def scalar_whitelist(functions: dict, constants: dict) -> ExpressionWhitelist:
    """The scalar-columns whitelist with the caller's callables bound.

    The schema checks syntax with placeholder callables; the runner binds
    the real (numpy) ones.  Both go through here so the permitted set is
    the one defined above.
    """
    return ExpressionWhitelist(
        functions={name: functions[name] for name in SCALAR_FUNCTIONS},
        constants={name: constants[name] for name in SCALAR_CONSTANTS},
        binary_ops=SCALAR_BINARY_OPS,
        unary_ops=SCALAR_UNARY_OPS,
    )


def _placeholder(*_args: float) -> float:  # never called: syntax check only
    return 0.0


_CHECK_WHITELIST = scalar_whitelist(
    {name: _placeholder for name in SCALAR_FUNCTIONS},
    {name: 0.0 for name in SCALAR_CONSTANTS},
)


class ScalarColumn(SchemaModel):
    """One new s-file column: a name, a formula, and the columns it reads."""

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "The new column's name in the s-file, units included if you like, "
            "e.g. 'UC_TopView x (mm)'. It must not be one of the scanner's own "
            "columns: a job never overwrites raw data."
        ),
    )
    expr: str = Field(
        ...,
        min_length=1,
        description=(
            "The formula, e.g. 'x * 0.0244' or 'a * b / c'. Numbers, the "
            "symbols in inputs, + - * / ** %, and the functions "
            + ", ".join(SCALAR_FUNCTIONS)
            + " (and pi, e)."
        ),
    )
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Symbol → existing s-file column, e.g. {x: 'UC_TopView centroid_x'}. "
            "A column made earlier in this job may be used by its name."
        ),
    )
    description: Optional[str] = Field(None, description="Free-text note.")

    @field_validator("inputs")
    @classmethod
    def _symbols_are_names(cls, inputs: dict[str, str]) -> dict[str, str]:
        for symbol, column in inputs.items():
            if not symbol.isidentifier():
                raise ValueError(f"input symbol {symbol!r} is not a valid name")
            if symbol in _RESERVED:
                raise ValueError(
                    f"input symbol {symbol!r} is a function or constant name"
                )
            if not column.strip():
                raise ValueError(f"input {symbol!r} names no column")
        return inputs

    @model_validator(mode="after")
    def _expression_parses(self) -> "ScalarColumn":
        try:
            compile_expression(self.expr, set(self.inputs), _CHECK_WHITELIST)
        except ExpressionWhitelistError as exc:
            raise ValueError(f"expr {self.expr!r}: {exc}") from exc
        return self


class ScalarJobRuntime(SchemaModel):
    """How the job runs among a group's analyses."""

    priority: int = Field(
        default=1000,
        ge=0,
        description=(
            "Run order within a group: lower runs first. The default, 1000, "
            "runs after the recipes whose columns a formula may read."
        ),
    )


class ScalarJob(VersionedSchemaModel):
    """New s-file columns computed from existing ones, shot by shot.

    The file's stem is the ID groups and the portal refer to.  Nothing is
    written under ``analysis/ScanNNN/``; the job's one product is its
    columns, merged into the s-file under the shared lock.
    """

    schema_version: int = Field(
        1,
        description=(
            "Format version of this job file. Leave at 1 — tools update this "
            "automatically when the file format changes."
        ),
    )
    kind: Literal["scalar_columns"] = Field(
        ...,
        description="scalar_columns: compute new s-file columns from existing ones.",
    )
    columns: list[ScalarColumn] = Field(
        ...,
        min_length=1,
        description="The new columns, evaluated in this order.",
    )
    output_name: Optional[str] = Field(
        None,
        description="Label the portal's Analysis tab lists the job under. Defaults to the file name.",
    )
    description: Optional[str] = Field(
        None, description="Free-text note about this job."
    )
    scan: ScalarJobRuntime = Field(
        default_factory=ScalarJobRuntime,
        description="How the job runs among a group's analyses.",
    )
    _source_id: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _names_are_unique(self) -> "ScalarJob":
        seen: set[str] = set()
        for column in self.columns:
            if column.name in seen:
                raise ValueError(f"column {column.name!r} is defined twice")
            seen.add(column.name)
        return self

    # The format-neutral names the analysis documents carry, so a host
    # lists a job beside the recipes without asking which document it holds.

    @property
    def source_id(self) -> Optional[str]:
        """Filename-derived job ID when loaded from disk, if known."""
        return self._source_id

    @property
    def device(self) -> str:
        """No device: the job's label (the hosts' fallback ID), else its kind."""
        return self.output_name or self._source_id or self.kind

    @property
    def effective_output_name(self) -> str:
        """``output_name`` if set, else the file stem."""
        return self.device

    @property
    def data_folder(self) -> str:
        """No device folder: the job reads the s-file alone (an empty name)."""
        return ""

    @property
    def destructive(self) -> bool:
        """Never: the job only adds or replaces its own columns."""
        return False


__all__ = [
    "SCALAR_CONSTANTS",
    "SCALAR_FUNCTIONS",
    "ScalarColumn",
    "ScalarJob",
    "ScalarJobRuntime",
    "scalar_whitelist",
]
