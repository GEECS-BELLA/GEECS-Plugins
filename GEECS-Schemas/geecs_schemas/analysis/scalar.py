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

A column may instead be a **condition**: ``cases``, rules checked in
order (the first that holds wins), each with a label, and ``otherwise``
for a shot no rule matches::

      - name: target in beam
        inputs: {x: Target Stage Position}
        cases:
          - {when: "4.0 <= x <= 6.5", label: Al foil}
          - {when: "10 <= x <= 12", label: Cu foil}
        otherwise: nothing

The s-file gets the number of the case that matched (0 for the first,
then 1, 2 ...; ``otherwise`` is the number after the last case), and the
runner writes the labels beside the job's sidecar so a page can say
"target in beam: Al foil".  A shot missing any of the column's inputs is
unknown (a blank cell) rather than ``otherwise``.

Columns are evaluated in order, so a later formula may use an earlier
column's ``name`` as an input.  The formula language is the restricted one
the CA gateway's derived channels use (:mod:`geecs_schemas.restricted_expr`):
numbers, the symbols named in ``inputs``, ``+ - * / ** %``, the
comparisons ``< <= > >= == !=`` (chains like ``a <= x <= b`` included),
``and`` / ``or`` / ``not``, and the functions in :data:`SCALAR_FUNCTIONS` —
never attribute access, imports or any other Python.  A comparison is 1
where it holds and 0 where it does not.  The syntax is checked here, when the file is loaded, so
a typo fails in the editor and the catalog rather than at run time.
"""

from __future__ import annotations

import ast
from typing import Literal, Optional

from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

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
SCALAR_UNARY_OPS: tuple[type[ast.unaryop], ...] = (ast.UAdd, ast.USub, ast.Not)
SCALAR_COMPARE_OPS: tuple[type[ast.cmpop], ...] = (
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)
SCALAR_BOOL_OPS: tuple[type[ast.boolop], ...] = (ast.And, ast.Or)

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
        bool_ops=SCALAR_BOOL_OPS,
        compare_ops=SCALAR_COMPARE_OPS,
    )


def _placeholder(*_args: float) -> float:  # never called: syntax check only
    return 0.0


_CHECK_WHITELIST = scalar_whitelist(
    {name: _placeholder for name in SCALAR_FUNCTIONS},
    {name: 0.0 for name in SCALAR_CONSTANTS},
)


def _check(text: str, symbols: set[str], what: str) -> None:
    try:
        compile_expression(text, symbols, _CHECK_WHITELIST)
    except ExpressionWhitelistError as exc:
        raise ValueError(f"{what} {text!r}: {exc}") from exc


class ScalarCase(SchemaModel):
    """One rule of a condition column: when it holds, the shot gets its label."""

    when: str = Field(
        ...,
        min_length=1,
        description=(
            "The rule, over the symbols in inputs: comparisons < <= > >= == !=, "
            "ranges like '4.0 <= x <= 6.5', and/or/not, e.g. "
            "'x > 10 and y < 2'. Rules are checked top to bottom; the first one "
            "that holds wins."
        ),
    )
    label: str = Field(
        ...,
        min_length=1,
        description=(
            "What it means, as pages show it, e.g. 'Al foil in beam'. The "
            "s-file stores the case's number (0 for the first case, 1 for the "
            "next, ...)."
        ),
    )


class ScalarColumn(SchemaModel):
    """One new s-file column: a formula, or condition rules, over existing columns."""

    # A form hint (the config editor's ``x-choice``): a column is one of
    # two kinds, each owning its fields — the editor shows a switch and
    # only the chosen kind's fields, and writes only those.
    model_config = ConfigDict(
        json_schema_extra={
            "x-choice": {
                "label": "kind",
                "options": [
                    {
                        "title": "formula",
                        "keys": ["expr"],
                        "help": "A number per shot from a formula over the inputs.",
                    },
                    {
                        "title": "condition",
                        "keys": ["cases", "otherwise"],
                        "help": (
                            "A label per shot: rules checked in order, the first "
                            "that holds wins; the s-file stores the rule's number."
                        ),
                    },
                ],
            }
        }
    )

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "The new column's name in the s-file, units included if you like, "
            "e.g. 'UC_TopView x (mm)'. It must not be one of the scanner's own "
            "columns: a job never overwrites raw data."
        ),
    )
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "The columns this column reads, each under a short symbol: "
            "symbol → existing s-file column, e.g. {x: 'UC_TopView centroid_x'}. "
            "Both kinds use them: a formula's expr and a condition's rules name "
            "the symbols. A column made earlier in this job may be used by its name."
        ),
        # the config editor's row words (x-rows): one empty row on a new column
        json_schema_extra={
            "x-rows": {
                "key": "symbol, e.g. x",
                "value": "s-file column",
                "add": "+ input",
                "start": True,
            }
        },
    )
    expr: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "The formula, e.g. 'x * 0.0244' or 'a * b / c'. Numbers, the "
            "symbols in inputs, + - * / ** %, comparisons (1 where true, 0 "
            "where false), and/or/not, and the functions "
            + ", ".join(SCALAR_FUNCTIONS)
            + " (and pi, e). For a column of kind formula; a condition has cases instead."
        ),
    )
    cases: Optional[list[ScalarCase]] = Field(
        None,
        min_length=1,
        description=(
            "For a column of kind condition: rules checked in order, each with "
            "a label, e.g. when '4.0 <= x <= 6.5' → 'Al foil'. The first rule "
            "that holds wins; a shot missing an input is unknown. The Images "
            "page lists the scan's label at the top."
        ),
    )
    otherwise: Optional[str] = Field(
        None,
        description=(
            "With cases: the label for a shot no rule matches (stored as the "
            "number after the last case). Defaults to 'other'."
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
        if self.expr is not None and self.cases is not None:
            raise ValueError(
                f"column {self.name!r} has both a formula (expr) and condition "
                "cases: a column is one kind — keep one"
            )
        if self.expr is None and self.cases is None:
            raise ValueError(
                f"column {self.name!r} is empty: a formula needs expr, a condition "
                "at least one case"
            )
        if self.otherwise is not None and self.cases is None:
            raise ValueError(f"column {self.name!r}: otherwise needs cases")
        symbols = set(self.inputs)
        if self.expr is not None:
            _check(self.expr, symbols, "expr")
        for place, case in enumerate(self.cases or []):
            _check(case.when, symbols, f"case {place + 1} when")
        return self

    @property
    def is_condition(self) -> bool:
        """Whether the column is condition rules (``cases``) rather than a formula."""
        return self.cases is not None

    @property
    def labels(self) -> list[str]:
        """A condition's labels by stored number: the cases', then ``otherwise``."""
        if self.cases is None:
            return []
        return [c.label for c in self.cases] + [self.otherwise or "other"]


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

    The file's stem is the ID groups and the portal refer to.  The job's
    product is its columns, merged into the s-file under the shared lock;
    under ``analysis/ScanNNN/`` it writes only its scalar sidecar and, for
    condition columns, their labels.
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
    "ScalarCase",
    "ScalarColumn",
    "ScalarJob",
    "ScalarJobRuntime",
    "scalar_whitelist",
]
