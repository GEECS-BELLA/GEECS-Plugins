"""The ``scalar_columns`` job: new s-file columns computed from existing ones.

The runner of a :class:`~geecs_schemas.analysis.ScalarJob` document.  It
reads no device folder: it takes the scan's s-file (``analysis/sNNN.txt``,
already loaded by the base class as ``auxiliary_data``), evaluates each
column's formula over every shot (numpy, element-wise, in the job's order
so a later formula may read an earlier one), and merges the new columns
back into the s-file under the shared lock, keyed on ``Shotnumber`` —
the same write a recipe's scalars take, sidecar included.

Refusals, each before anything is written:

- a column ``name`` that is one of the scanner's own columns
  (``scans/ScanNNN/ScanDataScanNNN.txt``): a job never overwrites raw
  data — re-running replaces only the job's own earlier columns;
- an input column the s-file lacks: the whole job is ``no_data``, its
  message naming the missing column and the closest names the s-file
  does have.

Non-numeric cells read as NaN, and so does a result that is not finite
(a division by zero, the log of a negative): a bad shot is a blank cell,
never a failed run.  A shot missing any of a column's inputs is NaN too —
for a comparison or a condition that means "unknown", never "false".

A **condition** column (``cases``) stores the number of the first rule
that holds (``otherwise`` the number after the last); its labels are
written beside the sidecar (``geecs_data_utils.scan_conditions``), which is
where the portal's Images page reads them.  Comparisons and ``and`` /
``or`` / ``not`` are evaluated element-wise: the expression's tree is
rewritten so Python's short-circuit operators become numpy's logical ones
and a chain ``a <= x <= b`` becomes ``(a <= x) & (x <= b)``.

Honours the ``ScanAnalyzer`` contract the task queue, the portal and MCP
call: ``run_analysis(scan_tag)`` returns one label saying what was added.
Nothing here creates a directory under ``scans/``.
"""

from __future__ import annotations

import ast
import difflib
import logging
from functools import reduce
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

import numpy as np
import pandas as pd
from geecs_data_utils.scan_conditions import labels_path, write_condition_labels
from geecs_schemas.analysis import ScalarJob
from geecs_schemas.analysis.scalar import ScalarColumn, scalar_whitelist
from geecs_schemas.restricted_expr import compile_expression

from scan_analysis.base import DataUnavailableWarning, ScanAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["ScalarColumnsAnalyzer"]

#: The key the s-file merge joins on.
KEY = "Shotnumber"

_WHITELIST = scalar_whitelist(
    {
        "abs": np.abs,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "arctan2": np.arctan2,
        "hypot": np.hypot,
        "minimum": np.minimum,
        "maximum": np.maximum,
    },
    {"pi": float(np.pi), "e": float(np.e)},
)


def _logical(op: Callable) -> Callable:
    def apply(*values):
        return reduce(op, [np.asarray(v, dtype=bool) for v in values])

    return apply


#: The helpers the rewritten tree calls (names no input symbol can take:
#: symbols are identifiers, and these start with an underscore pair).
_HELPERS = {
    "__and": _logical(np.logical_and),
    "__or": _logical(np.logical_or),
    "__not": lambda value: np.logical_not(np.asarray(value, dtype=bool)),
}


class _ElementWise(ast.NodeTransformer):
    """``and``/``or``/``not`` and comparison chains → numpy's element-wise forms."""

    @staticmethod
    def _call(name: str, args: list[ast.expr]) -> ast.Call:
        return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        name = "__and" if isinstance(node.op, ast.And) else "__or"
        return self._call(name, list(node.values))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return self._call("__not", [node.operand])
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        lefts = [node.left, *node.comparators[:-1]]
        pairs = [
            ast.Compare(left=left, ops=[op], comparators=[right])
            for left, op, right in zip(lefts, node.ops, node.comparators)
        ]
        return self._call("__and", pairs)


def _element_wise(source: str, symbols: set[str]) -> Callable[[Mapping], np.ndarray]:
    """Validate *source* against the whitelist, then compile it element-wise."""
    compile_expression(source, symbols, _WHITELIST)  # the refusals are the schema's
    tree = ast.parse(source, mode="eval")
    tree = ast.fix_missing_locations(_ElementWise().visit(tree))
    code = compile(tree, "<scalar-column>", "eval")
    base = {
        "__builtins__": {},
        **_WHITELIST.functions,
        **_WHITELIST.constants,
        **_HELPERS,
    }

    def evaluate(values: Mapping) -> np.ndarray:
        return np.asarray(eval(code, {**base, **values}), dtype=float)

    return evaluate


def _column_function(column: ScalarColumn) -> Callable[[Mapping], np.ndarray]:
    """One column's values from its inputs: the formula, or the case number."""
    symbols = set(column.inputs)
    if column.cases is None:
        return _element_wise(str(column.expr), symbols)
    rules = [_element_wise(case.when, symbols) for case in column.cases]

    def cases(values: Mapping) -> np.ndarray:
        n = len(next(iter(values.values()))) if values else 1
        codes = np.full(n, float(len(rules)))  # otherwise
        decided = np.zeros(n, dtype=bool)
        for number, rule in enumerate(rules):
            with np.errstate(invalid="ignore"):
                holds = np.broadcast_to(rule(values) > 0, (n,))
            hit = holds & ~decided
            codes[hit] = number
            decided |= hit
        return codes

    return cases


def _close_names(name: str, columns: list[str]) -> str:
    """A hint for a missing column: the s-file's nearest names, if any."""
    near = difflib.get_close_matches(name, columns, n=3, cutoff=0.6)
    return f" (did you mean {', '.join(repr(n) for n in near)}?)" if near else ""


class ScalarColumnsAnalyzer(ScanAnalyzer):
    """Compute a :class:`ScalarJob`'s columns and merge them into the s-file.

    Parameters
    ----------
    job : ScalarJob
        The validated document (its formulas were syntax-checked at load).
    """

    def __init__(self, job: ScalarJob, **kwargs):
        super().__init__(device_name=job.device, **kwargs)
        self.job = job
        self._compiled = [_column_function(column) for column in job.columns]

    def _raw_columns(self) -> set[str]:
        """The scanner's own column names (header of ``ScanDataScanNNN.txt``)."""
        if self.scan_directory is None:
            return set()
        raw = (
            Path(self.scan_directory) / f"ScanData{Path(self.scan_directory).name}.txt"
        )
        try:
            with raw.open(encoding="utf-8", errors="replace") as handle:
                header = handle.readline()
        except OSError:
            return set()
        return {
            name.strip() for name in header.rstrip("\r\n").split("\t") if name.strip()
        }

    def compute(self, table: pd.DataFrame) -> pd.DataFrame:
        """The job's columns over *table*: ``Shotnumber`` + one column each.

        Raises
        ------
        DataUnavailableWarning
            When an input column is not in *table* (nor made earlier in the job).
        ValueError
            When *table* has no ``Shotnumber`` column.
        """
        if KEY not in table.columns:
            raise ValueError(f"the s-file has no {KEY!r} column to merge on")
        available = [str(c) for c in table.columns]
        made: dict[str, np.ndarray] = {}
        missing = [
            f"{column.name!r} needs {source!r}{_close_names(source, available)}"
            for column in self.job.columns
            for source in column.inputs.values()
            if source not in table.columns
            and source not in {c.name for c in self.job.columns}
        ]
        if missing:
            raise DataUnavailableWarning(
                "input columns not in this scan's s-file: " + "; ".join(missing)
            )
        n = len(table)
        for column, compiled in zip(self.job.columns, self._compiled):
            values = {}
            for symbol, source in column.inputs.items():
                if source in made:
                    values[symbol] = made[source]
                else:
                    values[symbol] = pd.to_numeric(
                        table[source], errors="coerce"
                    ).to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                result = np.asarray(compiled(values), dtype=float)
            result = np.broadcast_to(result, (n,)).copy()
            result[~np.isfinite(result)] = np.nan
            for given in values.values():
                # a missing input is unknown, never "false" or "otherwise"
                result[~np.isfinite(np.broadcast_to(given, (n,)))] = np.nan
            made[column.name] = result
        out = pd.DataFrame({KEY: table[KEY].to_numpy()})
        for name, values in made.items():
            out[name] = values
        return out

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        """Refuse raw-column names, compute, then merge into the s-file."""
        clashes = sorted({c.name for c in self.job.columns} & self._raw_columns())
        if clashes:
            raise ValueError(
                "these names are the scanner's own columns and are never "
                f"overwritten: {', '.join(repr(c) for c in clashes)} — rename them"
            )
        updates = self.compute(self.auxiliary_data)
        # The sidecar first, like a recipe's scalars: it survives a
        # regenerated s-file.
        self.write_scalar_sidecar(updates)
        if self._merge_auxiliary_data(updates, key=KEY) is None:
            raise RuntimeError(
                f"could not write {self.auxiliary_file_path} (locked or unreadable); "
                "nothing was changed — run again"
            )
        self._write_labels()
        names = [c.name for c in self.job.columns]
        counts = {
            name: int(np.isfinite(updates[name].to_numpy(dtype=float)).sum())
            for name in names
        }
        detail = ", ".join(
            f"{name!r} ({counts[name]} of {len(updates)} shots)" for name in names
        )
        target = (
            Path(self.auxiliary_file_path).name
            if self.auxiliary_file_path
            else "the s-file"
        )
        return [f"wrote {len(names)} column(s) to {target}: {detail}"]

    def _write_labels(self) -> Optional[Path]:
        """The condition columns' labels, beside the sidecar (none: no file)."""
        conditions = {c.name: c.labels for c in self.job.columns if c.is_condition}
        sidecar = self.scalar_sidecar_path()
        if not conditions or sidecar is None:
            return None
        scan_name = sidecar.name.split("_", 1)[0]
        job_id = getattr(self, "id", None) or self.job.device
        return write_condition_labels(
            labels_path(sidecar.parent, scan_name, job_id), job_id, conditions
        )

    def cleanup(self) -> None:
        """Nothing is held between scans."""
