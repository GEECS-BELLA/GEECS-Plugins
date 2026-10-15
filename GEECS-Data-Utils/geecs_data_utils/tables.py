"""A small tab-delimited table in memory: named columns of strings, read-only.

The analysis core's table inputs (a spectrometer's calibration surveys) are
bound as these: the scan host reads the file
(:func:`geecs_data_utils.io.tables.read_table`), the core never does. Cells
stay the strings they were read as — a consumer converts the columns it
understands (:meth:`Table.floats`), as the legacy ``csv.DictReader`` readers
did — and a table pickles as plain tuples, so a process-pool worker gets a
copy at no cost. No I/O lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Table:
    """Named columns over rows of string cells, in file order."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        """Own the cells as tuples: distinct column names, one cell per column."""
        columns = tuple(str(name) for name in self.columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("A table needs distinct column names")
        rows = tuple(tuple(str(cell) for cell in row) for row in self.rows)
        if any(len(row) != len(columns) for row in rows):
            raise ValueError("Every table row needs one cell per column")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)

    def __len__(self) -> int:
        """The number of rows."""
        return len(self.rows)

    def strings(self, column: str) -> tuple[str, ...]:
        """One column's cells as read."""
        index = self._index(column)
        return tuple(row[index] for row in self.rows)

    def floats(self, column: str) -> NDArray[np.float64]:
        """One column as float64: ``float()`` of each cell, so a non-number raises."""
        return np.array(
            [float(cell) for cell in self.strings(column)], dtype=np.float64
        )

    def row(self, index: int) -> Mapping[str, str]:
        """One row as a read-only ``column -> cell`` mapping."""
        return MappingProxyType(dict(zip(self.columns, self.rows[index], strict=True)))

    def _index(self, column: str) -> int:
        if column not in self.columns:
            raise KeyError(
                f"Table has no column {column!r}; it has {list(self.columns)}"
            )
        return self.columns.index(column)


__all__ = ["Table"]
