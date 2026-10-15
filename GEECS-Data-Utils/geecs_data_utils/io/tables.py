"""Read a tab-delimited text table with a header row into a :class:`Table`."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

from geecs_data_utils.tables import Table


def read_table(path: Union[str, Path]) -> Table:
    """Read a tab-delimited table: the first row names the columns, the rest are cells.

    Every cell is kept as the string it was read as and blank lines are
    skipped — the rows ``csv.DictReader`` yields, which the legacy
    spectrometer calibration readers used. A row with another number of
    cells than the header is an error, never padded or truncated.
    """
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Table {path} is empty")
        rows = [tuple(row) for row in reader if row]
    for number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(
                f"Table {path} line {number}: {len(row)} cells for "
                f"{len(header)} columns"
            )
    return Table(tuple(header), tuple(rows))


__all__ = ["read_table"]
