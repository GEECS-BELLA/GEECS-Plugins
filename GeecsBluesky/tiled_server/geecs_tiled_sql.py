"""GEECS overrides of Tiled's ``SQLAdapter`` for the catalog's tabular storage.

Deploy material for the **Tiled server**, not part of the ``geecs_bluesky``
package: the server runs in its own environment (``TILED_SETUP.md``), so
this file imports ``tiled`` and ``pyarrow`` only and is installed by
copying it beside the server's ``config.yml``.  Wire it in through the
catalog tree's ``adapters_by_mimetype`` and put its directory on the
server's ``PYTHONPATH`` (Tiled resolves the import outside the window in
which it prepends the config directory to ``sys.path``)::

    trees:
      - path: /
        tree: catalog
        args:
          ...
          adapters_by_mimetype:
            application/x-tiled-sql-table: geecs_tiled_sql:GeecsSQLAdapter

Two fixes, each scoped to the backend that needs it.

**SQLite — typed reads (GEECS-Plugins#1020).**  The ADBC SQLite driver
infers each result column's Arrow type from the rows of its *first batch*
(1024 by default) and ignores the declared column type.  SQLite stores a
float NaN as NULL, so a REAL column that is all-NaN for a scan's first
1024 shots is typed INT64, and the first real value after that fails the
whole read::

    OSError: [SQLite] Type mismatch in column N: expected INT64 but got DOUBLE

Setting the statement option ``adbc.sqlite.query.batch_rows`` to the
dataset's row count puts every row in the inference window: a column with
any value is typed from it, and an all-NULL column (still INT64) is cast
to its declared type by Tiled's own ``data.cast(target_schema)``.  No
measurable cost: the result was one Arrow table either way.

**PostgreSQL — NULL array elements.**  The ADBC PostgreSQL driver (1.11,
1.12) writes a NULL *element* of an array column as ``0.0`` — a plausible
reading, silently wrong.  Tiled's server turns every NaN into a null on the
way in (``deserialize_arrow`` reads the upload through pandas), so a NaN
telemetry sample inside a vector column would come back as zero.  Filling
null elements with NaN just before the ingest restores what was written;
NaN itself, ``inf`` and a NULL *whole* array all survive the driver.

Verified against Tiled 0.2.14 with adbc-driver-sqlite / -postgresql 1.11
and 1.12; the hooks used (``SQLAdapter.storage``, ``.dialect``,
``.connect()``, ``_read_full_table_or_partition``, ``append_partition``)
are the 0.2.x adapter's own.
"""

from __future__ import annotations

import math
from contextlib import closing
from typing import Any, List, Optional, Union

import pandas
import pyarrow
import pyarrow.compute
from tiled.adapters.sql import SQLAdapter

BATCH_ROWS_OPTION = "adbc.sqlite.query.batch_rows"


class _OneBatchConnection:
    """A connection proxy whose cursors read the whole result as one batch."""

    def __init__(self, conn: Any, rows: int) -> None:
        self._conn = conn
        self._rows = max(int(rows), 1)

    def cursor(self) -> Any:
        cur = self._conn.cursor()
        cur.adbc_statement.set_options(**{BATCH_ROWS_OPTION: self._rows})
        return cur

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class _OneBatchStorage:
    """Storage proxy: the same dialect, every connection wrapped for one batch."""

    def __init__(self, storage: Any, rows: int) -> None:
        self._storage = storage
        self._rows = rows

    @property
    def dialect(self) -> str:
        return self._storage.dialect

    def connect(self) -> _OneBatchConnection:
        return _OneBatchConnection(self._storage.connect(), self._rows)


def _is_float_list(arrow_type: pyarrow.DataType) -> bool:
    return (
        pyarrow.types.is_list(arrow_type) or pyarrow.types.is_large_list(arrow_type)
    ) and pyarrow.types.is_floating(arrow_type.value_type)


def fill_null_list_elements(table: pyarrow.Table) -> pyarrow.Table:
    """Replace null *elements* of every floating-point list column with NaN.

    A null list (the whole vector missing for a row) is kept null.  Columns
    of any other type are returned untouched.
    """
    for i, field in enumerate(table.schema):
        if not _is_float_list(field.type):
            continue
        arr = table.column(i).combine_chunks()
        if arr.values.null_count == 0:
            continue
        filled = pyarrow.compute.fill_null(arr.values, math.nan)
        rebuilt = type(arr).from_arrays(arr.offsets, filled, mask=arr.is_null())
        table = table.set_column(i, field, rebuilt)
    return table


class GeecsSQLAdapter(SQLAdapter):
    """``SQLAdapter`` with typed SQLite reads and NaN-safe PostgreSQL arrays."""

    def _count_rows(self, partition: Optional[int]) -> int:
        query = (
            f'SELECT count(*) FROM "{self.table_name}" '
            f"WHERE _dataset_id={self.dataset_id}"
        )
        if partition is not None:
            query += f" AND _partition_id={int(partition)}"
        with closing(self.storage.connect()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                (n,) = cursor.fetchone()
            conn.commit()
        return int(n)

    def _read_full_table_or_partition(
        self, fields: Optional[List[str]] = None, partition: Optional[int] = None
    ) -> pyarrow.Table:
        if self.storage.dialect != "sqlite":
            return super()._read_full_table_or_partition(
                fields=fields, partition=partition
            )
        real_storage = self.storage
        self.storage = _OneBatchStorage(real_storage, self._count_rows(partition))
        try:
            return super()._read_full_table_or_partition(
                fields=fields, partition=partition
            )
        finally:
            self.storage = real_storage

    def append_partition(
        self,
        partition: int,
        data: Union[
            List[pyarrow.RecordBatch],
            pyarrow.RecordBatch,
            pandas.DataFrame,
            pyarrow.Table,
        ],
    ) -> None:
        """Append *data* to *partition*; on PostgreSQL, NaN-fill null array elements first.

        The ADBC PostgreSQL driver writes a NULL element of an array column
        as ``0.0``; the parent's ingest is otherwise unchanged.
        """
        if self.storage.dialect == "postgresql":
            if isinstance(data, pandas.DataFrame):
                data = pyarrow.Table.from_pandas(data)
            elif isinstance(data, pyarrow.RecordBatch):
                data = pyarrow.Table.from_batches([data])
            elif isinstance(data, list):
                data = pyarrow.Table.from_batches(data)
            data = fill_null_list_elements(data)
        super().append_partition(partition, data)
