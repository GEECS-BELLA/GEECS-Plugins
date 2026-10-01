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
whole read (a TEXT column NULL for those rows fails the same way)::

    OSError: [SQLite] Type mismatch in column N: expected INT64 but got DOUBLE

Setting the statement option ``adbc.sqlite.query.batch_rows`` above any
dataset's row count makes the whole result one batch, so every column is
typed from every row, and an all-NULL column (still INT64) is cast to its
declared type by Tiled's own ``data.cast(target_schema)``.  One query, no
row count first — the driver appends rows as they come and reserves
nothing per batch (measured: a batch size of 10^9 costs no memory or
time), and the option must fit a C ``int`` (``INT_MAX`` is rejected).
The option is set on every cursor of a SQLite storage by a wrapper
installed **once, at construction**: the adapter's state never changes
during a request, so concurrent reads through one adapter instance — if
a Tiled version ever shared instances — cannot disturb each other.

**PostgreSQL — NULL array elements.**  The ADBC PostgreSQL driver (1.11,
1.12) writes a NULL *element* of an array column as ``0.0`` — a plausible
reading, silently wrong.  Tiled's server turns every NaN into a null on the
way in (``deserialize_arrow`` reads the upload through pandas), so a NaN
telemetry sample inside a vector column would come back as zero.  Filling
null elements with NaN just before the ingest restores what was written;
NaN itself, ``inf`` and a NULL *whole* array all survive the driver.

Verified against Tiled 0.2.14 and 0.2.18 with adbc-driver-sqlite /
-postgresql 1.11 and 1.12; the hooks used (``SQLAdapter.storage`` as the
one source of connections, ``.dialect``, ``.connect()``,
``append_partition``) are the 0.2.x adapter's own.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Union

import pandas
import pyarrow
import pyarrow.compute
from tiled.adapters.sql import SQLAdapter
from tiled.structures.core import Spec
from tiled.structures.table import TableStructure
from tiled.type_aliases import JSON

BATCH_ROWS_OPTION = "adbc.sqlite.query.batch_rows"
#: Larger than any dataset (a GEECS run is thousands of rows), within the
#: driver's accepted range (a C ``int``; ``INT_MAX`` itself is refused).
ONE_BATCH_ROWS = 10**9


class _OneBatchConnection:
    """A connection proxy whose cursors read the whole result as one batch."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def cursor(self) -> Any:
        cur = self._conn.cursor()
        cur.adbc_statement.set_options(**{BATCH_ROWS_OPTION: ONE_BATCH_ROWS})
        return cur

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class _OneBatchStorage:
    """A SQLite storage whose every connection is wrapped for one-batch reads.

    Immutable once built; everything but ``connect`` is the wrapped
    storage's own (``dialect``, ``uri``, ``dispose``, ...).
    """

    def __init__(self, storage: Any) -> None:
        self._storage = storage

    def connect(self) -> _OneBatchConnection:
        return _OneBatchConnection(self._storage.connect())

    def __getattr__(self, name: str) -> Any:
        # Only public attributes delegate: ``copy``/``pickle`` probe dunder
        # and private names on a bare instance, and an unguarded delegator
        # would recurse into itself looking for ``_storage``.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._storage, name)


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

    def __init__(
        self,
        data_uri: str,
        structure: TableStructure,
        table_name: str,
        dataset_id: int,
        *,
        metadata: Optional[JSON] = None,
        specs: Optional[List[Spec]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_uri,
            structure,
            table_name,
            dataset_id,
            metadata=metadata,
            specs=specs,
            **kwargs,
        )
        if self.storage.dialect == "sqlite":
            self.storage = _OneBatchStorage(self.storage)

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
