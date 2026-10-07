"""The stream table as a Parquet file beside the s-file.

GEECS registers every run from the spool after its stop, when the table
is known in full, so instead of the stock writer's appendable SQL table
each event stream is written as **one Parquet file in the scan folder** —
``ScanNNN/ScanDataScanNNN-<stream>.parquet``, the s-file's sibling
(:func:`geecs_data_utils.data.sfile.stream_table_parquet_path_for`) — and
registered from ``readable_storage`` exactly as the camera stacks are
(``application/x-parquet``).  The scan folder is the record; Tiled is the
index; a reader (``read_primary_scalars``) cannot tell which store served
the table.  :func:`geecs_bluesky.tiled.writer.make_tiled_writer` takes
``tables="parquet"`` (default) or ``"appendable"`` (the stock path).

Two rules:

- **The writer never creates a scan folder.**  The folder was claimed by
  the engine at ``open_run``; a missing one is an anomaly surfaced as a
  registration failure (the registrar's backoff), never papered over with
  ``mkdir``.
- **The URI is the Tiled host's view**
  (:func:`geecs_bluesky.data_paths.tiled_host_path`, the camera stacks'
  ``plugin_save_path`` pattern): ``config.ini``
  ``[Paths] geecs_tiled_host_data_base_path`` names the share as the Tiled
  host mounts it, and a path that cannot be translated is refused rather
  than registered wrong.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import pyarrow
import pyarrow.parquet
from bluesky.callbacks.tiled_writer import BATCH_SIZE, TiledWriter, _RunWriter
from bluesky.utils import truncate_json_overflow
from tiled.structures.core import StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management
from tiled.structures.table import TableStructure

from geecs_bluesky.data_paths import tiled_host_path
from geecs_bluesky.tiled.writer import DEFAULT_TABLE_STORE, TABLE_STORES
from geecs_data_utils.data.sfile import stream_table_parquet_path_for

logger = logging.getLogger(__name__)

PARQUET_MIMETYPE = "application/x-parquet"

# ── pure helpers ──────────────────────────────────────────────────────────


def file_uri(posix_path: str) -> str:
    """The ``file://localhost/...`` form Tiled stores for an asset."""
    return f"file://localhost{posix_path}"


def table_from_rows(rows: list[dict[str, Any]]) -> pyarrow.Table:
    """The event rows as an Arrow table, an all-null column typed string (the stock rule)."""
    table = pyarrow.Table.from_pylist(rows)
    schema = table.schema
    for i, field in enumerate(table.schema):
        if pyarrow.types.is_null(field.type):
            schema = schema.set(i, field.with_type(pyarrow.string()))
        elif pyarrow.types.is_list(field.type) and pyarrow.types.is_null(
            field.type.value_type
        ):
            schema = schema.set(i, field.with_type(pyarrow.list_(pyarrow.string())))
    return table.cast(schema) if schema != table.schema else table


def write_parquet_atomically(table: pyarrow.Table, path: Path) -> None:
    """Write beside the target and rename, so a reader never sees a partial file.

    Raises ``FileNotFoundError`` when the scan folder is not there: the
    writer never creates one (the repository's scan-folder invariant).
    """
    folder = path.parent
    if not folder.is_dir():
        raise FileNotFoundError(
            f"scan folder {folder} does not exist — the Tiled writer never creates one"
        )
    tmp = path.with_name(path.name + ".tmp")
    pyarrow.parquet.write_table(table, tmp)
    os.replace(tmp, path)


def parquet_data_source(table: pyarrow.Table, uri: str) -> DataSource:
    """The catalog's record of one Parquet file: Tiled's stock table adapter, external management."""
    return DataSource(
        structure_family=StructureFamily.table,
        mimetype=PARQUET_MIMETYPE,
        structure=TableStructure.from_arrow_table(table, npartitions=1),
        management=Management.external,
        assets=[Asset(data_uri=uri, is_directory=False, parameter="data_uris", num=0)],
    )


# ── the writer ────────────────────────────────────────────────────────────


class GeecsRunWriter(_RunWriter):
    """The stock per-run writer with the stream table handed over as Parquet.

    Everything but the table — the run container, the stream containers,
    the external camera datasets, the stop — is the stock writer's.  The
    table hand-over (``_write_internal_data``) writes the stream's rows to
    the scan folder and registers the file; a second hand-over for the same
    stream (a run longer than the batch size) rewrites the file with every
    row so far and updates the registration, so the record is always one
    file.

    Parameters
    ----------
    table_store :
        ``"parquet"`` (this hand-over) or ``"appendable"`` (the stock one).
    tiled_path :
        ``local path -> the Tiled host's path`` for the file's URI;
        :func:`~geecs_bluesky.data_paths.tiled_host_path` by default.  Tests
        inject a mapping.
    """

    def __init__(
        self,
        client: Any,
        batch_size: int = BATCH_SIZE,
        *,
        table_store: str = DEFAULT_TABLE_STORE,
        tiled_path: Callable[[str], str] = tiled_host_path,
    ) -> None:
        if table_store not in TABLE_STORES:
            raise ValueError(
                f"table_store must be one of {TABLE_STORES}, got {table_store!r}"
            )
        super().__init__(client, batch_size=batch_size)
        self.table_store = table_store
        self._tiled_path = tiled_path
        self._scan_folder: Path | None = None
        self._parquet_rows: dict[str, list[dict[str, Any]]] = {}
        self._parquet_nodes: dict[str, Any] = {}

    def start(self, doc: Any) -> None:
        """Remember the claimed scan folder, then the stock start."""
        folder = doc.get("scan_folder")
        self._scan_folder = Path(folder) if folder else None
        super().start(doc)

    def _write_internal_data(
        self, data_cache: list[dict[str, Any]], desc_node: Any
    ) -> None:
        if self.table_store != "parquet":
            return super()._write_internal_data(data_cache, desc_node)
        stream = desc_node.item["id"]
        if self._scan_folder is None:
            logger.warning(
                "stream %s: no scan_folder in the start document — its table goes to "
                "Tiled's appendable store",
                stream,
            )
            return super()._write_internal_data(data_cache, desc_node)

        rows = self._parquet_rows.setdefault(stream, [])
        rows.extend(data_cache)
        table = table_from_rows(rows)
        path = stream_table_parquet_path_for(self._scan_folder, stream)
        uri = file_uri(
            self._tiled_path(str(path))
        )  # refused before anything is written
        write_parquet_atomically(table, path)
        data_source = parquet_data_source(table, uri)

        node = self._parquet_nodes.get(stream)
        if node is None:
            metadata = truncate_json_overflow(
                {k: v for k, v in self.data_keys.items() if k in table.column_names}
            )
            node = desc_node.new(
                StructureFamily.table,
                [data_source],
                key="internal",
                metadata=metadata,
                access_tags=self.access_tags,
            )
            self._parquet_nodes[stream] = node
        else:
            self._update_data_source_for_node(node, data_source)
        logger.debug("stream %s: %d rows → %s", stream, table.num_rows, path.name)


class GeecsTiledWriter(TiledWriter):
    """The stock ``TiledWriter`` building :class:`GeecsRunWriter` per run.

    ``_factory`` mirrors the stock one — the normalizer in front of the
    run writer, the optional JSON-lines backup around both — with the one
    substitution.
    """

    def __init__(
        self,
        client: Any,
        *,
        table_store: str = DEFAULT_TABLE_STORE,
        tiled_path: Callable[[str], str] = tiled_host_path,
        **kwargs: Any,
    ) -> None:
        super().__init__(client, **kwargs)
        self.table_store = table_store
        self._tiled_path = tiled_path

    def _factory(self, name: str, doc: dict) -> tuple[list, list]:
        cb = run_writer = GeecsRunWriter(
            self.client,
            batch_size=self._batch_size,
            table_store=self.table_store,
            tiled_path=self._tiled_path,
        )
        if self._normalizer:
            cb = self._normalizer(
                patches=self.patches, spec_to_mimetype=self.spec_to_mimetype
            )
            cb.subscribe(run_writer)
        if self.backup_directory:  # pragma: no cover - never configured here
            from bluesky.callbacks.tiled_writer import (
                JSONLinesWriter,
                _ConditionalBackup,
            )

            cb = _ConditionalBackup(cb, [JSONLinesWriter(self.backup_directory)])
        return [cb], []
