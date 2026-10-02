"""Port the runs Tiled stores as SQL tables to Parquet files beside their s-files.

The history half of the #1020 storage arc.  Every run registered before
GeecsBluesky 0.110.0 has its stream tables in Tiled's SQL storage
(``tabular.db``, mimetype ``application/x-tiled-sql-table``); every run
since has them as ``ScanNNN/ScanDataScanNNN-<stream>.parquet`` in the scan
folder (:mod:`geecs_bluesky.tiled_parquet`).  ``geecs-tiled-port-tables``
makes the history look like the present, one table node at a time, through
Tiled's public API and nothing else:

1. read the node's table as the server serves it — the #1033 override
   types every column, and Tiled casts to the registered structure — as an
   Arrow table (``export`` to Arrow IPC, never pandas, so types and nulls
   are exactly the catalog's);
2. write it as the Parquet sibling of the s-file in the run's scan folder
   (:func:`~geecs_data_utils.data.sfile.stream_table_parquet_path_for`;
   beside its target and renamed; the folder must exist — **never
   created**);
3. detach the SQL data source from its storage (:func:`detach_sql_storage`:
   the same data-source update the writer uses, with no table parameters
   and external management — Tiled's node delete would otherwise delete
   the dataset's rows from ``tabular.db``), delete the table node, and
   register it again under the same key with a Parquet data source (the
   stream's metadata — the data keys — carried over), exactly what the
   writer does for a new run.  Tiled's delete keeps an asset other nodes
   still reference, so the shared ``tabular.db`` asset survives until
   nothing points at it;
4. read the new node back and compare every column with what step 1 saw.

A plain data-source update to Parquet was ruled out: Tiled 0.2.14 adds the
Parquet asset but keeps the SQLite one, and hands both to the Parquet
adapter, which refuses the second.

**Scan folders recorded from other hosts.**  A start document's
``scan_folder`` is the path the *engine* saw: ``/mnt/hdna2/data/…`` on the
worker, ``/Volumes/hdna2/data/…`` on a Mac, ``Z:/data/…`` on Windows.
``--alias SRC=DST`` maps each foreign root onto this host's mount before
the folder is looked for.  The Parquet's URI is the Tiled host's view
(:func:`~geecs_bluesky.data_paths.tiled_host_path`, the writer's rule).

**What stays.**  A run whose start document names no scan folder (the
pre-claim development runs of 2026-05..07) is skipped and reported; its
table stays in SQL storage, which the override keeps readable.  Nothing in
``tabular.db`` is ever modified or deleted by this command.

**Ledger and rollback.**  Every item appends one JSON line to the ledger
(``--ledger``): run, stream, table, rows, status, and the SQL data source
as it was.  ``--restore UID STREAM`` re-registers that SQL data source
from the ledger (external management, the same table and dataset id), the
rollback for one node; the Parquet file is left in place.

Idempotent: a node already on Parquet is skipped.  ``--dry-run`` plans and
reports without touching anything.  ``--limit N`` ports the first N
portable items — the rehearsal knob.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pyarrow
import pyarrow.ipc
from tiled.structures.core import StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management
from tiled.structures.table import TableStructure

from geecs_bluesky.data_paths import tiled_host_path
from geecs_bluesky.tiled_parquet import (
    PARQUET_MIMETYPE,
    file_uri,
    parquet_data_source,
    write_parquet_atomically,
)
from geecs_data_utils.data.sfile import stream_table_parquet_path_for

logger = logging.getLogger(__name__)

SQL_TABLE_MIMETYPE = "application/x-tiled-sql-table"
ARROW_MIMETYPE = "application/vnd.apache.arrow.file"
TABLE_KEY = "internal"

# ── planning ──────────────────────────────────────────────────────────────


def alias_scan_folder(folder: str, aliases: dict[str, str]) -> Path:
    r"""The engine's recorded scan folder as this host mounts it.

    *aliases* maps a recorded root (``Z:/data``, ``/Volumes/hdna2/data``)
    onto the local root; the longest matching root wins; separators are
    normalized, so ``Z:\\data`` and ``Z:/data`` are the same root.  No
    match means the path is already local.
    """
    posix = folder.replace("\\", "/")
    for src in sorted(aliases, key=len, reverse=True):
        root = src.replace("\\", "/").rstrip("/")
        if posix == root or posix.startswith(root + "/"):
            rest = posix[len(root) :].lstrip("/")
            return (
                Path(aliases[src]) / PurePosixPath(rest) if rest else Path(aliases[src])
            )
    if PureWindowsPath(folder).drive:  # an unaliased Windows path can never exist here
        return Path(posix)
    return Path(posix)


@dataclass
class PortItem:
    """One stream table and what the port decided about it."""

    run_uid: str
    stream: str
    scan_number: int | None
    recorded_folder: str | None
    scan_folder: str | None  # aliased, local
    table_name: str | None
    dataset_id: int | None
    status: str  # port | already-parquet | skip-no-folder | skip-missing-folder | other-mimetype
    parquet: str | None = None
    rows: int | None = None
    error: str | None = None
    old_data_source: dict[str, Any] | None = None

    @property
    def portable(self) -> bool:
        """Whether the port will act on this item."""
        return self.status == "port"


def _data_source_record(ds: Any) -> dict[str, Any]:
    """A JSON-able snapshot of a data source (the rollback needs it)."""
    return {
        "mimetype": ds.mimetype,
        "management": str(getattr(ds.management, "value", ds.management)),
        "parameters": dict(ds.parameters or {}),
        "structure": ds.structure
        if isinstance(ds.structure, dict)
        else asdict(ds.structure),
        "assets": [
            {
                "data_uri": a.data_uri,
                "is_directory": a.is_directory,
                "parameter": a.parameter,
                "num": a.num,
            }
            for a in ds.assets
        ],
    }


def plan_run(run: Any, aliases: dict[str, str]) -> list[PortItem]:
    """Every stream of *run* with a table node, classified."""
    start = run.metadata.get("start", {})
    recorded = start.get("scan_folder")
    scan_number = start.get("scan_number")
    items: list[PortItem] = []
    for stream, node in run.items():
        base = getattr(node, "base", node)
        try:
            if TABLE_KEY not in base:
                continue
            table = base[TABLE_KEY]
            sources = table.data_sources()
        except Exception as exc:  # noqa: BLE001 - a stream node without a table part
            logger.debug("%s/%s: no table (%s)", run.item["id"], stream, exc)
            continue
        if not sources:
            continue
        ds = sources[0]
        params = dict(ds.parameters or {})
        item = PortItem(
            run_uid=run.item["id"],
            stream=stream,
            scan_number=scan_number,
            recorded_folder=recorded,
            scan_folder=None,
            table_name=params.get("table_name"),
            dataset_id=params.get("dataset_id"),
            status="port",
            old_data_source=_data_source_record(ds),
        )
        if ds.mimetype == PARQUET_MIMETYPE:
            item.status = "already-parquet"
        elif ds.mimetype != SQL_TABLE_MIMETYPE:
            item.status = "other-mimetype"
        elif not recorded:
            item.status = "skip-no-folder"
        else:
            folder = alias_scan_folder(recorded, aliases)
            item.scan_folder = str(folder)
            if not folder.is_dir():
                item.status = "skip-missing-folder"
            else:
                item.parquet = str(stream_table_parquet_path_for(folder, stream))
        items.append(item)
    return items


def plan(
    client: Any, aliases: dict[str, str], *, runs: Iterable[str] | None = None
) -> list[PortItem]:
    """Classify every table node of every run (or of *runs*, by uid)."""
    client = client.include_data_sources()
    items: list[PortItem] = []
    uids = list(runs) if runs is not None else list(client.keys())
    for uid in uids:
        run = client[uid]
        if "start" not in run.metadata:
            continue  # not a Bluesky run
        items.extend(plan_run(run, aliases))
    return items


# ── the port of one table node ────────────────────────────────────────────


def read_arrow(table_node: Any) -> pyarrow.Table:
    """The node's table as the server serves it, as Arrow — the registered columns only.

    The server serializes through pandas, which can add its index as a
    reserved ``__index_level_0__`` column and pandas metadata to the schema;
    both are dropped, since Tiled refuses reserved names in a structure and
    the Parquet should carry the catalog's columns and nothing else.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "table.arrow"
        table_node.export(str(path), format=ARROW_MIMETYPE)
        with pyarrow.ipc.open_file(path) as reader:
            table = reader.read_all()
    keep = [name for name in table.column_names if not name.startswith("_")]
    return table.select(keep).replace_schema_metadata(None)


def tables_equal(a: pyarrow.Table, b: pyarrow.Table) -> bool:
    """Same columns, same rows, NaN equal to NaN."""
    if a.schema.names != b.schema.names or a.num_rows != b.num_rows:
        return False
    for name in a.schema.names:
        ca, cb = a[name].combine_chunks(), b[name].combine_chunks()
        if pyarrow.types.is_floating(ca.type) and pyarrow.types.is_floating(cb.type):
            if not np.array_equal(
                ca.to_numpy(zero_copy_only=False),
                cb.to_numpy(zero_copy_only=False),
                equal_nan=True,
            ):
                return False
        elif ca.to_pylist() != cb.to_pylist():
            return False
    return True


def port_item(
    client: Any,
    item: PortItem,
    *,
    tiled_path: Callable[[str], str] = tiled_host_path,
    dry_run: bool = False,
) -> PortItem:
    """Port one table node; *item* comes back with ``status`` ``ported`` or ``failed``."""
    if not item.portable:
        return item
    run = client[item.run_uid]
    base = getattr(run[item.stream], "base", run[item.stream])
    table = base[TABLE_KEY]
    metadata = dict(table.metadata)
    arrow = read_arrow(table)
    item.rows = arrow.num_rows
    parquet = Path(item.parquet)
    uri = file_uri(tiled_path(str(parquet)))  # refused before anything is written
    if dry_run:
        item.status = "would-port"
        return item
    write_parquet_atomically(arrow, parquet)
    data_source = parquet_data_source(arrow, uri)
    # Tiled deletes a *writable* data source's rows from the storage database
    # when its node goes — the SQL table is the rollback, so first make the
    # catalog treat it as external (not Tiled's to delete), then drop only
    # the catalog record.
    detach_sql_storage(table)
    table.delete()
    try:
        base.new(StructureFamily.table, [data_source], key=TABLE_KEY, metadata=metadata)
    except Exception as exc:  # noqa: BLE001 - put the SQL registration back
        item.error = f"re-registration failed: {exc!r}; SQL data source restored"
        restore_sql_table(base, item.old_data_source, metadata)
        item.status = "failed"
        return item
    back = read_arrow(base[TABLE_KEY])
    if not tables_equal(arrow, back):
        item.error = "read-back differs from the SQL table; SQL data source restored"
        base[TABLE_KEY].delete(external_only=False)
        restore_sql_table(base, item.old_data_source, metadata)
        item.status = "failed"
        return item
    item.status = "ported"
    return item


def detach_sql_storage(table_node: Any) -> None:
    """Make deleting the node leave its SQL rows alone.

    Tiled's node delete removes a SQL-backed data source's rows from the
    storage database (``DELETE … WHERE _dataset_id``, then ``DROP TABLE``
    once empty) **whatever its management**, keyed on the data source's own
    ``table_name`` / ``dataset_id`` parameters — and skips that step when
    they are absent.  So, through the same ``PUT /data_source`` the writer
    uses, the data source is rewritten with **no parameters** and external
    management; the node is unreadable for the milliseconds until it is
    deleted, and the SQL table — the rollback — is never touched.  Found by
    the opt-in test (2026-10-02): the first version deleted the node
    directly and lost the table.
    """
    from tiled.client.utils import handle_error
    from tiled.utils import safe_json_dump

    (ds,) = table_node.data_sources()
    detached = DataSource(
        id=ds.id,
        structure_family=ds.structure_family,
        mimetype=ds.mimetype,
        structure=TableStructure.from_json(ds.structure)
        if isinstance(ds.structure, dict)
        else ds.structure,
        parameters={},
        management=Management.external,
        assets=list(ds.assets),
    )
    handle_error(
        table_node.context.http_client.put(
            table_node.uri.replace("/metadata/", "/data_source/", 1),
            content=safe_json_dump({"data_source": detached}),
        )
    )


def restore_sql_table(base: Any, old: dict[str, Any], metadata: dict[str, Any]) -> Any:
    """Register the SQL table again (external management: the same table and dataset id)."""
    (asset,) = [a for a in old["assets"] if a.get("parameter") == "data_uri"] or old[
        "assets"
    ][:1]
    data_source = DataSource(
        structure_family=StructureFamily.table,
        mimetype=SQL_TABLE_MIMETYPE,
        structure=TableStructure.from_json(old["structure"]),
        parameters=dict(old["parameters"]),
        management=Management.external,
        assets=[
            Asset(
                data_uri=asset["data_uri"],
                is_directory=False,
                parameter="data_uri",
                num=None,
            )
        ],
    )
    return base.new(
        StructureFamily.table, [data_source], key=TABLE_KEY, metadata=metadata
    )


# ── the command ───────────────────────────────────────────────────────────


@dataclass
class Ledger:
    """Append-only JSON lines, one record per item."""

    path: Path
    records: list[dict[str, Any]] = field(default_factory=list)

    def append(self, item: PortItem) -> None:
        """Append one item's record (plus the time) and keep it in memory."""
        record = asdict(item)
        record["at"] = time.time()
        self.records.append(record)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    @classmethod
    def read(cls, path: Path) -> Iterator[dict[str, Any]]:
        """Every record of a ledger file, in order."""
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def parse_aliases(values: Iterable[str]) -> dict[str, str]:
    """``SRC=DST`` pairs."""
    aliases: dict[str, str] = {}
    for value in values:
        src, sep, dst = value.partition("=")
        if not sep or not src or not dst:
            raise argparse.ArgumentTypeError(f"--alias expects SRC=DST, got {value!r}")
        aliases[src] = dst
    return aliases


def summarize(items: list[PortItem]) -> dict[str, int]:
    """Items per status."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    """The ``geecs-tiled-port-tables`` command line."""
    parser = argparse.ArgumentParser(
        prog="geecs-tiled-port-tables",
        description="Port the runs Tiled stores as SQL tables to Parquet files beside their s-files.",
    )
    parser.add_argument(
        "--tiled-uri", default=None, help="the catalog (default: config.ini [tiled])"
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="SRC=DST",
        help="a recorded scan-folder root and this host's mount of it (repeatable)",
    )
    parser.add_argument("--ledger", type=Path, default=Path("tiled-port-ledger.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=None, help="port at most N portable items"
    )
    parser.add_argument(
        "--run", action="append", default=None, metavar="UID", help="only these runs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="plan and report, touch nothing"
    )
    parser.add_argument(
        "--restore",
        nargs=2,
        metavar=("UID", "STREAM"),
        help="re-register one node's SQL data source from the ledger (the rollback)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    tiled_uri, api_key = args.tiled_uri, None
    if tiled_uri is None:
        from geecs_bluesky.tiled_integration import read_tiled_config

        tiled_uri, api_key = read_tiled_config()
    if not tiled_uri:
        logger.error(
            "no Tiled URI: pass --tiled-uri or configure [tiled] uri in config.ini"
        )
        return 2
    from tiled.client import from_uri

    client = from_uri(tiled_uri, api_key=api_key).include_data_sources()

    if args.restore:
        uid, stream = args.restore
        records = [
            r
            for r in Ledger.read(args.ledger)
            if r["run_uid"] == uid
            and r["stream"] == stream
            and r.get("old_data_source")
        ]
        if not records:
            logger.error("no ledger record for %s/%s in %s", uid, stream, args.ledger)
            return 2
        base = getattr(client[uid][stream], "base", client[uid][stream])
        metadata = dict(base[TABLE_KEY].metadata) if TABLE_KEY in base else {}
        if TABLE_KEY in base:
            base[TABLE_KEY].delete(external_only=False)
        restore_sql_table(base, records[-1]["old_data_source"], metadata)
        logger.info(
            "%s/%s: SQL data source restored (the Parquet file is left in place)",
            uid,
            stream,
        )
        return 0

    aliases = parse_aliases(args.alias)
    items = plan(client, aliases, runs=args.run)
    counts = summarize(items)
    logger.info("plan: %s", counts)
    portable = [i for i in items if i.portable]
    if args.limit is not None:
        portable = portable[: args.limit]
    ledger = Ledger(args.ledger)
    failed = 0
    for n, item in enumerate(portable, 1):
        label = (
            f"Scan{item.scan_number:03d}"
            if item.scan_number is not None
            else item.run_uid[:8]
        )
        t0 = time.time()
        try:
            port_item(client, item, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 - one item never stops the port
            item.status, item.error = "failed", repr(exc)
        if item.status == "failed":
            failed += 1
            logger.error(
                "[%d/%d] %s/%s: FAILED — %s",
                n,
                len(portable),
                label,
                item.stream,
                item.error,
            )
        else:
            logger.info(
                "[%d/%d] %s/%s: %s, %s rows → %s (%.1f s)",
                n,
                len(portable),
                label,
                item.stream,
                item.status,
                item.rows,
                item.parquet,
                time.time() - t0,
            )
        ledger.append(item)
    for item in items:
        if not item.portable:
            ledger.append(item)
    logger.info("done: %s; %d failed", summarize(items), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
