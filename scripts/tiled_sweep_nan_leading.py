"""Sweep Tiled's SQLite tabular storage for datasets the stock reader cannot serve.

GEECS-Plugins#1020: the ADBC SQLite driver types each result column from
its first batch of rows (1024) and ignores the declared type, so a REAL or
TEXT column that is NULL for a dataset's first 1024 rows and has a value
later fails the whole read (``Type mismatch in column N: expected INT64
but got DOUBLE`` / ``STRING``).  This lists every such dataset, with the
run it belongs to.  INTEGER columns are typed INT64 either way and are not
checked.
The fix is the server-side adapter override in
``GeecsBluesky/tiled_server/geecs_tiled_sql.py`` (``TILED_SETUP.md``); this
sweep is the monitor that says whether any run depends on it.

Read-only: both databases are opened in ``mode=ro``.  Run it on the Tiled
host, where the files are (about 12 s for 220 tables)::

    ssh <tiled-host> 'cd ~/tiled && nice -n 15 python3 -' < scripts/tiled_sweep_nan_leading.py
    ssh <tiled-host> 'cd ~/tiled && python3 - --tabular tabular.db.bak-20261001-scan032' < scripts/tiled_sweep_nan_leading.py

Exit status is the number of affected datasets (capped at 125), so a cron
or a runbook step can test it.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from typing import Dict, List, Tuple

DEFAULT_BATCH = 1024
# Stay under SQLite's expression limits when a table has ~1300 columns.
COLUMNS_PER_QUERY = 400


def catalogued_datasets(catalog: sqlite3.Connection) -> Dict[Tuple[str, int], tuple]:
    """Map (table_name, dataset_id) → (run uid, stream path, scan number, start time)."""
    where: Dict[Tuple[str, int], tuple] = {}
    for node_id, params in catalog.execute(
        "select node_id, parameters from data_sources where mimetype like '%sql%'"
    ):
        p = json.loads(params or "{}")
        if "table_name" not in p:
            continue
        chain = catalog.execute(
            "select n.key, n.metadata from nodes_closure cl join nodes n "
            "on n.id = cl.ancestor where cl.descendant = ? order by cl.depth desc",
            (node_id,),
        ).fetchall()
        keys = [k for k, _ in chain if k]
        start: dict = {}
        for _, md in chain:
            md = json.loads(md or "{}")
            if "start" in md:
                start = md["start"]
                break
        where[(p["table_name"], p["dataset_id"])] = (
            keys[0] if keys else "?",
            "/".join(keys[1:]),
            start.get("scan_number"),
            start.get("time"),
        )
    return where


def affected_datasets(
    tabular: sqlite3.Connection, batch: int
) -> Tuple[List[str], List[tuple]]:
    """Return (table names, hits): hits = (table, dataset_id, rows, [(column, non-null count)])."""
    tables = [
        r[0]
        for r in tabular.execute(
            "select name from sqlite_master where type='table' and name like 'table_%'"
        )
    ]
    hits = []
    for tn in tables:
        cols = [
            r[1]
            for r in tabular.execute(f'pragma table_info("{tn}")')
            if r[2].upper() in ("REAL", "TEXT") and r[1] != "time"
        ]
        for i in range(0, len(cols), COLUMNS_PER_QUERY):
            chunk = cols[i : i + COLUMNS_PER_QUERY]
            sel = ", ".join(
                f'sum("{c}" is not null and rn <= {batch}), count("{c}")' for c in chunk
            )
            q = (
                f"select _dataset_id, count(*), {sel} from (select *, row_number() over "
                f'(partition by _dataset_id order by rowid) as rn from "{tn}") '
                "group by _dataset_id"
            )
            for row in tabular.execute(q):
                ds, n, vals = row[0], row[1], row[2:]
                if n <= batch:
                    continue
                bad = [
                    (c, vals[2 * j + 1])
                    for j, c in enumerate(chunk)
                    if vals[2 * j] == 0 and vals[2 * j + 1] > 0
                ]
                if bad:
                    hits.append((tn, ds, n, bad))
    return tables, hits


def main(argv: List[str] | None = None) -> int:
    """Run the sweep; return the number of affected datasets (capped at 125)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--catalog", default="catalog.db", help="Tiled catalog (default: catalog.db)"
    )
    ap.add_argument(
        "--tabular",
        default="tabular.db",
        help="SQLite tabular storage (default: tabular.db)",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"ADBC inference window in rows (default: {DEFAULT_BATCH})",
    )
    args = ap.parse_args(argv)

    catalog = sqlite3.connect(f"file:{args.catalog}?mode=ro", uri=True)
    tabular = sqlite3.connect(f"file:{args.tabular}?mode=ro", uri=True)
    where = catalogued_datasets(catalog)
    tables, hits = affected_datasets(tabular, args.batch)

    print(
        f"scanned {len(tables)} tables, {len(where)} catalogued datasets ({args.tabular})"
    )
    for tn, ds, n, bad in hits:
        uid, path, scan, t = where.get((tn, ds), ("UNCATALOGUED", "", None, None))
        print(f"\n{uid}  {path}  scan={scan}  t={t}  rows={n}  ({tn} ds={ds})")
        for c, k in bad:
            print(f"    {c}: {k} non-null")
    print(f"\n{len(hits)} affected dataset(s)")
    return min(len(hits), 125)


if __name__ == "__main__":
    sys.exit(main())
