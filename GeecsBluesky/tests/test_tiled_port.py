"""``geecs_bluesky.tiled_port`` — the history port of SQL-stored stream tables to Parquet.

The pure parts (aliasing, the plan's classification, table equality, the
ledger) are hermetic.  The port itself needs a Tiled with SQL storage and
is tested against a real in-process server (opt-in, as
``test_tiled_writer_catalog.py``): a run written by the **stock** writer
into an appendable SQL table, ported, read back, rolled back.
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow
import pytest

pytest.importorskip("tiled.client")
tiled_port = pytest.importorskip("geecs_bluesky.tiled_port")
from geecs_bluesky.tiled_port import (  # noqa: E402
    Ledger,
    PortItem,
    alias_scan_folder,
    parse_aliases,
    summarize,
    tables_equal,
)


class TestAliasing:
    def test_foreign_roots_map_onto_the_local_mount(self, tmp_path):
        aliases = {
            "Z:/data": str(tmp_path / "data"),
            "/Volumes/hdna2/data": str(tmp_path / "data"),
        }
        rel = Path("Undulator") / "Y2026" / "08-Aug" / "26_0820" / "scans" / "Scan003"
        assert (
            alias_scan_folder(
                "Z:/data/Undulator/Y2026/08-Aug/26_0820/scans/Scan003", aliases
            )
            == tmp_path / "data" / rel
        )
        assert (
            alias_scan_folder(
                "Z:\\data\\Undulator\\Y2026\\08-Aug\\26_0820\\scans\\Scan003", aliases
            )
            == tmp_path / "data" / rel
        )
        assert (
            alias_scan_folder(
                "/Volumes/hdna2/data/Undulator/Y2026/08-Aug/26_0820/scans/Scan003",
                aliases,
            )
            == tmp_path / "data" / rel
        )

    def test_a_local_path_passes_through_and_the_longest_root_wins(self, tmp_path):
        assert alias_scan_folder("/mnt/hdna2/data/x/scans/Scan001", {}) == Path(
            "/mnt/hdna2/data/x/scans/Scan001"
        )
        aliases = {"/a": "/short", "/a/data": "/long"}
        assert alias_scan_folder("/a/data/scans/Scan001", aliases) == Path(
            "/long/scans/Scan001"
        )
        assert alias_scan_folder("/a/other/Scan001", aliases) == Path(
            "/short/other/Scan001"
        )
        assert alias_scan_folder("/a/database/Scan001", aliases) == Path(
            "/short/database/Scan001"
        )  # no prefix bleed

    def test_parse_aliases(self):
        assert parse_aliases(
            ["Z:/data=/mnt/hdna2/data", "/Volumes/hdna2/data=/mnt/hdna2/data"]
        ) == {
            "Z:/data": "/mnt/hdna2/data",
            "/Volumes/hdna2/data": "/mnt/hdna2/data",
        }
        with pytest.raises(Exception):
            parse_aliases(["nonsense"])


class TestEqualityAndLedger:
    def test_tables_equal_treats_nan_as_equal_and_sees_real_differences(self):
        a = pyarrow.table(
            {
                "x": pyarrow.array([math.nan, 1.5, None]),
                "s": pyarrow.array(["a", None, "c"]),
                "k": pyarrow.array([1, 2, 3]),
            }
        )
        b = pyarrow.table(
            {
                "x": pyarrow.array([math.nan, 1.5, None]),
                "s": pyarrow.array(["a", None, "c"]),
                "k": pyarrow.array([1, 2, 3]),
            }
        )
        assert tables_equal(a, b)
        assert not tables_equal(a, b.slice(0, 2))
        assert not tables_equal(a, b.set_column(2, "k", pyarrow.array([1, 2, 4])))
        assert not tables_equal(a, b.rename_columns(["x", "s", "kk"]))

    def test_ledger_round_trips_items(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger.jsonl")
        item = PortItem(
            "uid1",
            "primary",
            7,
            "Z:/data/scans/Scan007",
            "/mnt/x/scans/Scan007",
            "table_a",
            3,
            "ported",
            parquet="/mnt/x/scans/Scan007/ScanDataScan007-primary.parquet",
            rows=10,
            old_data_source={"mimetype": "application/x-tiled-sql-table", "assets": []},
        )
        ledger.append(item)
        ledger.append(
            PortItem("uid2", "primary", None, None, None, None, None, "skip-no-folder")
        )
        records = list(Ledger.read(tmp_path / "ledger.jsonl"))
        assert [r["status"] for r in records] == ["ported", "skip-no-folder"]
        assert (
            records[0]["old_data_source"]["mimetype"] == "application/x-tiled-sql-table"
            and "at" in records[0]
        )
        assert summarize(
            [item, PortItem("u", "s", None, None, None, None, None, "port")]
        ) == {"ported": 1, "port": 1}


# ── opt-in: against a real in-process Tiled with SQL storage ─────────────

tiled_server_app = pytest.importorskip("tiled.server.app")
pytest.importorskip("tiled.catalog")


@pytest.fixture
def sql_catalog(tmp_path: Path):
    from tiled.catalog import from_uri as catalog_from_uri
    from tiled.client import from_context
    from tiled.client.context import Context
    from tiled.config import Authentication
    from tiled.server.app import build_app

    tree = catalog_from_uri(
        f"sqlite:///{tmp_path / 'catalog.db'}",
        writable_storage=[
            f"sqlite:///{tmp_path / 'tables.db'}",
            str(tmp_path / "storage"),
        ],
        readable_storage=[str(tmp_path)],
        init_if_not_exists=True,
    )
    app = build_app(tree, authentication=Authentication(single_user_api_key="test"))
    with Context.from_app(app, api_key="test") as context:
        yield from_context(context)


def _register_sql_run(
    client, tmp_path: Path, scan_number: int, scan_folder: Path | None, num: int = 3
) -> str:
    """A run the STOCK writer registers as an appendable SQL table (the pre-0.110.0 shape)."""
    from bluesky import RunEngine
    from bluesky import plans as bp

    from geecs_bluesky.tiled_spool import SpoolCallback, SpoolLayout
    from geecs_bluesky.tiled_writer import SpoolRegistrar, make_tiled_writer

    # The readable with a GEECS row's value kinds lives in the writer's own
    # opt-in test; load it by path (tests/ is not a package).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_catalog_test", Path(__file__).with_name("test_tiled_writer_catalog.py")
    )
    catalog_test = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(catalog_test)
    _Det = catalog_test._Det

    layout = SpoolLayout(tmp_path / f"state-{scan_number}")
    RE = RunEngine()
    RE.subscribe(SpoolCallback(layout))
    md = {"scan_number": scan_number}
    if scan_folder is not None:
        md["scan_folder"] = str(scan_folder)
    (uid,) = RE(bp.count([_Det()], num=num), **md)
    registrar = SpoolRegistrar(
        layout,
        "http://unused.test",
        client_factory=lambda: client,
        writer_factory=lambda c: make_tiled_writer(c, tables="appendable"),
        reachable=lambda uri: True,
        held=lambda path: False,
        orphan_after_s=1800.0,
    )
    heartbeat = registrar.sweep()
    assert heartbeat.done == 1 and heartbeat.failed == 0, heartbeat.last_error
    return uid


def test_port_moves_a_sql_table_to_parquet_and_back(
    tmp_path: Path, sql_catalog
) -> None:
    from geecs_data_utils.tiled_catalog import read_primary_scalars

    from geecs_bluesky.tiled_port import (
        PARQUET_MIMETYPE,
        SQL_TABLE_MIMETYPE,
        TABLE_KEY,
        plan,
        port_item,
        read_arrow,
        restore_sql_table,
    )

    # Three runs: one recorded under a foreign root (aliased), one with no
    # scan folder (stays), one whose folder is gone (stays).
    folder = tmp_path / "data" / "scans" / "Scan021"
    folder.mkdir(parents=True)
    uid = _register_sql_run(
        sql_catalog, tmp_path, 21, Path("/Volumes/hdna2/data/scans/Scan021")
    )
    uid_nofolder = _register_sql_run(sql_catalog, tmp_path, 22, None)
    uid_gone = _register_sql_run(
        sql_catalog, tmp_path, 23, tmp_path / "data" / "scans" / "Scan023"
    )
    before = read_arrow(sql_catalog[uid]["primary"].base[TABLE_KEY])
    assert (
        sql_catalog[uid]["primary"].base[TABLE_KEY].data_sources()[0].mimetype
        == SQL_TABLE_MIMETYPE
    )

    aliases = {"/Volumes/hdna2/data": str(tmp_path / "data")}
    items = plan(sql_catalog, aliases)
    by_uid = {i.run_uid: i for i in items}
    assert by_uid[uid].status == "port" and by_uid[uid].parquet == str(
        folder / "ScanDataScan021-primary.parquet"
    )
    assert by_uid[uid_nofolder].status == "skip-no-folder"
    assert by_uid[uid_gone].status == "skip-missing-folder"
    assert not (tmp_path / "data" / "scans" / "Scan023").exists()  # never created

    # dry run touches nothing
    dry = port_item(sql_catalog, by_uid[uid], tiled_path=lambda p: p, dry_run=True)
    assert (
        dry.status == "would-port" and dry.rows == 3 and not Path(dry.parquet).exists()
    )
    assert (
        sql_catalog[uid]["primary"].base[TABLE_KEY].data_sources()[0].mimetype
        == SQL_TABLE_MIMETYPE
    )

    # the port
    by_uid[uid].status = "port"
    item = port_item(sql_catalog, by_uid[uid], tiled_path=lambda p: p)
    assert item.status == "ported", item.error
    assert Path(item.parquet).exists()
    node = sql_catalog[uid]["primary"].base[TABLE_KEY]
    assert tables_equal(
        before, read_arrow(node)
    )  # the Parquet serves what the SQL table held
    (ds,) = node.data_sources()
    assert ds.mimetype == PARQUET_MIMETYPE
    assert ds.assets[0].data_uri.endswith(
        "/scans/Scan021/ScanDataScan021-primary.parquet"
    )
    assert (
        node.metadata.get("det_s", {}).get("dtype") == "number"
    )  # the data keys came along
    frame = read_primary_scalars(sql_catalog[uid]["primary"])
    assert (
        len(frame) == 3
        and frame["det_n"].isna().all()
        and frame["det_str"].tolist() == ["ON"] * 3
    )

    # idempotent
    again = {i.run_uid: i for i in plan(sql_catalog, aliases)}
    assert again[uid].status == "already-parquet"

    # rollback: the SQL table is still there, re-registered as external
    metadata = dict(node.metadata)
    node.delete(external_only=False)
    restore_sql_table(sql_catalog[uid]["primary"].base, item.old_data_source, metadata)
    restored = sql_catalog[uid]["primary"].base[TABLE_KEY]
    assert restored.data_sources()[0].mimetype == SQL_TABLE_MIMETYPE
    assert len(read_primary_scalars(sql_catalog[uid]["primary"])) == 3
    assert Path(item.parquet).exists()  # left in place
