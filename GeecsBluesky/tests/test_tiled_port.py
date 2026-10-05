"""``geecs_bluesky.tiled_port`` — the history port of SQL-stored stream tables to Parquet.

The pure parts (aliasing, the plan's classification and duplicate refusal,
table equality, the ledger, the restore guards) are hermetic.  The port
itself needs a Tiled with SQL storage and is tested against a real
in-process server (opt-in, as ``test_tiled_writer_catalog.py``): a run
written by the **stock** writer into an appendable SQL table, ported, read
back, rolled back through the ledger — and a still-SQL node refused.
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow
import pytest

pytest.importorskip("tiled.client")
tiled_port = pytest.importorskip("geecs_bluesky.tiled_port")
from geecs_bluesky.tiled_port import (  # noqa: E402
    PARQUET_MIMETYPE,
    SQL_TABLE_MIMETYPE,
    Ledger,
    PortItem,
    alias_scan_folder,
    is_live_sql_source,
    is_restorable_record,
    parse_aliases,
    refuse_duplicate_targets,
    restore_sql_table,
    summarize,
    tables_equal,
)


def _item(uid: str, status: str = "port", parquet: str | None = None, **kw) -> PortItem:
    return PortItem(
        uid,
        kw.get("stream", "primary"),
        kw.get("scan", 7),
        None,
        None,
        "table_a",
        3,
        status,
        parquet=parquet,
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

    def test_a_local_path_passes_through_and_the_longest_root_wins(self):
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
        with pytest.raises(ValueError, match="SRC=DST"):
            parse_aliases(["nonsense"])


class TestPlanGuards:
    def test_two_items_headed_for_one_file_are_both_refused(self):
        a = _item("uid1", parquet="/data/scans/Scan007/ScanDataScan007-primary.parquet")
        b = _item("uid2", parquet="/data/scans/Scan007/ScanDataScan007-primary.parquet")
        c = _item("uid3", parquet="/data/scans/Scan008/ScanDataScan008-primary.parquet")
        out = refuse_duplicate_targets([a, b, c], tiled_path=lambda p: p)
        assert [i.status for i in out] == [
            "skip-duplicate-target",
            "skip-duplicate-target",
            "port",
        ]

    def test_a_target_an_existing_parquet_node_serves_is_refused(self):
        existing = _item(
            "uid1",
            status="already-parquet",
            parquet="file://localhost/data/scans/Scan007/ScanDataScan007-primary.parquet",
        )
        new = _item(
            "uid2", parquet="/data/scans/Scan007/ScanDataScan007-primary.parquet"
        )
        out = refuse_duplicate_targets([existing, new], tiled_path=lambda p: p)
        assert (
            out[0].status == "already-parquet"
            and out[1].status == "skip-duplicate-target"
        )

    def test_is_live_sql_source(self):
        class _DS:
            def __init__(self, mimetype, parameters):
                self.mimetype, self.parameters = mimetype, parameters

        assert is_live_sql_source(
            _DS(SQL_TABLE_MIMETYPE, {"table_name": "t", "dataset_id": 0})
        )
        assert not is_live_sql_source(_DS(SQL_TABLE_MIMETYPE, {}))  # detached
        assert not is_live_sql_source(
            _DS(PARQUET_MIMETYPE, {"table_name": "t", "dataset_id": 1})
        )

    def test_a_detached_record_is_not_restorable_but_the_intent_record_is(self):
        intent = {
            "mimetype": SQL_TABLE_MIMETYPE,
            "parameters": {"table_name": "t", "dataset_id": 4},
            "assets": [],
        }
        after_detach = {"mimetype": SQL_TABLE_MIMETYPE, "parameters": {}, "assets": []}
        assert is_restorable_record(intent)
        assert not is_restorable_record(after_detach)
        assert not is_restorable_record(
            {
                "mimetype": PARQUET_MIMETYPE,
                "parameters": {"table_name": "t", "dataset_id": 4},
            }
        )
        assert not is_restorable_record(None)

    def test_restore_refuses_a_record_that_is_not_a_sql_table(self):
        with pytest.raises(ValueError, match="nothing to restore"):
            restore_sql_table(
                object(),
                {"mimetype": PARQUET_MIMETYPE, "parameters": {}, "assets": []},
                {},
            )
        with pytest.raises(ValueError, match="nothing to restore"):
            restore_sql_table(
                object(),
                {"mimetype": SQL_TABLE_MIMETYPE, "parameters": {}, "assets": []},
                {},
            )


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
        item = _item(
            "uid1",
            status="ported",
            parquet="/mnt/x/scans/Scan007/ScanDataScan007-primary.parquet",
        )
        item.rows = 10
        item.old_data_source = {"mimetype": SQL_TABLE_MIMETYPE, "assets": []}
        ledger.append(item)
        ledger.append(_item("uid2", status="skip-no-folder"))
        records = list(Ledger.read(tmp_path / "ledger.jsonl"))
        assert [r["status"] for r in records] == ["ported", "skip-no-folder"]
        assert (
            records[0]["old_data_source"]["mimetype"] == SQL_TABLE_MIMETYPE
            and "at" in records[0]
        )
        assert summarize([item, _item("u")]) == {"ported": 1, "port": 1}


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


def _sql_rows(tmp_path: Path, table_name: str, dataset_id: int) -> int | str:
    """Rows of one dataset straight from the SQL store (the rollback's substance)."""
    import sqlite3

    db = sqlite3.connect(f"file:{tmp_path / 'tables.db'}?mode=ro", uri=True)
    tables = [
        r[0]
        for r in db.execute(
            "select name from sqlite_master where type='table' and name like 'table_%'"
        )
    ]
    if table_name not in tables:
        return "TABLE GONE"
    return db.execute(
        f'select count(*) from "{table_name}" where _dataset_id=?', (dataset_id,)
    ).fetchone()[0]


def _register_sql_run(
    client,
    tmp_path: Path,
    scan_number: int,
    scan_folder: Path | str | None,
    num: int = 3,
) -> str:
    """A run the STOCK writer registers as an appendable SQL table (the pre-0.110.0 shape)."""
    import importlib.util

    from bluesky import RunEngine
    from bluesky import plans as bp

    from geecs_bluesky.tiled_spool import SpoolCallback, SpoolLayout
    from geecs_bluesky.tiled_writer import SpoolRegistrar, make_tiled_writer

    # The readable with a GEECS row's value kinds lives in the writer's own
    # opt-in test; load it by path (tests/ is not a package).
    spec = importlib.util.spec_from_file_location(
        "_catalog_test", Path(__file__).with_name("test_tiled_writer_catalog.py")
    )
    catalog_test = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(catalog_test)

    layout = SpoolLayout(tmp_path / f"state-{scan_number}")
    RE = RunEngine()
    RE.subscribe(SpoolCallback(layout))
    md = {"scan_number": scan_number}
    if scan_folder is not None:
        md["scan_folder"] = str(scan_folder)
    (uid,) = RE(bp.count([catalog_test._Det()], num=num), **md)
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


def test_port_moves_a_sql_table_to_parquet_and_the_ledger_rolls_it_back(
    tmp_path: Path, sql_catalog
) -> None:
    from geecs_data_utils.tiled_catalog import read_primary_scalars

    from geecs_bluesky.tiled_port import (
        TABLE_KEY,
        plan,
        port_item,
        read_arrow,
        restore_from_ledger,
    )

    # Three runs: one recorded under a foreign root (aliased), one with no
    # scan folder (stays), one whose folder is gone (stays).
    folder = tmp_path / "data" / "scans" / "Scan021"
    folder.mkdir(parents=True)
    uid = _register_sql_run(
        sql_catalog, tmp_path, 21, "/Volumes/hdna2/data/scans/Scan021"
    )
    uid_nofolder = _register_sql_run(sql_catalog, tmp_path, 22, None)
    uid_gone = _register_sql_run(
        sql_catalog, tmp_path, 23, tmp_path / "data" / "scans" / "Scan023"
    )
    before = read_arrow(sql_catalog[uid]["primary"].base[TABLE_KEY])
    (sql_ds,) = sql_catalog[uid]["primary"].base[TABLE_KEY].data_sources()
    table_name, dataset_id = (
        sql_ds.parameters["table_name"],
        sql_ds.parameters["dataset_id"],
    )
    assert (
        sql_ds.mimetype == SQL_TABLE_MIMETYPE
        and _sql_rows(tmp_path, table_name, dataset_id) == 3
    )

    aliases = {"/Volumes/hdna2/data": str(tmp_path / "data")}
    items = plan(sql_catalog, aliases, tiled_path=lambda p: p)
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

    # the port, with the ledger's intent record written before the catalog is touched
    ledger = Ledger(tmp_path / "ledger.jsonl")
    by_uid[uid].status = "port"
    item = port_item(
        sql_catalog, by_uid[uid], tiled_path=lambda p: p, before_mutation=ledger.append
    )
    ledger.append(item)
    assert item.status == "ported", item.error
    assert [r["status"] for r in Ledger.read(ledger.path)] == ["port", "ported"]
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
    # and the SQL rows — the rollback — are still there
    assert _sql_rows(tmp_path, table_name, dataset_id) == 3

    # idempotent
    again = {i.run_uid: i for i in plan(sql_catalog, aliases, tiled_path=lambda p: p)}
    assert again[uid].status == "already-parquet"

    # --restore on a node still on SQL never deletes it: a skipped record is
    # not restorable at all, and a `failed` record (an item that failed
    # before its detach) is refused with its rows untouched.
    ledger.append(by_uid[uid_gone])  # status skip-missing-folder
    with pytest.raises(LookupError, match="no SQL record"):
        restore_from_ledger(sql_catalog, ledger.path, uid_gone, "primary")
    failed_before_detach = by_uid[uid_gone]
    failed_before_detach.status, failed_before_detach.error = (
        "failed",
        "simulated: write refused",
    )
    ledger.append(failed_before_detach)
    with pytest.raises(RuntimeError, match="still on its SQL table"):
        restore_from_ledger(sql_catalog, ledger.path, uid_gone, "primary")
    (gone_ds,) = sql_catalog[uid_gone]["primary"].base[TABLE_KEY].data_sources()
    assert gone_ds.mimetype == SQL_TABLE_MIMETYPE
    assert (
        _sql_rows(
            tmp_path,
            gone_ds.parameters["table_name"],
            gone_ds.parameters["dataset_id"],
        )
        == 3
    )

    # --restore on the ported node: SQL again, readable, Parquet left in place
    account = restore_from_ledger(sql_catalog, ledger.path, uid, "primary")
    assert "restored" in account
    restored = sql_catalog[uid]["primary"].base[TABLE_KEY]
    assert restored.data_sources()[0].mimetype == SQL_TABLE_MIMETYPE
    assert len(read_primary_scalars(sql_catalog[uid]["primary"])) == 3
    assert Path(item.parquet).exists()
    # an unknown node has no record
    with pytest.raises(LookupError):
        restore_from_ledger(sql_catalog, ledger.path, uid_nofolder, "baseline")
