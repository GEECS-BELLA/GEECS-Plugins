"""``geecs_tiled_sql.py`` — the Tiled host's adapter override (#1020).

Runs in the venv built from ``../requirements.txt`` (the Tiled server stack
and the ADBC drivers — see ``../pytest.ini``).  The SQLite tests go through
the same ``SQLAdapter`` machinery and ADBC driver the server uses, against
a real file: first the stock adapter fails on the #1020 shape, then the
override reads it.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import pathlib
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow
import pytest
from tiled import storage as storage_mod
from tiled.adapters import sql
from tiled.structures.core import StructureFamily
from tiled.structures.data_source import DataSource, Management
from tiled.structures.table import TableStructure

MODULE = pathlib.Path(__file__).resolve().parents[1] / "geecs_tiled_sql.py"
SQL_TABLE_MIMETYPE = "application/x-tiled-sql-table"
N_NAN = 2000  # more than the driver's 1024-row inference window


@pytest.fixture(scope="module")
def mod():
    """The override module, imported from its deploy location (not a package)."""
    spec = importlib.util.spec_from_file_location("geecs_tiled_sql", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def nan_leading_table() -> pyarrow.Table:
    """The #1020 shape: a float column NULL for 2000 rows with a value after."""
    n = N_NAN + 1
    return pyarrow.table(
        {
            "time": pyarrow.array(np.arange(n, dtype=float)),
            "x": pyarrow.array(np.r_[np.full(N_NAN, np.nan), 1.5]),
            "allnan": pyarrow.array(np.full(n, np.nan)),
            "flag": pyarrow.array(np.r_[np.zeros(N_NAN, bool), True]),
            "name": pyarrow.array(["a"] * N_NAN + ["b"]),
            # a TEXT column NULL-leading fails the stock read the same way
            "late_text": pyarrow.array([None] * N_NAN + ["late"]),
            "k": pyarrow.array(np.arange(n, dtype="int64")),
        }
    )


@pytest.fixture
def sqlite_dataset(tmp_path):
    """A dataset written into a SQLite tabular store by the stock adapter.

    Yields the adapter constructor arguments
    ``(storage uri, structure, table_name, dataset_id)``.
    """
    storage = storage_mod.EmbeddedSQLStorage(f"sqlite:///{tmp_path}/tabular.db")
    storage_mod.register_storage(storage)
    table = nan_leading_table()
    data_source = DataSource(
        structure_family=StructureFamily.table,
        structure=TableStructure.from_arrow_table(table, npartitions=1),
        mimetype=SQL_TABLE_MIMETYPE,
        parameters={},
        management=Management.writable,
    )
    data_source = sql.SQLAdapter.init_storage(storage, data_source)
    args = (
        storage.uri,
        data_source.structure,
        data_source.parameters["table_name"],
        data_source.parameters["dataset_id"],
    )
    sql.SQLAdapter(*args).append_partition(0, table)
    try:
        yield args
    finally:
        storage_mod.unregister_storage(storage)
        storage.dispose()


def assert_typed_frame(frame) -> None:
    assert len(frame) == N_NAN + 1
    assert str(frame["x"].dtype) == "float64"
    assert frame["x"].iloc[-1] == 1.5 and int(frame["x"].isna().sum()) == N_NAN
    # all-NULL stays typed by its declaration, not INT64
    assert str(frame["allnan"].dtype) == "float64" and frame["allnan"].isna().all()
    assert str(frame["flag"].dtype) == "bool" and bool(frame["flag"].iloc[-1]) is True
    assert frame["name"].iloc[-1] == "b"
    assert frame["late_text"].iloc[-1] == "late"
    assert int(frame["late_text"].isna().sum()) == N_NAN
    assert str(frame["k"].dtype) == "int64" and frame["k"].iloc[-1] == N_NAN


class TestSQLiteTypedRead:
    def test_the_stock_adapter_fails_on_the_1020_shape(self, sqlite_dataset):
        # The test bites: without the override this is production's failure.
        # Also a deliberate canary: adbc-driver-sqlite is unpinned above, so
        # a driver release that types columns from their declaration turns
        # this red — the signal that the override can be retired.
        with pytest.raises(Exception, match="Type mismatch in column"):
            sql.SQLAdapter(*sqlite_dataset).read()

    def test_the_override_reads_every_column_by_its_declared_type(
        self, mod, sqlite_dataset
    ):
        assert_typed_frame(mod.GeecsSQLAdapter(*sqlite_dataset).read())

    def test_column_selected_and_partition_reads(self, mod, sqlite_dataset):
        adapter = mod.GeecsSQLAdapter(*sqlite_dataset)
        assert adapter.read(["x"])["x"].iloc[-1] == 1.5
        assert adapter.read_partition(0, ["x", "k"])["x"].iloc[-1] == 1.5
        assert adapter["x"].read()[-1] == 1.5

    def test_the_storage_is_wrapped_once_at_construction(self, mod, sqlite_dataset):
        adapter = mod.GeecsSQLAdapter(*sqlite_dataset)
        real = storage_mod.get_storage(sqlite_dataset[0])
        assert isinstance(adapter.storage, mod._OneBatchStorage)
        # everything but connect() is the wrapped storage's own
        assert adapter.storage.dialect == "sqlite" and adapter.storage.uri == real.uri
        before = adapter.storage
        adapter.read()
        assert adapter.storage is before  # nothing swapped during a request

    def test_concurrent_reads_through_one_adapter_instance(self, mod, sqlite_dataset):
        # The scenario an external review raised: requests sharing an adapter.
        # With the wrapper installed at construction there is no state to
        # race on; every concurrent read must come back typed.
        adapter = mod.GeecsSQLAdapter(*sqlite_dataset)
        with ThreadPoolExecutor(max_workers=8) as pool:
            frames = list(pool.map(lambda _: adapter.read(), range(16)))
        for frame in frames:
            assert_typed_frame(frame)

    def test_the_override_also_appends_through_the_wrapped_storage(
        self, mod, sqlite_dataset
    ):
        # An ingest statement carries the batch-size option too; it must be
        # inert there.  Append a second partition and read both back.
        adapter = mod.GeecsSQLAdapter(*sqlite_dataset)
        adapter.append_partition(1, nan_leading_table())
        assert len(adapter.read()) == 2 * (N_NAN + 1)
        assert_typed_frame(adapter.read_partition(1))

    def test_the_wrapper_survives_copy_and_pickle_probes(self, mod, sqlite_dataset):
        adapter = mod.GeecsSQLAdapter(*sqlite_dataset)
        clone = copy.copy(adapter.storage)  # would recurse without the guard
        assert clone.dialect == "sqlite"
        with pytest.raises(AttributeError):
            adapter.storage.__getstate__probe__  # noqa: B018 — a private name is not delegated

    def test_from_catalog_with_ordering_and_primary_key(self, mod, tmp_path):
        # The server's own construction path: init_storage with the ordering
        # parameters the TiledWriter may set, then from_catalog with the
        # data source's parameters as keyword arguments.
        storage = storage_mod.EmbeddedSQLStorage(f"sqlite:///{tmp_path}/ordered.db")
        storage_mod.register_storage(storage)
        try:
            table = nan_leading_table()
            data_source = DataSource(
                structure_family=StructureFamily.table,
                structure=TableStructure.from_arrow_table(table, npartitions=1),
                mimetype=SQL_TABLE_MIMETYPE,
                parameters={
                    "order_by_args": [{"column": "k", "direction": "asc"}],
                    "primary_key": ["k"],
                },
                management=Management.writable,
            )
            data_source = sql.SQLAdapter.init_storage(storage, data_source)
            node = types.SimpleNamespace(metadata_={"origin": "test"}, specs=[])
            adapter = mod.GeecsSQLAdapter.from_catalog(
                data_source, node, **data_source.parameters
            )
            assert isinstance(adapter, mod.GeecsSQLAdapter)
            assert isinstance(adapter.storage, mod._OneBatchStorage)
            assert adapter.order_by_args == [{"column": "k", "direction": "asc"}]
            assert adapter.primary_key == ["k"]
            assert adapter.metadata() == {"origin": "test"}
            adapter.append_partition(0, table)
            assert_typed_frame(adapter.read())
            with pytest.raises(Exception, match="Type mismatch in column"):
                sql.SQLAdapter.from_catalog(
                    data_source, node, **data_source.parameters
                ).read()
        finally:
            storage_mod.unregister_storage(storage)
            storage.dispose()

    def test_the_one_batch_size_is_accepted_by_the_driver(self, mod, sqlite_dataset):
        # The constant must sit inside the driver's accepted range (a C int):
        # a value it refuses raises here, at set_options, before any read.
        storage = storage_mod.get_storage(sqlite_dataset[0])
        conn = mod._OneBatchStorage(storage).connect()
        try:
            with conn.cursor() as cur:
                cur.execute("select 1")
                assert cur.fetchone() == (1,)
        finally:
            conn.close()


class TestFillNullListElements:
    def test_null_elements_become_nan_and_null_lists_stay_null(self, mod):
        table = pyarrow.table(
            {
                "i": pyarrow.array([0, 1, 2]),
                "v": pyarrow.array(
                    [[math.nan, None, 1.5, math.inf], [2.5, None, None, 3.5], None],
                    type=pyarrow.list_(pyarrow.float64()),
                ),
                "s": pyarrow.array(
                    [["a", None], ["b"], None], type=pyarrow.list_(pyarrow.string())
                ),
                "x": pyarrow.array([None, 1.0, 2.0]),
            }
        )
        out = mod.fill_null_list_elements(table)
        v = out["v"].to_pylist()
        assert v[2] is None
        assert [math.isnan(e) for e in v[0]] == [True, True, False, False]
        assert v[0][3] == math.inf
        assert [math.isnan(e) for e in v[1]] == [False, True, True, False]
        # other columns untouched: string lists and scalar floats keep their nulls
        assert out["s"].to_pylist() == [["a", None], ["b"], None]
        assert out["x"].null_count == 1
        assert out.schema == table.schema

    def test_large_list_is_handled_and_clean_columns_are_left_alone(self, mod):
        table = pyarrow.table(
            {
                "v": pyarrow.array(
                    [[None, 1.0]], type=pyarrow.large_list(pyarrow.float64())
                ),
                "w": pyarrow.array([[1.0, 2.0]], type=pyarrow.list_(pyarrow.float64())),
            }
        )
        out = mod.fill_null_list_elements(table)
        assert math.isnan(out["v"].to_pylist()[0][0])
        assert out["w"].to_pylist() == [[1.0, 2.0]]


@pytest.fixture
def postgres_adapter(mod):
    """An override adapter bound to a (never connected) PostgreSQL storage."""
    storage = storage_mod.RemoteSQLStorage("postgresql://db.invalid:5432/tiled_tables")
    storage_mod.register_storage(storage)
    schema = pyarrow.schema(
        [("seq_num", pyarrow.int64()), ("v", pyarrow.list_(pyarrow.float64()))]
    )
    structure = TableStructure.from_schema(schema, npartitions=1)
    try:
        yield mod.GeecsSQLAdapter(storage.uri, structure, "table_x", 7)
    finally:
        storage_mod.unregister_storage(storage)


class TestDialectRouting:
    def test_postgres_storage_is_not_wrapped(self, mod, postgres_adapter):
        assert isinstance(postgres_adapter.storage, storage_mod.RemoteSQLStorage)
        assert not isinstance(postgres_adapter.storage, mod._OneBatchStorage)

    def test_postgres_append_fills_null_list_elements_before_the_parent(
        self, mod, postgres_adapter, monkeypatch
    ):
        handed_over = {}

        def record(self, partition, data):
            handed_over["partition"], handed_over["data"] = partition, data

        monkeypatch.setattr(sql.SQLAdapter, "append_partition", record)
        frame = pyarrow.table(
            {
                "seq_num": [0, 1],
                "v": pyarrow.array(
                    [[None, 1.0], None], type=pyarrow.list_(pyarrow.float64())
                ),
            }
        ).to_pandas()
        postgres_adapter.append_partition(3, frame)
        assert handed_over["partition"] == 3
        v = handed_over["data"]["v"].to_pylist()
        assert math.isnan(v[0][0]) and v[0][1] == 1.0 and v[1] is None

    def test_sqlite_append_is_the_parents_untouched(
        self, mod, sqlite_dataset, monkeypatch
    ):
        handed_over = {}
        monkeypatch.setattr(
            sql.SQLAdapter,
            "append_partition",
            lambda self, partition, data: handed_over.setdefault("data", data),
        )
        table = nan_leading_table()
        mod.GeecsSQLAdapter(*sqlite_dataset).append_partition(0, table)
        assert handed_over["data"] is table
