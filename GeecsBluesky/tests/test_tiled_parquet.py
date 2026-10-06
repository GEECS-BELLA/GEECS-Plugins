"""``geecs_bluesky.tiled.parquet`` — the stream table as a Parquet file beside the s-file.

Hermetic: the Tiled client is faked at the two calls the writer makes for a
table (``desc_node.new`` and the data-source update), so what is pinned is
the file on disk, its contents, and the registration Tiled would receive.
The real-server round trip is the opt-in test in
``test_tiled_writer_catalog.py``.  The file's name is GEECS-Data-Utils'
(``stream_table_parquet_path_for``, tested there); the Tiled-host path
mapping is ``data_paths``' and is tested here beside its one consumer.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("tiled.client")
tiled_parquet = pytest.importorskip("geecs_bluesky.tiled.parquet")
from bluesky.callbacks.tiled_writer import TiledWriter, _RunWriter  # noqa: E402

from geecs_bluesky import data_paths  # noqa: E402
from geecs_bluesky.tiled import writer as tiled_writer  # noqa: E402
from geecs_bluesky.tiled.parquet import (  # noqa: E402
    PARQUET_MIMETYPE,
    GeecsRunWriter,
    GeecsTiledWriter,
    file_uri,
    table_from_rows,
    write_parquet_atomically,
)

N_NAN = 2000  # the #1020 shape: a float column NaN for its first rows

# ── fakes: the Tiled client at the calls the writer makes ─────────────────


class _FakeTableNode:
    def __init__(self, key: str, data_sources: list, metadata: dict) -> None:
        self.key = key
        self.sources = list(data_sources)
        self.metadata = metadata


class _FakeContainer:
    """A container node: children by key, the stream node doubles as its own ``.base``."""

    def __init__(self, key: str = "", metadata: dict | None = None) -> None:
        self.key = key
        self.metadata = dict(metadata or {})
        self.item = {"id": key}
        self.children: dict[str, object] = {}
        self.new_calls: list[dict] = []

    @property
    def base(self) -> _FakeContainer:
        return self

    def create_container(self, *, key, metadata=None, specs=None, access_tags=None):
        child = _FakeContainer(key, metadata)
        self.children[key] = child
        return child

    def new(
        self,
        structure_family,
        data_sources,
        *,
        key=None,
        metadata=None,
        specs=None,
        access_tags=None,
    ):
        self.new_calls.append(
            {
                "structure_family": structure_family,
                "data_sources": list(data_sources),
                "key": key,
                "metadata": metadata,
                "access_tags": access_tags,
            }
        )
        node = _FakeTableNode(key, data_sources, metadata or {})
        self.children[key] = node
        return node

    def update_metadata(self, *, metadata, drop_revision=False):
        self.metadata.update(metadata)


class _FakeClient(_FakeContainer):
    def include_data_sources(self):
        return self


# ── documents ─────────────────────────────────────────────────────────────


def _docs(uid: str, scan_folder: Path | None, n: int, stream: str = "primary"):
    start = {"uid": uid, "time": 1.0e9, "scan_number": 7}
    if scan_folder is not None:
        start["scan_folder"] = str(scan_folder)
    desc_uid = f"{uid}-d"
    yield "start", start
    yield (
        "descriptor",
        {
            "uid": desc_uid,
            "run_start": uid,
            "name": stream,
            "time": 1.0e9 + 0.1,
            "data_keys": {
                "x": {
                    "source": "ca://X",
                    "dtype": "number",
                    "shape": [],
                    "units": "mm",
                },
                "label": {"source": "ca://L", "dtype": "string", "shape": []},
            },
            "object_keys": {"dev": ["x", "label"]},
            "configuration": {},
            "hints": {},
        },
    )
    for i in range(n):
        last = i == n - 1
        yield (
            "event",
            {
                "uid": f"{uid}-e{i}",
                "descriptor": desc_uid,
                "seq_num": i + 1,
                "time": 1.0e9 + 1.0 + i,
                "data": {
                    "x": 1.5 if last else math.nan,
                    "label": "late" if last else None,
                },
                "timestamps": {"x": 1.0e9 + 1.0 + i, "label": 1.0e9 + 1.0 + i},
                "filled": {},
            },
        )
    yield (
        "stop",
        {
            "uid": f"{uid}-s",
            "run_start": uid,
            "time": 1.0e9 + 10.0,
            "exit_status": "success",
            "num_events": {stream: n},
        },
    )


def _replay(writer, docs) -> None:
    for name, doc in docs:
        writer(name, doc) if callable(writer) else getattr(writer, name)(doc)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def scan_folder(data_root: Path) -> Path:
    folder = data_root / "Y2026" / "10-Oct" / "26_1002" / "scans" / "Scan007"
    folder.mkdir(parents=True)
    return folder


def _as_tiled_host(data_root: Path):
    """A test's stand-in for ``tiled_host_path``: the share mounted at /mnt/hdna2/data over there."""
    return lambda p: "/mnt/hdna2/data" + p[len(str(data_root)) :]


# ── pure helpers ──────────────────────────────────────────────────────────


class TestHelpers:
    def test_rows_keep_their_types_and_an_all_null_column_is_text(self):
        rows = [{"x": math.nan, "n": None}, {"x": 1.5, "n": None}]
        table = table_from_rows(rows)
        assert str(table.schema.field("x").type) == "double"
        assert str(table.schema.field("n").type) == "string"

    def test_the_write_is_atomic_and_never_creates_the_folder(
        self, scan_folder, data_root
    ):
        table = table_from_rows([{"x": 1.0}])
        target = scan_folder / "t.parquet"
        write_parquet_atomically(table, target)
        assert target.exists() and not list(scan_folder.glob("*.tmp"))
        missing = data_root / "scans" / "Scan099" / "t.parquet"
        with pytest.raises(FileNotFoundError, match="never creates"):
            write_parquet_atomically(table, missing)
        assert not missing.parent.exists()

    def test_file_uri_is_tileds_form(self):
        assert (
            file_uri("/mnt/hdna2/data/x.parquet")
            == "file://localhost/mnt/hdna2/data/x.parquet"
        )


class TestTiledHostPath:
    """``data_paths.tiled_host_path``: the stacks' translation pattern, strict."""

    def test_translation_onto_the_tiled_hosts_posix_mount(self, data_root):
        local = data_root / "scans" / "Scan007" / "f.parquet"
        mapped = data_paths.translate_save_path_for_tiled_host(
            local, local_base_path=data_root, tiled_host_base_path="/mnt/hdna2/data"
        )
        assert mapped == "/mnt/hdna2/data/scans/Scan007/f.parquet"
        with pytest.raises(ValueError, match="not under the local data root"):
            data_paths.translate_save_path_for_tiled_host(
                Path("/elsewhere/f.parquet"),
                local_base_path=data_root,
                tiled_host_base_path="/mnt",
            )

    def test_unset_means_the_local_path_is_the_tiled_hosts(self, monkeypatch):
        monkeypatch.setattr(data_paths, "read_tiled_host_data_base_path", lambda: None)
        assert (
            data_paths.tiled_host_path("/local/data/scans/Scan007/f.parquet")
            == "/local/data/scans/Scan007/f.parquet"
        )

    def test_set_with_an_unknown_local_root_refuses_rather_than_registering_wrong(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            data_paths, "read_tiled_host_data_base_path", lambda: "/mnt/hdna2/data"
        )
        monkeypatch.setattr(data_paths, "_local_base_path", lambda: None)
        with pytest.raises(RuntimeError, match="refusing to register"):
            data_paths.tiled_host_path("/local/data/scans/Scan007/f.parquet")

    def test_a_root_unknown_at_import_is_re_read_once_per_ask(self, monkeypatch):
        # The share mounted after the writer started: the next ask reloads
        # the config and finds the root, so the service heals without a restart.
        from geecs_data_utils import ScanPaths

        monkeypatch.setattr(ScanPaths, "paths_config", SimpleNamespace(base_path=None))
        monkeypatch.setattr(
            ScanPaths,
            "reload_paths_config",
            classmethod(
                lambda cls: setattr(
                    cls, "paths_config", SimpleNamespace(base_path="/local/data")
                )
            ),
        )
        assert data_paths._local_base_path() == "/local/data"
        assert data_paths._local_base_path() == "/local/data"  # no second reload needed

    def test_set_with_a_known_local_root_translates(self, monkeypatch):
        monkeypatch.setattr(
            data_paths, "read_tiled_host_data_base_path", lambda: "/mnt/hdna2/data"
        )
        monkeypatch.setattr(data_paths, "_local_base_path", lambda: "/local/data")
        assert (
            data_paths.tiled_host_path("/local/data/scans/Scan007/f.parquet")
            == "/mnt/hdna2/data/scans/Scan007/f.parquet"
        )


# ── the run writer ────────────────────────────────────────────────────────


class TestGeecsRunWriter:
    def test_a_run_lands_as_one_parquet_file_registered_as_a_table(
        self, scan_folder, data_root
    ):
        client = _FakeClient()
        writer = GeecsRunWriter(client, tiled_path=_as_tiled_host(data_root))
        _replay(writer, _docs("run1", scan_folder, N_NAN + 1))

        path = scan_folder / "ScanDataScan007-primary.parquet"
        assert path.exists() and not list(scan_folder.glob("*.tmp"))
        table = pq.read_table(path)
        assert table.num_rows == N_NAN + 1
        assert set(table.column_names) == {
            "seq_num",
            "time",
            "x",
            "label",
            "ts_x",
            "ts_label",
        }
        x = table["x"].to_numpy(zero_copy_only=False)
        assert str(table.schema.field("x").type) == "double"
        assert int(np.isnan(x).sum()) == N_NAN and x[-1] == 1.5
        assert (
            table["label"].to_pylist()[-1] == "late"
            and table["label"].null_count == N_NAN
        )
        assert table["seq_num"].to_pylist()[:3] == [1, 2, 3]

        stream = client.children["run1"].children["primary"]
        (call,) = stream.new_calls
        assert call["key"] == "internal" and str(call["structure_family"]) in (
            "StructureFamily.table",
            "table",
        )
        (ds,) = call["data_sources"]
        assert ds.mimetype == PARQUET_MIMETYPE and str(ds.management) in (
            "Management.external",
            "external",
        )
        assert ds.structure.npartitions == 1 and set(ds.structure.columns) == set(
            table.column_names
        )
        (asset,) = ds.assets
        assert (
            asset.data_uri
            == "file://localhost/mnt/hdna2/data/Y2026/10-Oct/26_1002/scans/Scan007/ScanDataScan007-primary.parquet"
        )
        assert (
            asset.parameter == "data_uris"
            and asset.num == 0
            and asset.is_directory is False
        )
        assert (
            call["metadata"]["x"]["units"] == "mm"
        )  # the data keys ride along, as the stock writer does
        assert client.children["run1"].metadata["stop"]["exit_status"] == "success"

    def test_a_run_longer_than_the_batch_rewrites_one_file_and_updates_the_registration(
        self, scan_folder, monkeypatch
    ):
        client = _FakeClient()
        writer = GeecsRunWriter(client, batch_size=3, tiled_path=lambda p: p)
        updates: list = []
        monkeypatch.setattr(
            GeecsRunWriter,
            "_update_data_source_for_node",
            lambda self, node, ds: updates.append((node, ds)),
        )
        _replay(writer, _docs("run2", scan_folder, 5))
        files = sorted(p.name for p in scan_folder.iterdir())
        assert files == ["ScanDataScan007-primary.parquet"]
        assert pq.read_table(scan_folder / files[0]).num_rows == 5
        stream = client.children["run2"].children["primary"]
        assert len(stream.new_calls) == 1  # registered once ...
        ((node, ds),) = updates  # ... updated once, with the final shape
        assert node is stream.children["internal"] and ds.structure.npartitions == 1

    def test_a_missing_scan_folder_fails_the_registration_and_creates_nothing(
        self, data_root
    ):
        client = _FakeClient()
        writer = GeecsRunWriter(client, tiled_path=lambda p: p)
        gone = data_root / "scans" / "Scan042"
        with pytest.raises(FileNotFoundError, match="never creates"):
            _replay(writer, _docs("run3", gone, 2))
        assert not gone.exists()
        assert client.children["run3"].children["primary"].new_calls == []

    def test_a_path_the_tiled_host_cannot_read_is_refused_before_anything_is_written(
        self, scan_folder
    ):
        def refuse(path: str) -> str:
            raise RuntimeError("refusing to register")

        client = _FakeClient()
        writer = GeecsRunWriter(client, tiled_path=refuse)
        with pytest.raises(RuntimeError, match="refusing"):
            _replay(writer, _docs("run3b", scan_folder, 2))
        assert not list(scan_folder.iterdir())  # no file, no tmp
        assert client.children["run3b"].children["primary"].new_calls == []

    def test_without_a_scan_folder_the_stock_store_is_used_with_a_warning(
        self, caplog, monkeypatch
    ):
        client = _FakeClient()
        stock: list = []
        monkeypatch.setattr(
            _RunWriter,
            "_write_internal_data",
            lambda self, cache, node: stock.append(len(cache)),
        )
        writer = GeecsRunWriter(client, tiled_path=lambda p: p)
        with caplog.at_level("WARNING"):
            _replay(writer, _docs("run4", None, 2))
        assert stock == [2]
        assert "no scan_folder" in caplog.text

    def test_the_appendable_store_is_the_stock_path_untouched(
        self, scan_folder, monkeypatch
    ):
        client = _FakeClient()
        stock: list = []
        monkeypatch.setattr(
            _RunWriter,
            "_write_internal_data",
            lambda self, cache, node: stock.append(len(cache)),
        )
        writer = GeecsRunWriter(
            client, table_store="appendable", tiled_path=lambda p: p
        )
        _replay(writer, _docs("run5", scan_folder, 2))
        assert stock == [2]
        assert not list(scan_folder.glob("*.parquet"))

    def test_an_unknown_store_is_refused(self):
        with pytest.raises(ValueError, match="table_store"):
            GeecsRunWriter(_FakeClient(), table_store="csv")


# ── the factory and the writer's wiring ──────────────────────────────────


class TestFactory:
    def test_geecs_tiled_writer_builds_the_subclass_behind_the_normalizer(
        self, scan_folder
    ):
        client = _FakeClient()
        writer = GeecsTiledWriter(client, tiled_path=lambda p: p)
        _replay(writer, _docs("run6", scan_folder, 2))
        assert (scan_folder / "ScanDataScan007-primary.parquet").exists()
        (call,) = client.children["run6"].children["primary"].new_calls
        assert call["data_sources"][0].mimetype == PARQUET_MIMETYPE

    def test_the_default_tiled_path_is_the_data_paths_one(self):
        assert GeecsRunWriter(_FakeClient())._tiled_path is data_paths.tiled_host_path
        assert GeecsTiledWriter(_FakeClient())._tiled_path is data_paths.tiled_host_path

    def test_make_tiled_writer_picks_the_store(self):
        assert isinstance(
            tiled_writer.make_tiled_writer(_FakeClient()), GeecsTiledWriter
        )
        stock = tiled_writer.make_tiled_writer(_FakeClient(), tables="appendable")
        assert type(stock) is TiledWriter
        with pytest.raises(ValueError):
            tiled_writer.make_tiled_writer(_FakeClient(), tables="csv")

    def test_the_store_vocabulary_has_one_home(self):
        assert tiled_parquet.TABLE_STORES is tiled_writer.TABLE_STORES
        assert (
            tiled_parquet.DEFAULT_TABLE_STORE
            == tiled_writer.DEFAULT_TABLE_STORE
            == "parquet"
        )

    def test_the_registrar_threads_the_choice_into_its_default_factory(
        self, tmp_path, monkeypatch
    ):
        from geecs_bluesky.tiled.spool import SpoolLayout

        seen: list = []
        monkeypatch.setattr(
            tiled_writer,
            "make_tiled_writer",
            lambda client, *, tables: seen.append(tables) or (lambda n, d: None),
        )
        registrar = tiled_writer.SpoolRegistrar(
            SpoolLayout(tmp_path),
            "http://unused.test",
            tables="appendable",
            reachable=lambda uri: True,
        )
        registrar._writer_factory(SimpleNamespace())
        assert seen == ["appendable"]
        with pytest.raises(ValueError):
            tiled_writer.SpoolRegistrar(
                SpoolLayout(tmp_path),
                "http://unused.test",
                tables="csv",
                reachable=lambda uri: True,
            )

    def test_the_command_line_default_is_parquet(self):
        args = tiled_writer.build_parser().parse_args([])
        assert args.tables == "parquet"
        assert (
            tiled_writer.build_parser().parse_args(["--tables", "appendable"]).tables
            == "appendable"
        )
