"""The writer against a real Tiled catalog, in-process — opt-in.

Skips unless ``tiled[server]`` and ``h5py`` are importable (the ``tiled``
extra is client-only; CI skips this, as it skips the suite's other server
test).  What the fakes in ``test_tiled_writer.py`` cannot show: the default writer
factory (the stock ``TiledWriter``), the real normalizer's schema validation
of a synthesized stop, and ``_delete_existing`` against a real container.
File-backed on purpose: it is the catalog shape the deployed server has.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tiled.server.app")
pytest.importorskip("tiled.catalog")
h5py = pytest.importorskip("h5py")

from bluesky import RunEngine  # noqa: E402
from bluesky import plans as bp  # noqa: E402
from tiled.catalog import from_uri as catalog_from_uri  # noqa: E402
from tiled.client import from_context  # noqa: E402
from tiled.client.context import Context  # noqa: E402
from tiled.config import Authentication  # noqa: E402
from tiled.server.app import build_app  # noqa: E402

from geecs_bluesky.tiled_spool import (  # noqa: E402
    SpoolCallback,
    SpoolLayout,
    encode_line,
    spool_state,
)
from geecs_bluesky.tiled_writer import SpoolRegistrar  # noqa: E402


class _Det:
    """A readable with the value kinds a GEECS row carries."""

    name = "det"
    parent = None
    hints: dict = {}

    def read(self):
        t = time.time()
        return {
            "det_s": {"value": np.float32(1.5), "timestamp": t},
            "det_i": {"value": np.int16(7), "timestamp": t},
            "det_n": {"value": np.float64("nan"), "timestamp": t},
            "det_str": {"value": "ON", "timestamp": t},
        }

    def describe(self):
        return {
            "det_s": {
                "source": "sim",
                "dtype": "number",
                "shape": [],
                "dtype_numpy": "<f4",
            },
            "det_i": {
                "source": "sim",
                "dtype": "integer",
                "shape": [],
                "dtype_numpy": "<i2",
            },
            "det_n": {
                "source": "sim",
                "dtype": "number",
                "shape": [],
                "dtype_numpy": "<f8",
            },
            "det_str": {"source": "sim", "dtype": "string", "shape": []},
        }

    def read_configuration(self):
        return {"gain": {"value": np.int64(3), "timestamp": 0.0}}

    def describe_configuration(self):
        return {"gain": {"source": "sim", "dtype": "integer", "shape": []}}


@pytest.fixture
def catalog(tmp_path: Path):
    tree = catalog_from_uri(
        f"sqlite:///{tmp_path / 'catalog.db'}",
        writable_storage=[f"sqlite:///{tmp_path / 'tables.db'}"],
        readable_storage=[str(tmp_path)],
        init_if_not_exists=True,
    )
    app = build_app(tree, authentication=Authentication(single_user_api_key="test"))
    with Context.from_app(app, api_key="test") as context:
        yield from_context(context)


def _registrar(layout: SpoolLayout, client) -> SpoolRegistrar:
    return SpoolRegistrar(
        layout,
        "http://unused.test",
        client_factory=lambda: client,
        reachable=lambda uri: True,
        held=lambda path: False,
        orphan_after_s=1800.0,
    )


def _rows(client, uid: str) -> int:
    return len(client[uid]["primary"].base["internal"].read())


def test_spooled_count_round_trips_through_the_real_writer(
    tmp_path: Path, catalog
) -> None:
    layout = SpoolLayout(tmp_path / "state")
    RE = RunEngine()
    RE.subscribe(SpoolCallback(layout))
    (uid,) = RE(bp.count([_Det()], num=2))
    (path,) = layout.pending_files()
    assert spool_state(path).value == "complete"

    registrar = _registrar(layout, catalog)
    heartbeat = registrar.sweep()
    assert (
        heartbeat.done == 1 and heartbeat.failed == 0 and heartbeat.last_error is None
    )
    run = catalog[uid]
    assert run.metadata["stop"]["exit_status"] == "success"
    assert _rows(catalog, uid) == 2
    assert run["primary"].metadata["configuration"]["det"]["data"] == {"gain": 3}

    # Idempotence against a real server: the writer died before the rename.
    done = layout.done_files()[0]
    done.rename(done.with_name(done.name[: -len(".done")]))
    heartbeat = registrar.sweep()
    assert heartbeat.done == 2 and heartbeat.failed == 0
    assert catalog[uid].metadata["stop"]["exit_status"] == "success"
    assert _rows(catalog, uid) == 2


def test_external_datasets_register_and_read_back(tmp_path: Path, catalog) -> None:
    layout = SpoolLayout(tmp_path / "state")
    layout.ensure()
    h5 = tmp_path / "cam.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset(
            "/cam1",
            data=np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4),
            chunks=(1, 4, 4),
        )
        f.create_dataset(
            "/cam2", data=np.ones((2, 3, 3), dtype=np.uint16), chunks=(1, 3, 3)
        )
    uid = str(uuid.uuid4())
    desc = f"{uid}-d"
    t0 = time.time()
    docs: list[tuple[str, dict]] = [
        ("start", {"uid": uid, "time": t0, "scan_number": 5}),
        (
            "descriptor",
            {
                "uid": desc,
                "run_start": uid,
                "name": "primary",
                "time": t0 + 0.1,
                "data_keys": {
                    "cam1": {
                        "source": "pva",
                        "dtype": "array",
                        "shape": [1, 4, 4],
                        "external": "STREAM:",
                        "dtype_numpy": "<u2",
                    },
                    "cam2": {
                        "source": "pva",
                        "dtype": "array",
                        "shape": [1, 3, 3],
                        "external": "STREAM:",
                        "dtype_numpy": "<u2",
                    },
                },
                "object_keys": {"cam1": ["cam1"], "cam2": ["cam2"]},
                "configuration": {},
                "hints": {},
            },
        ),
    ]
    for key, shape in (("cam1", [1, 4, 4]), ("cam2", [1, 3, 3])):
        sres = f"{uid}-sres-{key}"
        docs.append(
            (
                "stream_resource",
                {
                    "uid": sres,
                    "data_key": key,
                    "mimetype": "application/x-hdf5",
                    "uri": f"file://localhost{h5}",
                    "parameters": {"dataset": f"/{key}", "chunk_shape": shape},
                    "run_start": uid,
                },
            )
        )
        for i in range(2):
            docs.append(
                (
                    "stream_datum",
                    {
                        "uid": f"{sres}-{i}",
                        "stream_resource": sres,
                        "descriptor": desc,
                        "indices": {"start": i, "stop": i + 1},
                        "seq_nums": {"start": i + 1, "stop": i + 2},
                    },
                )
            )
    docs.append(
        (
            "stop",
            {
                "uid": f"{uid}-s",
                "run_start": uid,
                "time": t0 + 1,
                "exit_status": "success",
                "num_events": {"primary": 2},
            },
        )
    )
    layout.file_for(uid, t0).write_text("".join(encode_line(n, d) for n, d in docs))
    heartbeat = _registrar(layout, catalog).sweep()
    assert heartbeat.done == 1 and heartbeat.failed == 0, heartbeat.last_error
    run = catalog[uid]
    assert run["primary"]["cam1"].read().shape == (2, 4, 4)
    assert run["primary"]["cam2"].read().shape == (2, 3, 3)


def test_orphan_empty_and_corrupt_files_through_the_real_writer(
    tmp_path: Path, catalog
) -> None:
    layout = SpoolLayout(tmp_path / "state")
    layout.ensure()
    t0 = time.time()
    old = t0 - 7200.0

    # An orphan: the synthesized stop passes the real normalizer's validation.
    ouid = str(uuid.uuid4())
    od = f"{ouid}-d"
    orphan = layout.file_for(ouid, t0)
    orphan.write_text(
        "".join(
            encode_line(n, d)
            for n, d in [
                ("start", {"uid": ouid, "time": t0}),
                (
                    "descriptor",
                    {
                        "uid": od,
                        "run_start": ouid,
                        "name": "primary",
                        "time": t0 + 0.1,
                        "configuration": {},
                        "hints": {},
                        "object_keys": {"sim": ["x"]},
                        "data_keys": {
                            "x": {"source": "sim", "dtype": "number", "shape": []}
                        },
                    },
                ),
                (
                    "event",
                    {
                        "uid": f"{ouid}-e",
                        "descriptor": od,
                        "seq_num": 1,
                        "time": t0 + 0.2,
                        "data": {"x": 1.0},
                        "timestamps": {"x": t0},
                    },
                ),
            ]
        )
    )
    os.utime(orphan, (old, old))
    # An empty file (a start the engine could not spool).
    euid = str(uuid.uuid4())
    empty = layout.file_for(euid, t0)
    empty.write_text("")
    os.utime(empty, (old, old))
    # A corrupt file whose start registers before the bad line.
    cuid = str(uuid.uuid4())
    corrupt = layout.file_for(cuid, t0)
    corrupt.write_text(
        encode_line("start", {"uid": cuid, "time": t0})
        + "{not json\n"
        + encode_line(
            "stop",
            {
                "uid": f"{cuid}-s",
                "run_start": cuid,
                "time": t0 + 1,
                "exit_status": "success",
            },
        )
    )

    heartbeat = _registrar(layout, catalog).sweep()
    assert heartbeat.done == 1 and heartbeat.failed == 2
    assert catalog[ouid].metadata["stop"]["exit_status"] == "fail"
    assert _rows(catalog, ouid) == 1
    assert euid not in catalog
    assert cuid not in catalog  # the partial container was removed with the set-aside
    assert {p.name.split("-", 1)[1] for p in layout.failed_files()} == {
        f"{euid}.jsonl.failed",
        f"{cuid}.jsonl.failed",
    }


def test_parquet_table_registers_from_the_scan_folder_and_reads_back(
    tmp_path: Path, catalog
) -> None:
    """The default writer: the stream table is a Parquet file beside the s-file,
    registered from ``readable_storage`` like a camera stack, and the reader
    seam (``read_primary_scalars``) cannot tell it from the SQL table."""
    from geecs_data_utils.tiled_catalog import read_primary_scalars

    from geecs_bluesky.tiled_parquet import PARQUET_MIMETYPE, GeecsTiledWriter

    scan_folder = tmp_path / "scans" / "Scan011"
    scan_folder.mkdir(parents=True)
    layout = SpoolLayout(tmp_path / "state")
    RE = RunEngine()
    RE.subscribe(SpoolCallback(layout))
    (uid,) = RE(bp.count([_Det()], num=3), scan_number=11, scan_folder=str(scan_folder))
    registrar = SpoolRegistrar(
        layout,
        "http://unused.test",
        client_factory=lambda: catalog,
        # batch_size=2: the second batch takes the real data-source update
        # path (structure string→double, same asset) on a table node.
        writer_factory=lambda client: GeecsTiledWriter(
            client, batch_size=2, tiled_path=lambda p: p
        ),
        reachable=lambda uri: True,
        held=lambda path: False,
        orphan_after_s=1800.0,
    )
    heartbeat = registrar.sweep()
    assert heartbeat.done == 1 and heartbeat.failed == 0, heartbeat.last_error

    parquet = scan_folder / "ScanDataScan011-primary.parquet"
    assert parquet.exists()
    table = catalog[uid]["primary"].base["internal"]
    (source,) = table.data_sources()
    assert source.mimetype == PARQUET_MIMETYPE
    assert source.assets[0].data_uri.endswith(
        "/scans/Scan011/ScanDataScan011-primary.parquet"
    )
    frame = read_primary_scalars(catalog[uid]["primary"])
    assert len(frame) == 3
    assert str(frame["det_n"].dtype) == "float64" and frame["det_n"].isna().all()
    assert frame["det_str"].tolist() == ["ON"] * 3 and frame["det_i"].tolist() == [
        7,
        7,
        7,
    ]
    assert list(frame["seq_num"]) == [1, 2, 3]

    # Idempotence: a replay (the writer died before the rename) rewrites the
    # same file and re-registers it once.
    done = layout.done_files()[0]
    done.rename(done.with_name(done.name[: -len(".done")]))
    assert registrar.sweep().failed == 0
    assert len(read_primary_scalars(catalog[uid]["primary"])) == 3
    assert sorted(p.name for p in scan_folder.iterdir()) == [
        "ScanDataScan011-primary.parquet"
    ]
