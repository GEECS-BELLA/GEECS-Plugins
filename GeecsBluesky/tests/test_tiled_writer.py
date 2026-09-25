"""geecs-tiled-writer: the sweep, the heartbeat, the concurrent stop.

Hermetic: the Tiled client and the writer callback are fakes.  What the
service promises — replay in order, mark done, leave an unfinished run
alone until its stop lands (or the orphan deadline passes), attempt
nothing while the server is unreachable, retry then set aside, register
idempotently, prune, and say all of it in the heartbeat.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from geecs_bluesky.tiled_spool import (
    SpoolLayout,
    SpoolState,
    encode_line,
    run_uid_of,
    spool_state,
)
from geecs_bluesky.tiled_writer import (
    STALE_AFTER_SWEEPS,
    SpoolRegistrar,
    WriterHeartbeat,
    external_groups,
    main,
    read_heartbeat,
    synthesized_stop,
    write_heartbeat,
)

URI = "http://tiled.test:8000"


def _docs(uid: str, at: float, scan: int | None = None) -> list[tuple[str, dict]]:
    start: dict[str, Any] = {"uid": uid, "time": at}
    if scan is not None:
        start["scan_number"] = scan
    return [
        ("start", start),
        (
            "descriptor",
            {
                "uid": f"{uid}-d",
                "run_start": uid,
                "name": "primary",
                "time": at + 0.5,
                "data_keys": {},
            },
        ),
        (
            "event",
            {
                "uid": f"{uid}-e",
                "descriptor": f"{uid}-d",
                "seq_num": 1,
                "time": at + 1.0,
                "data": {},
                "timestamps": {},
            },
        ),
        (
            "stop",
            {
                "uid": f"{uid}-s",
                "run_start": uid,
                "time": at + 2.0,
                "exit_status": "success",
                "num_events": {"primary": 1},
            },
        ),
    ]


def _write(layout: SpoolLayout, docs: list[tuple[str, dict]]) -> Path:
    """Write a spool file the way the engine does, without holding it open."""
    layout.ensure()
    start = docs[0][1]
    path = layout.file_for(str(start["uid"]), float(start["time"]))
    path.write_text("".join(encode_line(name, doc) for name, doc in docs))
    return path


class _Recorder:
    """A writer factory: every callback it builds records into ``calls``.

    ``fail`` maps a run uid to how many times its ``start`` still raises —
    a registration that fails for that run alone, as Tiled would.
    """

    def __init__(self, fail: dict[str, int] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.built = 0
        self.fail: dict[str, int] = dict(fail or {})

    def __call__(self, client: Any):
        self.built += 1

        def callback(name: str, doc: dict) -> None:
            if name == "start" and self.fail.get(doc["uid"], 0) > 0:
                self.fail[doc["uid"]] -= 1
                raise RuntimeError("tiled said no")
            self.calls.append((name, doc))

        return callback


class _FakeNode:
    def __init__(self) -> None:
        self.deleted: list[tuple[bool, bool]] = []

    def delete(self, recursive: bool = False, external_only: bool = True) -> None:
        self.deleted.append((recursive, external_only))


class _FakeClient:
    """``client[uid]`` raises KeyError unless the uid was registered before."""

    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = set(existing or ())
        self.nodes: dict[str, _FakeNode] = {}

    def __getitem__(self, key: str) -> _FakeNode:
        if key not in self.existing:
            raise KeyError(key)
        return self.nodes.setdefault(key, _FakeNode())


class _Clock:
    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _registrar(layout: SpoolLayout, recorder: _Recorder, **kwargs) -> SpoolRegistrar:
    clients = kwargs.pop("clients", None)
    client = kwargs.pop("client", None) or _FakeClient()
    kwargs.setdefault("clock", _Clock())
    kwargs.setdefault("reachable", lambda uri: True)
    if clients is None:
        clients = []
    kwargs.setdefault("client_factory", lambda: clients.append(client) or client)
    return SpoolRegistrar(layout, URI, writer_factory=recorder, **kwargs)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def test_complete_file_is_replayed_in_order_and_marked_done(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout = SpoolLayout(tmp_path)
    docs = _docs("run-a", 1000.0, scan=12)
    _write(layout, docs)
    recorder = _Recorder()
    registrar = _registrar(layout, recorder)
    with caplog.at_level(logging.INFO):
        heartbeat = registrar.sweep()
    assert recorder.calls == docs
    assert layout.pending_files() == []
    assert [p.name for p in layout.done_files()] == ["1000-run-a.jsonl.done"]
    assert heartbeat.done == 1 and heartbeat.pending == 0 and heartbeat.failed == 0
    assert heartbeat.last_ok == registrar._clock() and heartbeat.last_error is None
    assert heartbeat.registered == ["run-a"]
    assert "Scan012 (run-a): registered" in caplog.text
    # The heartbeat is on disk, readable, and current.
    on_disk = read_heartbeat(layout.heartbeat_path)
    assert on_disk is not None and on_disk.done == 1
    assert not on_disk.is_stale(now=registrar._clock())


def test_files_replay_oldest_first(tmp_path: Path) -> None:
    layout = SpoolLayout(tmp_path)
    _write(layout, _docs("run-late", 2000.0))
    _write(layout, _docs("run-early", 1000.0))
    recorder = _Recorder()
    _registrar(layout, recorder).sweep()
    starts = [doc["uid"] for name, doc in recorder.calls if name == "start"]
    assert starts == ["run-early", "run-late"]


def test_in_progress_file_waits_for_its_stop(tmp_path: Path) -> None:
    layout = SpoolLayout(tmp_path)
    docs = _docs("run-b", 1000.0)
    path = _write(layout, docs[:-1])  # no stop yet
    recorder = _Recorder()
    registrar = _registrar(layout, recorder)
    heartbeat = registrar.sweep()
    assert (
        recorder.calls == [] and heartbeat.in_progress == 1 and heartbeat.pending == 0
    )
    assert spool_state(path) is SpoolState.IN_PROGRESS
    with open(path, "a") as fh:
        fh.write(encode_line(*docs[-1]))
    heartbeat = registrar.sweep()
    assert [name for name, _ in recorder.calls] == [
        "start",
        "descriptor",
        "event",
        "stop",
    ]
    assert heartbeat.in_progress == 0 and heartbeat.done == 1


def test_orphan_is_registered_with_a_synthesized_fail_stop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The worker died mid-run: after the deadline the run is closed as failed."""
    layout = SpoolLayout(tmp_path)
    path = _write(layout, _docs("run-c", 1000.0)[:-1])
    clock = _Clock()
    old = clock.now - 3600.0
    os.utime(path, (old, old))
    recorder = _Recorder()
    registrar = _registrar(layout, recorder, clock=clock, orphan_after_s=1800.0)
    with caplog.at_level(logging.WARNING):
        heartbeat = registrar.sweep()
    assert heartbeat.done == 1
    name, stop = recorder.calls[-1]
    assert name == "stop"
    assert stop["run_start"] == "run-c" and stop["exit_status"] == "fail"
    assert stop["time"] == pytest.approx(old)
    assert "no stop document spooled (orphan)" in caplog.text


def test_unfinished_file_younger_than_the_deadline_is_not_an_orphan(
    tmp_path: Path,
) -> None:
    layout = SpoolLayout(tmp_path)
    path = _write(layout, _docs("run-d", 1000.0)[:-1])
    clock = _Clock()
    recent = clock.now - 60.0
    os.utime(path, (recent, recent))
    recorder = _Recorder()
    heartbeat = _registrar(layout, recorder, clock=clock, orphan_after_s=1800.0).sweep()
    assert recorder.calls == [] and heartbeat.in_progress == 1


def test_synthesized_stop_is_schema_valid() -> None:
    """The writer's normalizer validates every document; a synthesized stop must pass."""
    event_model = pytest.importorskip("event_model")
    validator = event_model.schema_validators[event_model.DocumentNames.stop]
    validator.validate(synthesized_stop("run-x", at=1234.5))


def test_unreachable_server_attempts_nothing_and_counts_no_attempt(
    tmp_path: Path,
) -> None:
    layout = SpoolLayout(tmp_path)
    _write(layout, _docs("run-e", 1000.0))
    recorder = _Recorder()
    reachable = {"value": False}
    registrar = _registrar(
        layout, recorder, reachable=lambda uri: reachable["value"], max_attempts=1
    )
    heartbeat = registrar.sweep()
    assert recorder.calls == [] and recorder.built == 0
    assert heartbeat.tiled_reachable is False and heartbeat.pending == 1
    assert "unreachable" in (heartbeat.last_error or "")
    assert layout.failed_files() == []
    # Back: the run registers on its FIRST attempt — the outage was not one.
    reachable["value"] = True
    heartbeat = registrar.sweep()
    assert heartbeat.done == 1 and heartbeat.tiled_reachable is True
    assert heartbeat.last_error is None


def test_failures_retry_then_set_the_run_aside(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout = SpoolLayout(tmp_path)
    _write(layout, _docs("run-f", 1000.0))
    _write(layout, _docs("run-g", 1001.0))
    recorder = _Recorder(fail={"run-f": 2})  # run-f fails twice; run-g succeeds
    clients: list = []
    registrar = _registrar(layout, recorder, clients=clients, max_attempts=2)
    with caplog.at_level(logging.WARNING):
        first = registrar.sweep()
    assert first.failed == 0 and first.pending == 1 and first.done == 1
    assert [doc["uid"] for name, doc in recorder.calls if name == "start"] == ["run-g"]
    assert "run-f: RuntimeError: tiled said no" in (first.last_error or "")
    assert "attempt 1 of 2" in caplog.text
    assert [run_uid_of(p) for p in layout.pending_files()] == ["run-f"]
    second = registrar.sweep()
    assert second.failed == 1 and second.pending == 0
    assert [p.name for p in layout.failed_files()] == ["1000-run-f.jsonl.failed"]
    assert "giving up" in caplog.text
    # A failure drops the client so the next attempt gets a fresh one: one
    # built for run-f (dropped), one for run-g (kept, reused by run-f's retry).
    assert len(clients) == 2


def test_failure_count_is_per_run(tmp_path: Path) -> None:
    layout = SpoolLayout(tmp_path)
    _write(layout, _docs("run-h", 1000.0))
    recorder = _Recorder(fail={"run-h": 1, "run-i": 2})
    registrar = _registrar(layout, recorder, max_attempts=3)
    registrar.sweep()
    registrar.sweep()
    assert layout.failed_files() == [] and len(layout.done_files()) == 1
    _write(layout, _docs("run-i", 1002.0))
    registrar.sweep()
    registrar.sweep()
    # run-i's two failures do not inherit run-h's one: still pending, not failed.
    assert [run_uid_of(p) for p in layout.pending_files()] == ["run-i"]
    assert layout.failed_files() == []


def test_existing_container_is_deleted_before_replay(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A writer that died between registering and renaming: register again, once."""
    layout = SpoolLayout(tmp_path)
    _write(layout, _docs("run-j", 1000.0))
    client = _FakeClient(existing={"run-j"})
    recorder = _Recorder()
    with caplog.at_level(logging.WARNING):
        heartbeat = _registrar(layout, recorder, client=client).sweep()
    assert client.nodes["run-j"].deleted == [(True, False)]
    assert heartbeat.done == 1 and len(recorder.calls) == 4
    assert "a container already exists" in caplog.text


def test_prune_keeps_recent_done_and_every_failed_file(tmp_path: Path) -> None:
    layout = SpoolLayout(tmp_path)
    layout.ensure()
    clock = _Clock()
    old_done = layout.spool_dir / "1-old.jsonl.done"
    new_done = layout.spool_dir / "2-new.jsonl.done"
    old_failed = layout.spool_dir / "3-bad.jsonl.failed"
    for path in (old_done, new_done, old_failed):
        path.write_text("")
    ancient = clock.now - 30 * 86400.0
    os.utime(old_done, (ancient, ancient))
    os.utime(old_failed, (ancient, ancient))
    os.utime(new_done, (clock.now - 3600.0, clock.now - 3600.0))
    heartbeat = _registrar(layout, _Recorder(), clock=clock, keep_days=7.0).sweep()
    assert not old_done.exists() and new_done.exists() and old_failed.exists()
    assert heartbeat.failed == 1


def test_empty_spool_still_writes_a_heartbeat(tmp_path: Path) -> None:
    layout = SpoolLayout(tmp_path)
    heartbeat = _registrar(layout, _Recorder()).sweep()
    assert heartbeat.pending == 0 and heartbeat.in_progress == 0
    assert read_heartbeat(layout.heartbeat_path) == heartbeat


# ---------------------------------------------------------------------------
# The heartbeat file
# ---------------------------------------------------------------------------


def test_heartbeat_round_trip_atomic_and_staleness(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    heartbeat = WriterHeartbeat(
        pid=1,
        version="0.103.0",
        started_at=100.0,
        last_sweep=200.0,
        sweep_interval=2.0,
        tiled_uri=URI,
        tiled_reachable=True,
        last_ok=150.0,
    )
    write_heartbeat(path, heartbeat)
    assert not path.with_name("heartbeat.json.tmp").exists()
    assert read_heartbeat(path) == heartbeat
    assert heartbeat.is_stale(now=200.0 + STALE_AFTER_SWEEPS * 2.0) is False
    assert heartbeat.is_stale(now=200.0 + STALE_AFTER_SWEEPS * 2.0 + 0.1) is True


def test_heartbeat_reader_tolerates_absence_and_newer_fields(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "heartbeat.json"
    assert read_heartbeat(path) is None
    path.write_text(
        '{"pid": 1, "version": "x", "started_at": 1, "last_sweep": 2, '
        '"sweep_interval": 2.0, "tiled_uri": "u", "tiled_reachable": true, '
        '"a_field_from_the_future": 42}'
    )
    assert read_heartbeat(path) is not None
    path.write_text("{not json")
    with caplog.at_level(logging.WARNING):
        assert read_heartbeat(path) is None
    assert "unreadable" in caplog.text


# ---------------------------------------------------------------------------
# The concurrent stop
# ---------------------------------------------------------------------------


class _Desc:
    def __init__(self, stream: str) -> None:
        self.item = {"id": stream}


def test_external_groups_by_stream_and_key_in_arrival_order() -> None:
    desc_nodes = {"d1": _Desc("primary"), "d2": _Desc("cam2_stream")}
    sres = {
        "r1": {"data_key": "cam1"},
        "r2": {"data_key": "cam1"},  # a re-prepare: same key, second resource
        "r3": {"data_key": "cam1-lineout"},
        "r4": {"data_key": "cam2"},
    }
    cache = {
        "r1": {"stream_resource": "r1", "descriptor": "d1"},
        "r3": {"stream_resource": "r3", "descriptor": "d1"},
        "r2": {"stream_resource": "r2", "descriptor": "d1"},
        "r4": {"stream_resource": "r4", "descriptor": "d2"},
        "r5": {
            "stream_resource": "r5",
            "descriptor": "d1",
        },  # no resource doc: its own group
    }
    groups = external_groups(cache, sres, desc_nodes)
    assert [[d["stream_resource"] for d in g] for g in groups] == [
        ["r1", "r2"],
        ["r3"],
        ["r4"],
        ["r5"],
    ]


def _concurrent_run_writer(max_workers: int):
    pytest.importorskip("bluesky.callbacks.tiled_writer")
    from geecs_bluesky.tiled_writer import make_concurrent_writer_classes

    _writer_cls, run_writer_cls = make_concurrent_writer_classes(max_workers)
    writer = run_writer_cls(client=object())
    writer._desc_nodes = {"d1": _Desc("primary")}
    writer._stream_resource_cache = {
        "r1": {"data_key": "cam1"},
        "r2": {"data_key": "cam1"},
        "r3": {"data_key": "cam2"},
        "r4": {"data_key": "cam3"},
    }
    writer._external_data_cache = {
        f"r{i}": {"stream_resource": f"r{i}", "descriptor": "d1"} for i in (1, 2, 3, 4)
    }
    return writer


def _instrument(writer, monkeypatch: pytest.MonkeyPatch, delay: float = 0.05):
    """Record each registration's window and the peak overlap; stub the stock stop."""
    from bluesky.callbacks.tiled_writer import _RunWriter

    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    windows: dict[str, tuple[float, float]] = {}

    def _write(datum):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        started = time.monotonic()
        time.sleep(delay)
        with lock:
            state["active"] -= 1
        windows[datum["stream_resource"]] = (started, time.monotonic())

    seen_by_stock: list[dict] = []
    monkeypatch.setattr(writer, "_write_external_data", _write)
    monkeypatch.setattr(
        _RunWriter,
        "stop",
        lambda self, doc: seen_by_stock.append(dict(self._external_data_cache)),
    )
    return state, windows, seen_by_stock


def test_concurrent_stop_overlaps_groups_and_keeps_a_key_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _concurrent_run_writer(max_workers=4)
    state, windows, seen_by_stock = _instrument(writer, monkeypatch)
    writer.stop({"uid": "stop"})
    assert set(windows) == {"r1", "r2", "r3", "r4"}
    assert state["peak"] >= 2, "the groups were registered one after another"
    # r1 and r2 share a data key: the second starts after the first ends.
    assert windows["r2"][0] >= windows["r1"][1]
    # The stock stop ran afterwards with nothing left to register.
    assert seen_by_stock == [{}]


def test_concurrent_stop_max_workers_one_is_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The knob bites: with one worker nothing overlaps (and the test above would fail)."""
    writer = _concurrent_run_writer(max_workers=1)
    state, windows, _seen = _instrument(writer, monkeypatch)
    writer.stop({"uid": "stop"})
    assert len(windows) == 4 and state["peak"] == 1


def test_concurrent_stop_propagates_a_group_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bluesky.callbacks.tiled_writer import _RunWriter

    writer = _concurrent_run_writer(max_workers=4)

    def _write(datum):
        if datum["stream_resource"] == "r3":
            raise RuntimeError("registration refused")

    monkeypatch.setattr(writer, "_write_external_data", _write)
    monkeypatch.setattr(_RunWriter, "stop", lambda self, doc: None)
    with pytest.raises(RuntimeError, match="registration refused"):
        writer.stop({"uid": "stop"})


def test_concurrent_tiled_writer_factory_builds_the_concurrent_run_writer() -> None:
    tiled_writer = pytest.importorskip("bluesky.callbacks.tiled_writer")
    from geecs_bluesky.tiled_writer import make_concurrent_writer_classes

    writer_cls, run_writer_cls = make_concurrent_writer_classes(2)

    class _Client:
        def include_data_sources(self):
            return self

    plain = writer_cls(_Client(), normalizer=None)
    callbacks, _ = plain._factory("start", {"uid": "x"})
    assert isinstance(callbacks[0], run_writer_cls)
    assert callbacks[0].max_workers == 2

    normalized = writer_cls(_Client())
    callbacks, _ = normalized._factory("start", {"uid": "x"})
    assert isinstance(callbacks[0], tiled_writer.RunNormalizer)
    assert any(
        isinstance(cb, run_writer_cls) for cb in callbacks[0]._token_refs.values()
    )


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _dead_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_main_once_sweeps_and_writes_the_heartbeat(tmp_path: Path) -> None:
    uri = f"http://127.0.0.1:{_dead_port()}"
    rc = main(["--once", "--state-dir", str(tmp_path), "--tiled-uri", uri])
    assert rc == 0
    heartbeat = read_heartbeat(tmp_path / "heartbeat.json")
    assert heartbeat is not None
    assert heartbeat.tiled_uri == uri and heartbeat.tiled_reachable is False
    assert (tmp_path / "spool").is_dir()


def test_main_without_a_catalog_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "geecs_bluesky.tiled_integration.read_tiled_config", lambda: (None, None)
    )
    assert main(["--once", "--state-dir", str(tmp_path)]) == 2
