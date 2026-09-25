"""geecs-tiled-writer: the sweep, the retry policy, the command.

Hermetic: the Tiled client and the writer callback are fakes (the real
catalog round trip is ``test_tiled_writer_catalog.py``, opt-in).  What the
service promises — replay in order, mark done, leave an unfinished run
alone while its engine holds it or until the orphan deadline, attempt
nothing while the server is unreachable, back off then set aside,
set a corrupt file aside at once, register idempotently, prune, and say
all of it in the heartbeat.
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from geecs_bluesky.tiled_spool import (
    STALE_WHILE_REGISTERING_S,
    SpoolLayout,
    SpoolState,
    WriterHeartbeat,
    encode_line,
    read_heartbeat,
    run_uid_of,
    spool_is_held,
    spool_state,
)
from geecs_bluesky.tiled_writer import (
    SpoolRegistrar,
    main,
    synthesized_stop,
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
    # Liveness is asserted explicitly where it matters (the lock tests);
    # everywhere else no engine holds anything.
    kwargs.setdefault("held", lambda path: False)
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


def test_held_file_is_a_live_run_however_long_it_is_quiet(tmp_path: Path) -> None:
    """A paused run goes silent past any deadline; the engine's lock says it is alive."""
    fcntl = pytest.importorskip("fcntl")
    layout = SpoolLayout(tmp_path)
    path = _write(layout, _docs("run-l", 1000.0)[:-1])
    clock = _Clock()
    old = clock.now - 7200.0
    os.utime(path, (old, old))
    engine = open(path, "a")  # the engine's handle, lock held for the run
    fcntl.flock(engine.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    recorder = _Recorder()
    registrar = _registrar(
        layout, recorder, clock=clock, orphan_after_s=1800.0, held=spool_is_held
    )
    try:
        heartbeat = registrar.sweep()
        assert recorder.calls == [] and heartbeat.in_progress == 1
        assert layout.pending_files() == [path]  # not renamed under the engine
    finally:
        engine.close()  # the worker exits: the lock goes with it
    heartbeat = registrar.sweep()
    assert heartbeat.done == 1 and recorder.calls[-1][1]["exit_status"] == "fail"


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


# ---------------------------------------------------------------------------
# The retry policy
# ---------------------------------------------------------------------------


def test_failures_back_off_then_set_the_run_aside(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout = SpoolLayout(tmp_path)
    _write(layout, _docs("run-f", 1000.0))
    _write(layout, _docs("run-g", 1001.0))
    recorder = _Recorder(fail={"run-f": 3})  # run-f keeps failing; run-g succeeds
    clock = _Clock()
    client = _FakeClient()
    registrar = _registrar(
        layout, recorder, clock=clock, client=client, max_attempts=2, sweep_interval=2.0
    )
    with caplog.at_level(logging.WARNING):
        first = registrar.sweep()
    assert first.failed == 0 and first.pending == 1 and first.done == 1
    assert [doc["uid"] for name, doc in recorder.calls if name == "start"] == ["run-g"]
    assert "run-f: RuntimeError: tiled said no" in (first.last_error or "")
    assert "attempt 1 of 2" in caplog.text and "next try in 4 s" in caplog.text
    # Inside the backoff window: no attempt, and the sweep is clean.
    built = recorder.built
    clock.now += 1.0
    second = registrar.sweep()
    assert recorder.built == built and second.pending == 1 and second.last_error is None
    # The failed attempt left a partial container behind; the deadline passes.
    client.existing.add("run-f")
    clock.now += 4.0
    third = registrar.sweep()
    assert recorder.built == built + 1
    assert third.failed == 1 and third.pending == 0
    assert [p.name for p in layout.failed_files()] == ["1000-run-f.jsonl.failed"]
    assert client.nodes["run-f"].deleted, (
        "the half-registered container stayed in Tiled"
    )
    assert "giving up" in caplog.text and "set aside" in (third.last_error or "")


def test_backoff_doubles_from_the_sweep_interval_and_caps(tmp_path: Path) -> None:
    layout = SpoolLayout(tmp_path)
    _write(layout, _docs("run-x", 1000.0))
    recorder = _Recorder(fail={"run-x": 20})
    clock = _Clock()
    registrar = _registrar(
        layout,
        recorder,
        clock=clock,
        sweep_interval=2.0,
        max_backoff_s=10.0,
        max_attempts=10,
    )
    registrar.sweep()
    built = 1
    assert recorder.built == built
    for delay in (4.0, 8.0, 10.0, 10.0):
        clock.now += delay - 0.5
        registrar.sweep()
        assert recorder.built == built, (
            f"attempted {delay - 0.5:.1f} s in, before its {delay:.0f} s"
        )
        clock.now += 0.5
        registrar.sweep()
        built += 1
        assert recorder.built == built
    assert layout.failed_files() == []  # five attempts of ten: still pending


def test_failure_count_is_per_run(tmp_path: Path) -> None:
    layout = SpoolLayout(tmp_path)
    _write(layout, _docs("run-h", 1000.0))
    recorder = _Recorder(fail={"run-h": 1, "run-i": 2})
    clock = _Clock()
    registrar = _registrar(
        layout, recorder, clock=clock, max_attempts=3, sweep_interval=2.0
    )
    registrar.sweep()  # run-h attempt 1 fails
    clock.now += 4.0
    registrar.sweep()  # run-h registers
    assert layout.failed_files() == [] and len(layout.done_files()) == 1
    _write(layout, _docs("run-i", 1002.0))
    registrar.sweep()  # run-i attempt 1
    clock.now += 4.0
    registrar.sweep()  # run-i attempt 2
    # run-i's two failures do not inherit run-h's one: still pending, not failed.
    assert [run_uid_of(p) for p in layout.pending_files()] == ["run-i"]
    assert layout.failed_files() == []


def test_corrupt_file_is_set_aside_at_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A retry reads the same bytes: a corrupt file is an operator's call, not a backoff."""
    layout = SpoolLayout(tmp_path)
    path = _write(layout, _docs("run-k", 1000.0))
    lines = path.read_text().splitlines(keepends=True)
    lines[1] = "{not json\n"
    path.write_text("".join(lines))
    client = _FakeClient(
        existing={"run-k"}
    )  # the start got registered before the bad line
    recorder = _Recorder()
    with caplog.at_level(logging.ERROR):
        heartbeat = _registrar(layout, recorder, client=client).sweep()
    assert heartbeat.failed == 1 and heartbeat.done == 0 and heartbeat.pending == 0
    assert [p.name for p in layout.failed_files()] == ["1000-run-k.jsonl.failed"]
    assert "line 2 unreadable" in (heartbeat.last_error or "")
    assert "cannot be registered from" in caplog.text
    assert client.nodes["run-k"].deleted  # no half-run left in Tiled
    assert recorder.built == 1  # one attempt, no retry


def test_empty_file_is_never_registered_as_a_success(tmp_path: Path) -> None:
    """A start the engine could not spool: after the deadline it is set aside, not done."""
    layout = SpoolLayout(tmp_path)
    layout.ensure()
    path = layout.file_for("run-empty", 1000.0)
    path.write_text("")
    clock = _Clock()
    old = clock.now - 7200.0
    os.utime(path, (old, old))
    recorder = _Recorder()
    heartbeat = _registrar(layout, recorder, clock=clock, orphan_after_s=1800.0).sweep()
    assert heartbeat.done == 0 and heartbeat.failed == 1
    assert recorder.calls == []
    assert "no start document" in (heartbeat.last_error or "")
    assert [p.name for p in layout.failed_files()] == ["1000-run-empty.jsonl.failed"]


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
# The writer the registrar builds
# ---------------------------------------------------------------------------


def test_default_writer_is_the_stock_tiled_writer() -> None:
    """No subclass over bluesky internals: the registrar replays through the stock writer."""
    tiled_writer = pytest.importorskip("bluesky.callbacks.tiled_writer")
    from geecs_bluesky.tiled_writer import make_tiled_writer

    class _Client:
        def include_data_sources(self):
            return self

    writer = make_tiled_writer(_Client())
    assert type(writer) is tiled_writer.TiledWriter


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


def test_module_runs_as_a_script(tmp_path: Path) -> None:
    """``python -m geecs_bluesky.tiled_writer`` is the by-hand path on a checkout with no reinstall.

    Found on the worker: without the ``__main__`` block the module imported
    and exited silently, no log line, no heartbeat.
    """
    import subprocess
    import sys

    uri = f"http://127.0.0.1:{_dead_port()}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "geecs_bluesky.tiled_writer",
            "--once",
            "--state-dir",
            str(tmp_path),
            "--tiled-uri",
            uri,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "geecs-tiled-writer" in result.stderr and "swept:" in result.stderr
    assert read_heartbeat(tmp_path / "heartbeat.json") is not None


def test_main_without_a_catalog_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "geecs_bluesky.tiled_integration.read_tiled_config", lambda: (None, None)
    )
    assert main(["--once", "--state-dir", str(tmp_path)]) == 2


# ---------------------------------------------------------------------------
# The heartbeat during a registration (the deploy PR's review, finding 1)
# ---------------------------------------------------------------------------


def test_heartbeat_names_the_run_before_its_registration_starts(
    tmp_path: Path,
) -> None:
    """A registration is ~25 s of silence: the heartbeat on disk says so first.

    A writer stand-in reads the heartbeat file mid-registration (at the
    stop document, the moment the stock writer does its ~500 HTTP calls)
    and a reader 30 s later must not call it stale; after the sweep the
    run is no longer named.
    """
    from geecs_bluesky.tiled_spool import heartbeat_verdict

    layout = SpoolLayout(tmp_path)
    _write(layout, _docs("run-a", 1000.0, scan=12))
    _write(layout, _docs("run-b", 2000.0))
    clock = _Clock()
    seen: list[tuple[str, WriterHeartbeat | None]] = []

    class _Peeking(_Recorder):
        def __call__(self, client):
            inner = super().__call__(client)

            def callback(name: str, doc: dict) -> None:
                if name == "stop":
                    seen.append(
                        (doc["run_start"], read_heartbeat(layout.heartbeat_path))
                    )
                inner(name, doc)

            return callback

    registrar = _registrar(layout, _Peeking(), clock=clock)
    final = registrar.sweep()
    assert [uid for uid, _ in seen] == ["run-a", "run-b"]
    first, second = (hb for _, hb in seen)
    assert first is not None and first.registering == "run-a"
    assert (
        first.registering_since == clock.now and first.pending == 1
    )  # run-b waits behind
    assert second is not None and second.registering == "run-b" and second.pending == 0
    # 30 s into it (the measured 25–28 s): alive, and the verdict says what it is doing
    assert not first.is_stale(now=clock.now + 30)
    verdict = heartbeat_verdict(first, now=clock.now + 30)
    assert verdict.level == "ok" and "registering run-a" in verdict.reason
    # …but a registration that never returns is stale
    assert first.is_stale(now=clock.now + STALE_WHILE_REGISTERING_S + 1)
    assert heartbeat_verdict(first, now=clock.now + STALE_WHILE_REGISTERING_S + 1).stale
    # the sweep's closing heartbeat names nothing and counts nothing pending
    assert final.registering is None and final.registering_since is None
    assert final.pending == 0 and final.done == 2
    on_disk = read_heartbeat(layout.heartbeat_path)
    assert on_disk is not None and on_disk.registering is None
    # between registrations the three-sweep rule applies
    assert not on_disk.is_stale(now=clock.now + 6)
    assert on_disk.is_stale(now=clock.now + 6.1)
