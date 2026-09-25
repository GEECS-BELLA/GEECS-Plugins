"""The engine-side Tiled spool: one JSON Lines file per run, durable at the stop.

What the writer service relies on: every document of a run lands in the
run's file the moment it is emitted (flushed), the stop is fsynced, a
file whose last line is the stop is *complete*, and a truncated last line
(the worker died mid-write) is a readable file that is not complete.  And
what the engine relies on: spooling never reaches the network and a spool
failure never fails the run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from geecs_bluesky.tiled_integration import (
    SafeDocumentCallback,
    subscribe_tiled_spool,
)
from geecs_bluesky.tiled_spool import (
    DEFAULT_STATE_DIR,
    ENV_STATE_DIR,
    SpoolCallback,
    SpoolError,
    SpoolLayout,
    SpoolState,
    default_state_dir,
    iter_documents,
    run_uid_of,
    spool_state,
)

RUN: list[tuple[str, dict]] = [
    ("start", {"uid": "run-1", "time": 1700000000.5, "scan_number": 12}),
    (
        "descriptor",
        {
            "uid": "d-1",
            "run_start": "run-1",
            "name": "primary",
            "time": 1700000000.9,
            "data_keys": {"x": {"shape": [], "dtype": "number", "source": "PV:X"}},
        },
    ),
    (
        "event",
        {
            "uid": "e-1",
            "descriptor": "d-1",
            "seq_num": 1,
            "time": 1700000001.0,
            "data": {"x": 1.5},
            "timestamps": {"x": 1700000001.0},
        },
    ),
    (
        "stop",
        {
            "uid": "s-1",
            "run_start": "run-1",
            "time": 1700000002.0,
            "exit_status": "success",
            "num_events": {"primary": 1},
        },
    ),
]


def _spool(tmp_path: Path, **kwargs) -> tuple[SpoolLayout, SpoolCallback]:
    layout = SpoolLayout(tmp_path / "state")
    return layout, SpoolCallback(layout, **kwargs)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def test_round_trip_one_file_per_run(tmp_path: Path) -> None:
    layout, spool = _spool(tmp_path)
    for name, doc in RUN:
        spool(name, doc)
    files = layout.pending_files()
    assert [p.name for p in files] == ["1700000000-run-1.jsonl"]
    assert run_uid_of(files[0]) == "run-1"
    assert list(iter_documents(files[0])) == RUN
    assert spool_state(files[0]) is SpoolState.COMPLETE
    assert spool.run_uid is None  # closed at the stop


def test_every_document_is_flushed_and_only_the_stop_is_fsynced(
    tmp_path: Path,
) -> None:
    """A second reader sees each line as it is emitted; fsync is paid once."""
    synced: list[int] = []
    layout, spool = _spool(tmp_path, fsync=synced.append)
    spool(*RUN[0])
    path = layout.pending_files()[0]
    assert path.read_text().count("\n") == 1
    assert spool_state(path) is SpoolState.IN_PROGRESS
    spool(*RUN[1])
    spool(*RUN[2])
    assert path.read_text().count("\n") == 3
    assert synced == []
    spool(*RUN[3])
    assert len(synced) == 1
    assert spool_state(path) is SpoolState.COMPLETE


def test_numpy_values_spool_as_plain_json(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    layout, spool = _spool(tmp_path)
    spool("start", {"uid": "r", "time": 1.0})
    spool(
        "event",
        {
            "uid": "e",
            "descriptor": "d",
            "seq_num": 1,
            "time": 2.0,
            "data": {"a": np.float64(2.5), "b": np.int32(3), "c": np.array([1, 2])},
            "timestamps": {},
        },
    )
    spool("stop", {"uid": "s", "run_start": "r", "time": 3.0, "exit_status": "success"})
    docs = dict(iter_documents(layout.pending_files()[0]))
    assert docs["event"]["data"] == {"a": 2.5, "b": 3, "c": [1, 2]}
    assert type(docs["event"]["data"]["a"]) is float


def test_unserializable_document_is_loud_and_scoped_to_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The guard the engine wears: the run goes on, the next run spools again."""
    layout, spool = _spool(tmp_path)
    safe = SafeDocumentCallback(spool, label="TiledSpool")
    with caplog.at_level(logging.ERROR):
        safe("start", {"uid": "r1", "time": 1.0})
        safe("event", {"uid": "e", "descriptor": "d", "data": {"bad": object()}})
        safe(
            "stop",
            {"uid": "s", "run_start": "r1", "time": 2.0, "exit_status": "success"},
        )
        safe("start", {"uid": "r2", "time": 3.0})
        safe(
            "stop",
            {"uid": "s2", "run_start": "r2", "time": 4.0, "exit_status": "success"},
        )
    assert (
        "TiledSpool failed while handling event document during run r1" in caplog.text
    )
    states = {run_uid_of(p): spool_state(p) for p in layout.pending_files()}
    assert states == {"r1": SpoolState.IN_PROGRESS, "r2": SpoolState.COMPLETE}
    # r1's file holds exactly the start: the refused document was never half-written.
    r1 = next(p for p in layout.pending_files() if run_uid_of(p) == "r1")
    assert [name for name, _ in iter_documents(r1)] == ["start"]


def test_nested_start_is_refused(tmp_path: Path) -> None:
    _layout, spool = _spool(tmp_path)
    spool("start", {"uid": "outer", "time": 1.0})
    with pytest.raises(SpoolError, match="nested"):
        spool("start", {"uid": "inner", "time": 2.0})


def test_document_without_a_run_is_refused(tmp_path: Path) -> None:
    _layout, spool = _spool(tmp_path)
    with pytest.raises(SpoolError, match="no run open"):
        spool("event", {"uid": "e"})


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------


def test_truncated_last_line_reads_as_in_progress_and_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout, spool = _spool(tmp_path)
    for name, doc in RUN:
        spool(name, doc)
    path = layout.pending_files()[0]
    path.write_text(path.read_text()[:-20])  # the stop line cut mid-way
    assert spool_state(path) is SpoolState.IN_PROGRESS
    with caplog.at_level(logging.WARNING):
        docs = list(iter_documents(path))
    assert [name for name, _ in docs] == ["start", "descriptor", "event"]
    assert "truncated write" in caplog.text


def test_stop_line_without_its_newline_is_not_complete(tmp_path: Path) -> None:
    """The newline is the commit mark: a stop record still being written is not a stop."""
    layout, spool = _spool(tmp_path)
    for name, doc in RUN:
        spool(name, doc)
    path = layout.pending_files()[0]
    path.write_text(path.read_text().rstrip("\n"))
    assert spool_state(path) is SpoolState.IN_PROGRESS
    # Read back, the record itself is intact — nothing is lost, only not yet committed.
    assert [name for name, _ in iter_documents(path)][-1] == "stop"


def test_corrupt_middle_line_raises(tmp_path: Path) -> None:
    layout, spool = _spool(tmp_path)
    for name, doc in RUN:
        spool(name, doc)
    path = layout.pending_files()[0]
    lines = path.read_text().splitlines(keepends=True)
    lines[1] = "{not json\n"
    path.write_text("".join(lines))
    with pytest.raises(SpoolError, match="line 2"):
        list(iter_documents(path))


def test_empty_file_is_in_progress(tmp_path: Path) -> None:
    path = tmp_path / "1-x.jsonl"
    path.write_text("")
    assert spool_state(path) is SpoolState.IN_PROGRESS


def test_run_uid_of_strips_every_suffix() -> None:
    assert run_uid_of(Path("1700000000-abc-def.jsonl")) == "abc-def"
    assert run_uid_of(Path("1700000000-abc-def.jsonl.done")) == "abc-def"
    assert run_uid_of(Path("1700000000-abc-def.jsonl.failed")) == "abc-def"


# ---------------------------------------------------------------------------
# Where the spool lives
# ---------------------------------------------------------------------------


def test_default_state_dir_precedence() -> None:
    env = {ENV_STATE_DIR: "/explicit", "STATE_DIRECTORY": "/var/lib/a:/var/lib/b"}
    assert default_state_dir(env, honour_state_directory=True) == Path("/explicit")
    env.pop(ENV_STATE_DIR)
    assert default_state_dir(env, honour_state_directory=True) == Path("/var/lib/a")
    # The engine never reads systemd's: its unit may own a state directory
    # of its own one day, and the spool must not silently move with it.
    expected = DEFAULT_STATE_DIR.expanduser()
    assert default_state_dir(env, honour_state_directory=False) == expected
    assert default_state_dir({}, honour_state_directory=True) == expected


# ---------------------------------------------------------------------------
# The subscription
# ---------------------------------------------------------------------------


class _Engine:
    """RunEngine stand-in recording subscriptions (never a real RE needed)."""

    def __init__(self) -> None:
        self.subscribed: list = []

    def subscribe(self, callback) -> int:
        self.subscribed.append(callback)
        return len(self.subscribed)


def test_subscribe_tiled_spool_needs_a_configured_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No [tiled] uri → nothing subscribed, nothing created (a box with no writer)."""
    monkeypatch.setattr(
        "geecs_bluesky.tiled_integration.read_tiled_config", lambda: (None, None)
    )
    engine = _Engine()
    with caplog.at_level(logging.WARNING):
        assert subscribe_tiled_spool(engine, tmp_path / "state") is None
    assert engine.subscribed == []
    assert "No Tiled URI configured" in caplog.text
    assert not (tmp_path / "state").exists()


def test_subscribe_tiled_spool_never_opens_a_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The engine spools; reachability is the writer's concern."""
    monkeypatch.setattr(
        "geecs_bluesky.tiled_integration.read_tiled_config",
        lambda: ("http://192.0.2.1:8000", "key"),
    )

    def _no_network(*args, **kwargs):
        raise AssertionError("the engine must not open a socket for Tiled")

    monkeypatch.setattr("socket.create_connection", _no_network)
    engine = _Engine()
    token = subscribe_tiled_spool(engine, tmp_path / "state")
    assert token == 1
    assert isinstance(engine.subscribed[0], SafeDocumentCallback)
    for name, doc in RUN:
        engine.subscribed[0](name, doc)
    layout = SpoolLayout(tmp_path / "state")
    assert [run_uid_of(p) for p in layout.pending_files()] == ["run-1"]
    assert spool_state(layout.pending_files()[0]) is SpoolState.COMPLETE


def test_subscribe_tiled_spool_reads_the_state_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "geecs_bluesky.tiled_integration.read_tiled_config",
        lambda: ("http://192.0.2.1:8000", None),
    )
    monkeypatch.setenv(ENV_STATE_DIR, str(tmp_path / "from-env"))
    assert subscribe_tiled_spool(_Engine()) == 1
    assert (tmp_path / "from-env" / "spool").is_dir()


def test_make_run_engine_tiled_spools_and_never_builds_a_tiled_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``make_run_engine(tiled=True)``: the spool is the engine's whole Tiled path."""
    pytest.importorskip("aioca")
    from bluesky.plan_stubs import close_run, open_run

    from geecs_bluesky.run_engine import make_run_engine

    monkeypatch.setattr(
        "geecs_bluesky.tiled_integration.read_tiled_config",
        lambda: ("http://192.0.2.1:8000", None),
    )
    monkeypatch.setenv(ENV_STATE_DIR, str(tmp_path / "state"))

    def _no_client(*args, **kwargs):
        raise AssertionError("the engine built a Tiled client")

    monkeypatch.setattr("tiled.client.from_uri", _no_client, raising=False)
    RE = make_run_engine(mock=True, tiled=True)

    def plan():
        yield from open_run(md={"scan_number": 7})
        yield from close_run()

    RE(plan())
    layout = SpoolLayout(tmp_path / "state")
    (path,) = layout.pending_files()
    assert spool_state(path) is SpoolState.COMPLETE
    names = [name for name, _ in iter_documents(path)]
    assert names[0] == "start" and names[-1] == "stop"
