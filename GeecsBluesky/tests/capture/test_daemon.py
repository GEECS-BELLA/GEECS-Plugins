"""CaptureDaemon + ScanCaptureSession over a fake frame source.

Real Hdf5StackWriter against tmp dirs (integration of the seam), fake p4p.
"""

from __future__ import annotations

import threading
import time

import pytest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

import geecs_bluesky.capture.daemon as daemon_mod  # noqa: E402
from geecs_bluesky.capture.daemon import CaptureDaemon  # noqa: E402
from geecs_bluesky.capture.discovery import CameraTarget  # noqa: E402


class FakeSource:
    """FrameSource test double: exposes the callbacks for direct injection."""

    def __init__(self) -> None:
        self.subscribed: list[CameraTarget] = []
        self.on_frame = None
        self.on_connection = None
        self.closed = False

    def subscribe(self, targets, on_frame, on_connection) -> None:
        """Record the subscription and capture the callbacks."""
        self.subscribed = list(targets)
        self.on_frame = on_frame
        self.on_connection = on_connection

    def close(self) -> None:
        """Record teardown."""
        self.closed = True


def _targets() -> list[CameraTarget]:
    return [
        CameraTarget(
            "UC_CamA", "Point Grey Camera", "undulator:uc_cama:image", "192.168.6.100"
        ),
        CameraTarget(
            "UC_CamB", "Point Grey Camera", "undulator:uc_camb:image", "192.168.6.100"
        ),
    ]


def _start_doc(tmp_path, *, uid="run-1", devices=("UC_CamA",), create_dirs=True):
    paths = {}
    for device in devices:
        d = tmp_path / "ScanXXX" / device
        if create_dirs:
            d.mkdir(parents=True)  # the ENGINE creates these
        paths[device] = str(d)
    return {
        "uid": uid,
        "time": 1000.0,
        "scan_number": 7,
        "experiment": "Undulator",
        "nonscalar_save_paths": paths,
    }


def _wait_written(path, n, timeout=10.0) -> None:
    """Poll until the stack file holds *n* frames (writer thread is async)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                with h5py.File(path, "r") as f:
                    if "frames" in f and f["frames"].shape[0] >= n:
                        return
            except OSError:
                pass  # writer mid-append
        time.sleep(0.05)
    raise AssertionError(f"{path} never reached {n} frames")


def test_full_scan_capture_flow(tmp_path) -> None:
    """Start → frames (with dup + stale) → stop: dedupe, filter, finalize."""
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )

    daemon("start", _start_doc(tmp_path))
    assert [t.device for t in source.subscribed] == ["UC_CamA"]

    frame = np.full((3, 3), 5, dtype=np.uint16)
    source.on_frame("UC_CamA", frame, 995.0, 1001.0)  # stale (pre-start cache)
    source.on_frame("UC_CamA", frame, 1001.0, 1001.5)  # real shot 1
    source.on_frame("UC_CamA", frame, 1002.0, 1002.5)  # real shot 2
    source.on_frame("UC_CamA", frame, 1002.0, 1003.5)  # idle re-push duplicate
    source.on_frame("UC_Ghost", frame, 1002.0, 1003.5)  # unknown device ignored
    source.on_connection("UC_CamA", False)  # initial Disconnected — absorbed?
    # No: received > 0 by now, so this counts as a REAL disconnect.

    daemon("stop", {"run_start": "run-1", "exit_status": "success"})
    assert source.closed

    with h5py.File(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", "r") as f:
        assert f["frames"].shape == (2, 3, 3)
        assert list(f["acq_timestamp"][:]) == [1001.0, 1002.0]
        assert f.attrs["frames_written"] == 2
        assert f.attrs["frames_received"] == 4
        assert f.attrs["duplicates_dropped"] == 1
        assert f.attrs["stale_skipped"] == 1
        assert f.attrs["disconnect_events"] == 1
        assert bool(f.attrs["finalized"]) is True
        assert f.attrs["scan_number"] == 7
        # The counter identity closes.
        assert f.attrs["frames_received"] == (
            f.attrs["frames_written"]
            + f.attrs["duplicates_dropped"]
            + f.attrs["stale_skipped"]
            + f.attrs["shape_errors"]
            + f.attrs["queue_drops"]
            + f.attrs["late_frames"]
            + f.attrs["writer_create_failures"]
        )


def test_dirs_created_after_start_doc(tmp_path) -> None:
    """THE production ordering: start doc first, engine mkdir later, then frames.

    The engine's save-enable plan creates scans/ScanNNN/<device>/ AFTER the
    start document (defer_save_on); writers must therefore be lazy. This is
    the regression pin for the review's HIGH finding.
    """
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )
    daemon("start", _start_doc(tmp_path, create_dirs=False))
    assert [t.device for t in source.subscribed] == ["UC_CamA"]

    # Engine creates the dir only now (save-enable), before the first trigger.
    (tmp_path / "ScanXXX" / "UC_CamA").mkdir(parents=True)
    frame = np.full((2, 2), 1, dtype=np.uint16)
    source.on_frame("UC_CamA", frame, 1001.0, 1001.5)
    source.on_frame("UC_CamA", frame, 1002.0, 1002.5)
    _wait_written(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", 2)

    daemon("stop", {"run_start": "run-1"})
    with h5py.File(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", "r") as f:
        assert f["frames"].shape == (2, 2, 2)
        assert f.attrs["writer_create_failures"] == 0
        assert bool(f.attrs["finalized"]) is True


def test_missing_device_dir_drops_counted_never_created(tmp_path) -> None:
    """A device whose dir never appears: frames drop counted, no mkdir, no file."""
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )
    daemon("start", _start_doc(tmp_path, devices=("UC_CamA", "UC_CamB")))
    # Remove CamB's dir to simulate it never being created by the engine.
    (tmp_path / "ScanXXX" / "UC_CamB").rmdir()
    assert sorted(t.device for t in source.subscribed) == ["UC_CamA", "UC_CamB"]

    frame = np.full((2, 2), 1, dtype=np.uint16)
    source.on_frame("UC_CamB", frame, 1001.0, 1001.5)
    time.sleep(0.3)  # let the writer thread attempt (and fail) creation

    daemon("stop", {"run_start": "run-1"})
    assert not (tmp_path / "ScanXXX" / "UC_CamB").exists()
    assert not (tmp_path / "ScanXXX" / "UC_CamB" / "UC_CamB.h5").exists()


def test_placeholder_timestamp_zero_is_always_stale(tmp_path) -> None:
    """The gateway's (1,1) placeholder (timestamp 0.0) can never lock geometry."""
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )
    daemon("start", _start_doc(tmp_path))

    placeholder = np.zeros((1, 1), dtype=np.uint8)
    real = np.full((4, 4), 3, dtype=np.uint16)
    source.on_frame("UC_CamA", placeholder, 0.0, 1001.0)  # placeholder post
    source.on_frame("UC_CamA", real, 1001.0, 1001.5)
    source.on_frame("UC_CamA", real, 1002.0, 1002.5)
    _wait_written(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", 2)

    daemon("stop", {"run_start": "run-1"})
    with h5py.File(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", "r") as f:
        assert f["frames"].shape == (2, 4, 4)  # geometry from the REAL frame
        assert f.attrs["stale_skipped"] == 1
        assert f.attrs["shape_errors"] == 0


def test_queue_overflow_is_counted_per_device(tmp_path, monkeypatch) -> None:
    """put_nowait Full → per-device queue_drops; identity still closes."""
    monkeypatch.setattr(daemon_mod, "WRITER_QUEUE_MAX", 1)
    release = threading.Event()

    class BlockingWriter:
        def __init__(self, *args, **kwargs) -> None:
            self.frames = 0

        def append(self, frame, acq_ts, recv_ts) -> None:
            release.wait(timeout=10.0)
            self.frames += 1

        def finalize(self, counters) -> int:
            return self.frames

        def abort(self) -> None:
            pass

    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator",
        targets=_targets(),
        source_factory=lambda: source,
        writer_factory=BlockingWriter,
    )
    daemon("start", _start_doc(tmp_path))

    frame = np.full((2, 2), 1, dtype=np.uint16)
    source.on_frame("UC_CamA", frame, 1001.0, 1001.5)  # taken by writer thread
    time.sleep(0.2)  # writer now blocked inside append
    source.on_frame("UC_CamA", frame, 1002.0, 1002.5)  # fills queue (max 1)
    source.on_frame("UC_CamA", frame, 1003.0, 1003.5)  # queue Full → dropped
    release.set()

    session = daemon._session
    daemon("stop", {"run_start": "run-1"})
    counters = None
    # Summary logged; recover counters from the session's device state.
    dev = session._devices["UC_CamA"]
    counters = dev.counters()
    assert counters["queue_drops"] == 1
    assert counters["frames_received"] == 3
    assert counters["frames_received"] == (
        counters["frames_written"]
        + counters["duplicates_dropped"]
        + counters["stale_skipped"]
        + counters["shape_errors"]
        + counters["queue_drops"]
        + counters["late_frames"]
        + counters["writer_create_failures"]
    )


def test_initial_disconnect_absorbed_before_frames(tmp_path) -> None:
    """p4p's subscribe-time Disconnected is absorbed; later ones count."""
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )
    daemon("start", _start_doc(tmp_path))

    source.on_connection("UC_CamA", False)  # initial — absorbed
    frame = np.full((2, 2), 1, dtype=np.uint16)
    source.on_frame("UC_CamA", frame, 1001.0, 1001.5)
    source.on_connection("UC_CamA", False)  # real loss — counted
    _wait_written(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", 1)

    daemon("stop", {"run_start": "run-1"})
    with h5py.File(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", "r") as f:
        assert f.attrs["disconnect_events"] == 1


def test_start_without_save_paths_ignored(tmp_path) -> None:
    """Runs that save nothing (or non-scan runs) open no session."""
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )
    daemon("start", {"uid": "run-x", "time": 1.0})
    daemon("stop", {"run_start": "run-x"})
    assert source.subscribed == []


def test_second_start_closes_first_unfinalized(tmp_path) -> None:
    """Overlapping starts close the stale session without the finalized stamp."""
    sources = [FakeSource(), FakeSource()]
    it = iter(sources)
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: next(it)
    )
    daemon("start", _start_doc(tmp_path, uid="run-1"))
    frame = np.full((2, 2), 1, dtype=np.uint16)
    sources[0].on_frame("UC_CamA", frame, 1001.0, 1001.5)
    _wait_written(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", 1)

    daemon("start", _start_doc(tmp_path / "second", uid="run-2"))
    assert sources[0].closed
    with h5py.File(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", "r") as f:
        assert "finalized" not in f.attrs

    daemon("stop", {"run_start": "run-2"})


def test_mismatched_stop_never_finalizes(tmp_path) -> None:
    """A stop for a different run closes the session UN-finalized."""
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )
    daemon("start", _start_doc(tmp_path, uid="run-1"))
    frame = np.full((2, 2), 1, dtype=np.uint16)
    source.on_frame("UC_CamA", frame, 1001.0, 1001.5)
    _wait_written(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", 1)

    daemon("stop", {"run_start": "run-OTHER"})
    assert source.closed
    with h5py.File(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", "r") as f:
        assert "finalized" not in f.attrs


def test_daemon_shutdown_closes_open_session(tmp_path) -> None:
    """shutdown() aborts an in-flight session (daemon exit mid-scan)."""
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )
    daemon("start", _start_doc(tmp_path))
    frame = np.full((2, 2), 1, dtype=np.uint16)
    source.on_frame("UC_CamA", frame, 1001.0, 1001.5)
    _wait_written(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", 1)
    daemon.shutdown()
    assert source.closed
    with h5py.File(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", "r") as f:
        assert "finalized" not in f.attrs
