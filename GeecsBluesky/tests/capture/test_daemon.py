"""CaptureDaemon + ScanCaptureSession over a fake frame source.

Real Hdf5StackWriter against tmp dirs (integration of the seam), fake p4p.
"""

from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

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


def _start_doc(tmp_path, *, uid="run-1", missing_device=False):
    cam_a = tmp_path / "ScanXXX" / "UC_CamA"
    cam_a.mkdir(parents=True)  # the ENGINE creates these — the test plays engine
    paths = {"UC_CamA": str(cam_a)}
    if missing_device:
        paths["UC_CamB"] = str(tmp_path / "ScanXXX" / "UC_CamB")  # never created
    return {
        "uid": uid,
        "time": 1000.0,
        "scan_number": 7,
        "experiment": "Undulator",
        "nonscalar_save_paths": paths,
    }


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
    source.on_connection("UC_CamA", False)

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


def test_missing_device_dir_skipped_never_created(tmp_path) -> None:
    """A camera whose engine dir is absent is skipped — and NOT mkdir'ed."""
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )
    daemon("start", _start_doc(tmp_path, missing_device=True))
    assert [t.device for t in source.subscribed] == ["UC_CamA"]
    assert not (tmp_path / "ScanXXX" / "UC_CamB").exists()
    daemon("stop", {"run_start": "run-1"})


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
    doc2 = _start_doc(tmp_path / "second", uid="run-2")
    daemon("start", doc2)
    assert sources[0].closed

    with h5py.File(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", "r") as f:
        assert "finalized" not in f.attrs

    daemon("stop", {"run_start": "run-2"})


def test_daemon_shutdown_closes_open_session(tmp_path) -> None:
    """shutdown() aborts an in-flight session (daemon exit mid-scan)."""
    source = FakeSource()
    daemon = CaptureDaemon(
        experiment="Undulator", targets=_targets(), source_factory=lambda: source
    )
    daemon("start", _start_doc(tmp_path))
    daemon.shutdown()
    assert source.closed
    with h5py.File(tmp_path / "ScanXXX" / "UC_CamA" / "UC_CamA.h5", "r") as f:
        assert "finalized" not in f.attrs
