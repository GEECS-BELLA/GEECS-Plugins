"""Hdf5StackWriter: container contract, invariants, failure modes."""

from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

from geecs_bluesky.capture.writer import SCHEMA_ID, Hdf5StackWriter  # noqa: E402


def _writer(tmp_path, **overrides):
    kwargs = {
        "device": "UC_Cam",
        "experiment": "Undulator",
        "scan_number": 42,
        "source_pv": "undulator:uc_cam:image",
    }
    kwargs.update(overrides)
    return Hdf5StackWriter(tmp_path, **kwargs)


def test_writer_appends_and_finalizes(tmp_path) -> None:
    """Frames + aligned timestamps land; finalize stamps counters."""
    w = _writer(tmp_path)
    frames = [np.full((4, 5), i, dtype=np.uint16) for i in range(3)]
    for i, frame in enumerate(frames):
        w.append(frame, 1000.0 + i, 2000.0 + i)
    n = w.finalize({"frames_written": 3, "duplicates_dropped": 1})
    assert n == 3

    with h5py.File(tmp_path / "UC_Cam.h5", "r") as f:
        assert f.attrs["schema"] == SCHEMA_ID
        assert f.attrs["device"] == "UC_Cam"
        assert f.attrs["scan_number"] == 42
        assert bool(f.attrs["finalized"]) is True
        assert f.attrs["duplicates_dropped"] == 1
        assert f["frames"].shape == (3, 4, 5)
        assert f["frames"].dtype == np.uint16
        assert f["frames"].compression == "gzip"
        assert f["frames"].shuffle is True
        assert list(f["acq_timestamp"][:]) == [1000.0, 1001.0, 1002.0]
        assert list(f["recv_timestamp"][:]) == [2000.0, 2001.0, 2002.0]
        assert (f["frames"][2] == 2).all()


def test_writer_refuses_missing_directory(tmp_path) -> None:
    """The daemon never creates directories — the writer enforces it."""
    with pytest.raises(FileNotFoundError):
        _writer(tmp_path / "does_not_exist")
    assert not (tmp_path / "does_not_exist").exists()


def test_writer_rejects_shape_change(tmp_path) -> None:
    """A mid-scan geometry change raises instead of corrupting the stack."""
    w = _writer(tmp_path)
    w.append(np.zeros((4, 5), dtype=np.uint16), 1.0, 2.0)
    with pytest.raises(ValueError):
        w.append(np.zeros((6, 7), dtype=np.uint16), 3.0, 4.0)
    assert w.finalize({}) == 1


def test_writer_abort_leaves_valid_unfinalized_file(tmp_path) -> None:
    """Abort closes the file; written frames survive, finalized stays unset."""
    w = _writer(tmp_path)
    w.append(np.ones((2, 2), dtype=np.uint8), 1.0, 2.0)
    w.abort()
    with h5py.File(tmp_path / "UC_Cam.h5", "r") as f:
        assert f["frames"].shape == (1, 2, 2)
        assert "finalized" not in f.attrs
