"""Reader tests for capture frame stacks (io/scan_stack.py)."""

from __future__ import annotations

import pickle

import h5py
import numpy as np
import pytest

from geecs_data_utils.io.scan_stack import (
    LABVIEW_EPOCH_OFFSET,
    ShotRef,
    find_stack_file,
    is_stack_file,
    read_shot,
    read_stack_timestamps,
)


def _write_stack(device_dir, n=3, schema="geecs-capture/1"):
    """Write a minimal contract-conformant stack, as the daemon would."""
    device_dir.mkdir(parents=True, exist_ok=True)
    path = device_dir / f"{device_dir.name}.h5"
    with h5py.File(path, "w", libver="latest") as f:
        f.attrs["schema"] = schema
        f.attrs["device"] = device_dir.name
        f.create_dataset(
            "frames",
            data=np.stack([np.full((4, 5), i, dtype=np.uint16) for i in range(n)]),
            chunks=(1, 4, 5),
        )
        f.create_dataset("acq_timestamp", data=np.arange(n) + 1000.0)
        f.create_dataset("recv_timestamp", data=np.arange(n) + 2000.0)
    return path


def test_find_and_validate_stack(tmp_path) -> None:
    """find_stack_file locates <device>/<device>.h5 and validates the schema."""
    device_dir = tmp_path / "UC_Cam"
    path = _write_stack(device_dir)
    assert find_stack_file(device_dir) == path
    assert is_stack_file(path)


def test_wrong_schema_is_not_a_stack(tmp_path) -> None:
    """Dispatch is on the schema attribute, never the extension."""
    device_dir = tmp_path / "UC_Cam"
    _write_stack(device_dir, schema="something-else/9")
    assert find_stack_file(device_dir) is None


def test_absent_or_garbage_file_is_none(tmp_path) -> None:
    """Absent stack means 'not captured' — never an error."""
    device_dir = tmp_path / "UC_Cam"
    device_dir.mkdir()
    assert find_stack_file(device_dir) is None
    (device_dir / "UC_Cam.h5").write_bytes(b"not hdf5 at all")
    assert find_stack_file(device_dir) is None


def test_read_shot_and_timestamps(tmp_path) -> None:
    """read_shot returns the exact frame; timestamps convert epochs."""
    path = _write_stack(tmp_path / "UC_Cam")
    frame = read_shot(path, 2)
    assert frame.shape == (4, 5)
    assert (frame == 2).all()
    unix = read_stack_timestamps(path)
    lv = read_stack_timestamps(path, labview_epoch=True)
    assert list(unix) == [1000.0, 1001.0, 1002.0]
    assert list(lv - unix) == [LABVIEW_EPOCH_OFFSET] * 3

    with pytest.raises(IndexError):
        read_shot(path, 3)
    with pytest.raises(TypeError):
        read_shot(path)  # plain path with no index


def test_shotref_behaves_as_path_and_pickles(tmp_path) -> None:
    """ShotRef travels like a Path AND survives process-pool pickling."""
    path = _write_stack(tmp_path / "UC_Cam")
    ref = ShotRef(path, 1)
    assert ref.exists()
    assert ref.parent == path.parent
    assert ref.shot_index == 1
    assert (read_shot(ref) == 1).all()

    clone = pickle.loads(pickle.dumps(ref))
    assert isinstance(clone, ShotRef)
    assert clone.shot_index == 1
    assert str(clone) == str(path)
    assert (read_shot(clone) == 1).all()


def test_derived_shotref_refused_cleanly(tmp_path) -> None:
    """Paths derived from a ShotRef carry no index — clean TypeError."""
    path = _write_stack(tmp_path / "UC_Cam")
    ref = ShotRef(path, 1)
    derived = ref.parent / path.name  # ShotRef-typed on 3.11, no index
    with pytest.raises(TypeError):
        read_shot(derived)
