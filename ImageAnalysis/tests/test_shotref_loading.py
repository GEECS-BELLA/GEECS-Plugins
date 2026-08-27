"""ImageAnalyzer.load_image resolves capture-stack ShotRefs to frames."""

from __future__ import annotations

import h5py
import numpy as np

from geecs_data_utils.io.scan_stack import ShotRef
from image_analysis.base import ImageAnalyzer


def _write_stack(tmp_path, n=3):
    path = tmp_path / "UC_Cam.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "geecs-capture/1"
        f.create_dataset(
            "frames",
            data=np.stack([np.full((4, 5), i, dtype=np.uint16) for i in range(n)]),
            chunks=(1, 4, 5),
        )
        f.create_dataset("acq_timestamp", data=np.arange(n) + 1000.0)
    return path


def test_load_image_resolves_shotref(tmp_path) -> None:
    """A ShotRef loads the single referenced frame from the stack."""
    path = _write_stack(tmp_path)
    analyzer = ImageAnalyzer()
    frame = analyzer.load_image(ShotRef(path, 2))
    assert frame.shape == (4, 5)
    assert (frame == 2).all()


def test_load_image_list_mixes_refs_and_paths(tmp_path) -> None:
    """List loading handles ShotRefs like any other per-item path."""
    path = _write_stack(tmp_path)
    analyzer = ImageAnalyzer()
    frames = analyzer.load_image([ShotRef(path, 0), ShotRef(path, 1)])
    assert [int(f[0, 0]) for f in frames] == [0, 1]
