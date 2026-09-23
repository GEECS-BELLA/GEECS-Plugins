"""ImageAnalyzer/LineAnalyzer loading resolves capture-stack ShotRefs to frames."""

from __future__ import annotations

import h5py
import numpy as np

from geecs_data_utils.io.scan_stack import FRAMES_DATASET, TIMESTAMPS_DATASET, ShotRef
from image_analysis.base import ImageAnalyzer


def _write_stack(tmp_path, n=3):
    path = tmp_path / "UC_Cam.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            FRAMES_DATASET,
            data=np.stack([np.full((4, 5), i, dtype=np.uint16) for i in range(n)]),
            chunks=(1, 4, 5),
        )
        f.create_dataset(TIMESTAMPS_DATASET, data=np.arange(n) + 1000.0)
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


def _write_trace_stack(tmp_path, *, frames, x0=0.0, dx=2.5e-9):
    """An ARRAY capture stack, as the PVA gateway's file plugin writes one."""
    path = tmp_path / "U_ICT.h5"
    prefix = "/entry/instrument/NDAttributes/u_ict-hdf-scopetrace_channel0"
    n = len(frames)
    with h5py.File(path, "w") as f:
        f.create_dataset(FRAMES_DATASET, data=np.asarray(frames, dtype=float))
        f.create_dataset(f"{prefix}-frame_acq_timestamp", data=np.arange(n) + 1000.0)
        f.create_dataset(f"{prefix}-wave_x0", data=np.full(n, x0))
        f.create_dataset(f"{prefix}-wave_dx", data=np.full(n, dx))
        f.create_dataset(
            f"{prefix}-wave_samples", data=np.full(n, float(len(frames[0])))
        )
    return path


def test_a_line_analyzer_reads_a_scan_s_scope_trace(tmp_path) -> None:
    """The whole 1D path, unchanged: schema config -> reader -> Nx2 volts vs seconds.

    A Bluesky scan captures scope traces into one stack per device rather
    than one file per shot, so the only thing the analyzer is handed
    differently is a ShotRef.  Nothing above ``read_1d_data`` learns a new
    concept — which is the point of routing the stack through the same
    ``Data1DType``.
    """
    from geecs_schemas.analysis.processing_1d import Data1DLoading, Line1DConfig
    from image_analysis.analyzers.line_analyzer import LineAnalyzer

    volts = np.array([[0.0, 0.5, 1.0, 0.5], [0.0, 0.1, 0.2, 0.1]])
    path = _write_trace_stack(tmp_path, frames=volts, x0=-1e-9, dx=2.5e-9)

    analyzer = LineAnalyzer(
        Line1DConfig(
            description="scope trace out of a capture stack",
            data_loading=Data1DLoading(data_type="pva_stack"),
        )
    )
    data = analyzer.load_image(ShotRef(path, 1))

    np.testing.assert_allclose(
        data[:, 0], [-1e-9, 1.5e-9, 4e-9, 6.5e-9], rtol=0, atol=1e-21
    )
    np.testing.assert_array_equal(data[:, 1], volts[1])
    assert analyzer.data_metadata["x_units"] == "s"


def test_the_frame_index_survives_into_the_auxiliary_data(tmp_path) -> None:
    """``aux["file_path"]`` must stay a ShotRef, or a downstream re-read gets frame 0."""
    from geecs_schemas.analysis.processing_1d import Data1DLoading, Line1DConfig
    from image_analysis.analyzers.line_analyzer import LineAnalyzer

    path = _write_trace_stack(tmp_path, frames=np.array([[1.0, 2.0], [3.0, 4.0]]))
    analyzer = LineAnalyzer(
        Line1DConfig(
            description="scope trace out of a capture stack",
            data_loading=Data1DLoading(data_type="pva_stack"),
        )
    )
    captured = {}
    analyzer.analyze_image = lambda data, aux: captured.update(aux) or {}

    analyzer.analyze_image_file(ShotRef(path, 1))

    assert getattr(captured["file_path"], "shot_index", None) == 1
