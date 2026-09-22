"""Reading one shot of an ARRAY capture stack as x-vs-y (io/array1d.py + io/scan_stack.py).

The PVA gateway serves images and arrays through one file plugin, so a
scan folder can now hold three kinds of stack.  These tests pin the two
rules that keep the array kinds honest: the stack says which kind it is
(the plugin's own ``wave_*`` attributes, not a shape guess), and a frame
reaches a consumer at its TRUE length, never at the padded one.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from geecs_data_utils.io import (
    Data1DConfig,
    Data1DType,
    ShotRef,
    read_1d_data,
    stack_content_kind,
)

PVA_STACK = Data1DConfig(data_type=Data1DType.PVA_STACK)


def _write_stack(path, frames, *, device, variable, axis=None):
    """A stack as the file plugin writes it: frames + per-frame attributes.

    *axis* is the per-frame ``(x0, dx, samples)`` triple of an array
    stack; ``None`` writes no ``wave_*`` attribute at all, which is how
    an image stack looks.
    """
    frames = np.asarray(frames)
    prefix = f"/entry/instrument/NDAttributes/{device}-hdf-{variable}"
    with h5py.File(path, "w") as f:
        f.create_dataset("/entry/data/data", data=frames)
        f.create_dataset(
            f"{prefix}-frame_acq_timestamp", data=np.arange(len(frames)) + 1000.0
        )
        if axis is not None:
            x0, dx, samples = (np.asarray(a, dtype=float) for a in zip(*axis))
            f.create_dataset(f"{prefix}-wave_x0", data=x0)
            f.create_dataset(f"{prefix}-wave_dx", data=dx)
            f.create_dataset(f"{prefix}-wave_samples", data=samples)
    return path


def _waveform_stack(tmp_path, frames, axis):
    return _write_stack(
        tmp_path / "scope.h5",
        frames,
        device="u_ict",
        variable="scopetrace_channel0",
        axis=axis,
    )


def _lineout_stack(tmp_path, frames):
    """A lineout stack: the axis rides in column 0, so wave_* is all NaN."""
    return _write_stack(
        tmp_path / "spec.h5",
        frames,
        device="u_spec",
        variable="interpspec",
        axis=[(np.nan, np.nan, np.nan)] * len(frames),
    )


def _padded(rows_per_shot, ceiling, columns=2):
    """Lineout frames of the given true lengths, NaN-padded to *ceiling* rows."""
    frames = np.full((len(rows_per_shot), ceiling, columns), np.nan)
    for shot, rows in enumerate(rows_per_shot):
        frames[shot, :rows] = (
            np.arange(rows * columns, dtype=float).reshape(rows, columns) + shot
        )
    return frames


# --------------------------------------------------------------------------
# What the stack says it is
# --------------------------------------------------------------------------


def test_an_image_stack_is_classified_by_the_absence_of_the_axis_attributes(tmp_path):
    """Rank alone cannot tell (N, H, W) pixels from (N, n, 2) rows — the plugin's attributes can."""
    path = _write_stack(
        tmp_path / "cam.h5",
        np.zeros((3, 4, 2)),  # deliberately two columns wide: shaped like a lineout
        device="uc_cam",
        variable="image",
    )
    assert stack_content_kind(path) == "image"


def test_array_kinds_are_classified_by_frame_rank(tmp_path):
    assert (
        stack_content_kind(
            _waveform_stack(tmp_path, np.zeros((2, 6)), [(0.0, 1e-9, 6)] * 2)
        )
        == "waveform"
    )
    assert stack_content_kind(_lineout_stack(tmp_path, _padded([2, 2], 4))) == "lineout"


def test_an_array_stack_of_another_shape_is_refused(tmp_path):
    """An array stack is (n,) or (n, 2); anything else is not something to guess at."""
    path = _write_stack(
        tmp_path / "odd.h5",
        np.zeros((2, 4, 3)),
        device="u_dev",
        variable="thing",
        axis=[(np.nan, np.nan, np.nan)] * 2,
    )
    with pytest.raises(ValueError, match="neither a waveform"):
        stack_content_kind(path)


# --------------------------------------------------------------------------
# Waveforms: values on the wire, the axis in two numbers
# --------------------------------------------------------------------------


def test_a_waveform_axis_is_rebuilt_from_x0_and_dx(tmp_path):
    volts = np.array([[0.1, 0.2, 0.3, 0.4], [1.1, 1.2, 1.3, 1.4]])
    path = _waveform_stack(tmp_path, volts, [(-2e-9, 4e-9, 4)] * 2)

    result = read_1d_data(ShotRef(path, 1), PVA_STACK)

    np.testing.assert_allclose(
        result.data[:, 0], [-2e-9, 2e-9, 6e-9, 10e-9], rtol=0, atol=1e-21
    )
    np.testing.assert_array_equal(result.data[:, 1], volts[1])
    # Seconds is the wire format's definition of x0/dx, so the reader may
    # name it; every other unit on this path rides in the analyzer config.
    assert (result.x_units, result.x_label) == ("s", "Time")
    assert result.y_units is None
    # The file is the only place a consumer can learn WHICH variable the
    # folder's stack holds.
    assert result.y_label == "scopetrace_channel0"


def test_wave_samples_shortens_a_padded_waveform(tmp_path):
    """A waveform stack with a ceiling pads like a lineout; wave_samples says where the record ends."""
    frames = np.array([[1.0, 2.0, 3.0, np.nan, np.nan]])
    path = _waveform_stack(tmp_path, frames, [(0.0, 1.0, 3)])

    result = read_1d_data(ShotRef(path, 0), PVA_STACK)

    np.testing.assert_array_equal(result.data[:, 1], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.data[:, 0], [0.0, 1.0, 2.0])


def test_the_declared_record_length_rules_over_the_frame(tmp_path):
    """A waveform says how long its record is; the reader must not re-derive it.

    Sniffing where the NaN pad starts would agree here only by accident —
    these samples are all real, and only ``wave_samples`` says the record
    ends at three.
    """
    path = _waveform_stack(
        tmp_path, np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), [(0.0, 1.0, 3)]
    )

    result = read_1d_data(ShotRef(path, 0), PVA_STACK)

    np.testing.assert_array_equal(result.data[:, 1], [1.0, 2.0, 3.0])


def test_a_declared_record_covering_padding_is_refused(tmp_path):
    """wave_samples and the frame disagree — neither is quietly believed."""
    path = _waveform_stack(
        tmp_path, np.array([[1.0, 2.0, np.nan, np.nan]]), [(0.0, 1.0, 4)]
    )
    with pytest.raises(ValueError, match="contains padding"):
        read_1d_data(ShotRef(path, 0), PVA_STACK)


def test_wave_samples_longer_than_the_frame_is_refused(tmp_path):
    path = _waveform_stack(tmp_path, np.ones((1, 4)), [(0.0, 1.0, 9)])
    with pytest.raises(ValueError, match="wave_samples says 9"):
        read_1d_data(ShotRef(path, 0), PVA_STACK)


def test_a_waveform_without_an_axis_falls_back_to_the_sample_index(tmp_path):
    """Unlabelled sample numbers, never a time axis of NaNs."""
    path = _waveform_stack(tmp_path, np.array([[5.0, 6.0, 7.0]]), [(np.nan, np.nan, 3)])

    result = read_1d_data(ShotRef(path, 0), PVA_STACK)

    np.testing.assert_array_equal(result.data[:, 0], [0.0, 1.0, 2.0])
    assert result.x_units is None and result.x_label is None


# --------------------------------------------------------------------------
# Lineouts: the axis is column 0, and the padding must not survive the read
# --------------------------------------------------------------------------


def test_a_lineout_is_split_into_its_axis_and_its_values(tmp_path):
    frames = np.full((1, 4, 2), np.nan)
    frames[0, :3] = [[100.0, 7.0], [101.0, 8.0], [102.0, 9.0]]
    path = _lineout_stack(tmp_path, frames)

    result = read_1d_data(ShotRef(path, 0), PVA_STACK)

    np.testing.assert_array_equal(result.data[:, 0], [100.0, 101.0, 102.0])
    np.testing.assert_array_equal(result.data[:, 1], [7.0, 8.0, 9.0])
    assert result.y_label == "interpspec"


def test_trimming_the_padding_restores_the_shape_guard(tmp_path):
    """The reason un-padding is not cosmetic.

    The gateway pads every frame to the devicetype ceiling, so two shots
    whose spectra are different lengths arrive SAME-SHAPED.  A consumer
    that averages per-shot lineouts guards on shape (ScanAnalysis'
    ``average_data`` returns None for inhomogeneous shapes) — and on
    padded frames that guard passes, averaging column 1 index-wise over
    axes that do not line up.  Read at their true lengths, the shapes
    differ again and the guard bites.
    """
    frames = _padded([3, 5], ceiling=8)
    path = _lineout_stack(tmp_path, frames)

    with h5py.File(path, "r") as f:
        raw = f["/entry/data/data"]
        assert raw[0].shape == raw[1].shape  # the trap: padding hides the difference

    shapes = {
        read_1d_data(ShotRef(path, shot), PVA_STACK).data.shape for shot in (0, 1)
    }
    assert shapes == {(3, 2), (5, 2)}


def test_padding_between_values_is_refused(tmp_path):
    """A hole is a frame written wrong — never silently shortened to the hole."""
    frames = np.full((1, 5, 2), np.nan)
    frames[0, 0] = [1.0, 2.0]
    frames[0, 3] = [3.0, 4.0]
    path = _lineout_stack(tmp_path, frames)

    with pytest.raises(ValueError, match="padding between values"):
        read_1d_data(ShotRef(path, 0), PVA_STACK)


def test_an_all_padding_frame_is_refused(tmp_path):
    path = _lineout_stack(tmp_path, np.full((1, 5, 2), np.nan))
    with pytest.raises(ValueError, match="entirely padding"):
        read_1d_data(ShotRef(path, 0), PVA_STACK)


# --------------------------------------------------------------------------
# Refusals at the seam
# --------------------------------------------------------------------------


def test_an_image_stack_is_not_x_vs_y(tmp_path):
    path = _write_stack(
        tmp_path / "cam.h5", np.zeros((2, 4, 5)), device="uc_cam", variable="image"
    )
    with pytest.raises(ValueError, match="image stack is pixels"):
        read_1d_data(ShotRef(path, 0), PVA_STACK)


def test_a_plain_path_carries_no_frame_index(tmp_path):
    path = _waveform_stack(tmp_path, np.ones((2, 3)), [(0.0, 1.0, 3)] * 2)
    with pytest.raises(ValueError, match="must travel with the path"):
        read_1d_data(path, PVA_STACK)


def test_a_frame_index_outside_the_stack_is_refused(tmp_path):
    path = _waveform_stack(tmp_path, np.ones((2, 3)), [(0.0, 1.0, 3)] * 2)
    with pytest.raises(IndexError, match="outside stack of 2 frames"):
        read_1d_data(ShotRef(path, 2), PVA_STACK)


def test_auxiliary_columns_are_not_a_stack_concept(tmp_path):
    path = _lineout_stack(tmp_path, _padded([2], 4))
    config = Data1DConfig(
        data_type=Data1DType.PVA_STACK, auxiliary_columns={"charge": 3}
    )
    with pytest.raises(ValueError, match="not supported for pva_stack"):
        read_1d_data(ShotRef(path, 0), config)
