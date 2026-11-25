"""The LabVIEW-waveform HDF5 scope reader (``tek_scope_hdf5``) and its extra channels."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from geecs_data_utils.io.array1d import Data1DConfig, Data1DType, read_1d_data

h5py = pytest.importorskip("h5py")

N = 50


def _write(path, *, increments=(5e-11,) * 4, lengths=(N,) * 4) -> None:
    """A 4-channel scope file: channel k holds raw counts k*1000 + i, volts = raw*0.01 + k."""
    with h5py.File(path, "w") as f:
        axes = f.create_group("wfm_group0/axes")
        for k in range(4):
            t = axes.create_group(f"axis{2 * k}")
            t.attrs["start"] = 0.0
            t.attrs["increment"] = increments[k]
            v = axes.create_group(f"axis{2 * k + 1}")
            v.create_dataset(
                "data_vector/data", data=k * 1000 + np.arange(lengths[k], dtype="i2")
            )
            v.create_dataset("scale_coef", data=np.array([float(k), 0.01]))


def _volts(k: int) -> np.ndarray:
    return (k * 1000 + np.arange(N)) * 0.01 + k


def test_reads_the_requested_channel_in_volts(tmp_path):
    path = tmp_path / "s.dat"
    _write(path)
    result = read_1d_data(path, Data1DConfig(data_type="tek_scope_hdf5", trace_index=3))
    np.testing.assert_allclose(result.data[:, 0], np.arange(N) * 5e-11)
    np.testing.assert_allclose(result.data[:, 1], _volts(3))
    assert (result.x_units, result.y_units) == ("s", "V")
    assert result.auxiliary_column_data == {}


def test_extra_channels_are_read_by_trace_index(tmp_path):
    path = tmp_path / "s.dat"
    _write(path)
    config = Data1DConfig(
        data_type=Data1DType.TEK_SCOPE_HDF5,
        trace_index=3,
        auxiliary_columns={"photodiode": 1},
    )
    result = read_1d_data(path, config)
    np.testing.assert_allclose(result.data[:, 1], _volts(3))
    np.testing.assert_allclose(result.auxiliary_column_data["photodiode"], _volts(1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"increments": (5e-11, 1e-10, 5e-11, 5e-11)},
        {"lengths": (N, N - 1, N, N)},
    ],
)
def test_an_extra_channel_on_another_time_axis_is_refused(tmp_path, kwargs):
    path = tmp_path / "s.dat"
    _write(path, **kwargs)
    config = Data1DConfig(
        data_type="tek_scope_hdf5", trace_index=3, auxiliary_columns={"pd": 1}
    )
    with pytest.raises(ValueError, match="not sampled like"):
        read_1d_data(path, config)


def test_a_channel_past_the_file_is_refused(tmp_path):
    path = tmp_path / "s.dat"
    _write(path)
    config = Data1DConfig(
        data_type="tek_scope_hdf5", trace_index=0, auxiliary_columns={"pd": 4}
    )
    with pytest.raises(ValueError, match="only contains 4 traces"):
        read_1d_data(path, config)


def test_a_scope_extra_channel_is_checked_against_the_trace_not_the_columns():
    # Index 1 is y_column's default, but for a scope file it is a channel.
    Data1DConfig(data_type="tek_scope_hdf5", trace_index=3, auxiliary_columns={"pd": 1})
    with pytest.raises(ValidationError, match="measured trace"):
        Data1DConfig(
            data_type="tek_scope_hdf5", trace_index=3, auxiliary_columns={"pd": 3}
        )
    with pytest.raises(ValidationError, match="measured trace"):
        Data1DConfig(data_type="tsv", auxiliary_columns={"w": 1})
