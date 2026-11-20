"""The Thorlabs FTS export reader (``read_1d_data`` with ``thorlabs_fts``)."""

from __future__ import annotations

import numpy as np
import pytest

from geecs_data_utils.io.array1d import (
    Data1DConfig,
    Data1DType,
    is_thorlabs_fts,
    read_1d_data,
)


def _write(path, body: str) -> None:
    path.write_bytes(body.replace("\n", "\r\n").encode())


HEADER = "#Thorlabs FTS\n[SpectrumHeader]\n#XAxisUnit;nm_vac\n#YAxisUnit;intensity\n"


def test_reads_the_data_block_and_names_the_axes(tmp_path):
    path = tmp_path / "s.csv"
    _write(path, HEADER + "[Data]\n7.9e+02;1.0e+01\n7.905e+02;2.0e+01\n[EndOfFile]\n")
    assert is_thorlabs_fts(path)
    result = read_1d_data(path, Data1DConfig(data_type=Data1DType.THORLABS_FTS))
    np.testing.assert_allclose(result.data, [[790.0, 10.0], [790.5, 20.0]])
    assert (result.x_label, result.x_units) == ("Wavelength (vacuum)", "nm")
    assert result.y_label == "intensity"


def test_a_file_with_no_data_rows_is_refused(tmp_path):
    path = tmp_path / "s.csv"
    _write(path, HEADER + "[Data]\n[EndOfFile]\n")
    with pytest.raises(ValueError, match="no \\[Data\\] rows"):
        read_1d_data(path, Data1DConfig(data_type=Data1DType.THORLABS_FTS))


def test_a_bad_data_row_is_refused(tmp_path):
    path = tmp_path / "s.csv"
    _write(path, HEADER + "[Data]\n7.9e+02;oops\n[EndOfFile]\n")
    with pytest.raises(ValueError, match="bad data row"):
        read_1d_data(path, Data1DConfig(data_type=Data1DType.THORLABS_FTS))


def test_other_files_are_not_fts(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("wavelength,counts\n1,2\n")
    assert not is_thorlabs_fts(path)
    assert not is_thorlabs_fts(tmp_path / "absent.csv")


def test_auxiliary_columns_are_read_or_refused(tmp_path):
    path = tmp_path / "s.csv"
    _write(
        path, HEADER + "[Data]\n7.9e+02;1.0e+01;3\n7.905e+02;2.0e+01;4\n[EndOfFile]\n"
    )
    result = read_1d_data(
        path,
        Data1DConfig(data_type=Data1DType.THORLABS_FTS, auxiliary_columns={"dark": 2}),
    )
    np.testing.assert_allclose(result.auxiliary_column_data["dark"], [3, 4])
    with pytest.raises(ValueError, match="requested columns"):
        read_1d_data(
            path,
            Data1DConfig(data_type=Data1DType.THORLABS_FTS, auxiliary_columns={"w": 4}),
        )
