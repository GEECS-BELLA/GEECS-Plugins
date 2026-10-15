"""Tables are read-only string columns; the reader keeps the legacy DictReader rows."""

import pickle

import numpy as np
import pytest

from geecs_data_utils.io import read_table
from geecs_data_utils.tables import Table


def test_table_owns_its_cells_and_converts_on_request():
    table = Table(("cam number", "FOV [mm]"), [[1, "10.5"], ("2", 20)])
    assert table.rows == (("1", "10.5"), ("2", "20")) and len(table) == 2
    np.testing.assert_array_equal(table.floats("FOV [mm]"), [10.5, 20.0])
    assert table.strings("cam number") == ("1", "2")
    assert dict(table.row(1)) == {"cam number": "2", "FOV [mm]": "20"}
    with pytest.raises(TypeError):
        table.row(0)["cam number"] = "9"
    with pytest.raises(KeyError, match="no column 'set'"):
        table.floats("set")
    assert pickle.loads(pickle.dumps(table)) == table


@pytest.mark.parametrize(
    "columns, rows",
    [((), []), (("a", "a"), []), (("a", "b"), [["1"]])],
)
def test_malformed_tables_are_refused(columns, rows):
    with pytest.raises(ValueError):
        Table(columns, rows)


def test_read_table_keeps_cells_as_strings_and_skips_blank_lines(tmp_path):
    path = tmp_path / "cam.txt"
    path.write_text(
        "cam number\tscreen\tsensitivity\n1\tback\t0.36974\n\n2\tfront\t1.21\n"
    )
    table = read_table(path)
    assert table.columns == ("cam number", "screen", "sensitivity")
    assert table.rows == (("1", "back", "0.36974"), ("2", "front", "1.21"))
    np.testing.assert_array_equal(table.floats("sensitivity"), [0.36974, 1.21])
    path.write_text("a\tb\n1\t2\t3\n")
    with pytest.raises(ValueError, match="line 2: 3 cells for 2 columns"):
        read_table(path)
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_table(path)
