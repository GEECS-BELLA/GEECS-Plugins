"""Scan conditions: the labels file and the per-scan summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geecs_data_utils.scan_conditions import (
    labels_path,
    read_condition_labels,
    scan_conditions,
    write_condition_labels,
)

LABELS = {"target": ["Al foil", "Cu foil", "nothing"]}


def test_write_then_read(tmp_path):
    path = labels_path(tmp_path, "Scan005", "beamline")
    assert path.name == "Scan005_beamline_labels.json"
    write_condition_labels(path, "beamline", LABELS)
    assert read_condition_labels(tmp_path) == LABELS
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no .part left


def test_the_writer_never_creates_a_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_condition_labels(tmp_path / "missing" / "x_labels.json", "j", LABELS)
    assert not (tmp_path / "missing").exists()


def test_the_reader_is_tolerant(tmp_path):
    assert read_condition_labels(tmp_path / "nope") == {}
    (tmp_path / "Scan001_bad_labels.json").write_text("{not json")
    (tmp_path / "Scan001_odd_labels.json").write_text('{"columns": []}')
    write_condition_labels(tmp_path / "Scan001_ok_labels.json", "ok", LABELS)
    assert read_condition_labels(tmp_path) == LABELS


def test_one_label_mixed_and_unknown():
    frame = pd.DataFrame(
        {"target": [0, 0, 0], "door": [1, 0, np.nan], "gone": [np.nan] * 3}
    )
    found = scan_conditions(
        frame,
        {**LABELS, "door": ["open", "shut"], "gone": ["a"], "absent": ["x"]},
    )
    by = {c.column: c for c in found}
    assert set(by) == {"target", "door", "gone"}  # a column the scan lacks: none
    assert by["target"].label == "Al foil" and not by["target"].mixed
    assert by["target"].detail == ""
    assert by["door"].label == "mixed"
    assert by["door"].detail == "open 1 · shut 1 · unknown 1"
    assert by["gone"].label == "unknown"


def test_a_number_without_a_label():
    (cond,) = scan_conditions(pd.DataFrame({"target": [7, 7]}), LABELS)
    assert cond.label == "value 7"
