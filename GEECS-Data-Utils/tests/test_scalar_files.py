"""Scalar sinks preserve other writers' data and never create parent folders."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import time

import numpy as np
import pandas as pd
import pytest

from geecs_data_utils.scalar_files import (
    merge_sfile,
    merge_updates,
    prepare_updates,
    write_scalar_sidecar,
)


def test_normalization_owns_data_and_retains_first_case_variant():
    data = pd.DataFrame(
        [[2, 99, 5], [np.nan, 88, 7]], columns=["shotnumber", "Shotnumber", "value"]
    )
    original = data.copy(deep=True)
    result = prepare_updates(data)
    assert result.to_dict("list") == {"Shotnumber": [2], "value": [5]}
    pd.testing.assert_frame_equal(data, original)


def test_identically_named_key_columns_do_not_survive_normalization():
    data = pd.DataFrame([[2, 99, 5]], columns=["Shotnumber", "Shotnumber", "value"])
    assert prepare_updates(data).to_dict("list") == {"Shotnumber": [2], "value": [5]}


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        pd.DataFrame(),
        pd.DataFrame({"other": [1]}),
        pd.DataFrame({"Shotnumber": [np.nan]}),
    ],
)
def test_invalid_updates_are_not_written(tmp_path, data):
    assert merge_sfile(tmp_path / "s1.txt", data) is None
    assert write_scalar_sidecar(tmp_path / "sidecar.txt", data) is None
    assert not list(tmp_path.iterdir())


def test_merge_preserves_cells_column_order_and_missing_value_policy():
    current = pd.DataFrame(
        {"Shotnumber": [3, 1, 2], "motor": [30, 10, 20], "beam": [3, 1, 2]}
    )
    updates = pd.DataFrame(
        {"Shotnumber": [2, 4, 2], "beam": [9, 4, np.nan], "spectrum": [4, 8, 6]}
    )
    original = current.copy(deep=True)
    update_copy = updates.copy(deep=True)
    merged = merge_updates(current, updates).set_index("Shotnumber")
    assert list(merged.index) == [1, 2, 3, 4]
    assert list(merged.columns) == ["motor", "beam", "spectrum"]
    assert merged.loc[2, "beam"] == 2  # Missing update keeps the old finite cell.
    assert merged.loc[2, "spectrum"] == 6
    assert merged.loc[3, "motor"] == 30
    assert merged.loc[4, "beam"] == 4
    pd.testing.assert_frame_equal(current, original)
    pd.testing.assert_frame_equal(updates, update_copy)


def test_sfile_and_sidecar_deliberately_treat_nan_updates_differently(tmp_path):
    sfile, sidecar = tmp_path / "s1.txt", tmp_path / "Scan001_beam.txt"
    pd.DataFrame({"Shotnumber": [1], "beam": [7.0]}).to_csv(
        sfile, sep="\t", index=False
    )
    updates = pd.DataFrame({"Shotnumber": [1, 1], "beam": [5.0, np.nan]})
    merged = merge_sfile(sfile, updates)
    assert merged.loc[0, "beam"] == 7.0
    assert write_scalar_sidecar(sidecar, updates) == sidecar
    assert np.isnan(pd.read_csv(sidecar, sep="\t").loc[0, "beam"])
    assert not sfile.with_suffix(".txt.lock").exists()


def test_new_sfile_uses_last_duplicate_and_sorts_identity(tmp_path):
    result = merge_sfile(
        tmp_path / "s1.txt", pd.DataFrame({"Shotnumber": [2, 1, 1], "x": [20, 9, 10]})
    )
    assert result.to_dict("list") == {"Shotnumber": [1, 2], "x": [10, 20]}


def test_existing_lock_is_never_broken(tmp_path):
    path = tmp_path / "s1.txt"
    lock = path.with_suffix(".txt.lock")
    lock.write_text("other owner")
    assert (
        merge_sfile(path, pd.DataFrame({"Shotnumber": [1], "x": [1]}), timeout=0)
        is None
    )
    assert lock.read_text() == "other owner"
    assert not path.exists()


@pytest.mark.parametrize("contents", ["", "bad\tcolumns\n1\t2\n"])
def test_unreadable_or_keyless_existing_sfile_is_not_overwritten(tmp_path, contents):
    path = tmp_path / "s1.txt"
    path.write_text(contents)
    assert merge_sfile(path, pd.DataFrame({"Shotnumber": [1], "x": [1]})) is None
    assert path.read_text() == contents
    assert not path.with_suffix(".txt.lock").exists()


def test_write_failure_releases_our_lock(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    path = tmp_path / "s1.txt"
    with pytest.raises(OSError, match="simulated"):
        merge_sfile(path, pd.DataFrame({"Shotnumber": [1], "x": [1]}))
    assert not path.with_suffix(".txt.lock").exists()


def test_missing_parent_is_never_created(tmp_path):
    path = tmp_path / "scans" / "Scan001" / "s1.txt"
    updates = pd.DataFrame({"Shotnumber": [1], "x": [1]})
    assert merge_sfile(path, updates) is None
    with pytest.raises(OSError):
        write_scalar_sidecar(path, updates)
    assert not list(tmp_path.iterdir())


def test_concurrent_analyzers_preserve_both_sets_of_columns(tmp_path, monkeypatch):
    path = tmp_path / "s1.txt"
    pd.DataFrame({"Shotnumber": [1, 2], "motor": [10, 20]}).to_csv(
        path, sep="\t", index=False
    )
    reader = pd.read_csv
    start = Barrier(2)

    def slow_read(*args, **kwargs):
        result = reader(*args, **kwargs)
        time.sleep(0.03)
        return result

    monkeypatch.setattr(pd, "read_csv", slow_read)

    def run(name):
        start.wait(timeout=2)
        return merge_sfile(
            path, pd.DataFrame({"Shotnumber": [1], name: [5]}), interval=0.005
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["beam", "spectrum"]))
    assert all(result is not None for result in results)
    saved = reader(path, sep="\t").set_index("Shotnumber")
    assert saved.loc[1, "beam"] == saved.loc[1, "spectrum"] == 5
    assert saved.loc[2, "motor"] == 20
