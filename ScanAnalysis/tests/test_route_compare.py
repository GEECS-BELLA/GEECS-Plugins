"""One definition of equal route outputs: exact, except noscan averages by ulps."""

import h5py
import numpy as np
import pandas as pd

from scan_analysis.route_compare import compare_snapshots, snapshot_analysis_tree


def test_snapshot_decodes_hdf5_tables_and_counts_other_files(tmp_path):
    analysis = tmp_path / "analysis"
    (analysis / "Scan001" / "Out").mkdir(parents=True)
    with h5py.File(analysis / "Scan001" / "Out" / "D_1_processed.h5", "w") as handle:
        handle.create_dataset("image", data=np.arange(4.0).reshape(2, 2))
    pd.DataFrame({"Shotnumber": [1], "x": [2.5]}).to_csv(
        analysis / "s1.txt", sep="\t", index=False
    )
    (analysis / "Scan001" / "Out" / "D_1_processed_visual.png").write_bytes(b"png")
    files = snapshot_analysis_tree(analysis)
    assert sorted(files) == [
        "Scan001/Out/D_1_processed.h5",
        "Scan001/Out/D_1_processed_visual.png",
        "s1.txt",
    ]
    kind, key, data, dtype = files["Scan001/Out/D_1_processed.h5"]
    assert (kind, key, dtype) == ("h5", "image", np.dtype("float64"))
    np.testing.assert_array_equal(data, np.arange(4.0).reshape(2, 2))
    assert files["s1.txt"][0] == "table" and list(files["s1.txt"][1]["x"]) == [2.5]
    assert files["Scan001/Out/D_1_processed_visual.png"] == ("file", 3)


def test_compare_reports_every_kind_of_difference():
    frame = pd.DataFrame({"Shotnumber": [1, 2], "x": [1.0, 2.0]})
    ones = np.ones((2, 2))
    legacy = {
        "a.h5": ("h5", "image", ones, ones.dtype),
        "b.h5": ("h5", "image", ones, ones.dtype),
        "c.h5": ("h5", "image", ones, ones.dtype),
        "D_average_processed.h5": ("h5", "image", ones, ones.dtype),
        "s7.txt": ("table", frame),
        "only_legacy.png": ("file", 10),
    }
    core = {
        "a.h5": ("h5", "image", ones, ones.dtype),
        "b.h5": ("h5", "image", ones + 1e-9, ones.dtype),
        "c.h5": ("h5", "data", ones, ones.dtype),
        "D_average_processed.h5": ("h5", "image", ones * (1 + 1e-16), ones.dtype),
        "s7.txt": ("table", frame.assign(x=[1.0, 3.0])),
        "only_core.png": ("file", 99),
    }
    problems = compare_snapshots(legacy, core, average_ulps=4)
    assert problems[:2] == [
        "only legacy wrote only_legacy.png",
        "only core wrote only_core.png",
    ]
    assert any(p.startswith("b.h5: arrays differ") for p in problems)
    assert any(p.startswith("c.h5: image/") for p in problems)
    assert not any(p.startswith("D_average_processed") for p in problems)
    assert any(p.startswith("s7.txt:") for p in problems)
    assert len(problems) == 5
    assert compare_snapshots(legacy, legacy) == []


def test_average_tolerance_is_a_few_ulps_of_the_stored_dtype():
    base = np.full((3, 3), 5.0, dtype=np.float32)
    one_ulp = np.nextafter(base, np.float32(6.0))
    far = base * np.float32(1 + 1e-5)

    def entry(data):
        return {"D_average_processed.h5": ("h5", "image", data, data.dtype)}

    assert compare_snapshots(entry(base), entry(one_ulp)) == []
    assert compare_snapshots(entry(base), entry(far)) != []
    assert compare_snapshots(entry(base), entry(one_ulp), average_ulps=0) != []
