"""One definition of equal route outputs: exact, except noscan averages by ulps."""

import h5py
import numpy as np
import pandas as pd

from scan_analysis.route_compare import (
    compare_snapshots,
    read_golden,
    snapshot_analysis_tree,
    snapshot_scan_products,
    write_golden,
)


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


def test_nonuniform_axis_allowance_excludes_only_the_width_columns():
    """#1029: rms/fwhm may differ between routes on a nonuniform axis; nothing else may."""
    frame = pd.DataFrame(
        {
            "Shotnumber": [1, 2],
            "D_rms": [1.0, 2.0],
            "D_fwhm": [3.0, 4.0],
            "D_CoM": [5.0, 6.0],
        }
    )
    legacy = {"s7.txt": ("table", frame)}
    widths = {"s7.txt": ("table", frame.assign(D_rms=[1.5, 2.5], D_fwhm=[3.5, 4.5]))}
    assert compare_snapshots(legacy, widths) != []
    assert compare_snapshots(legacy, widths, nonuniform_axis=True) == []
    centroid = {"s7.txt": ("table", frame.assign(D_CoM=[5.5, 6.5]))}
    assert compare_snapshots(legacy, centroid, nonuniform_axis=True) != []
    missing = {"s7.txt": ("table", frame.drop(columns=["D_fwhm"]))}
    problems = compare_snapshots(legacy, missing, nonuniform_axis=True)
    assert problems == ["s7.txt: width columns ['D_rms', 'D_fwhm'] vs ['D_rms']"]


def test_average_tolerance_is_a_few_ulps_of_the_stored_dtype():
    base = np.full((3, 3), 5.0, dtype=np.float32)
    one_ulp = np.nextafter(base, np.float32(6.0))
    far = base * np.float32(1 + 1e-5)

    def entry(data):
        return {"D_average_processed.h5": ("h5", "image", data, data.dtype)}

    assert compare_snapshots(entry(base), entry(one_ulp)) == []
    assert compare_snapshots(entry(base), entry(far)) != []
    assert compare_snapshots(entry(base), entry(one_ulp), average_ulps=0) != []


def test_scan_products_decode_spectra_count_images_and_skip_inputs(tmp_path):
    trees = {}
    for route, (payload, charge) in (
        ("legacy", (b"\x00\x01", "1.5")),
        ("core", (b"\x00\x02", "1.5000001")),
    ):
        scan = tmp_path / route / "Scan003"
        (scan / "Cam").mkdir(parents=True)
        (scan / "Cam" / "raw.png").write_bytes(route.encode())  # an input
        (scan / "ScanInfoScan003.ini").write_text(route)
        (scan / "Cam-interp").mkdir()
        (scan / "Cam-interp" / "raw.png").write_bytes(payload)
        (scan / "Cam-interpSpec").mkdir()
        (scan / "Cam-interpSpec" / "raw.tsv").write_text(
            f"Energy [MeV]\tCharge\n10.0\t{charge}\n"
        )
        (scan / "Cam-other").mkdir()
        (scan / "Cam-other" / "blob.bin").write_bytes(payload)
        trees[route] = snapshot_scan_products(scan, {"Cam", "ScanInfoScan003.ini"})
    assert sorted(trees["core"]) == [
        "Scan003/Cam-interp/raw.png",
        "Scan003/Cam-interpSpec/raw.tsv",
        "Scan003/Cam-other/blob.bin",
    ]
    assert trees["core"]["Scan003/Cam-interp/raw.png"][0] == "file"
    assert trees["core"]["Scan003/Cam-interpSpec/raw.tsv"][0] == "table"
    # the image by presence, the spectrum as numbers, raw bytes exactly
    problems = compare_snapshots(trees["legacy"], trees["core"])
    assert problems == [
        'Scan003/Cam-interpSpec/raw.tsv: column "Charge": Series are different',
        "Scan003/Cam-other/blob.bin: bytes differ",
    ]
    assert compare_snapshots(trees["legacy"], trees["core"], rtol=1e-6) == [
        "Scan003/Cam-other/blob.bin: bytes differ"
    ]


def test_the_golden_tolerance_is_relative_to_scale_and_bites_a_1e4_change():
    """GOLDEN_RTOL: cross-platform drift passes; a x1.0001 core change does not."""
    from scan_analysis.route_compare import GOLDEN_RTOL

    image = np.array([[0.0, 1000.0], [2500.0, np.nan]])
    table = pd.DataFrame({"Shotnumber": [1, 2], "x": [0.0, 3000.0], "k": ["a", "b"]})

    def tree(array, frame):
        return {
            "D_1_processed.h5": ("h5", "image", array, array.dtype),
            "s1.txt": ("table", frame),
        }

    base = tree(image, table)
    # last-digit drift (and a zero beside large values): inside the allowance
    drift = tree(
        image + np.array([[1e-13, 1e-9], [-2e-9, 0.0]]),
        table.assign(x=[1e-12, 3000.0 + 1e-9]),
    )
    assert compare_snapshots(base, drift, rtol=GOLDEN_RTOL) == []
    assert compare_snapshots(base, drift) != []
    # the bite: a x1.0001 change is 1e-4 relative, a hundred times the allowance
    nudged = tree(image * 1.0001, table.assign(x=[0.0, 3000.0 * 1.0001]))
    problems = compare_snapshots(base, nudged, rtol=GOLDEN_RTOL)
    assert len(problems) == 2
    assert "relative 0.0001 > rtol 1e-06" in problems[0]
    assert 'column "x" differs (relative 0.0001' in problems[1]
    # NaN placement and a non-numeric cell stay exact
    assert (
        compare_snapshots(base, tree(np.nan_to_num(image), table), rtol=GOLDEN_RTOL)
        != []
    )
    assert (
        compare_snapshots(
            base, tree(image, table.assign(k=["a", "c"])), rtol=GOLDEN_RTOL
        )
        != []
    )


def test_a_golden_round_trips_every_kind_bit_for_bit(tmp_path):
    """Arrays keep dtype and value, tables their column dtypes and text cells."""
    image = np.arange(6, dtype=np.float32).reshape(2, 3) / 7
    table = pd.DataFrame(
        {
            "Shotnumber": [1, 2, 3],
            "Bin #": [1, 1, 2],
            "x": [0.1 + 0.2, np.nan, 1e-300],
            "label": ["a", None, "c"],
            "flag": [True, False, True],
        }
    )
    snapshot = {
        "Scan001/Out/D_1_processed.h5": ("h5", "image", image, image.dtype),
        "Scan001/Out/D_1_processed_visual.png": ("file", 1234),
        "Scan001/Cam-interp/raw.png": ("bytes", "ab" * 32),
        "s1.txt": ("table", table),
    }
    write_golden(snapshot, tmp_path / "case", extra={"display": ["a.png"]})
    assert sorted(p.name for p in (tmp_path / "case").iterdir()) == [
        "arrays.npz",
        "manifest.json",
    ]
    stored = read_golden(tmp_path / "case")
    assert stored.extra == {"display": ["a.png"]}
    assert sorted(stored.files) == sorted(snapshot)
    assert compare_snapshots(stored.files, snapshot, average_ulps=0) == []
    kind, key, data, dtype = stored.files["Scan001/Out/D_1_processed.h5"]
    assert (kind, key, dtype) == ("h5", "image", np.dtype("float32"))
    assert data.dtype == np.float32
    assert stored.files["Scan001/Out/D_1_processed_visual.png"] == ("file", None)
    assert stored.files["Scan001/Cam-interp/raw.png"] == ("bytes", "ab" * 32)
    pd.testing.assert_frame_equal(stored.files["s1.txt"][1], table, check_exact=True)
    # A changed sample is reported, so the golden really bites.
    nudged = {**snapshot}
    nudged["Scan001/Out/D_1_processed.h5"] = (
        "h5",
        "image",
        np.nextafter(image, np.float32(9)),
        image.dtype,
    )
    assert compare_snapshots(stored.files, nudged) != []
