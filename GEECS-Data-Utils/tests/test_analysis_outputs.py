"""A scan's per-shot analysis outputs as a source (io/analysis_outputs.py).

``{day}/analysis/ScanNNN/<folder>/ScanNNN_<stem>_SSS.<ext>`` files are found
by name, read through the image and trace ladders, and a shot with no file
is refused, never a neighbour's. Nothing is ever created.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from geecs_data_utils.io import analysis_outputs, shot_images, shot_traces


def _spectrum(path, shot):
    """A three-column output table: momentum, charge, charge density."""
    rows = "\n".join(f"{x:.3f}\t{x * shot:.3f}\t{2 * x * shot:.3f}" for x in (1, 2, 3))
    path.write_text("Momentum_GeV/c\tCharge_fC\tChargeDen_pC/GeV/c\n" + rows + "\n")


@pytest.fixture()
def scan_folder(tmp_path):
    """Scan075 with a raw camera and an analysis tree of three output folders."""
    day = tmp_path / "Expt" / "Y2026" / "11-Nov" / "26_1103"
    scan = day / "scans" / "Scan075"
    (scan / "CAM-X").mkdir(parents=True)
    out = day / "analysis" / "Scan075"
    spec = out / "py-magspec-interpSpec"
    image = out / "py-magspec"
    mixed = out / "py-ebeam"
    for folder in (spec, image, mixed):
        folder.mkdir(parents=True)
    for shot in (1, 2, 3):
        _spectrum(spec / f"Scan075_py-magspec-interpSpec_{shot:03d}.txt", shot)
        pixels = np.full((4, 6), 100 * shot, dtype=np.uint16)
        Image.fromarray(pixels).save(image / f"Scan075_py-magspec_{shot:03d}.png")
        Image.fromarray(pixels).save(mixed / f"Scan075_py-ebeam_{shot:03d}.png")
        _spectrum(mixed / f"Scan075_py-ebeamX_{shot:03d}.txt", shot)
    (out / "Scan075_online_Summary.txt").write_text("not a folder\n")
    (spec / "Scan074_py-magspec-interpSpec_001.txt").write_text("another scan's\n")
    return scan


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_the_outputs_are_the_analysis_folders_subfolders(scan_folder):
    assert analysis_outputs.output_folders(scan_folder) == [
        "analysis:py-ebeam",
        "analysis:py-magspec",
        "analysis:py-magspec-interpSpec",
    ]


def test_a_scan_with_no_analysis_folder_has_none(tmp_path):
    scan = tmp_path / "d" / "scans" / "Scan001"
    scan.mkdir(parents=True)
    assert analysis_outputs.output_folders(scan) == []
    assert not (tmp_path / "d" / "analysis").exists()


def test_a_folder_holds_one_series_per_stem(scan_folder):
    names = [
        s.name for s in analysis_outputs.output_series(scan_folder, "analysis:py-ebeam")
    ]
    assert names == ["analysis:py-ebeam/py-ebeam", "analysis:py-ebeam/py-ebeamX"]


def test_a_bare_folder_is_its_own_series_an_image_first(scan_folder):
    found = analysis_outputs.resolve_output(scan_folder, "analysis:py-ebeam")
    assert (found.stem, found.ext) == ("py-ebeam", "png")
    found = analysis_outputs.resolve_output(
        scan_folder, "analysis:py-magspec-interpSpec"
    )
    assert (found.stem, found.ext, found.width) == ("py-magspec-interpSpec", "txt", 3)


@pytest.mark.parametrize(
    "name",
    [
        "analysis:../scans",
        "analysis:py-ebeam/../py-magspec",
        "analysis:py-ebeam/nope",
        "analysis:",
        "py-ebeam",
    ],
)
def test_a_name_outside_the_listing_resolves_to_nothing(scan_folder, name):
    assert analysis_outputs.resolve_output(scan_folder, name) is None


def test_an_image_output_reads_through_the_image_ladder(scan_folder):
    kind = shot_images.device_kind(scan_folder, "analysis:py-magspec")
    assert (kind.kind, kind.ext) == ("native", "png")
    found = shot_images.load_row_array(scan_folder, None, "analysis:py-magspec", 2)
    assert found.kind == "native"
    assert found.array.max() == 200
    assert found.path.name == "Scan075_py-magspec_002.png"


def test_an_image_bin_averages_its_shots(scan_folder):
    found = shot_images.average_rows(scan_folder, None, "analysis:py-magspec", [1, 3])
    assert float(found.array.max()) == pytest.approx(200.0)


def test_a_shot_with_no_file_is_refused_never_a_neighbours(scan_folder):
    found = shot_images.load_row_array(scan_folder, None, "analysis:py-magspec", 4)
    assert found.array is None
    assert found.reason == "file not found"


def test_the_row_announced_shot_names_the_file(scan_folder):
    frame = pd.DataFrame({"Shotnumber": [1, 2], "Announced Shot": [3, 1]})
    found = shot_images.load_row_array(scan_folder, frame, "analysis:py-magspec", 1)
    assert found.path.name == "Scan075_py-magspec_003.png"


def test_a_text_output_is_a_trace_with_a_channel_per_y_column(scan_folder):
    probe = shot_traces.trace_device(scan_folder, "analysis:py-magspec-interpSpec")
    assert (probe.kind, probe.content, probe.table) == ("native", "lineout", True)
    assert probe.channels == ("Charge_fC", "ChargeDen_pC/GeV/c")
    one = shot_traces.load_row_trace(
        scan_folder, None, "analysis:py-magspec-interpSpec", 2, channel=1, probe=probe
    )
    assert one.result.data.tolist() == [[1.0, 4.0], [2.0, 8.0], [3.0, 12.0]]
    assert (one.result.x_label, one.result.x_units) == ("Momentum", "GeV/c")
    assert (one.result.y_label, one.result.y_units) == ("ChargeDen", "pC/GeV/c")


def test_a_named_series_in_a_mixed_folder_is_its_own_kind(scan_folder):
    assert not shot_traces.trace_device(scan_folder, "analysis:py-ebeam").kind
    probe = shot_traces.trace_device(scan_folder, "analysis:py-ebeam/py-ebeamX")
    assert probe.kind == "native"
    mean = shot_traces.average_trace_rows(
        scan_folder, None, "analysis:py-ebeam/py-ebeamX", [1, 3], probe=probe
    )
    assert mean.result.data[:, 1].tolist() == [2.0, 4.0, 6.0]


def test_a_channel_past_the_columns_is_refused(scan_folder):
    one = shot_traces.load_row_trace(
        scan_folder, None, "analysis:py-magspec-interpSpec", 1, channel=2
    )
    assert one.result is None


def test_a_pooled_bin_reads_each_scans_own_outputs(scan_folder, tmp_path):
    other = scan_folder.parent / "Scan076"
    other.mkdir()
    out = scan_folder.parent.parent / "analysis" / "Scan076" / "py-magspec"
    out.mkdir(parents=True)
    Image.fromarray(np.full((4, 6), 600, dtype=np.uint16)).save(
        out / "Scan076_py-magspec_001.png"
    )
    parts = [
        shot_images.ScanRows(scan_folder, None, (1,)),
        shot_images.ScanRows(other, None, (1,)),
    ]
    found = shot_images.average_scan_rows(parts, "analysis:py-magspec")
    assert float(found.array.max()) == pytest.approx(350.0)


@pytest.mark.parametrize(
    "text",
    [
        "a\tb\n1\t2\n3\n",  # a short row
        "1\t2\nx\ty\n",  # prose past the first line
        "only\n1\n2\n",  # one column
        "",
    ],
)
def test_what_is_not_a_table(tmp_path, text):
    path = tmp_path / "t.txt"
    path.write_text(text)
    assert shot_traces.read_text_table(path) is None


def test_a_table_with_no_header_names_its_columns(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("1\t2\t3\n4\t5\t6\n")
    names, values = shot_traces.read_text_table(path)
    assert names == ("column 0", "column 1", "column 2")
    assert values.shape == (2, 3)


def test_a_bracketed_header_gives_label_and_units(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("Energy [MeV]\tCharge [pC]\n1\t2\n")
    one = shot_traces.read_table_trace(path, 0)
    assert (one.x_label, one.x_units, one.y_label, one.y_units) == (
        "Energy",
        "MeV",
        "Charge",
        "pC",
    )


def test_reading_leaves_the_tree_untouched(scan_folder):
    root = scan_folder.parent.parent
    before = _tree(root)
    shot_images.load_row_array(scan_folder, None, "analysis:py-magspec", 9)
    shot_traces.trace_device(scan_folder, "analysis:py-missing")
    analysis_outputs.output_folders(root / "scans" / "Scan099")
    assert _tree(root) == before


def test_the_label_marks_an_output():
    assert analysis_outputs.output_label("analysis:py-ebeam/py-ebeamX") == (
        "analysis · py-ebeam › py-ebeamX"
    )
    assert analysis_outputs.output_label("CAM-X") == "CAM-X"


def _result(x, y):
    from geecs_data_utils.io.array1d import Data1DResult

    return Data1DResult(data=np.column_stack((x, y)))


def test_per_shot_axes_average_index_wise_x_too():
    a = _result([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    b = _result([1.2, 2.2, 3.2], [30.0, 40.0, np.nan])
    assert shot_traces.average_traces([a, b]) is None  # the strict rule
    mean = shot_traces.average_traces([a, b], per_shot_axes=True)
    assert mean.data[:, 0] == pytest.approx([1.1, 2.1, 3.1])
    assert mean.data[:, 1].tolist() == [20.0, 30.0, 30.0]


def test_per_shot_axes_still_refuse_another_length():
    a = _result([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    b = _result([1.0, 2.0], [1.0, 2.0])
    assert shot_traces.average_traces([a, b], per_shot_axes=True) is None


def test_a_shared_axis_keeps_the_first_traces_x():
    a = _result([1.0, 2.0], [1.0, 3.0])
    mean = shot_traces.average_traces([a, _result([1.0, 2.0], [3.0, 5.0])], True)
    assert mean.data.tolist() == [[1.0, 2.0], [2.0, 4.0]]


def test_an_outputs_bin_averages_moving_axes_and_says_so(scan_folder):
    spec = scan_folder.parent.parent / "analysis" / "Scan075" / "py-magspec-interpSpec"
    for shot, shift in ((1, 0.0), (2, 0.1)):
        rows = "\n".join(f"{x + shift}\t{shot}\t{shot}" for x in (1.0, 2.0, 3.0))
        (spec / f"Scan075_py-magspec-interpSpec_{shot:03d}.txt").write_text(
            "Momentum_GeV/c\tCharge_fC\tChargeDen\n" + rows + "\n"
        )
    mean = shot_traces.average_trace_rows(
        scan_folder, None, "analysis:py-magspec-interpSpec", [1, 2]
    )
    assert mean.result.data[:, 0] == pytest.approx([1.05, 2.05, 3.05])
    assert mean.result.data[:, 1].tolist() == [1.5, 1.5, 1.5]
    assert mean.note == shot_traces.AXES_AVERAGED_NOTE


def test_a_raw_text_device_keeps_the_strict_rule(tmp_path):
    day = tmp_path / "Expt" / "Y2026" / "11-Nov" / "26_1103"
    scan = day / "scans" / "Scan001"
    device = scan / "SPEC-X"
    device.mkdir(parents=True)
    for shot, shift in ((1, 0.0), (2, 0.1)):
        rows = "\n".join(f"{x + shift}\t{shot}" for x in (1.0, 2.0, 3.0))
        (device / f"Scan001_SPEC-X_{shot:03d}.txt").write_text(rows + "\n")
    mean = shot_traces.average_trace_rows(scan, None, "SPEC-X", [1, 2])
    assert mean.result is None
    assert "differ" in mean.reason
