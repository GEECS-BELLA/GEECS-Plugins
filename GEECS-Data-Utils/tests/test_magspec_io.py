"""A magspec image read with its momentum and angle axes (io/magspec.py).

Synthetic files shaped like a real ``analysis/ScanNNN/py-magspec*`` triple:
a 16-bit greyscale PNG carrying ``Comment: <n> aC/count``, a momentum
table with one row per image column and an angle table with one row per
image row. Nothing is ever created in the scan tree.
"""

from __future__ import annotations

import io

import numpy as np
import png
import pytest

from geecs_data_utils.frames import Axis, Frame
from geecs_data_utils.io import magspec
from geecs_data_utils.io.images import write_png16

HEIGHT, WIDTH = 4, 6
NAME = "analysis:py-magspec"


def _png(path, pixels, comment="1 aC/count"):
    """A 16-bit greyscale PNG, with a ``Comment`` text chunk when *comment* is given."""
    write_png16(path, pixels)
    if comment is None:
        return
    chunks = list(png.Reader(bytes=path.read_bytes()).chunks())
    chunks.insert(1, (b"tEXt", b"Comment\x00" + comment.encode("latin-1")))
    out = io.BytesIO()
    png.write_chunks(out, chunks)
    path.write_bytes(out.getvalue())


def _table(path, header, axis, charge):
    """A tab-separated axis table: axis, charge, charge density."""
    lines = [header] + [f"{x:.8e}\t{q:.8e}\t{2 * q:.8e}" for x, q in zip(axis, charge)]
    path.write_text("\n".join(lines) + "\n")


def _pixels(shot):
    return np.arange(HEIGHT * WIDTH, dtype=np.uint16).reshape(HEIGHT, WIDTH) + shot


def _momentum(shot):
    return np.linspace(0.01, 5.0, WIDTH) * (1 + shot / 1000)


ANGLE = np.linspace(-1.3, 1.3, HEIGHT)


def _write_shot(out, shot, *, width=WIDTH, height=HEIGHT, comment="1 aC/count"):
    tag = f"Scan076_{{}}_{shot:03d}"
    pixels = _pixels(shot)
    _png(out / "py-magspec" / f"{tag.format('py-magspec')}.png", pixels, comment)
    _table(
        out / "py-magspec-interpSpec" / f"{tag.format('py-magspec-interpSpec')}.txt",
        "Momentum_GeV/c\tCharge_fC\tChargeDen_pC/GeV/c",
        _momentum(shot)[:width] if width <= WIDTH else np.linspace(0.01, 5, width),
        pixels.sum(axis=0)[:width] / 1000 if width <= WIDTH else np.ones(width),
    )
    angle = ANGLE if height == HEIGHT else np.linspace(-1.3, 1.3, height)
    _table(
        out / "py-magspec-interpDiv" / f"{tag.format('py-magspec-interpDiv')}.txt",
        "Angle_mrad\tCharge_fC\tChargeDen_fC/mrad",
        angle,
        np.ones(height),
    )


@pytest.fixture()
def scan_folder(tmp_path):
    """Scan076 with a py-magspec triple for shots 1 and 2, and shot 3's image alone."""
    day = tmp_path / "Expt" / "Y2026" / "11-Nov" / "26_1103"
    scan = day / "scans" / "Scan076"
    scan.mkdir(parents=True)
    out = day / "analysis" / "Scan076"
    for name in ("py-magspec", "py-magspec-interpSpec", "py-magspec-interpDiv"):
        (out / name).mkdir(parents=True)
    _write_shot(out, 1)
    _write_shot(out, 2)
    _png(out / "py-magspec" / "Scan076_py-magspec_003.png", _pixels(3))
    return scan


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_a_shot_is_a_frame_with_angle_rows_and_momentum_columns(scan_folder):
    shot = magspec.magspec_shot(scan_folder, NAME, 2)
    assert shot is not None
    frame = shot.frame
    assert frame.data.shape == (HEIGHT, WIDTH)
    rows, columns = frame.axes
    assert (rows.label, rows.unit) == ("Angle", "mrad")
    assert (columns.label, columns.unit) == ("Momentum", "GeV/c")
    np.testing.assert_allclose(rows.values, ANGLE)
    np.testing.assert_allclose(columns.values, _momentum(2), rtol=1e-8)
    assert (frame.unit, frame.label) == ("aC", "Charge")
    np.testing.assert_array_equal(frame.data, _pixels(2))
    assert (frame.shot.device, frame.shot.shot_number) == (NAME, 2)


def test_each_shot_reads_its_own_momentum_axis(scan_folder):
    one = magspec.magspec_shot(scan_folder, NAME, 1).frame.axes[1].values
    two = magspec.magspec_shot(scan_folder, NAME, 2).frame.axes[1].values
    assert not np.array_equal(one, two)


def test_the_lineouts_carry_their_units(scan_folder):
    shot = magspec.magspec_shot(scan_folder, NAME, 1)
    charge = shot.spectrum.channels["Charge"]
    assert charge.unit == "fC"
    assert shot.spectrum.channels["ChargeDen"].unit == "pC/GeV/c"
    assert shot.divergence.channels["ChargeDen"].unit == "fC/mrad"
    # The spectrum's charge is the image's column sums in fC.
    np.testing.assert_allclose(
        charge.data, shot.frame.data.sum(axis=0) / 1000, rtol=1e-6
    )


def test_the_comment_scales_the_counts(tmp_path):
    path = tmp_path / "x.png"
    _png(path, _pixels(0), comment="2.5 fC/count")
    image, unit = magspec.read_charge_image(path)
    assert unit == "fC"
    np.testing.assert_allclose(image, _pixels(0) * 2.5)


def test_no_comment_is_one_attocoulomb_per_count(tmp_path):
    path = tmp_path / "x.png"
    _png(path, _pixels(0), comment=None)
    image, unit = magspec.read_charge_image(path)
    assert unit == "aC"
    np.testing.assert_array_equal(image, _pixels(0))


@pytest.mark.parametrize(
    "comment, expected",
    [
        ("1 aC/count", (1.0, "aC")),
        (" 1e3 aC / count ", (1000.0, "aC")),
        ("a camera", None),
        (None, None),
    ],
)
def test_charge_per_count(comment, expected):
    assert magspec.charge_per_count(comment) == expected


def test_a_shot_missing_an_axis_file_is_none_never_a_neighbours(scan_folder):
    assert magspec.magspec_shot(scan_folder, NAME, 3) is None
    assert magspec.magspec_shot(scan_folder, NAME, 9) is None


def test_an_output_without_axis_siblings_has_no_magspec_series(scan_folder):
    out = scan_folder.parent.parent / "analysis" / "Scan076"
    (out / "py-ebeam").mkdir()
    _png(out / "py-ebeam" / "Scan076_py-ebeam_001.png", _pixels(1))
    assert magspec.magspec_series(scan_folder, "analysis:py-ebeam") is None
    assert magspec.magspec_series(scan_folder, "CAM-X") is None
    assert magspec.magspec_series(scan_folder, NAME) is not None


def test_an_axis_of_the_wrong_length_is_refused(scan_folder):
    out = scan_folder.parent.parent / "analysis" / "Scan076"
    _write_shot(out, 4, width=WIDTH + 1)
    with pytest.raises(ValueError, match="columns wide"):
        magspec.magspec_shot(scan_folder, NAME, 4)
    _write_shot(out, 5, height=HEIGHT - 1)
    with pytest.raises(ValueError, match="rows high"):
        magspec.magspec_shot(scan_folder, NAME, 5)


NOT_A_TABLE = "not a tab-separated table"


@pytest.mark.parametrize(
    "text, match",
    [
        ("", NOT_A_TABLE),
        ("Momentum_GeV/c\n1.0\n", NOT_A_TABLE),
        ("Momentum_GeV/c\tCharge_fC\n", NOT_A_TABLE),
        ("Momentum_GeV/c\tCharge_fC\n1.0\t2.0\n1.0\t2.0\t3.0\n", NOT_A_TABLE),
        ("Momentum_GeV/c\tCharge_fC\n1.0\tx\n", NOT_A_TABLE),
        ("Momentum_GeV/c\tMomentum_fC\n1.0\t2.0\n", "distinct"),
        ("Momentum_GeV/c\tCharge_fC\nnan\t2.0\n", "finite"),
    ],
)
def test_a_malformed_table_is_refused(tmp_path, text, match):
    path = tmp_path / "t.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        magspec.read_lineout_table(path)


def test_a_header_with_no_unit_reads_as_a_bare_label(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("angle\tCharge\n-1.0\t0.5\n1.0\t0.25\n")
    table = magspec.read_lineout_table(path)
    assert (table.axis.label, table.axis.unit) == ("angle", "")
    assert table.channels["Charge"].unit == ""


def test_a_header_reads_as_the_trace_path_reads_it(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("Angle [mrad]\tCharge_Density_fC\n-1.0\t0.5\n1.0\t0.25\n")
    table = magspec.read_lineout_table(path)
    assert (table.axis.label, table.axis.unit) == ("Angle", "mrad")
    assert table.channels["Charge_Density"].unit == "fC"


@pytest.mark.parametrize("cut", [0, 60, -20])
def test_a_truncated_or_empty_png_is_a_value_error(tmp_path, cut):
    whole = tmp_path / "whole.png"
    _png(whole, _pixels(0))
    path = tmp_path / "cut.png"
    raw = whole.read_bytes()
    path.write_bytes(raw[:cut])
    with pytest.raises(ValueError, match="not a readable PNG"):
        magspec.read_charge_image(path)


def test_series_resolved_once_reads_every_shot(scan_folder, monkeypatch):
    series = magspec.magspec_series(scan_folder, NAME)

    def no_listing(*args):
        raise AssertionError("a resolved series lists no folder")

    monkeypatch.setattr(magspec.analysis_outputs, "resolve_output", no_listing)
    for number in (1, 2):
        shot = magspec.magspec_shot(scan_folder, NAME, number, series=series)
        np.testing.assert_array_equal(shot.frame.data, _pixels(number))
    assert magspec.magspec_shot(scan_folder, NAME, 3, series=series) is None


def test_reading_creates_nothing(scan_folder):
    root = scan_folder.parent.parent.parent
    before = _tree(root)
    magspec.magspec_shot(scan_folder, NAME, 1)
    magspec.magspec_shot(scan_folder, NAME, 3)
    magspec.magspec_shot(scan_folder.parent / "Scan099", NAME, 1)
    assert _tree(root) == before


def _image(data, angle, momentum, unit="aC"):
    axes = (
        Axis(angle, unit="mrad", label="Angle"),
        Axis(momentum, unit="GeV/c", label="Momentum"),
    )
    return Frame(np.asarray(data, dtype=float), axes, None, unit, "Charge")


def test_charge_density_is_charge_over_each_pixels_widths():
    # Rows 0.5 mrad apart, columns 0.25 GeV/c apart: a pixel is 0.125 mrad·GeV/c.
    frame = _image(np.full((3, 4), 2.0), [0.0, 0.5, 1.0], [1.0, 1.25, 1.5, 1.75])
    density = magspec.charge_density(frame)
    assert density.unit == "pC/mrad/(GeV/c)"
    assert density.label == "Charge density"
    # 2 aC = 2e-6 pC over 0.125 mrad·GeV/c.
    np.testing.assert_allclose(density.data, 2e-6 / 0.125)
    assert density.axes == frame.axes
    # Integrating the density over the pixels gives the charge back, in pC.
    assert density.data.sum() * 0.125 == pytest.approx(frame.data.sum() * 1e-6)


def test_charge_density_follows_a_descending_or_uneven_axis():
    frame = _image(np.ones((2, 3)), [1.0, 0.0], [1.0, 2.0, 4.0])
    density = magspec.charge_density(frame, unit="fC")
    # Column widths by central difference: 1, 1.5, 2; row widths 1.
    np.testing.assert_allclose(density.data, 1e-3 / np.array([[1, 1.5, 2]] * 2))
    assert density.unit == "fC/mrad/(GeV/c)"


@pytest.mark.parametrize(
    "frame, match",
    [
        (_image(np.ones((2, 2)), [0, 1], [0, 1], unit="counts"), "charge units"),
        (_image(np.ones((1, 2)), [0], [0, 1]), "one sample"),
        (_image(np.ones((2, 2)), [0, 1], [1, 1]), "repeats"),
    ],
)
def test_charge_density_refuses(frame, match):
    with pytest.raises(ValueError, match=match):
        magspec.charge_density(frame)


def test_shots_average_pixels_and_axes_index_by_index():
    one = _image([[0.0, 2.0]], [0.0], [1.0, 2.0])
    two = _image([[2.0, 4.0]], [0.0], [1.1, 2.2])
    mean = magspec.average_shots([one, two])
    np.testing.assert_allclose(mean.data, [[1.0, 3.0]])
    np.testing.assert_allclose(mean.axes[1].values, [1.05, 2.1])
    assert (mean.axes[1].unit, mean.unit) == ("GeV/c", "aC")
    assert magspec.average_shots([one]) is one
    with pytest.raises(ValueError, match="differ in size"):
        magspec.average_shots([one, _image([[1.0]], [0.0], [1.0])])
    with pytest.raises(ValueError, match="no shots"):
        magspec.average_shots([])
