"""A magspec output drawn on its physical axes, in the strip and the big view.

``analysis:py-magspec`` with its ``-interpSpec`` (momentum, one row per image
column) and ``-interpDiv`` (angle, one row per image row) siblings is a
calibrated output: its row offers "pixels | GeV/c · mrad", and on its axes
each tile is charge density, pC/mrad/(GeV/c), drawn with titled axes. The
reading is GEECS-Data-Utils' ``io.magspec``.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import png as pypng
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geecs_portal.plotter import images
from geecs_portal.plotter.settings import merged
from geecs_schemas.plot_settings import ImageCamera, PlotterSettings
from tests.plotter.conftest import CAM, post

DAY = "2026-10-02"
NAME = "analysis:py-magspec"
ANGLE = [0.0, 0.5, 1.0, 1.5]  # mrad: rows 0.5 mrad tall
#: Each shot's momentum step (GeV/c): the field moves it shot to shot.
STEP = {1: 0.25, 2: 0.5, 4: 0.25}


def _density(shot: int) -> float:
    """A shot's uniform image (10·shot aC per pixel) as pC/mrad/(GeV/c)."""
    return 10 * shot * 1e-6 / (0.5 * STEP[shot])


def _table(path: Path, header: str, axis) -> None:
    rows = "\n".join(f"{x}\t1.0\t1.0" for x in axis)
    path.write_text(header + "\n" + rows + "\n")


@pytest.fixture
def magspec(camera: Path) -> Path:
    """Scan001's py-magspec triple for shots 1, 2 and 4; shot 3's image alone."""
    out = camera.parent.parent / "analysis" / "Scan001"
    for name in ("py-magspec", "py-magspec-interpSpec", "py-magspec-interpDiv"):
        (out / name).mkdir(parents=True)
    tag = "Scan001_{}_{:03d}.{}"
    for shot in (1, 2, 3, 4):
        Image.fromarray(np.full((4, 6), 10 * shot, dtype=np.uint16)).save(
            out / "py-magspec" / tag.format("py-magspec", shot, "png")
        )
        if shot == 3:
            continue
        _table(
            out
            / "py-magspec-interpSpec"
            / tag.format("py-magspec-interpSpec", shot, "txt"),
            "Momentum_GeV/c\tCharge_fC\tChargeDen_pC/GeV/c",
            1.0 + STEP[shot] * np.arange(6),
        )
        _table(
            out
            / "py-magspec-interpDiv"
            / tag.format("py-magspec-interpDiv", shot, "txt"),
            "Angle_mrad\tCharge_fC\tChargeDen_fC/mrad",
            ANGLE,
        )
    return camera


def _get(client: TestClient, route: str, **query: object):
    return client.get(f"/api/images/{route}", params={"day": DAY, "scan": 1, **query})


def _png(response) -> Image.Image:
    assert response.status_code == 200, response.text
    return Image.open(io.BytesIO(response.content))


def test_a_magspec_row_offers_its_physical_axes(client: TestClient, magspec) -> None:
    page = post(client, scan=1, page="images", device=NAME)
    assert 'data-toggle-for="axes" data-value="true"' in page
    assert 'data-axes="false"' in page
    assert "axes=true" not in page


def test_a_camera_or_an_output_without_axis_tables_offers_none(
    client: TestClient, magspec: Path
) -> None:
    assert 'data-toggle-for="axes"' not in post(
        client, scan=1, page="images", device=CAM
    )
    plain = magspec.parent.parent / "analysis" / "Scan001" / "py-ebeam"
    plain.mkdir()
    Image.fromarray(np.ones((4, 6), dtype=np.uint16)).save(
        plain / "Scan001_py-ebeam_001.png"
    )
    page = post(client, scan=1, page="images", device="analysis:py-ebeam")
    assert 'data-toggle-for="axes"' not in page


def test_on_its_axes_every_url_asks_for_them_and_the_set_keeps_it(
    client: TestClient, magspec
) -> None:
    page = post(client, scan=1, page="images", device=NAME, axes="true")
    assert 'data-axes="true"' in page  # plotter.js adds axes=true to each tile
    levels_url = page.split('data-levels-url="')[1].split('"')[0]
    assert "axes=true" in levels_url
    # Each tile scaling itself: its URL's display half asks for the axes too.
    page = post(client, scan=1, page="images", device=NAME, zscale="tile")
    assert "cmap=white_blue_yellow&amp;axes=true" in page
    assert '<figcaption class="hint">pC/mrad/(GeV/c)</figcaption>' in page
    # A later page that does not send the field keeps it.
    assert 'data-axes="true"' in post(client, scan=1, page="images", device=NAME)


def test_a_tile_on_its_axes_is_a_figure_with_titled_axes(
    client: TestClient, magspec
) -> None:
    plain = _png(_get(client, "tile.png", device=NAME, strip="shots", index=1))
    assert plain.size == (6, 4)  # the pixels themselves
    tile = _png(
        _get(client, "tile.png", device=NAME, strip="shots", index=1, axes="true")
    )
    big = _png(
        _get(
            client,
            "tile.png",
            device=NAME,
            strip="shots",
            index=1,
            axes="true",
            bar="true",
        )
    )
    assert tile.size[0] > 100 and big.size[0] > tile.size[0]


def test_the_shared_scale_on_its_axes_is_in_charge_density(
    client: TestClient, magspec
) -> None:
    levels = _get(client, "levels", device=NAME, strip="shots", axes="true").json()
    assert levels["zmin"] == pytest.approx(min(_density(1), _density(2), _density(4)))
    assert levels["zmax"] == pytest.approx(max(_density(1), _density(2), _density(4)))
    # Shot 3 lacks its tables: no tile on its axes, so three tiles.
    assert levels["tiles"] == 3
    raw = _get(client, "levels", device=NAME, strip="shots").json()
    assert (raw["zmin"], raw["zmax"]) == (10.0, 40.0)


def test_a_bin_averages_density_and_axes_index_by_index(magspec: Path) -> None:
    frame = pd.DataFrame({"Bin #": [1, 1, 2, 2]})
    tile = images.calibrated_tile(magspec, frame, NAME, (1, 2))
    np.testing.assert_allclose(tile.data, (_density(1) + _density(2)) / 2)
    np.testing.assert_allclose(tile.axes[1].values, 1.0 + 0.375 * np.arange(6))
    np.testing.assert_allclose(tile.axes[0].values, ANGLE)
    assert tile.unit == "pC/mrad/(GeV/c)"
    # Shot 3 has no axis tables: left out of its bin, never a neighbour's axes.
    alone = images.calibrated_tile(magspec, frame, NAME, (3, 4))
    np.testing.assert_allclose(alone.data, _density(4))


def test_a_shot_without_its_axis_tables_is_a_404_naming_why(
    client: TestClient, magspec
) -> None:
    response = _get(
        client, "tile.png", device=NAME, strip="shots", index=3, axes="true"
    )
    assert response.status_code == 404
    assert "lacks its image or an axis table" in response.json()["detail"]


def test_axes_on_a_camera_are_ignored(client: TestClient, magspec) -> None:
    plain = _get(client, "tile.png", device=CAM, strip="shots", index=1)
    asked = _get(client, "tile.png", device=CAM, strip="shots", index=1, axes="true")
    assert asked.content == plain.content


def test_the_roi_crops_the_axes_with_the_pixels(magspec: Path) -> None:
    frame = pd.DataFrame({"Bin #": [1, 1, 2, 2]})
    tile = images.calibrated_tile(magspec, frame, NAME, (1,))
    part = images.crop_frame(tile, (1, 2, 4, 9))
    assert part.data.shape == (2, 3)
    np.testing.assert_allclose(part.axes[0].values, ANGLE[2:])
    np.testing.assert_allclose(part.axes[1].values, 1.0 + 0.25 * np.arange(1, 4))
    with pytest.raises(images.NoImage, match="outside"):
        images.crop_frame(tile, (10, 10, 12, 12))


def test_a_stacked_camera_keeps_its_own_axes_choice() -> None:
    settings = PlotterSettings(image_more=[ImageCamera(device=NAME)])
    after = merged(settings, {"axes@2": "true", "axes": "false"})
    assert after.image_more[0].axes is True and after.image_axes is False


def _tables_of(magspec: Path, shot: int) -> list[Path]:
    out = magspec.parent.parent / "analysis" / "Scan001"
    return [
        out / f"py-magspec-{kind}" / f"Scan001_py-magspec-{kind}_{shot:03d}.txt"
        for kind in ("interpSpec", "interpDiv")
    ]


def test_a_bin_read_before_its_tables_landed_is_read_again(magspec: Path) -> None:
    frame = pd.DataFrame({"Bin #": [1, 1, 2, 2]})
    held = [(p, p.read_text()) for p in _tables_of(magspec, 2)]
    for path, _ in held:
        path.unlink()  # shot 2's image is there, its tables not yet
    early = images.kept_calibrated_tile(magspec, frame, NAME, (1, 2))
    np.testing.assert_allclose(early.data, _density(1))  # shot 2 left out
    for path, text in held:
        path.write_text(text)  # the tables land: their folders' times move
        stamp = path.parent.stat().st_mtime_ns + 10**9
        os.utime(path.parent, ns=(stamp, stamp))
    late = images.kept_calibrated_tile(magspec, frame, NAME, (1, 2))
    np.testing.assert_allclose(late.data, (_density(1) + _density(2)) / 2)


def test_clean_ups_work_in_the_pngs_counts(magspec: Path) -> None:
    """A '10 aC/count' image: one count off is 10 aC off, not 1."""
    path = magspec.parent.parent / "analysis" / "Scan001" / "py-magspec"
    image = path / "Scan001_py-magspec_001.png"
    chunks = list(pypng.Reader(bytes=image.read_bytes()).chunks())
    chunks.insert(1, (b"tEXt", b"Comment\x0010 aC/count"))
    out = io.BytesIO()
    pypng.write_chunks(out, chunks)
    image.write_bytes(out.getvalue())
    series = images.magspec.magspec_series(magspec, NAME)
    part = images.ScanRows(magspec, pd.DataFrame({"Bin #": [1]}), (1,))
    found = images._calibrated_shot(part, 1, NAME, series, lambda c: c - 1)
    # (10 counts − 1) × 10 aC = 90 aC per pixel.
    np.testing.assert_allclose(found.data, 90e-6 / (0.5 * STEP[1]))


def test_centring_never_moves_charge_off_its_axes(magspec: Path) -> None:
    # A bright pixel off centre: centring would move it to the middle.
    pixels = np.full((4, 6), 10, dtype=np.uint16)
    pixels[0, 0] = 1000
    image = magspec.parent.parent / "analysis" / "Scan001" / "py-magspec"
    Image.fromarray(pixels).save(image / "Scan001_py-magspec_001.png")
    assert not np.array_equal(images.ImagePrep(centre=0.5).apply(pixels * 1.0), pixels)
    frame = pd.DataFrame({"Bin #": [1, 1, 2, 2]})
    centred = images.calibrated_tile(
        magspec, frame, NAME, (1,), images.ImagePrep(centre=0.5)
    )
    assert centred.data[0, 0] == centred.data.max()  # still at its momentum and angle
    plain = images.calibrated_tile(magspec, frame, NAME, (1,))
    np.testing.assert_array_equal(centred.data, plain.data)


def test_the_axis_tables_are_listed_once_while_their_folders_hold(
    magspec: Path, monkeypatch
) -> None:
    calls = []
    real = images.magspec.magspec_series
    monkeypatch.setattr(
        images.magspec,
        "magspec_series",
        lambda *a: calls.append(a) or real(*a),
    )
    images._has_magspec_series.cache_clear()
    for _ in range(3):
        assert images.calibrated_device(magspec, NAME)
    assert len(calls) == 1
