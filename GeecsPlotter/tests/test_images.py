"""The image strip: devices, tiles, the bin averages and the PNG routes.

Scan001 of the fake share gains a camera, ``UC_Cam``, whose shot ``n`` is a
flat frame of value ``10 * n``, with shot 3's file missing, and a vendor
device. Bin 1 is shots 1–2, bin 2 shots 3–4.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geecs_plotter import images, scans
from geecs_plotter.settings import SettingsError, merged, PlotterSettings
from tests.conftest import DAY, EXPERIMENT, post

CAM = "UC_Cam"
SCAN = {"day": "2026-10-02", "scan": 1}


@pytest.fixture
def camera(share: Path) -> Path:
    """Scan001's camera and vendor folders; shot 3 has no file."""
    folder = share / EXPERIMENT / "Y2026" / "10-Oct" / "26_1002" / "scans" / "Scan001"
    cam = folder / CAM
    cam.mkdir()
    for shot in (1, 2, 4):
        frame = np.full((4, 6), 10 * shot, dtype=np.uint16)
        frame[0, 0] = 1000  # a bright corner, so the window is not flat
        Image.fromarray(frame).save(cam / f"Scan001_{CAM}_{shot:03d}.png")
    haso = folder / "U_Haso"
    haso.mkdir()
    (haso / "Scan001_U_Haso_001.himg").write_bytes(b"proprietary")
    return folder


def _tree(root: Path) -> list[str]:
    return sorted(str(p) for p in root.rglob("*"))


def _opened(share: Path):
    run = scans.find_scan(DAY, 1, experiment=EXPERIMENT, base_path=share)
    return scans.load_frame(run), scans.scan_folder(run)


class TestTiles:
    def test_bins_are_the_bin_column_s(self, share: Path) -> None:
        frame, _ = _opened(share)
        assert images.bins(frame) == [(1, (1, 2)), (2, (3, 4))]

    def test_a_scan_without_bins_says_so(self, share: Path) -> None:
        run = scans.find_scan(DAY, 2, experiment=EXPERIMENT, base_path=share)
        with pytest.raises(images.NoImage, match="Bin #"):
            images.bins(scans.load_frame(run))

    def test_captions_carry_the_count_and_the_scan_variable(self, share: Path) -> None:
        frame, _ = _opened(share)
        values = frame["U_Jet Position"]
        bins = images.tiles(frame, "bins", values)
        assert [(t.index, t.rows, t.caption) for t in bins] == [
            (0, (1, 2), "bin 1 · 2 shots · 1"),
            (1, (3, 4), "bin 2 · 2 shots · 2"),
        ]
        shots = images.tiles(frame, "shots", shot_numbers=scans.shot_axis(frame))
        assert [t.caption for t in shots] == ["shot 1", "shot 2", "shot 3", "shot 4"]
        assert [t.rows for t in shots] == [(1,), (2,), (3,), (4,)]


class TestPixels:
    def test_a_bin_is_its_shots_average(self, share: Path, camera: Path) -> None:
        frame, folder = _opened(share)
        tile = images.tile_array(folder, frame, CAM, (1, 2))
        assert tile[2, 3] == pytest.approx(15.0)  # (10 + 20) / 2

    def test_a_shot_with_no_file_is_left_out_of_its_bin(
        self, share: Path, camera: Path
    ) -> None:
        frame, folder = _opened(share)
        tile = images.tile_array(folder, frame, CAM, (3, 4))
        assert tile[2, 3] == 40  # shot 4 alone, never shot 3's neighbour twice

    def test_a_tile_with_no_image_says_why(self, share: Path, camera: Path) -> None:
        frame, folder = _opened(share)
        with pytest.raises(images.NoImage, match="file not found"):
            images.tile_array(folder, frame, CAM, (3,))

    def test_a_missed_shot_is_refused(self, share: Path, camera: Path) -> None:
        frame, folder = _opened(share)
        frame = frame.assign(**{f"{CAM}-acq_timestamp": [1.0, float("nan"), 1.0, 1.0]})
        assert images.row_array(folder, frame, CAM, 2).reason == (
            "the device missed this shot"
        )

    def test_pictures_and_not(self, camera: Path) -> None:
        assert images.devices(camera) == [CAM, "U_Haso"]
        assert images.is_image_device(camera, CAM)
        assert not images.is_image_device(camera, "U_Haso")
        assert not images.is_image_device(camera, "../Scan002")


class TestRoutes:
    def test_devices(self, client: TestClient, camera: Path) -> None:
        body = client.get("/api/images/devices", params=SCAN).json()
        assert body == {"devices": [CAM, "U_Haso"]}

    def test_the_strip_lists_tiles_with_their_urls(
        self, client: TestClient, camera: Path
    ) -> None:
        body = client.get(
            "/api/images/strip", params={**SCAN, "device": CAM, "strip": "bins"}
        ).json()
        assert [t["rows"] for t in body["tiles"]] == [[1, 2], [3, 4]]
        assert body["tiles"][1]["url"].startswith("api/images/tile.png?")
        assert "index=1" in body["tiles"][1]["url"]
        tile = client.get("/" + body["tiles"][1]["url"])
        assert tile.status_code == 200
        assert tile.headers["content-type"] == "image/png"

    def test_a_tile_draws_with_the_colour_map(
        self, client: TestClient, camera: Path
    ) -> None:
        params = {**SCAN, "device": CAM, "strip": "shots", "index": 1}
        grey = client.get("/api/images/tile.png", params={**params, "cmap": "nope"})
        colour = client.get(
            "/api/images/tile.png", params={**params, "cmap": "viridis"}
        )
        assert np.asarray(Image.open(io.BytesIO(grey.content))).ndim == 2
        assert np.asarray(Image.open(io.BytesIO(colour.content))).shape == (4, 6, 3)

    @pytest.mark.parametrize(
        "params, reason",
        [
            ({"device": CAM, "strip": "shots", "index": 3}, "file not found"),
            ({"device": CAM, "strip": "bins", "index": 2}, "no bin 2 of 2"),
            ({"device": "U_Haso", "strip": "bins", "index": 0}, "no pictures"),
            ({"device": "../Scan002", "strip": "bins", "index": 0}, "no pictures"),
        ],
    )
    def test_refusals_are_404s_and_touch_nothing(
        self, client: TestClient, share: Path, camera: Path, params: dict, reason: str
    ) -> None:
        before = _tree(share)
        response = client.get("/api/images/tile.png", params={**SCAN, **params})
        assert response.status_code == 404
        assert reason in response.json()["detail"]
        assert _tree(share) == before

    def test_the_code_loads_the_same_pixels(
        self, client: TestClient, share: Path, camera: Path
    ) -> None:
        body = client.get(
            "/api/images/strip", params={**SCAN, "device": CAM, "strip": "bins"}
        ).json()
        code = body["code"].replace(
            "experiment='Testexp'", f"experiment='Testexp', base_path={str(share)!r}"
        )
        namespace: dict = {}
        exec(code, namespace)
        assert namespace["pixels"][2, 3] == 40  # the last tile: bin 2, shot 4


class TestPage:
    def test_the_images_tab_shows_the_strip(
        self, client: TestClient, camera: Path
    ) -> None:
        page = post(client, scan=1, tab="images")
        assert 'id="strip" class="strip grid"' in page
        assert page.count('class="tile"') == 2
        assert "bin 2 · 2 shots · 2" in page
        assert "api/images/tile.png?day=2026-10-02" in page
        assert "data-figure-url" not in page  # no XY plot on this tab

    def test_the_picked_camera_and_display_are_saved(
        self, client: TestClient, camera: Path
    ) -> None:
        page = post(
            client,
            scan=1,
            tab="images",
            device=CAM,
            strip="shots",
            layout="row",
            cols=3,
            cmap="magma",
            plo=2,
            phi=99,
        )
        assert "grid-auto-columns: calc((100% - 2 * var(--gap)) / 3)" in page
        assert page.count('class="tile"') == 4
        assert "cmap=magma&amp;plo=2.0&amp;phi=99.0" in page
        saved = client.get("/api/settings/default").json()
        assert saved["image_device"] == CAM and saved["image_strip"] == "shots"

    def test_a_camera_not_in_this_scan_is_kept(
        self, client: TestClient, camera: Path
    ) -> None:
        post(client, scan=1, tab="images", device=CAM)
        page = post(client, scan=2)
        assert f"{CAM} (not in this scan)" in page
        assert f"{CAM} is not in this scan" in page

    def test_the_xy_tab_is_the_default(self, client: TestClient) -> None:
        page = post(client, scan=1)
        assert "data-figure-url" in page
        assert 'id="strip"' not in page


@pytest.mark.parametrize(
    "field, value",
    [
        ("plo", "101"),
        ("phi", "nan"),
        ("layout", "diagonal"),
        ("strip", "all"),
        ("cols", "0"),
        ("cmap", "no spaces"),
        ("tab", "3d"),
    ],
)
def test_a_bad_image_control_is_refused(field: str, value: str) -> None:
    with pytest.raises(SettingsError):
        merged(PlotterSettings(), {field: value})
