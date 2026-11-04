"""The image strip: devices, tiles, the bin averages and the PNG routes.

Scan001 of the fake share gains a camera, ``UC_Cam``, whose shot ``n`` is a
flat frame of value ``10 * n``, with shot 3's file missing, and a vendor
device. Bin 1 is shots 1–2, bin 2 shots 3–4.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geecs_portal.plotter import images, scans
from geecs_portal.plotter.settings import SettingsError, merged, PlotterSettings
from tests.plotter.conftest import CAM, DAY, EXPERIMENT, post

SCAN = {"day": "2026-10-02", "scan": 1}


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
        # The value alone: the strip's title names the variable once.
        assert [(t.index, t.rows, t.caption) for t in bins] == [
            (0, (1, 2), "1 · 2 shots"),
            (1, (3, 4), "2 · 2 shots"),
        ]
        # Without a scan variable the bin is named by its number.
        assert images.tiles(frame, "bins")[1].caption == "bin 2 · 2 shots"
        shots = images.tiles(frame, "shots", shot_numbers=scans.shot_axis(frame))
        assert [t.caption for t in shots] == ["shot 1", "shot 2", "shot 3", "shot 4"]
        named = images.tiles(frame, "shots", values, scans.shot_axis(frame))
        assert named[2].caption == "shot 3 · 2"
        assert [t.rows for t in shots] == [(1,), (2,), (3,), (4,)]

    def test_one_shot_per_bin_is_its_first_or_its_highest_y(self, share: Path) -> None:
        frame, _ = _opened(share)
        values, numbers = frame["U_Jet Position"], scans.shot_axis(frame)
        first = images.tiles(frame, "pick", values, numbers)
        assert [(t.index, t.rows, t.caption, t.bin) for t in first] == [
            (1, (1,), "1 · shot 1", "1"),
            (3, (3,), "2 · shot 3", "2"),
        ]
        # Highest charge: 12 over 10 in bin 1; bin 2's NaN is skipped.
        top = images.tiles(
            frame, "pick", values, numbers, pick_by=frame["U_ICT charge"]
        )
        assert [t.rows for t in top] == [(2,), (4,)]
        assert images.pick_row((1, 2), pd.Series([float("nan")] * 2)) == 1


class TestPixels:
    def test_a_hit_filter_cleans_each_shot_and_keys_the_store(
        self, share: Path, camera: Path
    ) -> None:
        from geecs_data_utils.data.image_prep import ImagePrep

        frame, folder = _opened(share)
        raw = images.kept_tile_array(folder, frame, CAM, (1, 2))
        clean = images.kept_tile_array(
            folder, frame, CAM, (1, 2), ImagePrep(hits="hot")
        )
        assert raw[0, 0] == pytest.approx(1000.0)  # the bright corner, averaged
        assert clean[0, 0] == pytest.approx(15.0)  # gone from each shot
        assert clean[2, 3] == pytest.approx(15.0)

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
        with pytest.raises(images.NoImage, match="missed this shot"):
            images.tile_array(folder, frame, CAM, (2,))

    def test_rows_with_no_bin_are_a_bin_of_their_own(self, share: Path) -> None:
        """The binned plot counts them; the strip must not drop them."""
        frame, _ = _opened(share)
        frame = frame.assign(**{"Bin #": [1, float("nan"), 2, 2]})
        assert [t.caption for t in images.tiles(frame, "bins")] == [
            "bin 1 · 1 shot",
            "bin 2 · 2 shots",
            "bin none · 1 shot",
        ]

    def test_shot_captions_are_whole_numbers(self, share: Path) -> None:
        frame, _ = _opened(share)
        numbers = frame["Shotnumber"].astype(float)
        numbers.iloc[1] = float("nan")
        assert [t.caption for t in images.tiles(frame, "shots", shot_numbers=numbers)][
            :2
        ] == ["shot 1", "shot ? (row 2)"]

    def test_a_tile_with_no_image_is_remembered_with_its_reason(
        self, share: Path, camera: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, folder = _opened(share)
        calls = []
        real = images.tile_array
        monkeypatch.setattr(
            images, "tile_array", lambda *a: calls.append(a) or real(*a)
        )
        for _ in range(2):
            with pytest.raises(images.NoImage, match="file not found"):
                images.kept_tile_array(folder, frame, CAM, (3,))
        assert len(calls) == 1  # the second ask is answered from memory
        # ...but not for long: a passing failure is looked at again.
        monkeypatch.setattr(images, "_REFUSAL_SECONDS", -1.0)
        with pytest.raises(images.NoImage):
            images.kept_tile_array(folder, frame, CAM, (3,))
        assert len(calls) == 2

    def test_the_memory_is_bounded_in_bytes(
        self, share: Path, camera: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, folder = _opened(share)
        one = images.tile_array(folder, frame, CAM, (1,)).nbytes
        monkeypatch.setattr(images, "_KEPT_BYTES", one + 1)
        monkeypatch.setattr(images, "_kept", type(images._kept)())
        monkeypatch.setattr(images, "_kept_bytes", 0)
        images.kept_tile_array(folder, frame, CAM, (1,))
        images.kept_tile_array(folder, frame, CAM, (2,))
        assert len(images._kept) == 1 and images._kept_bytes <= one + 1

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
        first = post(client, scan=1, page="images")
        assert "pick a camera from the list" in first  # no probing for one
        assert 'id="strip"' not in first
        page = post(client, scan=1, page="images", device=CAM)
        assert 'id="strip" class="strip grid"' in page
        assert page.count('class="tile"') == 2
        assert ">2 · 2 shots<" in page and "U_Jet Position = " not in page
        assert f"{CAM} vs U_Jet Position · 2 bins" in page  # the strip's title
        assert "api/images/tile.png?day=2026-10-02" in page
        assert "data-figure-url" not in page  # no XY plot on this tab

    def test_the_picked_camera_and_display_are_saved(
        self, client: TestClient, camera: Path
    ) -> None:
        page = post(
            client,
            scan=1,
            page="images",
            device=CAM,
            strip="shots",
            layout="row",
            cols=3,
            cmap="magma",
            zscale="tile",  # each tile scales itself: its URL is on the page
        )
        assert "grid-auto-columns: calc((100% - 2 * var(--gap)) / 3)" in page
        assert page.count('class="tile"') == 4
        assert "index=1&amp;cmap=magma" in page
        saved = client.get("/api/settings/default").json()
        assert saved["image_device"] == CAM and saved["image_strip"] == "shots"

    def test_a_camera_not_in_this_scan_is_kept(
        self, client: TestClient, camera: Path
    ) -> None:
        post(client, scan=1, page="images", device=CAM)
        page = post(client, scan=2, page="images")
        assert f"{CAM} (not in this scan)" in page
        assert f"{CAM} is not in this scan" in page

    def test_the_xy_tab_is_the_default(self, client: TestClient) -> None:
        page = post(client, scan=1)
        assert "data-figure-url" in page
        assert 'id="strip"' not in page


@pytest.mark.parametrize(
    "field, value",
    [
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


class TestRoiAndZScale:
    def test_crop_is_the_region_cut_at_the_edge(self) -> None:
        pixels = np.arange(24).reshape(4, 6)
        assert images.crop(pixels, None) is pixels
        assert images.crop(pixels, (1, 2, 3, 4)).tolist() == [[13, 14], [19, 20]]
        assert images.crop(pixels, (4, 0, 99, 1)).tolist() == [[4, 5]]
        with pytest.raises(images.NoImage, match="6×4"):
            images.crop(pixels, (10, 10, 12, 12))

    def test_shared_levels_are_the_widest_of_the_tiles(
        self, share: Path, camera: Path
    ) -> None:
        frame, folder = _opened(share)
        tiles = images.tiles(frame, "shots")  # shot 3 has no file
        zmin, zmax, count = images.shared_levels(
            folder, frame, CAM, tiles, roi=(1, 1, 6, 4)
        )
        # Inside the ROI (off the bright corner) shot n is flat at 10 * n.
        assert (zmin, zmax, count) == (10.0, 40.0, 3)

    def test_the_levels_route(self, client: TestClient, camera: Path) -> None:
        body = client.get(
            "/api/images/levels",
            params={
                **SCAN,
                "device": CAM,
                "strip": "bins",
                "roi": "1,1,6,4",
            },
        ).json()
        assert body == {"zmin": 15.0, "zmax": 40.0, "tiles": 2}

    def test_a_tile_on_fixed_levels(self, client: TestClient, camera: Path) -> None:
        params = {
            **SCAN,
            "device": CAM,
            "strip": "shots",
            "index": 2,
            "roi": "1,1,3,2",
            "zmin": 0,
            "zmax": 40,
        }
        png = client.get("/api/images/tile.png", params=params)
        # Shot 2 is 20 everywhere in the ROI: half way up a 0–40 scale (the
        # gray map draws it in RGB).
        drawn = np.asarray(Image.open(io.BytesIO(png.content)))
        assert drawn.shape == (1, 2, 3)
        assert set(drawn.ravel().tolist()) <= {127, 128}

    def test_a_tile_and_the_levels_take_the_clean_ups(
        self, client: TestClient, camera: Path
    ) -> None:
        params = {**SCAN, "device": CAM, "strip": "shots", "index": 2}
        levels = client.get(
            "/api/images/levels", params={**params, "rows": "1,2", "hits": "hot"}
        ).json()
        # The bright corner is a hit: gone, the strip runs 10 to 20.
        assert (levels["zmin"], levels["zmax"], levels["tiles"]) == (10.0, 20.0, 2)
        centred = {**params, "hits": "hot", "centre": 0.5}
        assert client.get("/api/images/tile.png", params=centred).status_code == 200
        for bad in ({"hits": "gauss"}, {"centre": -1}, {"rows": "1,x"}):
            route = "levels" if "rows" in bad else "tile.png"
            response = client.get(f"/api/images/{route}", params={**params, **bad})
            assert response.status_code == 422, bad

    def test_the_page_carries_the_pick_and_the_clean_ups(
        self, client: TestClient, camera: Path
    ) -> None:
        page = post(
            client,
            scan=1,
            page="images",
            device=CAM,
            strip="pick",
            pick="ymax",
            y="U_ICT charge",
            hits="median3",
            centre="0.5",
        )
        assert page.count('class="tile"') == 2
        assert "2 · shot 4" in page and "1 · shot 2" in page
        assert "strip=shots&amp;index=4&amp;hits=median3&amp;centre=0.5" in page
        assert "rows=2%2C4" in page  # the shared scale reads the shots shown
        assert "2 bins, one shot each · hits removed, centred" in page

    def test_the_centre_threshold_is_above_zero_or_off(self) -> None:
        assert merged(PlotterSettings(), {"centre": "200"}).image_centre == 200.0
        assert merged(PlotterSettings(), {"centre": ""}).image_centre is None
        with pytest.raises(SettingsError, match="centre"):
            merged(PlotterSettings(), {"centre": "0"})

    @pytest.mark.parametrize("roi", ["1,2,3", "3,0,1,2", "-1,0,2,2", "a,b,c,d"])
    def test_a_bad_roi_is_a_422(
        self, client: TestClient, camera: Path, roi: str
    ) -> None:
        params = {**SCAN, "device": CAM, "strip": "shots", "index": 1, "roi": roi}
        assert client.get("/api/images/tile.png", params=params).status_code == 422

    def test_the_roi_is_kept_per_camera(self) -> None:
        settings = merged(PlotterSettings(), {"device": CAM, "roi": "1,2,30,40"})
        settings = merged(settings, {"device": "U_Other", "roi": "0,0,5,5"})
        assert settings.image_rois == {CAM: [1, 2, 30, 40], "U_Other": [0, 0, 5, 5]}
        cleared = merged(settings, {"device": CAM, "roi": ""})
        assert cleared.image_rois == {"U_Other": [0, 0, 5, 5]}
        assert settings.image_rois[CAM] == [1, 2, 30, 40]  # the old copy kept

    def test_the_page_draws_the_saved_roi_and_fixed_scale(
        self, client: TestClient, camera: Path
    ) -> None:
        client.post("/api/settings/default", data={"device": CAM, "roi": "1,1,6,4"})
        page = post(
            client, scan=1, page="images", device=CAM, zscale="fixed", zmin=0, zmax=50
        )
        assert "x 1–6, y 1–4" in page
        assert "roi=1%2C1%2C6%2C4&amp;zmin=0.0&amp;zmax=50.0" in page

    def test_a_shared_scale_leaves_the_tiles_to_the_script(
        self, client: TestClient, camera: Path
    ) -> None:
        page = post(client, scan=1, page="images", device=CAM, zscale="shared")
        assert 'data-zscale="shared"' in page
        assert "api/images/levels?day=2026-10-02" in page
        assert "&amp;cmap=" not in page  # no tile asks before the scale is read


@pytest.mark.parametrize(
    "field, value", [("zscale", "log"), ("zmin", "inf"), ("roi", "1,1,1,1")]
)
def test_a_bad_z_or_roi_control_is_refused(field: str, value: str) -> None:
    with pytest.raises(SettingsError):
        merged(PlotterSettings(image_device=CAM), {field: value})


def test_each_tile_runs_from_its_darkest_to_its_brightest_pixel(
    client: TestClient, camera: Path, settings_dir: Path
) -> None:
    """LabVIEW's autoscale, with no percentile window: a graded spot keeps
    its grades (a 99.7th-percentile white would put 400 at 122, not 100),
    even under a set saved with the old window's fields."""
    spot = np.full((10, 10), 10, dtype=np.uint16)
    spot[4, 4], spot[4, 5], spot[5, 4] = 100, 400, 1000
    folder = camera / "UC_Spot"
    folder.mkdir()
    Image.fromarray(spot).save(folder / "Scan001_UC_Spot_001.png")
    settings_dir.mkdir(exist_ok=True)
    (settings_dir / "default.json").write_text(
        '{"schema_version": 1, "image_plo": 10.0, "image_phi": 99.7}'
    )
    page = post(client, scan=1, page="images", device="UC_Spot", strip="shots")
    assert "phi=" not in page and "plo=" not in page
    tile = client.get(
        "/api/images/tile.png",
        params={
            **SCAN,
            "device": "UC_Spot",
            "strip": "shots",
            "index": 1,
            "cmap": "none",
        },
    )
    drawn = np.asarray(Image.open(io.BytesIO(tile.content)))
    assert (drawn[0, 0], drawn[4, 4], drawn[4, 5], drawn[5, 4]) == (0, 23, 100, 255)


def test_a_bin_read_as_a_decimal_is_captioned_whole(share: Path) -> None:
    frame, _ = _opened(share)
    frame = frame.assign(**{"Bin #": [1.0, 1.0, 2.0, 2.0]})
    assert [t.caption for t in images.tiles(frame, "bins")] == [
        "bin 1 · 2 shots",
        "bin 2 · 2 shots",
    ]


def test_a_value_two_bins_share_names_the_bins() -> None:
    """A scan back over its start: the two 17.5 bins read apart, the rest stay bare."""
    frame = pd.DataFrame({"Bin #": [1, 1, 2, 2, 3, 3]})
    values = pd.Series([17.5, 17.5, 20.0, 20.0, 17.5, 17.5])
    assert [t.caption for t in images.tiles(frame, "bins", values)] == [
        "17.5 · bin 1 · 2 shots",
        "20 · 2 shots",
        "17.5 · bin 3 · 2 shots",
    ]


def test_every_hit_filter_has_a_label_on_the_rail() -> None:
    from geecs_data_utils.data.image_prep import HIT_FILTERS

    assert set(images.HIT_LABELS) == set(HIT_FILTERS)


def test_fixed_levels_with_white_below_black_say_so(
    client: TestClient, camera: Path
) -> None:
    page = post(
        client, scan=1, page="images", device=CAM, zscale="fixed", zmin=50, zmax=10
    )
    assert 'id="level-warning" role="alert">' in page  # shown
    fine = post(client, scan=1, page="images", device=CAM, zmin=10, zmax=50)
    assert 'id="level-warning" role="alert" hidden>' in fine


def test_a_tile_can_carry_its_own_colour_bar(client: TestClient, camera: Path) -> None:
    query = {"day": DAY, "scan": 1, "device": CAM, "strip": "shots", "index": 1}
    plain = client.get("/api/images/tile.png", params=query)
    barred = client.get("/api/images/tile.png", params={**query, "cbar": "true"})
    assert plain.status_code == barred.status_code == 200
    width = Image.open(io.BytesIO(plain.content)).width
    assert Image.open(io.BytesIO(barred.content)).width > width


def test_the_grid_colour_bar_route(client: TestClient) -> None:
    ok = client.get("/api/images/colourbar.png", params={"zmin": 0, "zmax": 10})
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
    flat = client.get("/api/images/colourbar.png", params={"zmin": 5, "zmax": 5})
    assert flat.status_code == 422


def test_bars_follow_the_z_scale_and_can_be_turned_off(
    client: TestClient, camera: Path
) -> None:
    each = post(client, scan=1, page="images", device=CAM, zscale="tile")
    assert "cbar=true" in each and 'data-cbar="true"' in each
    assert "strip-bar-cell" in each
    fixed = post(
        client, scan=1, page="images", device=CAM, zscale="fixed", zmin=0, zmax=50
    )
    assert "cbar=true" not in fixed  # one bar for the grid instead
    off = post(client, scan=1, page="images", device=CAM, zscale="tile", cbar="false")
    assert "cbar=true" not in off and 'data-cbar="false"' in off


def test_the_grid_colour_bar_stands_upright_on_request(client: TestClient) -> None:
    query = {"zmin": 0, "zmax": 10}
    lying = Image.open(
        io.BytesIO(client.get("/api/images/colourbar.png", params=query).content)
    )
    upright = Image.open(
        io.BytesIO(
            client.get(
                "/api/images/colourbar.png", params={**query, "vertical": "true"}
            ).content
        )
    )
    assert lying.width > lying.height and upright.height > upright.width


def test_a_new_set_draws_white_blue_yellow_on_one_shared_scale(
    client: TestClient, camera: Path
) -> None:
    """The author's defaults: GEECSplotter's table, every tile on one scale."""
    settings = PlotterSettings()
    assert (settings.image_cmap, settings.image_zscale) == (
        "white_blue_yellow",
        "shared",
    )
    page = post(client, scan=1, page="images", device=CAM)
    assert 'data-zscale="shared"' in page
    assert '<option value="white_blue_yellow" selected>' in page


def _charge_filter(low: float, high: str = "inf") -> str:
    return (
        '{"groups":[{"conditions":[{"column":"U_ICT charge","low":%s,"high":"%s"}]}],'
        '"nan_policy":"exclude"}' % (low, high)
    )


class TestShotFilters:
    """The Plot page's shot filters keep the same shots on the Images page."""

    def test_a_bin_keeps_only_its_passing_shots(self, share: Path) -> None:
        frame, _ = _opened(share)  # charge 10, 12, NaN, 30 in bins 1, 1, 2, 2
        keep = images.shot_rows(frame, _charge_filter(11))
        assert keep == {2, 4}
        assert [t.rows for t in images.tiles(frame, "bins", keep=keep)] == [(2,), (4,)]
        assert [t.index for t in images.tiles(frame, "shots", keep=keep)] == [2, 4]

    def test_a_bin_with_no_passing_shot_drops_out(self, share: Path) -> None:
        frame, _ = _opened(share)
        keep = images.shot_rows(frame, _charge_filter(20))
        found = images.tiles(frame, "bins", keep=keep)
        assert [(t.index, t.rows, t.bin) for t in found] == [(0, (4,), "2")]

    def test_no_active_filter_keeps_every_shot(self, share: Path) -> None:
        frame, _ = _opened(share)
        assert images.shot_rows(frame, "") is None
        assert images.shot_rows(frame, '{"groups":[],"nan_policy":"exclude"}') is None

    def test_a_filter_on_a_missing_column_says_so(self, share: Path) -> None:
        frame, _ = _opened(share)
        bad = '{"groups":[{"conditions":[{"column":"nope","low":0,"high":1}]}]}'
        with pytest.raises(images.NoImage, match="nope"):
            images.shot_rows(frame, bad)

    def test_the_page_and_its_routes_keep_the_filtered_shots(
        self, client: TestClient, camera: Path
    ) -> None:
        filters = _charge_filter(20)
        page = post(client, scan=1, page="images", device=CAM, filters=filters)
        assert page.count('class="tile"') == 1  # bin 1 has no shot above 20
        assert "1 / 4 shots pass" in page
        assert "Edit filters" in page and 'id="filters-drawer"' in page
        # The one tile is bin 2's, at index 0 among the bins left: its route
        # averages shot 4 alone (its picture is shot 4's).
        query = {"day": DAY, "scan": 1, "device": CAM, "strip": "bins", "index": 0}
        alone = client.get("/api/images/tile.png", params={**query, "filters": filters})
        shot4 = client.get(
            "/api/images/tile.png", params={**query, "strip": "shots", "index": 4}
        )
        assert alone.status_code == 200 and alone.content == shot4.content
        levels = client.get(
            "/api/images/levels",
            params={"day": DAY, "scan": 1, "device": CAM, "filters": filters},
        ).json()
        assert levels["tiles"] == 1
