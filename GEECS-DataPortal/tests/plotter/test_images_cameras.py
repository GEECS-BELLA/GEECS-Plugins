"""Several cameras on the Images page: one row each, each with its own look."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_portal.plotter.settings import (
    MAX_CAMERAS,
    PlotterSettings,
    SettingsError,
    camera_field,
    merged,
)
from tests.plotter.conftest import CAM, post

CAM2 = "UC_Cam2"


@pytest.fixture
def two_cameras(camera: Path) -> Path:
    """Scan001 with a second camera, the first one's pictures under its name."""
    shutil.copytree(camera / CAM, camera / CAM2)
    for path in (camera / CAM2).iterdir():
        path.rename(path.with_name(path.name.replace(CAM, CAM2)))
    return camera


class TestSettings:
    def test_a_camera_is_added_set_and_dropped(self) -> None:
        s = merged(PlotterSettings(), {"addcam": "1"})
        s = merged(s, {"addcam": "1"})
        s = merged(
            s,
            {
                "device@2": CAM2,
                "cmap@2": "viridis",
                "zscale@2": "fixed",
                "zmin@2": "5",
                "zmax@2": "50",
                "hits@3": "median3",
                "centre@3": "0.5",
                "cbar@2": "false",
            },
        )
        two, three = s.image_more
        assert (two.device, two.cmap, two.zscale, two.zmin, two.zmax, two.cbar) == (
            CAM2,
            "viridis",
            "fixed",
            5.0,
            50.0,
            False,
        )
        assert (three.hits, three.centre) == ("median3", 0.5)
        s = merged(s, {"dropcam": "2"})
        assert [c.hits for c in s.image_more] == ["median3"]

    def test_a_camera_field_is_checked_as_the_first_cameras_is(self) -> None:
        s = merged(PlotterSettings(), {"addcam": "1"})
        with pytest.raises(SettingsError, match="zscale@2"):
            merged(s, {"zscale@2": "sideways"})
        with pytest.raises(SettingsError, match="centre@2"):
            merged(s, {"centre@2": "-1"})

    def test_a_region_is_saved_for_its_cameras_device(self) -> None:
        s = merged(PlotterSettings(), {"addcam": "1"})
        # The region posted before the device in the form still lands on it.
        s = merged(s, {"roi@2": "0,0,3,2", "device@2": CAM2})
        assert s.image_rois == {CAM2: [0, 0, 3, 2]}

    def test_a_field_for_a_camera_not_there_is_ignored(self) -> None:
        assert merged(PlotterSettings(), {"cmap@3": "viridis"}).image_more == []

    def test_no_more_than_the_most_cameras(self) -> None:
        s = PlotterSettings()
        for _ in range(MAX_CAMERAS + 2):
            s = merged(s, {"addcam": "1"})
        assert len(s.image_more) == MAX_CAMERAS - 1

    def test_the_first_cameras_fields_keep_their_names(self) -> None:
        assert camera_field("cmap", 1) == "cmap"
        assert camera_field("cmap", 4) == "cmap@4"


def test_two_cameras_are_two_rows_on_the_same_tiles(
    client: TestClient, two_cameras: Path
) -> None:
    post(client, scan=1, page="images", device=CAM, addcam="1")
    page = post(
        client,
        scan=1,
        page="images",
        **{"device@2": CAM2, "cmap@2": "magma", "zscale@2": "tile"},
    )
    assert 'id="strip" class="strip row"' in page  # several cameras: one row each
    assert 'id="strip@2" class="strip row"' in page
    assert f'data-device="{CAM2}"' in page and "cmap=magma" in page
    assert page.count('data-bin="1"') == 2 and page.count('data-bin="2"') == 2
    assert 'name="dropcam" value="2"' in page
    assert "2 cameras" in page


def test_every_camera_has_its_own_filtered_picker(
    client: TestClient, two_cameras: Path, settings_dir: Path
) -> None:
    post(client, scan=1, page="images", device=CAM, devf="Cam", addcam="1")
    page = post(client, scan=1, page="images", **{"devf@2": "cam2"})
    # each row's picker has its own filter box, saved under its own name
    assert 'id="device-filter" name="devf" value="Cam"' in page
    assert 'id="device@2-filter" name="devf@2" value="cam2"' in page
    assert 'data-filter-for="device@2"' in page
    saved = (settings_dir / "default.json").read_text()
    assert '"device_filter": "cam2"' in saved


def test_each_row_keeps_its_own_z_scale(client: TestClient, two_cameras: Path) -> None:
    post(client, scan=1, page="images", device=CAM, addcam="1")
    page = post(
        client,
        scan=1,
        page="images",
        **{"device@2": CAM2, "zscale@2": "fixed", "zmin@2": "0", "zmax@2": "40"},
    )
    first, second = page.split('id="strip@2"')
    assert "zmin=" not in first.split('id="strip"')[1]  # the first scales each tile
    assert "zmin=0.0&amp;zmax=40.0" in second


def test_a_camera_without_pictures_says_so_on_its_row(
    client: TestClient, two_cameras: Path
) -> None:
    post(client, scan=1, page="images", device=CAM, addcam="1")
    page = post(client, scan=1, page="images", **{"device@2": "U_Haso"})
    assert 'id="strip"' in page and 'id="strip@2"' not in page
    assert "U_Haso has no pictures" in page


def test_a_dropped_camera_leaves_the_first_as_it_was(
    client: TestClient, two_cameras: Path
) -> None:
    post(client, scan=1, page="images", device=CAM, addcam="1")
    post(client, scan=1, page="images", **{"device@2": CAM2})
    page = post(client, scan=1, page="images", dropcam="2")
    assert 'id="strip@2"' not in page and f'data-device="{CAM}"' in page
    assert 'id="strip" class="strip grid"' in page  # alone again: its own layout
