"""Background subtraction on the Images page: a background scan's average off each shot."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geecs_portal.plotter import images, scans
from tests.plotter.conftest import CAM, DAY, EXPERIMENT, post

BG = f"{DAY}:2"


@pytest.fixture
def background(camera: Path) -> Path:
    """Scan002 as the background: the camera reads 4 and 6 there (average 5)."""
    folder = camera.parent / "Scan002" / CAM
    folder.mkdir()
    for shot, level in ((1, 4), (2, 6)):
        Image.fromarray(np.full((4, 6), level, dtype=np.uint16)).save(
            folder / f"Scan002_{CAM}_{shot:03d}.png"
        )
    return folder


def _scan(share: Path, number: int):
    run = scans.find_scan(DAY, number, experiment=EXPERIMENT, base_path=share)
    return scans.load_frame(run), scans.scan_folder(run)


def test_each_shot_loses_the_background_scans_average(
    share: Path, background: Path
) -> None:
    frame, folder = _scan(share, 1)
    bg_frame, bg_folder = _scan(share, 2)
    back = images.Background(bg_folder, bg_frame, "Scan002")
    np.testing.assert_allclose(images.background_array(back, CAM), 5.0)
    raw = images.tile_array(folder, frame, CAM, (1,))
    taken = images.tile_array(folder, frame, CAM, (1,), background=back)
    np.testing.assert_allclose(taken, raw - 5.0)


def test_a_background_without_the_camera_says_so(share: Path, camera: Path) -> None:
    frame, folder = _scan(share, 1)
    bg_frame, bg_folder = _scan(share, 2)  # Scan002 has no camera folder here
    back = images.Background(bg_folder, bg_frame, "Scan002")
    with pytest.raises(images.NoImage, match="no background"):
        images.tile_array(folder, frame, CAM, (1,), background=back)


def test_the_routes_take_the_background_off(
    client: TestClient, background: Path
) -> None:
    query = {"day": DAY, "scan": 1, "device": CAM}
    plain = client.get("/api/images/levels", params=query).json()
    taken = client.get("/api/images/levels", params={**query, "bg": BG}).json()
    assert (taken["zmin"], taken["zmax"]) == (plain["zmin"] - 5, plain["zmax"] - 5)
    tile = {**query, "strip": "shots", "index": 1, "zmin": 0, "zmax": 1000}
    one = client.get("/api/images/tile.png", params={**tile, "bg": BG})
    assert one.status_code == 200
    drawn = np.asarray(Image.open(io.BytesIO(one.content)).convert("L"))
    assert drawn.shape[:2] == (4, 6)
    bad = client.get("/api/images/tile.png", params={**tile, "bg": "nonsense"})
    assert bad.status_code == 422


def test_the_page_carries_the_background_and_a_camera_can_leave_it(
    client: TestClient, background: Path
) -> None:
    page = post(client, scan=1, page="images", device=CAM, bg=BG, zscale="tile")
    assert f"bg={DAY}%3A2" in page and "minus background Scan002" in page
    assert 'data-toggle-for="bgsub"' in page
    off = post(client, scan=1, page="images", bgsub="false")
    assert "bg=" not in off.split('id="strip"')[1].split("</div>")[0]
    cleared = post(client, scan=1, page="images", bg="")
    assert 'data-toggle-for="bgsub"' not in cleared
