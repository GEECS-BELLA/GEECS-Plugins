"""The Images tab: the plotter's second page of a set (``/images``).

The image strip moved off the Plot page onto its own page, with the big
view under it; both pages save into the one set, and a change reopens the
page it was made on. The camera fixture is ``test_images.py``'s.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from geecs_portal.plotter import images
from tests.plotter.conftest import CAM, post

SCAN = {"day": "2026-10-02", "scan": 1}


def test_a_change_on_the_images_page_reopens_it(
    client: TestClient, camera: Path, settings_dir: Path
) -> None:
    response = client.post(
        "/",
        data={
            **{k: str(v) for k, v in SCAN.items()},
            "set": "default",
            "owner": "default",
            "page": "images",
            "device": CAM,
            "tab": "xy",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("./images?")
    saved = json.loads((settings_dir / "default.json").read_text())
    # The page is not a view the set keeps: the Plot page stays on its view.
    assert saved["tab"] == "xy" and saved["image_device"] == CAM


def test_the_images_page_is_the_strip_and_the_big_view(
    client: TestClient, camera: Path
) -> None:
    page = post(client, scan=1, page="images", device=CAM)
    assert 'id="strip"' in page and 'id="bigview"' in page
    assert 'data-mode="bins"' in page
    assert 'data-bin="1"' in page and 'data-bin="2"' in page
    assert 'name="page" value="images"' in page
    assert "images.js" in page and "data-figure-url" not in page
    assert 'data-value="map"' not in page  # the Plot page's switch is not here


def test_the_strip_can_be_copied_or_saved_as_one_picture(
    client: TestClient, camera: Path
) -> None:
    page = post(client, scan=1, page="images", device=CAM)
    assert "data-strip-copy" in page and "data-strip-save" in page
    assert "strip_export.js" in page
    # The picture's title line and file name name the scan and the camera.
    assert 'data-title="Scan001 · ' in page
    assert 'data-filename="Scan001_' in page and f"_{CAM}_bins" in page


def test_the_plot_page_has_no_image_view_and_an_old_images_set_opens_xy(
    client: TestClient, camera: Path, settings_dir: Path
) -> None:
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "default.json").write_text(
        json.dumps({"schema_version": 1, "tab": "images"})
    )
    page = client.get("/", params={**SCAN, "set": "default"}).text
    assert 'data-value="images"' not in page
    assert "data-figure-url" in page and 'id="strip"' not in page
    assert 'name="tab" value="xy"' in page


def test_without_a_set_the_images_page_asks_the_name_and_comes_back(
    client: TestClient,
) -> None:
    page = client.get("/images", params=SCAN).text
    assert 'id="ask-name"' in page and 'action="/images"' in page


def test_a_vendor_device_says_so(client: TestClient, camera: Path) -> None:
    page = post(client, scan=1, page="images", device="U_Haso")
    assert "a vendor format" in page and 'id="bigview"' not in page


def test_shot_tiles_carry_their_bin(share: Path, camera: Path) -> None:
    from tests.plotter.test_images import _opened

    frame, _ = _opened(share)
    tiles = images.tiles(frame, "shots")
    assert [t.bin for t in tiles] == ["1", "1", "2", "2"]
    assert [t.bin for t in images.tiles(frame, "bins")] == ["1", "2"]
