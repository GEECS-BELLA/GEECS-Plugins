"""Several scans on the Images page: every scan's shots on one strip, bins pooled by X.

The open scan (2026-10-02 Scan001: bin 1 at X=1, shots 1-2; bin 2 at X=2,
shots 3-4, shot 3 with no image) and the set's extra scan (2026-10-03
Scan001: bin 1 at X=2, bin 2 at X=3, one shot each). Pooled by X: X=1
(two shots), X=2 (three: two scans), X=3 (one).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geecs_portal.plotter import images, multi
from tests.plotter.conftest import CAM, EXPERIMENT, SCAN_VARIABLE, post

OTHER = "2026-10-03:1"
REFS = [(date(2026, 10, 2), 1), (date(2026, 10, 3), 1)]


@pytest.fixture
def other(camera: Path, share: Path) -> Path:
    """2026-10-03 Scan001: the camera at 100 on both shots, bins at X=2 and 3."""
    day = share / EXPERIMENT / "Y2026" / "10-Oct" / "26_1003"
    folder = day / "scans" / "Scan001"
    (folder / CAM).mkdir(parents=True)
    (folder / "ScanInfoScan001.ini").write_text(
        "[Scan Info]\n"
        'Scan No = "1"\n'
        f'Scan Parameter = "{SCAN_VARIABLE}"\n'
        'Start = "2"\nEnd = "3"\nStep size = "1"\nShots per step = "1"\n'
    )
    (day / "analysis").mkdir()
    pd.DataFrame(
        {"Shotnumber": [1, 2], "Bin #": [1, 2], SCAN_VARIABLE: [2.0, 3.0]}
    ).to_csv(day / "analysis" / "s1.txt", sep="\t", index=False)
    for shot in (1, 2):
        frame = np.full((4, 6), 100, dtype=np.uint16)
        frame[0, 0] = 1000
        Image.fromarray(frame).save(folder / CAM / f"Scan001_{CAM}_{shot:03d}.png")
    return folder


def _set(share: Path, tolerance=None) -> images.ScanSet:
    return multi.scan_set(REFS, tolerance, experiment=EXPERIMENT, base_path=share)


def test_the_scans_bins_pool_by_x(share: Path, other: Path) -> None:
    found = _set(share)
    bins = images.kept_bins(found.frame, None, found.bin_col)
    # Merged rows: the open scan's four, then the other's two.
    assert [rows for _, rows in bins] == [(1, 2), (3, 4, 5), (6,)]
    assert found.tolerance == pytest.approx(0.5)


def test_a_pooled_bin_averages_every_scans_shots(share: Path, other: Path) -> None:
    found = _set(share)
    pixels = images.tile_array(found, found.frame, CAM, (3, 4, 5))
    # Shot 3 has no image: shot 4 (40) and the other scan's shot 1 (100).
    assert pixels[1, 1] == pytest.approx(70.0)


def test_a_merged_row_is_read_from_its_own_scan(share: Path, other: Path) -> None:
    found = _set(share)
    part, row = found.locate(5)
    assert (part.day, part.scan, row) == (date(2026, 10, 3), 1, 1)
    assert images.tile_array(found, found.frame, CAM, (5,))[1, 1] == 100


def test_a_large_tolerance_pools_everything(share: Path, other: Path) -> None:
    found = _set(share, tolerance=10.0)
    assert len(images.kept_bins(found.frame, None, found.bin_col)) == 1


def test_the_page_shows_one_strip_over_both_scans(
    client: TestClient, other: Path
) -> None:
    page = post(client, scan=1, page="images", device=CAM, scans=OTHER, strip="bins")
    assert "2 scans pooled" in page
    assert "2 scans · 3 shots" in page  # X=2: two of the open scan, one of the other
    assert page.count('class="tile"') == 3
    assert "scans=2026-10-03%3A1" in page  # every tile names the other scan
    assert "Pool bins within" in page and 'placeholder="auto: 0.5"' in page
    assert 'data-multi="true"' in page


def test_a_shot_names_its_scan(client: TestClient, other: Path) -> None:
    page = post(client, scan=1, page="images", device=CAM, scans=OTHER, strip="shots")
    assert "10-02 S001 shot 1" in page and "10-03 S001 shot 2" in page
    # The big view's preview opens the other scan's own shot.
    assert (
        'data-uid="folder:Testexp:2026-10-03:1" data-day="2026-10-03" data-row="2"'
        in page
    )


def test_the_tile_and_levels_routes_take_the_scans(
    client: TestClient, other: Path
) -> None:
    base = {"day": "2026-10-02", "scan": 1, "device": CAM, "scans": OTHER}
    tile = client.get(
        "/api/images/tile.png", params={**base, "strip": "bins", "index": 1}
    )
    assert tile.status_code == 200, tile.text
    levels = client.get(
        "/api/images/levels", params={**base, "strip": "bins", "roi": "1,1,6,4"}
    ).json()
    # Bins at 15 (10 and 20), 70 and 100.
    assert levels == {"zmin": 15.0, "zmax": 100.0, "tiles": 3}
    one = client.get(
        "/api/images/levels",
        params={**base, "strip": "bins", "roi": "1,1,6,4", "mtol": 10},
    ).json()
    assert one["tiles"] == 1


def test_the_tolerance_saves_with_the_set(
    client: TestClient, other: Path, settings_dir: Path
) -> None:
    import json

    page = post(client, scan=1, page="images", device=CAM, scans=OTHER, mtol="0.25")
    assert (
        json.loads((settings_dir / "default.json").read_text())["merge_tolerance"]
        == 0.25
    )
    assert 'value="0.25"' in page and "mtol=0.25" in page
    post(client, scan=1, page="images", mtol="")
    assert (
        json.loads((settings_dir / "default.json").read_text())["merge_tolerance"]
        is None
    )


def test_a_scan_without_the_camera_leaves_its_shots_out(
    client: TestClient, other: Path
) -> None:
    for png in (other / CAM).iterdir():
        png.unlink()
    base = {"day": "2026-10-02", "scan": 1, "device": CAM, "scans": OTHER}
    alone = client.get(
        "/api/images/tile.png", params={**base, "strip": "bins", "index": 2}
    )
    assert alone.status_code == 404  # X=3 is the other scan's alone
    pooled = client.get(
        "/api/images/tile.png", params={**base, "strip": "bins", "index": 1}
    )
    assert pooled.status_code == 200  # X=2 still has the open scan's shot 4


def test_a_bad_scans_parameter_is_a_422(client: TestClient, other: Path) -> None:
    response = client.get(
        "/api/images/tile.png",
        params={"day": "2026-10-02", "scan": 1, "device": CAM, "scans": "yesterday"},
    )
    assert response.status_code == 422


def test_the_notebook_code_rebuilds_the_set(client: TestClient, other: Path) -> None:
    page = post(client, scan=1, page="images", device=CAM, scans=OTHER, strip="bins")
    assert "multi.scan_set(refs, None" in page
    assert "bin_col=folder.bin_col, scan_names=folder.names()" in page


def test_reading_several_scans_leaves_the_share_untouched(
    client: TestClient, other: Path, share: Path
) -> None:
    before = sorted(str(p) for p in share.rglob("*") if EXPERIMENT in str(p))
    post(client, scan=1, page="images", device=CAM, scans=OTHER)
    base = {"day": "2026-10-02", "scan": 1, "device": CAM, "scans": OTHER}
    for index in range(3):
        client.get(
            "/api/images/tile.png", params={**base, "strip": "bins", "index": index}
        )
    after = sorted(str(p) for p in share.rglob("*") if EXPERIMENT in str(p))
    assert after == before


def test_the_strip_follows_the_plot_pages_x(
    client: TestClient, other: Path, share: Path
) -> None:
    # A column the plot's X is picked by hand (autofill off): the captions
    # carry its values and the pooled bins are placed by it.
    delay = "U_Delay"
    for day, values in (
        ("26_1002", [10.0, 10.0, 30.0, 30.0]),
        ("26_1003", [20.0, 40.0]),
    ):
        path = share / EXPERIMENT / "Y2026" / "10-Oct" / day / "analysis" / "s1.txt"
        frame = pd.read_csv(path, sep="\t")
        frame[delay] = values
        frame.to_csv(path, sep="\t", index=False)
    page = post(
        client,
        scan=1,
        page="images",
        device=CAM,
        scans=OTHER,
        strip="bins",
        x=delay,
        xauto="false",
        mtol="0",
    )
    # By the delay the four bins interleave the scans: 10, 20, 30, 40
    # (a tolerance of 0: the default, half the step of 20, would pool pairs).
    import re

    captions = re.findall(r'data-caption="([^"]*)"', page)
    assert len(captions) == 4, (captions, re.findall(r'class="m">([^<]*)', page))
    assert "20 · 10-03 S001 · 1 shot" in captions
    assert "bx=U_Delay" in page
    assert f"vs {delay}" in page


def test_the_strip_listing_s_tiles_rebuild_its_own_pooling(
    client: TestClient, other: Path
) -> None:
    """A listing asked for with ``bx=`` writes that ``bx`` into every tile URL.

    Otherwise tile *i* of the listing names bin *i* of a differently
    pooled set (the scan variable's) when the browser fetches it.
    """
    base = {"day": "2026-10-02", "scan": 1, "device": CAM, "scans": OTHER}
    by_shot = client.get(
        "/api/images/strip", params={**base, "strip": "bins", "bx": ""}
    )
    assert by_shot.status_code == 200, by_shot.text
    urls = [tile["url"] for tile in by_shot.json()["tiles"]]
    assert urls and all("bx=&" in u or u.endswith("bx=") for u in urls), urls


def test_a_saved_bin_width_never_cuts_the_merged_bin_numbers() -> None:
    """The merged numbers are bin identities: a width meant for X is dropped."""
    from geecs_data_utils.data.binning import BinningConfig, bin_rows
    from geecs_data_utils.data.scan_merge import merge_scan_frames

    def scan(positions):
        return pd.DataFrame(
            {
                "Shotnumber": range(1, 2 * len(positions) + 1),
                "Bin #": [b for b in range(1, len(positions) + 1) for _ in "ab"],
                "pos": [p for p in positions for _ in "ab"],
            }
        )

    frame = merge_scan_frames(
        {
            (date(2026, 1, 1), 1): scan([1.0, 2.0, 3.0]),
            (date(2026, 1, 1), 2): scan([1.0, 2.0, 3.0]),
        }
    )
    pooled, cfg, _ = multi.merged_bins(
        frame, BinningConfig(bin_col="Bin #", bin_width=2.0), "pos"
    )
    assert cfg.bin_width is None and cfg.bin_edges is None and cfg.quantile_bins is None
    assert [len(rows) for _, rows in bin_rows(pooled, cfg)] == [4, 4, 4]
