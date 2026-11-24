"""A scope row overlays several channels on each trace tile.

The LeCroy's files hold four traces; the row's checkboxes pick which ones
each tile draws on one axis and one y scale, a bin overlaying each
channel's own mean. A shot whose file lacks a channel draws the others;
one whose file holds no waveform at all says so.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.plotter.conftest import CAM, post
from tests.plotter.test_image_traces import write_scope

SCOPE = "SCOPE-LeCroy"
DAY = "2026-10-02"
BASE = {"day": DAY, "scan": 1, "device": SCOPE}


def channel_counts(shot: int) -> list[np.ndarray]:
    """Shot *shot*'s four channels: channel n is flat at ``10 * n + shot``."""
    return [np.full(6, 10.0 * n + shot) for n in range(4)]


@pytest.fixture
def scope4(camera: Path) -> Path:
    """Scan001's four-channel scope.

    Shots 1 and 2 have all four channels; shot 3's file holds no waveform
    (the scope's stub when it was not armed); shot 4's only channels 0-2.
    """
    folder = camera / SCOPE
    folder.mkdir()
    for shot in (1, 2):
        write_scope(folder / f"Scan001_{SCOPE}_{shot:03d}.dat", channel_counts(shot))
    with h5py.File(folder / f"Scan001_{SCOPE}_003.dat", "w") as f:
        f.attrs["stub"] = 1
    write_scope(folder / f"Scan001_{SCOPE}_004.dat", channel_counts(4)[:3])
    return camera


def _figure(client: TestClient, **query: object) -> dict:
    response = client.get("/api/images/trace", params={**BASE, **query})
    assert response.status_code == 200, response.text
    return response.json()


def _png(response) -> Image.Image:
    assert response.status_code == 200, response.text
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def test_the_row_offers_every_channel_as_a_checkbox(
    client: TestClient, scope4: Path, settings_dir: Path
) -> None:
    page = post(client, scan=1, page="images", device=SCOPE, channel="3,0,2")
    saved = json.loads((settings_dir / "default.json").read_text())
    assert saved["image_channels"] == [0, 2, 3]
    for n in range(4):
        checked = " checked" if n in (0, 2, 3) else ""
        assert (
            f'<input type="checkbox" value="{n}" data-channel-box{checked}> Channel {n}'
            in page
        )
    assert 'name="channel" form="pick" value="0,2,3"' in page
    # Every tile and the shared y range name each channel once.
    assert "channel=0&amp;channel=2&amp;channel=3" in page
    assert "images.channel_traces(" in page and "[0, 2, 3]" in page


def test_a_stacked_scope_keeps_its_own_channels(
    client: TestClient, scope4: Path, settings_dir: Path
) -> None:
    post(client, scan=1, page="images", device=CAM, addcam="1")
    page = post(
        client,
        scan=1,
        page="images",
        device=CAM,
        **{"device@2": SCOPE, "channel@2": "1,3"},
    )
    saved = json.loads((settings_dir / "default.json").read_text())
    assert saved["image_more"][0]["channels"] == [1, 3]
    assert 'name="channel@2" form="pick" value="1,3"' in page


def test_no_channel_is_refused(
    client: TestClient, scope4: Path, settings_dir: Path
) -> None:
    post(client, scan=1, page="images", device=SCOPE, channel="1")
    post(client, scan=1, page="images", device=SCOPE, channel="")
    saved = json.loads((settings_dir / "default.json").read_text())
    assert saved["image_channels"] == [1]


def test_a_set_saved_with_one_channel_opens_on_it(
    client: TestClient, scope4: Path, settings_dir: Path
) -> None:
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "default.json").write_text(
        json.dumps({"day": DAY, "scan": 1, "image_device": SCOPE, "image_channel": 2})
    )
    page = client.get("/images", params={"set": "default", "day": DAY, "scan": 1}).text
    assert (
        '<input type="checkbox" value="2" data-channel-box checked> Channel 2' in page
    )


def test_the_big_view_overlays_each_channel_in_its_colour(
    client: TestClient, scope4: Path
) -> None:
    body = _figure(client, strip="shots", index=1, channel=[0, 2, 3])
    lines = body["figure"]["data"]
    assert [line["name"] for line in lines] == ["Channel 0", "Channel 2", "Channel 3"]
    assert [line["y"][0] for line in lines] == [1.0, 21.0, 31.0]
    assert len({line["line"]["color"] for line in lines}) == 3
    assert body["figure"]["layout"]["showlegend"] is True


def test_one_channel_has_no_legend(client: TestClient, scope4: Path) -> None:
    body = _figure(client, strip="shots", index=1, channel=2)
    (line,) = body["figure"]["data"]
    assert line["y"][0] == 21.0
    assert body["figure"]["layout"]["showlegend"] is False


def test_a_bin_overlays_each_channels_own_mean(
    client: TestClient, scope4: Path
) -> None:
    # Bin 1: shots 1 and 2, so channel n is 10 n + 1.5.
    lines = _figure(client, strip="bins", index=0, channel=[0, 1, 2, 3])["figure"][
        "data"
    ]
    assert [line["y"][0] for line in lines] == [1.5, 11.5, 21.5, 31.5]


def test_a_shot_missing_a_channel_draws_the_others(
    client: TestClient, scope4: Path
) -> None:
    body = _figure(client, strip="shots", index=4, channel=[1, 3])
    assert [line["name"] for line in body["figure"]["data"]] == ["Channel 1"]
    # The channel left out is said, never silently dropped.
    (lacks,) = body["missing"]
    assert lacks.startswith("Channel 3: ") and "only contains 3 traces" in lacks
    _png(
        client.get(
            "/api/images/tile.png",
            params={**BASE, "strip": "shots", "index": 4, "channel": [1, 3]},
        )
    )


def test_a_file_with_no_waveform_is_a_404_naming_why(
    client: TestClient, scope4: Path
) -> None:
    response = client.get(
        "/api/images/tile.png",
        params={**BASE, "strip": "shots", "index": 3, "channel": [0, 1, 2, 3]},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "the file holds no trace"


def test_the_shared_y_range_spans_every_channel(
    client: TestClient, scope4: Path
) -> None:
    response = client.get(
        "/api/images/levels",
        params={**BASE, "strip": "shots", "channel": [0, 3]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Channel 0 of shot 1 is the lowest (1), channel 3 of shot 2 the highest
    # (32): shot 4 has no channel 3, shot 3 no trace at all.
    assert (body["zmin"], body["zmax"], body["tiles"]) == (1.0, 32.0, 3)


def test_an_overlaid_tile_draws_each_channel_in_its_own_colour(
    client: TestClient, scope4: Path
) -> None:
    def colours(channels: list[int]) -> set:
        image = _png(
            client.get(
                "/api/images/tile.png",
                params={**BASE, "strip": "shots", "index": 1, "channel": channels},
            )
        )
        return set(image.getdata())

    one, three = colours([0]), colours([0, 1, 3])
    orange = (255, 127, 14)  # matplotlib's C1: channel 1 on every tile
    assert orange not in one and orange in three


@pytest.mark.parametrize("bad", [[16], [-1]])
def test_a_channel_out_of_range_is_a_422(
    client: TestClient, scope4: Path, bad: list[int]
) -> None:
    response = client.get(
        "/api/images/tile.png",
        params={**BASE, "strip": "shots", "index": 1, "channel": bad},
    )
    assert response.status_code == 422


def test_the_notebook_code_compiles(client: TestClient, scope4: Path) -> None:
    for window in ({}, {"xmin": 1e-9, "xmax": 4e-9}):
        body = client.get(
            "/api/images/strip", params={**BASE, "channel": [0, 2], **window}
        ).json()
        compile(body["code"], "<strip>", "exec")
