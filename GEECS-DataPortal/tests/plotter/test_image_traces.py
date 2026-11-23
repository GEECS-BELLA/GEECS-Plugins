"""A scope's row on the Images page: trace tiles over the same bins as a camera.

The scope's per-shot NI-HWS files (the LeCroy VI's ``.dat``) are read by
GEECS-Data-Utils' ``io.shot_traces``; here, the page draws a row of them,
alone or stacked under a camera, a bin's tile is its shots' mean, and
the big view gets the tile as a figure.
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

from tests.plotter.conftest import CAM, EXPERIMENT, post

SCOPE = "SCOPE-LeCroy"
DAY = "2026-10-02"


def write_scope(path: Path, channels: list) -> None:
    """An NI-HWS waveform file: per channel a 1 ns time axis and raw counts (volts = counts)."""
    with h5py.File(path, "w") as f:
        axes = f.create_group("wfm_group0/axes")
        traces = f.create_group("wfm_group0/traces")
        for i, counts in enumerate(channels):
            x = axes.create_group(f"axis{2 * i}")
            x.attrs["increment"] = 1e-9
            x.attrs["start"] = 0.0
            y = axes.create_group(f"axis{2 * i + 1}")
            y.create_dataset("data_vector/data", data=np.asarray(counts, dtype=float))
            y.create_dataset("scale_coef", data=np.array([0.0, 1.0]))
            traces.create_group(f"trace{i}").attrs["name"] = np.bytes_(f"Channel {i}")


@pytest.fixture
def scope(camera: Path) -> Path:
    """Scan001's scope: shots 1, 2 and 4 (shot 3 has no file), two channels.

    Channel 0 is flat at the shot number; channel 1 a ramp times the shot.
    Bin 1 is shots 1-2, bin 2 shots 3-4.
    """
    folder = camera / SCOPE
    folder.mkdir()
    for shot in (1, 2, 4):
        write_scope(
            folder / f"Scan001_{SCOPE}_{shot:03d}.dat",
            [np.full(6, float(shot)), np.arange(6.0) * shot],
        )
    return camera


def _png(response) -> Image.Image:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    return Image.open(io.BytesIO(response.content))


def test_a_scope_alone_is_a_row_of_trace_tiles(client: TestClient, scope: Path) -> None:
    page = post(client, scan=1, page="images", device=SCOPE)
    assert 'data-kind="trace"' in page and 'id="strip"' in page
    assert 'data-bin="1"' in page and 'data-bin="2"' in page
    assert "only camera" not in page
    # A scope's own controls: its channel; no colour map, clean-ups or ROI.
    assert 'name="channel"' in page and ">Channel 1<" in page
    assert 'name="cmap"' not in page and 'name="hits"' not in page
    assert "data-roi-open" not in page
    assert 'id="bigview"' in page and 'id="big-trace"' in page
    assert "images.trace_tile(" in page  # the notebook code reads the traces


def test_a_scope_stacks_under_a_camera(
    client: TestClient, scope: Path, settings_dir: Path
) -> None:
    post(client, scan=1, page="images", device=CAM, addcam="1")
    page = post(client, scan=1, page="images", **{"device@2": SCOPE})
    assert 'id="strip"' in page and 'id="strip@2"' in page
    assert page.count('data-kind="trace"') == 1 and page.count('data-kind="image"') == 1
    assert 'name="cmap"' in page and 'name="cmap@2"' not in page
    assert 'name="channel@2"' in page
    # The scope's tiles carry its channel; the camera's carry none.
    assert "channel=0" in page


def test_the_channel_saves_per_row(
    client: TestClient, scope: Path, settings_dir: Path
) -> None:
    page = post(client, scan=1, page="images", device=SCOPE, channel="1")
    saved = json.loads((settings_dir / "default.json").read_text())
    assert saved["image_channel"] == 1
    assert '<option value="1" selected>Channel 1</option>' in page
    assert "channel=1" in page


def _tile(client: TestClient, **query: object):
    base = {"day": DAY, "scan": 1, "device": SCOPE}
    return client.get("/api/images/tile.png", params={**base, **query})


def test_a_trace_tile_is_a_small_plot(client: TestClient, scope: Path) -> None:
    image = _png(_tile(client, strip="bins", index=0, channel=1))
    assert image.size == (320, 200)
    _png(_tile(client, strip="shots", index=4, zmin=0, zmax=30))


def test_a_shot_with_no_file_is_a_404_naming_why(
    client: TestClient, scope: Path
) -> None:
    response = _tile(client, strip="shots", index=3)
    assert response.status_code == 404
    assert response.json()["detail"] == "file not found"


def test_a_channel_the_scope_lacks_is_a_404(client: TestClient, scope: Path) -> None:
    response = _tile(client, strip="bins", index=0, channel=5)
    assert response.status_code == 404
    assert "no channel 5" in response.json()["detail"]


def _trace_y(client: TestClient, **query: object) -> list[float]:
    base = {"day": DAY, "scan": 1, "device": SCOPE}
    response = client.get("/api/images/trace", params={**base, **query})
    assert response.status_code == 200, response.text
    (line,) = response.json()["figure"]["data"]
    return line["y"]


def test_a_bin_is_its_shots_mean_sample_by_sample(
    client: TestClient, scope: Path
) -> None:
    # Bin 1: shots 1 and 2 → channel 1 is the ramp times 1.5.
    assert _trace_y(client, strip="bins", index=0, channel=1) == pytest.approx(
        list(np.arange(6.0) * 1.5)
    )
    # Bin 2: shot 3 has no file, so it is shot 4 alone.
    assert _trace_y(client, strip="bins", index=1) == pytest.approx([4.0] * 6)


def test_a_shot_tile_is_that_shot(client: TestClient, scope: Path) -> None:
    assert _trace_y(client, strip="shots", index=2, channel=1) == pytest.approx(
        list(np.arange(6.0) * 2)
    )


def test_the_shared_y_range_spans_every_tile(client: TestClient, scope: Path) -> None:
    response = client.get(
        "/api/images/levels",
        params={"day": DAY, "scan": 1, "device": SCOPE, "strip": "shots", "channel": 1},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Shots 1, 2 and 4 (shot 3 has none): the ramp up to 5 × 4.
    assert (body["zmin"], body["zmax"], body["tiles"]) == (0.0, 20.0, 3)


def test_the_strip_route_lists_a_scopes_tiles(client: TestClient, scope: Path) -> None:
    response = client.get(
        "/api/images/strip",
        params={"day": DAY, "scan": 1, "device": SCOPE, "channel": 1},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["tiles"]) == 2 and "channel=1" in body["tiles"][0]["url"]
    assert "trace_tile" in body["code"]


def test_a_camera_has_no_trace_route(client: TestClient, scope: Path) -> None:
    response = client.get(
        "/api/images/trace", params={"day": DAY, "scan": 1, "device": CAM}
    )
    assert response.status_code == 404


def test_reading_traces_leaves_the_share_untouched(
    client: TestClient, scope: Path, share: Path
) -> None:
    before = sorted(
        p.relative_to(share) for p in share.rglob("*") if EXPERIMENT in str(p)
    )
    post(client, scan=1, page="images", device=SCOPE)
    for index in (0, 1):
        _tile(client, strip="bins", index=index)
    _tile(client, strip="shots", index=3)
    after = sorted(
        p.relative_to(share) for p in share.rglob("*") if EXPERIMENT in str(p)
    )
    assert after == before


def test_the_shared_y_range_is_over_the_tiles_shown(
    client: TestClient, scope: Path
) -> None:
    params = {"day": DAY, "scan": 1, "device": SCOPE, "strip": "shots", "channel": 1}
    body = client.get("/api/images/levels", params={**params, "shown": "1,2"}).json()
    # Shots 1 and 2 alone: the ramp up to 5 × 2.
    assert (body["zmin"], body["zmax"], body["tiles"]) == (0.0, 10.0, 2)


# --- the row's x window ------------------------------------------------------

WINDOW = {"xmin": 2e-9, "xmax": 4e-9}  # samples 2, 3 and 4 of the 1 ns axis


def test_a_windowed_tile_is_drawn(client: TestClient, scope: Path) -> None:
    _png(_tile(client, strip="bins", index=0, channel=1, **WINDOW))
    _png(_tile(client, strip="bins", index=0, xmin=3e-9))  # one side only


def test_a_window_with_no_samples_is_a_404(client: TestClient, scope: Path) -> None:
    response = _tile(client, strip="bins", index=0, xmin=1.0, xmax=2.0)
    assert response.status_code == 404
    assert "no samples" in response.json()["detail"]


def test_a_reversed_window_is_a_422(client: TestClient, scope: Path) -> None:
    response = _tile(client, strip="bins", index=0, xmin=4e-9, xmax=2e-9)
    assert response.status_code == 422


def test_the_shared_y_range_is_taken_inside_the_window(
    client: TestClient, scope: Path
) -> None:
    params = {"day": DAY, "scan": 1, "device": SCOPE, "strip": "shots", "channel": 1}
    body = client.get("/api/images/levels", params={**params, **WINDOW}).json()
    # The ramp's samples 2-4 times the shot (1, 2, 4): from 2 up to 16.
    assert (body["zmin"], body["zmax"], body["tiles"]) == (2.0, 16.0, 3)


def test_the_big_view_opens_on_the_window_with_the_whole_trace(
    client: TestClient, scope: Path
) -> None:
    base = {"day": DAY, "scan": 1, "device": SCOPE, "strip": "shots", "index": 2}
    body = client.get(
        "/api/images/trace", params={**base, "channel": 1, **WINDOW}
    ).json()
    (line,) = body["figure"]["data"]
    assert len(line["y"]) == 6  # every sample: it zooms out past the window
    assert body["figure"]["layout"]["xaxis"]["range"] == pytest.approx([2e-9, 4e-9])


def test_the_window_saves_per_row_and_rides_on_the_tiles(
    client: TestClient, scope: Path, settings_dir: Path
) -> None:
    page = post(client, scan=1, page="images", device=SCOPE, xmin="2e-9", xmax="4e-9")
    saved = json.loads((settings_dir / "default.json").read_text())
    assert (saved["image_xmin"], saved["image_xmax"]) == (2e-9, 4e-9)
    assert 'name="xmin"' in page and "xmin=2e-09" in page and "xmax=4e-09" in page


def test_a_reversed_window_in_the_boxes_draws_the_whole_trace(
    client: TestClient, scope: Path
) -> None:
    page = post(client, scan=1, page="images", device=SCOPE, xmin="4e-9", xmax="2e-9")
    assert "must be above" in page
    assert "xmin=" not in page


def test_a_camera_row_has_no_x_window(client: TestClient, camera: Path) -> None:
    page = post(client, scan=1, page="images", device=CAM)
    assert 'name="xmin"' not in page


def test_the_notebook_code_crops_to_the_window(client: TestClient, scope: Path) -> None:
    page = post(client, scan=1, page="images", device=SCOPE, xmin="2e-9", xmax="4e-9")
    assert "images.windowed(trace, (2e-09, 4e-09))" in page


def test_a_windowed_tile_is_cropped(client: TestClient, scope: Path) -> None:
    """The window changes what a tile draws (a tile that ignored it would not)."""
    whole = _tile(client, strip="bins", index=0, channel=1).content
    cropped = _tile(client, strip="bins", index=0, channel=1, **WINDOW).content
    assert whole != cropped


def test_the_big_view_s_y_range_is_the_window_s_samples(
    client: TestClient, scope: Path
) -> None:
    # Index 2 of the shots strip is shot 2: its ramp 0, 2, ..., 10 has 4, 6
    # and 8 inside the window, padded by 4 % of that span (never 0 to 10).
    base = {"day": DAY, "scan": 1, "device": SCOPE, "strip": "shots", "index": 2}
    body = client.get(
        "/api/images/trace", params={**base, "channel": 1, **WINDOW}
    ).json()
    assert body["figure"]["layout"]["yaxis"]["range"] == pytest.approx([3.84, 8.16])


def test_a_window_missing_every_tile_says_so_on_the_levels(
    client: TestClient, scope: Path
) -> None:
    params = {"day": DAY, "scan": 1, "device": SCOPE, "strip": "shots", "channel": 1}
    response = client.get(
        "/api/images/levels", params={**params, "xmin": 1.0, "xmax": 2.0}
    )
    assert response.status_code == 404
    assert "x window" in response.json()["detail"]


def test_a_blank_window_box_is_no_limit(
    client: TestClient, scope: Path, settings_dir: Path
) -> None:
    post(client, scan=1, page="images", device=SCOPE, xmin="  ", xmax="4e-9")
    saved = json.loads((settings_dir / "default.json").read_text())
    assert (saved["image_xmin"], saved["image_xmax"]) == (None, 4e-9)


@pytest.mark.parametrize(
    ("units", "value", "text"),
    [("s", 2e-6, "2 µs"), ("V", 0.5, "500 mV"), (None, 1500.0, "1.5 k")],
)
def test_si_units_tick_with_a_prefix(units, value, text):
    from geecs_portal.plotter.routes.images import tick_formatter

    assert tick_formatter(units)(value) == text


@pytest.mark.parametrize("units", ["nm", "a.u."])
def test_other_units_tick_as_plain_numbers(units):
    # A spectrometer's wavelength in nm would otherwise read "1 knm".
    from geecs_portal.plotter.routes.images import tick_formatter

    assert tick_formatter(units) is None


def test_a_one_channel_spectrometer_ignores_a_scopes_saved_channel(
    client: TestClient, camera: Path, settings_dir: Path
) -> None:
    # A spectrometer's per-shot text files: one channel (wavelength, counts).
    spectro = camera / "SPEC-Thorlabs"
    spectro.mkdir()
    for shot in (1, 2, 3, 4):
        (spectro / f"Scan001_SPEC-Thorlabs_{shot:03d}.txt").write_text(
            "".join(f"{700 + i}\t{shot * i}\r\n" for i in range(5))
        )
    # The row last showed a scope on channel 3, then the spectrometer.
    post(client, scan=1, page="images", device=SCOPE, channel="3")
    page = post(client, scan=1, page="images", device="SPEC-Thorlabs", strip="shots")
    assert "channel=0" in page and "channel=3" not in page
    tile = client.get(
        "/api/images/tile.png",
        params={
            "day": DAY,
            "scan": 1,
            "device": "SPEC-Thorlabs",
            "strip": "shots",
            "index": 2,
            "channel": 0,
        },
    )
    assert tile.status_code == 200, tile.text
    # The saved channel stays the scope's.
    assert json.loads((settings_dir / "default.json").read_text())["image_channel"] == 3
