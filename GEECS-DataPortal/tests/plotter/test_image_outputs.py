"""Analysis outputs as rows on the Images page.

An analysis run's per-shot products under ``{day}/analysis/ScanNNN/<folder>/``
(``ScanNNN_<stem>_SSS.png`` images, ``.txt`` column tables) are listed
beside the device folders as ``analysis:<folder>``, and drawn like a
camera's or a scope's row: the same bins, tiles, levels and big view, and
several scans pooled. GEECS-Data-Utils' ``io.analysis_outputs`` finds them.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geecs_portal.plotter import images, multi
from tests.plotter.conftest import CAM, EXPERIMENT, SCAN_VARIABLE, post

DAY = "2026-10-02"
IMAGE = "analysis:py-magspec"
TABLE = "analysis:py-spec"


def _image(path: Path, level: int) -> None:
    pixels = np.full((4, 6), level, dtype=np.uint16)
    pixels[0, 0] = 1000
    Image.fromarray(pixels).save(path)


def _table(path: Path, shot: int) -> None:
    rows = "\n".join(f"{x}\t{x * shot}\t{10 * shot}" for x in (1.0, 2.0, 3.0))
    path.write_text("Momentum_GeV/c\tCharge_fC\tChargeDen_pC/GeV/c\n" + rows + "\n")


def _outputs(day: Path, scan: int, shots: tuple[int, ...], level: int = 10) -> Path:
    """``analysis/ScanNNN``: a magspec image, a spectrum table and a mixed folder."""
    out = day / "analysis" / f"Scan{scan:03d}"
    for name in ("py-magspec", "py-spec", "py-ebeam"):
        (out / name).mkdir(parents=True)
    for shot in shots:
        tag = f"Scan{scan:03d}"
        _image(out / "py-magspec" / f"{tag}_py-magspec_{shot:03d}.png", level * shot)
        _table(out / "py-spec" / f"{tag}_py-spec_{shot:03d}.txt", shot)
        _image(out / "py-ebeam" / f"{tag}_py-ebeam_{shot:03d}.png", level)
        _table(out / "py-ebeam" / f"{tag}_py-ebeamX_{shot:03d}.txt", shot)
    return out


@pytest.fixture
def outputs(camera: Path) -> Path:
    """Scan001's analysis outputs for shots 1, 2 and 4 (shot 3 has none)."""
    return _outputs(camera.parent.parent, 1, (1, 2, 4))


def _png(response) -> Image.Image:
    assert response.status_code == 200, response.text
    return Image.open(io.BytesIO(response.content))


def _get(client: TestClient, route: str, **query: object):
    base = {"day": DAY, "scan": 1}
    return client.get(f"/api/images/{route}", params={**base, **query})


def test_the_outputs_follow_the_device_folders(
    client: TestClient, outputs: Path
) -> None:
    listed = _get(client, "devices").json()["devices"]
    assert listed[-3:] == ["analysis:py-ebeam", IMAGE, TABLE]
    assert CAM in listed[:-3]


def test_the_picker_marks_them_as_analysis_outputs(
    client: TestClient, outputs: Path
) -> None:
    page = post(client, scan=1, page="images", device=CAM)
    assert f'<option value="{IMAGE}">analysis · py-magspec</option>' in page


def test_an_image_output_is_a_camera_row(client: TestClient, outputs: Path) -> None:
    page = post(client, scan=1, page="images", device=IMAGE)
    assert 'data-kind="image"' in page and 'data-bin="2"' in page
    assert 'name="cmap"' in page
    # The shot's own file; bin 1 averages shots 1 and 2 (10 and 20).
    assert _png(_get(client, "tile.png", device=IMAGE, strip="shots", index=2))
    pixels = images.tile_array(
        outputs.parent.parent / "scans" / "Scan001",
        pd.DataFrame({"Bin #": [1, 1, 2, 2]}),
        IMAGE,
        (1, 2),
    )
    assert pixels[1, 1] == pytest.approx(15.0)


def test_a_shot_with_no_output_is_a_404_naming_why(
    client: TestClient, outputs: Path
) -> None:
    response = _get(client, "tile.png", device=IMAGE, strip="shots", index=3)
    assert response.status_code == 404
    assert response.json()["detail"] == "file not found"


def test_the_shared_scale_reads_the_outputs(client: TestClient, outputs: Path) -> None:
    levels = _get(client, "levels", device=IMAGE, strip="shots").json()
    assert (levels["zmin"], levels["zmax"], levels["tiles"]) == (10.0, 1000.0, 3)


def test_a_table_output_is_a_trace_row_its_columns_the_channels(
    client: TestClient, outputs: Path
) -> None:
    page = post(client, scan=1, page="images", device=TABLE, channel="1")
    assert 'data-kind="trace"' in page
    assert '<option value="0">Charge_fC</option>' in page
    assert '<option value="1" selected>ChargeDen_pC/GeV/c</option>' in page
    assert _png(
        _get(client, "tile.png", device=TABLE, strip="bins", index=0, channel=1)
    )
    figure = _get(client, "trace", device=TABLE, strip="shots", index=4).json()
    assert (figure["points"], figure["shots"]) == (3, 1)
    levels = _get(client, "levels", device=TABLE, strip="shots", channel=1).json()
    assert (levels["zmin"], levels["zmax"]) == (10.0, 40.0)


def test_a_picked_folder_of_several_series_offers_each(
    client: TestClient, outputs: Path
) -> None:
    page = post(client, scan=1, page="images", device="analysis:py-ebeam")
    assert 'data-kind="image"' in page  # its own series: the image
    assert ">analysis · py-ebeam › py-ebeamX</option>" in page
    page = post(client, scan=1, page="images", device="analysis:py-ebeam/py-ebeamX")
    assert 'data-kind="trace"' in page


def test_pooled_scans_read_each_scans_own_outputs(
    camera: Path, outputs: Path, share: Path
) -> None:
    day = share / EXPERIMENT / "Y2026" / "10-Oct" / "26_1003"
    folder = day / "scans" / "Scan001"
    folder.mkdir(parents=True)
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
    _outputs(day, 1, (1, 2), level=100)
    found = multi.scan_set(
        [(date(2026, 10, 2), 1), (date(2026, 10, 3), 1)],
        None,
        experiment=EXPERIMENT,
        base_path=share,
    )
    # Merged row 5 is the other scan's shot 1: its own file (100).
    assert images.tile_array(found, found.frame, IMAGE, (5,))[1, 1] == 100
    trace = images.trace_tile(found, found.frame, TABLE, (4, 5), channel=1)
    assert trace.data[:, 1].tolist() == [25.0, 25.0, 25.0]


def test_listing_and_reading_create_nothing(
    client: TestClient, outputs: Path, share: Path
) -> None:
    before = sorted(share.rglob("*"))
    _get(client, "devices")
    _get(client, "tile.png", device="analysis:nope", strip="shots", index=1)
    _get(client, "tile.png", device=IMAGE, strip="shots", index=3)
    post(client, scan=2, page="images", device=IMAGE)
    after = sorted(p for p in share.rglob("*") if "plotter-settings" not in str(p))
    assert after == [p for p in before if "plotter-settings" not in str(p)]


def test_a_bin_of_moving_axes_averages_index_wise_and_says_so(
    client: TestClient, outputs: Path
) -> None:
    # Bin 1 is shots 1 and 2; shot 2's axis is moved by 0.1 (its own field).
    rows = "\n".join(f"{x + 0.1}\t{2 * x}\t20" for x in (1.0, 2.0, 3.0))
    (outputs / "py-spec" / "Scan001_py-spec_002.txt").write_text(
        "Momentum_GeV/c\tCharge_fC\tChargeDen_pC/GeV/c\n" + rows + "\n"
    )
    body = _get(client, "trace", device=TABLE, strip="bins", index=0, channel=1).json()
    assert body["shots"] == 2 and "averaged sample by sample" in body["note"]
    x = body["figure"]["data"][0]["x"]
    assert x == pytest.approx([1.05, 2.05, 3.05])
    assert _png(_get(client, "tile.png", device=TABLE, strip="bins", index=0))
    one = _get(client, "trace", device=TABLE, strip="shots", index=1).json()
    assert one["note"] == ""
