"""The Images page's scan conditions: condition columns' labels over the scan."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from geecs_data_utils.scan_conditions import labels_path, write_condition_labels

from tests.plotter.conftest import CAM, EXPERIMENT, post

DAY_DIR = Path(EXPERIMENT) / "Y2026" / "10-Oct" / "26_1002"
LABELS = {"target": ["Al foil", "Cu foil", "nothing"], "door": ["open", "shut"]}


def label_scan(share: Path, target: list, door: list) -> None:
    """Scan001's s-file gains condition columns, its analysis folder their labels."""
    day = share / DAY_DIR
    sfile = day / "analysis" / "s1.txt"
    frame = pd.read_csv(sfile, sep="\t").assign(target=target, door=door)
    frame.to_csv(sfile, sep="\t", index=False)
    folder = day / "analysis" / "Scan001"
    folder.mkdir()
    write_condition_labels(labels_path(folder, "Scan001", "where"), "where", LABELS)


def test_the_images_page_lists_each_condition(
    client: TestClient, share: Path, camera: Path
) -> None:
    label_scan(share, target=[0, 0, 0, 0], door=[0, 0, 1, None])
    page = post(client, scan=1, page="images", device=CAM)
    assert 'id="conditions"' in page
    assert "target:</span> <b>Al foil</b>" in page
    # the door changed during the scan: mixed, flagged, with the split
    assert "door:</span> <b>mixed</b>" in page
    assert "changed mid-scan" in page
    assert "open 2 · shut 1 · unknown 1" in page
    assert "conditions-help" in page  # the "?" that says where they come from


def test_a_scan_without_conditions_lists_none(client: TestClient, camera: Path) -> None:
    assert 'id="conditions"' not in post(client, scan=1, page="images", device=CAM)


def test_the_conditions_route(client: TestClient, share: Path) -> None:
    label_scan(share, target=[2, 2, 2, 2], door=[1, 1, 1, 1])
    body = client.get("/api/conditions", params={"day": "2026-10-02", "scan": 1})
    assert body.status_code == 200
    assert body.headers["cache-control"] == "no-cache"
    by = {c["column"]: c for c in body.json()["conditions"]}
    assert by["target"]["label"] == "nothing" and not by["target"]["mixed"]
    assert by["door"] == {
        "column": "door",
        "label": "shut",
        "mixed": False,
        "detail": "",
        "counts": {"shut": 4},
        "unknown": 0,
    }
    empty = client.get("/api/conditions", params={"day": "2026-10-02", "scan": 2})
    assert empty.json() == {"conditions": []}


def test_the_copied_grid_carries_the_condition_lines(
    client: TestClient, share: Path, camera: Path
) -> None:
    import html
    import json
    import re

    label_scan(share, target=[0, 0, 0, 0], door=[0, 0, 1, None])
    page = post(client, scan=1, page="images", device=CAM)
    found = re.search(r'data-conditions="([^"]*)"', page)
    assert found
    assert json.loads(html.unescape(found.group(1))) == [
        "target: Al foil",
        "door: mixed — changed mid-scan (open 2 · shut 1 · unknown 1)",
    ]
