"""The Images page's scan scalars: picked s-file columns summarized under the conditions."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_portal.plotter.settings import (
    MAX_SCALARS,
    PlotterSettings,
    SettingsError,
    merged,
)
from tests.plotter.conftest import CAM, SCAN_VARIABLE, post

CHARGE = "U_ICT charge"  # 10, 12, NaN, 30 in Scan001
CHARGE_ABOVE_11 = (
    '{"groups":[{"conditions":[{"column":"U_ICT charge","low":11,"high":"inf"}]}],'
    '"nan_policy":"exclude"}'
)


class TestSettings:
    def test_none_by_default_and_an_old_set_opens(self) -> None:
        s = PlotterSettings.model_validate({"schema_version": 1})
        assert s.image_scalars == [] and s.image_scalar_stat == "mean_std"

    def test_add_drop_and_the_statistic(self) -> None:
        s = merged(PlotterSettings(), {"scaladd": CHARGE})
        s = merged(s, {"scaladd": SCAN_VARIABLE})
        s = merged(s, {"scaladd": CHARGE})  # already there: once
        assert s.image_scalars == [CHARGE, SCAN_VARIABLE]
        s = merged(s, {"scaldrop": CHARGE, "scalstat": "median_iqr"})
        assert s.image_scalars == [SCAN_VARIABLE]
        assert s.image_scalar_stat == "median_iqr"
        with pytest.raises(SettingsError):
            merged(s, {"scalstat": "mode"})

    def test_at_most_max(self) -> None:
        s = PlotterSettings(image_scalars=[f"c{i}" for i in range(MAX_SCALARS)])
        with pytest.raises(SettingsError, match="at most"):
            merged(s, {"scaladd": "one more"})


def test_nothing_shows_until_a_scalar_is_picked(
    client: TestClient, camera: Path
) -> None:
    page = post(client, scan=1, page="images", device=CAM)
    assert 'id="scan-scalars"' not in page
    assert 'id="scalars-group"' in page and "none: pick a column" in page


def test_the_line_under_the_conditions(client: TestClient, camera: Path) -> None:
    post(client, scan=1, page="images", device=CAM, scaladd=SCAN_VARIABLE)
    page = post(client, scan=1, page="images", scaladd=CHARGE)
    assert 'id="scan-scalars"' in page
    assert "U_Jet Position: 1.5 ± 0.5774" in page
    assert "U_ICT charge: 17.33 ± 11.02 (n=3)" in page  # the NaN shot left out
    assert "mean ± std over 4 shots" in page
    # the copied grid's line
    found = re.search(r'data-scalars="([^"]*)"', page)
    line = json.loads(html.unescape(found.group(1)))
    assert line.startswith("U_Jet Position: 1.5 ± 0.5774  ·  U_ICT charge:")
    assert line.endswith("(mean ± std over 4 shots)")


def test_the_shot_filters_apply(client: TestClient, camera: Path) -> None:
    body = client.get(
        "/api/images/scalars",
        params={
            "day": "2026-10-02",
            "scan": 1,
            "cols": json.dumps([CHARGE, "not here"]),
            "stat": "median_iqr",
            "filters": CHARGE_ABOVE_11,
        },
    )
    assert body.status_code == 200 and body.headers["cache-control"] == "no-cache"
    got = body.json()
    assert got["shots"] == 2  # 12 and 30
    charge, absent = got["scalars"]
    assert charge["center"] == pytest.approx(21.0)
    assert charge["text"] == "U_ICT charge: 21 [16.5 – 25.5]"
    assert absent["missing"] and absent["text"] == "not here: not in this scan"
    assert "summarize_columns" in got["code"] and "shot_rows" in got["code"]


def test_the_route_refuses_bad_parameters(client: TestClient, camera: Path) -> None:
    base = {"day": "2026-10-02", "scan": 1}
    assert (
        client.get("/api/images/scalars", params={**base, "cols": "x"}).status_code
        == 400
    )
    bad_stat = {**base, "cols": "[]", "stat": "mode"}
    assert client.get("/api/images/scalars", params=bad_stat).status_code == 400
