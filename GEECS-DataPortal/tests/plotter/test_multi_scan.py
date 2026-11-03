"""Several scans on the XY plot: overlay (a trace per scan) and merged (one dataset)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.plotter.conftest import EXPERIMENT, SCAN_VARIABLE, post

from geecs_portal.plotter import figures
from geecs_portal.plotter.settings import PlotterSettings, ScanRef, merged

CHARGE = "U_ICT charge"
#: A column only the other day's scan has.
ONLY_THERE = "U_Only There"
OTHER_DAY = date(2026, 10, 3)


@pytest.fixture
def two_days(share: Path) -> Path:
    """The share plus Scan001 of 2026-10-03, stepping the jet too."""
    day = share / EXPERIMENT / "Y2026" / "10-Oct" / "26_1003"
    folder = day / "scans" / "Scan001"
    folder.mkdir(parents=True)
    (folder / "ScanInfoScan001.ini").write_text(
        "[Scan Info]\n"
        'Scan No = "1"\n'
        f'Scan Parameter = "{SCAN_VARIABLE}"\n'
        'Start = "1"\nEnd = "2"\nStep size = "1"\nShots per step = "1"\n'
        'ScanStartInfo = "the other day"\n'
    )
    (day / "analysis").mkdir()
    pd.DataFrame(
        {
            "Shotnumber": [1, 2],
            "Bin #": [1, 2],
            SCAN_VARIABLE: [1.0, 2.0],
            CHARGE: [20.0, 40.0],
            ONLY_THERE: [7.0, 8.0],
            # A Bluesky s-file's own scan column: the scan number per row.
            "scan": [1, 1],
        }
    ).to_csv(day / "analysis" / "s1.txt", sep="\t", index=False)
    return share


def figure(client: TestClient, **params: object) -> dict:
    query = {"day": "2026-10-02", "scan": 1, "y": CHARGE, "x": SCAN_VARIABLE}
    response = client.get("/api/figure", params={**query, **params})
    assert response.status_code == 200, response.text
    return response.json()


def points(data: list) -> list:
    """Each trace's name and points: what the snippet must rebuild (its colours
    are the notebook's, real hex where the page has theme tokens)."""
    return [
        (
            t.get("name"),
            t.get("x"),
            t.get("y"),
            *(t.get("error_y", {}).get(k) for k in ("array", "arrayminus")),
        )
        for t in data
    ]


def rebuilt(body: dict, share: Path) -> dict:
    """Run the response's snippet against *share*; the page form of its figure."""
    code = body["code"].replace(
        "experiment='Testexp'", f"experiment='Testexp', base_path={str(share)!r}"
    )
    namespace: dict = {}
    exec(code.replace("fig.show()\n", ""), namespace)
    return figures.page_figure(namespace["fig"])


class TestMerged:
    def test_every_scan_s_shots_are_one_dataset(self, client, two_days):
        body = figure(client, scans="2026-10-03:1", multi="merged")
        assert body["multi"] == "merged"
        assert body["scans"] == ["10-02 S001", "10-03 S001"]
        assert (body["pass"], body["total"]) == (6, 6)
        trace = body["figure"]["data"][0]
        assert trace["y"] == [10.0, 12.0, None, 30.0, 20.0, 40.0]
        assert trace["x"] == [1.0, 1.0, 2.0, 2.0, 1.0, 2.0]

    def test_the_code_rebuilds_the_merged_plot(self, client, two_days):
        body = figure(client, scans="2026-10-03:1", multi="merged")
        assert "multi.merged_frame(refs" in body["code"]
        assert points(rebuilt(body, two_days)["data"]) == points(body["figure"]["data"])

    def test_one_fit_to_the_pooled_shots(self, client, two_days):
        body = figure(
            client, scans="2026-10-03:1", multi="merged", yfit="polynomial", yorder=1
        )
        (fit,) = body["fits"]
        assert fit["points"] == 5  # shot 3's NaN is left out
        assert fit["scan"] == ""

    def test_bins_are_by_x_not_bin_number(self, client, two_days):
        # Bin 1 is X=1 in Scan001 of the 2nd but both shots of bin 1 of the
        # 3rd's are X=1 and 2: by Bin # they would mix two settings.
        body = figure(client, scans="2026-10-03:1", multi="merged", view="bin")
        trace = body["figure"]["data"][0]
        assert trace["x"] == [1.0, 2.0]
        assert trace["y"] == [12.0, 35.0]  # medians of {10, 12, 20} and {30, 40}
        assert "bin_col='U_Jet Position'" in body["code"]
        assert points(rebuilt(body, two_days)["data"]) == points(body["figure"]["data"])

    def test_binning_merged_scans_by_shot_number_is_refused(self, client, two_days):
        response = client.get(
            "/api/figure",
            params={
                "day": "2026-10-02",
                "scan": 1,
                "y": CHARGE,
                "scans": "2026-10-03:1",
                "multi": "merged",
                "view": "bin",
            },
        )
        assert response.status_code == 400
        assert "pick an X" in response.json()["detail"]

    def test_a_shot_filter_can_name_the_scan(self, client, two_days):
        scan_one = json.dumps(
            {"groups": [{"conditions": [{"column": "scan", "low": 1, "high": 1}]}]}
        )
        body = figure(client, scans="2026-10-02:2", multi="merged", filters=scan_one)
        assert (body["pass"], body["total"]) == (4, 6)
        assert points(rebuilt(body, two_days)["data"]) == points(body["figure"]["data"])


class TestOverlay:
    def test_a_trace_per_scan_in_its_own_colour(self, client, two_days):
        body = figure(client, scans="2026-10-03:1")
        assert body["multi"] == "overlay"
        data = body["figure"]["data"]
        assert [t["name"] for t in data] == ["10-02 S001", "10-03 S001"]
        assert [t["marker"]["color"] for t in data] == [
            "$tok:--trace-1",
            "$tok:--trace-2",
        ]
        assert data[0]["y"] == [10.0, 12.0, None, 30.0]
        assert data[1]["y"] == [20.0, 40.0]
        assert body["figure"]["layout"]["showlegend"] is True

    def test_the_scans_take_the_colours_in_order(self, client, two_days):
        body = figure(client, scans="2026-10-02:2,2026-10-03:1", x="")
        assert [t["marker"]["color"] for t in body["figure"]["data"]] == [
            "$tok:--trace-1",
            "$tok:--trace-2",
            "$tok:--trace-3",
        ]

    def test_eight_scans_take_eight_colours(self):
        assert (
            figures.page_figure(
                figures.overlay_figure(
                    [(str(i), {"a": [1.0]}, [1]) for i in range(8)],
                    "a",
                    palette=figures.THEMED_PALETTE,
                )
            )["data"][7]["marker"]["color"]
            == "$tok:--trace-8"
        )

    def test_the_code_rebuilds_the_overlay(self, client, two_days):
        body = figure(client, scans="2026-10-03:1", yfit="polynomial", yorder=1)
        assert "multi.overlay_groups(" in body["code"]
        assert points(rebuilt(body, two_days)["data"]) == points(body["figure"]["data"])

    def test_one_fit_per_scan(self, client, two_days):
        body = figure(client, scans="2026-10-03:1", yfit="polynomial", yorder=1)
        assert [(f["scan"], f["trace"], f["points"]) for f in body["fits"]] == [
            ("10-02 S001", 1, 3),
            ("10-03 S001", 2, 2),
        ]
        # A fit's dashed curve is in its scan's colour, after its points.
        assert body["figure"]["data"][-1]["line"]["color"] == "$tok:--trace-2"

    def test_the_chosen_y_alone_is_drawn(self, client, two_days):
        body = figure(
            client, scans="2026-10-03:1", y2=ONLY_THERE, oy=2, yfit="polynomial"
        )
        data = body["figure"]["data"]
        assert [t["name"] for t in data] == ["10-02 S001", "10-03 S001"]
        assert data[0]["y"] == [None] * 4  # Scan001 of the 2nd has no such column
        assert data[1]["y"] == [7.0, 8.0]
        assert body["fits"] == []  # Y2's own fit is none
        assert body["figure"]["layout"]["yaxis"]["title"]["text"] == ONLY_THERE

    def test_an_overlay_of_a_y_that_is_off_is_refused(self, client, two_days):
        response = client.get(
            "/api/figure",
            params={
                "day": "2026-10-02",
                "scan": 1,
                "y": CHARGE,
                "scans": "2026-10-03:1",
                "oy": 2,
            },
        )
        assert response.status_code == 400
        assert "Y2" in response.json()["detail"]

    def test_more_than_eight_scans_is_refused(self, client, two_days):
        scans = ",".join(f"2026-10-0{d}:{n}" for d in (4, 5) for n in range(1, 5))
        response = client.get(
            "/api/figure",
            params={"day": "2026-10-02", "scan": 1, "y": CHARGE, "scans": scans},
        )
        assert response.status_code == 400
        assert "at most 8" in response.json()["detail"]

    def test_each_scan_is_binned_on_its_own(self, client, two_days):
        body = figure(client, scans="2026-10-03:1", view="bin")
        data = body["figure"]["data"]
        assert [t["y"] for t in data] == [[11.0, 30.0], [20.0, 40.0]]
        assert [t["x"] for t in data] == [[1.0, 2.0], [1.0, 2.0]]
        assert points(rebuilt(body, two_days)["data"]) == points(data)


class TestScans:
    def test_the_main_scan_is_not_drawn_twice(self, client, two_days):
        body = figure(client, scans="2026-10-02:1,2026-10-03:1,2026-10-03:1")
        assert body["scans"] == ["10-02 S001", "10-03 S001"]

    def test_a_missing_extra_scan_is_a_404(self, client, two_days):
        response = client.get(
            "/api/figure",
            params={
                "day": "2026-10-02",
                "scan": 1,
                "y": CHARGE,
                "scans": "2026-10-09:4",
            },
        )
        assert response.status_code == 404
        assert "Scan004 on 2026-10-09" in response.json()["detail"]

    def test_a_malformed_list_is_a_400(self, client, two_days):
        response = client.get(
            "/api/figure",
            params={"day": "2026-10-02", "scan": 1, "y": CHARGE, "scans": "yesterday"},
        )
        assert response.status_code == 400

    def test_the_filter_count_counts_every_scan(self, client, two_days):
        body = client.get(
            "/api/filter-count",
            params={"day": "2026-10-02", "scan": 1, "scans": "2026-10-03:1"},
        ).json()
        assert body == {"pass": 6, "total": 6}

    def test_the_day_lists_its_scans(self, client, two_days):
        body = client.get("/api/day", params={"day": "2026-10-02"}).json()
        assert [s["scan"] for s in body["scans"]] == [1, 2]
        other = client.get("/api/day", params={"day": "2026-10-03"}).json()
        assert other["scans"][0]["description"] == "the other day"
        assert (
            client.get("/api/day", params={"day": "2026-10-09"}).json()["scans"] == []
        )

    def test_a_lookup_never_creates_a_scan_folder(self, client, two_days):
        before = sorted(two_days.rglob("*"))
        client.get("/api/day", params={"day": "2026-10-09"})
        client.get(
            "/api/figure",
            params={
                "day": "2026-10-02",
                "scan": 1,
                "y": CHARGE,
                "scans": "2026-10-09:4",
            },
        )
        assert sorted(two_days.rglob("*")) == before


class TestSettings:
    def test_an_old_set_loads_with_no_extra_scans(self):
        settings = PlotterSettings.model_validate({"schema_version": 1, "y": "a"})
        assert settings.extra_scans == []
        assert (settings.scan_view, settings.overlay_y) == ("overlay", 1)

    def test_the_form_saves_the_scans_and_the_view(self):
        settings = merged(
            PlotterSettings(),
            {
                "scans": "2026-10-03:1,2026-10-02:2,2026-10-03:1",
                "multi": "merged",
                "oy": "2",
            },
        )
        assert settings.extra_scans == [
            ScanRef(day=OTHER_DAY, scan=1),
            ScanRef(day=date(2026, 10, 2), scan=2),
        ]
        assert (settings.scan_view, settings.overlay_y) == ("merged", 2)
        again = PlotterSettings.model_validate_json(settings.model_dump_json())
        assert again == settings

    def test_an_empty_list_clears_the_scans(self):
        settings = PlotterSettings(extra_scans=[ScanRef(day=OTHER_DAY, scan=1)])
        assert merged(settings, {"scans": ""}).extra_scans == []

    @pytest.mark.parametrize(
        "field, value",
        [
            ("scans", "2026-10-03"),
            ("scans", "2026-10-03:0"),
            ("multi", "both"),
            ("oy", "4"),
        ],
    )
    def test_a_bad_value_is_refused(self, client, field, value):
        response = client.post(
            "/",
            data={
                "day": "2026-10-02",
                "scan": "1",
                "set": "default",
                "owner": "default",
                field: value,
            },
        )
        assert response.status_code == 422


class TestPage:
    def test_the_rail_shows_a_chip_per_scan(self, client, two_days):
        html = post(client, scan=1, scans="2026-10-03:1,2026-10-09:4")
        assert "10-03 S001" in html
        assert "10-09 S004" in html
        assert "not on the share" in html
        # The figure asks for the scans found, not the missing one.
        assert "scans=2026-10-03%3A1&amp;" in html
        assert 'data-scans="2026-10-03:1"' in html

    def test_the_lists_are_the_union_of_columns(self, client, two_days):
        html = post(client, scan=1, scans="2026-10-03:1")
        assert f"{ONLY_THERE} · in 1 of 2 scans" in html
        assert '<option value="scan"' in html

    def test_one_scan_shows_no_scan_view(self, client, two_days):
        html = post(client, scan=1)
        assert "Add scan…" in html
        assert 'data-toggle-for="multi"' not in html
        assert "scans=&amp;" in html

    def test_the_map_says_it_draws_the_main_scan(self, client, two_days):
        html = post(client, scan=1, scans="2026-10-03:1", tab="map")
        assert "the map draws 10-02 S001 alone" in html


def test_a_chip_s_colour_is_its_trace_s_with_a_missing_scan_first(client, two_days):
    """The figure colours the scans found; a missing one before them shifts nothing."""
    html = post(client, scan=1, scans="2026-10-09:4,2026-10-03:1")
    before = html[: html.index("10-03 S001")]
    # The last swatch before the chip's name is its own.
    assert before[before.rindex('class="swatch t') :].startswith('class="swatch t2"')
    body = figure(client, scans="2026-10-03:1")
    assert body["figure"]["data"][1]["marker"]["color"] == "$tok:--trace-2"
