"""Portal app tests — hermetic: fake catalogs, no network, no data root."""

from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
from fastapi.testclient import TestClient

from geecs_data_utils.tiled_catalog import (
    CatalogStatus,
    RunDetail,
    StubCatalog,
    metadata_rows,
    summary_from_metadata,
)

from geecs_portal.app import create_app

TEST_DAY = date(2026, 7, 12)

#: LabVIEW-epoch base for acq_timestamp fixtures (the wire convention).
_LV = 3_870_000_000.0


def _epoch(hour: int, minute: int = 0) -> float:
    return datetime(
        TEST_DAY.year, TEST_DAY.month, TEST_DAY.day, hour, minute
    ).timestamp()


def _start_doc(scan_number: int, **extra) -> dict:
    doc = {
        "uid": f"uid-{scan_number:03d}",
        "time": _epoch(9, scan_number),
        "geecs_event_schema": 1,
        "plan_name": "geecs_step_scan",
        "motor": None,
        "num_points": 1,
        "shots_per_step": 10,
        "experiment": "Undulator",
        "scan_number": scan_number,
        "save_sets": ["Amp4In"],
    }
    doc.update(extra)
    return doc


def _detail(
    scan_number: int = 2, with_data: bool = True, motor: str | None = None
) -> RunDetail:
    start = _start_doc(scan_number, motor=motor)
    stop = {"time": start["time"] + 30.0, "exit_status": "success"}
    data = None
    if with_data:
        data = pd.DataFrame(
            {
                "scan_event_index": [1, 2, 3],  # id machinery (1-based per schema)
                "cam-MaxCounts": [10.0, 12.5, 11.0],
                "cam-acq_timestamp": [_LV + 1.0, _LV + 2.0, _LV + 3.0],  # companion
                "cam-label": ["a", "b", "c"],  # non-numeric: never plottable
                "telemetry_dev-val": ["1.5", "2.5", "3.5"],  # dtype-tolerant
                "ts_cam-MaxCounts": [  # reader-side event times (Unix)
                    _epoch(9, 30) + i for i in range(3)
                ],
                "cam-dead": [float("nan")] * 3,  # all-NaN: not plottable
                "mono": [4.0, 5.0, 6.0],  # scan-variable readback
            }
        )
    return RunDetail(
        summary=summary_from_metadata(start["uid"], start, stop),
        start_doc=start,
        stop_doc=stop,
        data=data,
    )


class FakeCatalog:
    """ScanCatalog fake: two runs on TEST_DAY, one detail behind each uid."""

    def __init__(self, fail_listing: bool = False):
        self.fail_listing = fail_listing
        self.details = {f"uid-{n:03d}": _detail(n) for n in (1, 2)}

    def probe(self) -> CatalogStatus:
        return CatalogStatus(ok=True, label="fake catalog")

    def list_runs(self, experiment: str, day: date):
        if self.fail_listing:
            raise RuntimeError("listing boom")
        self.listed = (experiment, day)
        # Newest first — the ScanCatalog protocol contract the real
        # TiledScanCatalog implements (day_view renders catalog order).
        return sorted(
            (detail.summary for detail in self.details.values()),
            key=lambda summary: summary.start_time,
            reverse=True,
        )

    def load_run(self, uid: str) -> RunDetail:
        return self.details[uid]


class DownCatalog(FakeCatalog):
    """A catalog whose backend is unreachable (Tiled outage)."""

    def load_run(self, uid: str) -> RunDetail:
        raise ConnectionError("tiled unreachable")


def _client(catalog=None, **kwargs) -> TestClient:
    app = create_app(catalog if catalog is not None else FakeCatalog(), **kwargs)
    return TestClient(app)


class TestHealth:
    def test_health_reports_probe_and_version(self):
        response = _client().get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True and payload["catalog"] == "fake catalog"
        assert payload["version"]  # the /api cache-bust key, for scripts

    def test_health_with_stub_catalog(self):
        response = _client(StubCatalog()).get("/health")
        assert response.status_code == 200
        assert response.json()["ok"] is False


class TestDayView:
    def test_lists_runs_with_scan_numbers(self):
        catalog = FakeCatalog()
        client = _client(catalog, default_experiment="Undulator")
        response = client.get(f"/day/{TEST_DAY.isoformat()}")
        assert response.status_code == 200
        assert "Scan 001" in response.text and "Scan 002" in response.text
        assert catalog.listed == ("Undulator", TEST_DAY)

    def test_experiment_query_overrides_default(self):
        catalog = FakeCatalog()
        client = _client(catalog, default_experiment="Undulator")
        client.get(f"/day/{TEST_DAY.isoformat()}?experiment=Thomson")
        assert catalog.listed[0] == "Thomson"

    def test_bad_date_is_404(self):
        assert _client().get("/day/not-a-date").status_code == 404

    def test_catalog_failure_surfaces_not_500(self):
        response = _client(FakeCatalog(fail_listing=True)).get(
            f"/day/{TEST_DAY.isoformat()}"
        )
        assert response.status_code == 200
        assert "catalog error" in response.text

    def test_empty_day_renders_no_scans_message(self):
        response = _client(StubCatalog()).get(f"/day/{TEST_DAY.isoformat()}")
        assert response.status_code == 200
        assert "No scans recorded" in response.text

    def test_index_redirects_to_today(self):
        response = _client().get("/", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert response.headers["location"].startswith("/day/")

    def test_go_form_redirects_to_day(self):
        response = _client().get(
            "/go?day=2026-07-12&experiment=Bella PW", follow_redirects=False
        )
        assert response.status_code in (302, 307)
        assert response.headers["location"] == "/day/2026-07-12?experiment=Bella+PW"

    def test_filter_narrows_run_list(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get(f"/day/{TEST_DAY.isoformat()}?filter=scan 001")
        assert "Scan 001" in response.text
        assert "Scan 002" not in response.text

    def test_experiment_with_space_is_urlencoded_in_links(self):
        client = _client(FakeCatalog())
        response = client.get(f"/day/{TEST_DAY.isoformat()}?experiment=Bella PW")
        # sticky-query links use urlencode's plus form; either encoding is
        # valid in a query string — the pin is that no raw space leaks.
        assert (
            "experiment=Bella+PW" in response.text
            or "experiment=Bella%20PW" in response.text
        )
        assert "experiment=Bella PW" not in response.text


class TestRunView:
    def test_metadata_rows_render(self):
        response = _client().get("/run/uid-002")
        assert response.status_code == 200
        assert "Scan 002" in response.text
        assert "success" in response.text

    def test_page_has_tabs_and_the_vendored_plotly(self):
        response = _client().get("/run/uid-002")
        assert 'data-pane="overview"' in response.text
        assert 'data-pane="plot"' in response.text
        assert 'data-pane="images"' in response.text
        # The one committed JS asset — version-pinned, served locally.
        assert "/static/plotly-cartesian-" in response.text

    def test_vendored_plotly_is_served(self):
        response = _client().get("/static/plotly-cartesian-3.1.1.min.js")
        assert response.status_code == 200
        assert "plotly.js" in response.text[:200]

    def test_scan_stepper_neighbours_from_the_days_listing(self):
        # uid-001 is the older run (listing is newest first): its only
        # neighbour is uid-002 as "next"; uid-002 has uid-001 as "prev".
        client = _client(FakeCatalog(), default_experiment="Undulator")
        older = client.get("/run/uid-001")
        assert "/run/uid-002?" in older.text
        newer = client.get("/run/uid-002")
        assert "/run/uid-001?" in newer.text

    def test_analysis_state_rides_the_stepper_links(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get(
            "/run/uid-001?y=cam-MaxCounts&y=mono&view=bin"
            "&filters=%7B%22groups%22%3A%5B%5D%7D&tab=plot"
        )
        stepper_line = next(
            line for line in response.text.splitlines() if "/run/uid-002?" in line
        )
        assert "y=cam-MaxCounts" in stepper_line and "y=mono" in stepper_line
        assert "view=bin" in stepper_line
        assert "filters=" in stepper_line

    def test_unknown_run_is_404(self):
        assert _client().get("/run/nope").status_code == 404

    def test_day_list_renders_newest_first(self):
        response = _client(FakeCatalog(), default_experiment="Undulator").get(
            f"/day/{TEST_DAY.isoformat()}"
        )
        assert response.text.index("Scan 002") < response.text.index("Scan 001")


class TestCatalogOutage:
    """A Tiled outage must read as catalog-unavailable, never run-not-found."""

    def test_run_view_is_503(self):
        response = _client(DownCatalog()).get("/run/uid-002")
        assert response.status_code == 503
        assert "catalog unavailable" in response.json()["detail"]

    def test_plot_and_image_are_503(self):
        client = _client(DownCatalog())
        assert client.get("/run/uid-002/plot.png?y=cam-MaxCounts").status_code == 503
        assert client.get("/run/uid-002/image.png?device=cam&shot=1").status_code == 503


class TestCacheHeaders:
    def test_completed_run_plot_is_immutable(self):
        response = _client().get("/run/uid-002/plot.png?y=cam-MaxCounts")
        assert "immutable" in response.headers["cache-control"]

    def test_running_run_plot_must_revalidate(self):
        catalog = FakeCatalog()
        detail = _detail(8)
        catalog.details["uid-008"] = RunDetail(
            summary=summary_from_metadata(
                detail.start_doc["uid"], detail.start_doc, None
            ),
            start_doc=detail.start_doc,
            stop_doc=None,
            data=detail.data,
        )
        response = _client(catalog).get("/run/uid-008/plot.png?y=cam-MaxCounts")
        assert response.headers["cache-control"] == "no-cache"


class TestRunDayGuards:
    def test_no_start_time_and_no_day_never_resolves_todays_folder(self):
        # A run with a broken start time must not fall back to today's
        # same-numbered scan folder (scan numbers restart daily).
        catalog = FakeCatalog()
        detail = _detail(7)
        detail.start_doc["time"] = 0
        catalog.details["uid-007"] = RunDetail(
            summary=summary_from_metadata(
                detail.start_doc["uid"], detail.start_doc, None
            ),
            start_doc=detail.start_doc,
            stop_doc=None,
            data=detail.data,
        )
        client = _client(catalog)
        response = client.get("/run/uid-007/image.png?device=cam&shot=1")
        assert response.status_code == 404
        assert "not resolvable" in response.json()["detail"]


class TestPlot:
    def test_plot_returns_png(self):
        response = _client().get("/run/uid-002/plot.png?y=cam-MaxCounts")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")

    def test_plot_with_x_column(self):
        response = _client().get(
            "/run/uid-002/plot.png?y=cam-MaxCounts&x=scan_event_index"
        )
        assert response.status_code == 200
        assert response.content.startswith(b"\x89PNG")

    def test_non_numeric_column_is_404(self):
        assert _client().get("/run/uid-002/plot.png?y=cam-label").status_code == 404

    def test_all_nan_column_is_404(self):
        assert _client().get("/run/uid-002/plot.png?y=cam-dead").status_code == 404

    def test_numeric_string_telemetry_plots(self):
        response = _client().get("/run/uid-002/plot.png?y=telemetry_dev-val")
        assert response.status_code == 200
        assert response.content.startswith(b"\x89PNG")

    def test_dataless_run_is_404(self):
        catalog = FakeCatalog()
        catalog.details["uid-009"] = _detail(9, with_data=False)
        client = _client(catalog)
        assert client.get("/run/uid-009/plot.png?y=cam-MaxCounts").status_code == 404


class TestAnalysisApi:
    """The /api JSON endpoints: one-liners over the data-utils primitives."""

    _FILTERS = (
        '{"groups":[{"conditions":'
        '[{"column":"cam-MaxCounts","low":10.5,"high":13.0}]}]}'
    )

    def test_columns_carries_provenance_and_schema_semantics(self):
        payload = _client().get("/api/run/uid-002/columns").json()
        names = {c["name"] for c in payload["columns"]}
        assert "cam-MaxCounts" in names
        assert "telemetry_dev-val" in names  # numeric strings plot
        assert "cam-label" not in names  # non-numeric
        assert "cam-dead" not in names  # all-NaN
        assert "scan_event_index" not in names  # id machinery
        assert "cam-acq_timestamp" not in names  # companion machinery
        # No s-file resolvable for the fake: everything is run-doc.
        assert {c["provenance"] for c in payload["columns"]} == {"run"}
        assert payload["total"] == 3

    def test_columns_default_x_is_the_scan_variable(self):
        catalog = FakeCatalog()
        catalog.details["uid-007"] = _detail(7, motor="mono")
        payload = _client(catalog).get("/api/run/uid-007/columns").json()
        assert payload["default_x"] == "mono"

    def test_frame_returns_series_and_shot_key(self):
        payload = (
            _client().get("/api/run/uid-002/frame?cols=cam-MaxCounts&x=mono").json()
        )
        assert payload["series"]["cam-MaxCounts"] == [10.0, 12.5, 11.0]
        assert payload["series"]["mono"] == [4.0, 5.0, 6.0]
        assert payload["shot"] == [1.0, 2.0, 3.0]  # 1-based scan_event_index
        assert payload["pass"] == 3 and payload["total"] == 3
        assert "scan_frame" in payload["code"]

    def test_frame_applies_filters(self):
        payload = (
            _client()
            .get(
                "/api/run/uid-002/frame",
                params={"cols": "cam-MaxCounts", "filters": self._FILTERS},
            )
            .json()
        )
        assert payload["series"]["cam-MaxCounts"] == [12.5, 11.0]
        assert payload["pass"] == 2 and payload["total"] == 3
        assert "apply_filters" in payload["code"]

    def test_filters_that_empty_the_frame_are_not_a_404(self):
        response = _client().get(
            "/api/run/uid-002/frame",
            params={
                "cols": "cam-MaxCounts",
                "filters": '{"groups":[{"conditions":'
                '[{"column":"cam-MaxCounts","low":-2.0,"high":-1.0}]}]}',
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["series"]["cam-MaxCounts"] == []
        assert payload["pass"] == 0

    def test_binned_hand_computed_identity_bins(self):
        payload = (
            _client()
            .get(
                "/api/run/uid-002/binned",
                params={"cols": "cam-MaxCounts", "bincfg": '{"bin_col":"mono"}'},
            )
            .json()
        )
        assert payload["bins"] == [4.0, 5.0, 6.0]
        assert payload["counts"] == [1, 1, 1]
        series = payload["series"]["cam-MaxCounts"]
        assert series["center"] == [10.0, 12.5, 11.0]
        assert series["err_low"] == [0.0, 0.0, 0.0]  # one shot per bin
        assert "bin_frame" in payload["code"]
        assert "BinningConfig" in payload["code"]

    def test_binned_missing_bin_column_is_404(self):
        # The fixture has no "Bin #" column — the default config must
        # refuse honestly, not 500.
        response = _client().get(
            "/api/run/uid-002/binned", params={"cols": "cam-MaxCounts"}
        )
        assert response.status_code == 404
        assert "bin column" in response.json()["detail"]

    def test_filter_count(self):
        payload = (
            _client()
            .get("/api/run/uid-002/filter-count", params={"filters": self._FILTERS})
            .json()
        )
        assert payload == {"pass": 2, "total": 3}

    def test_bad_filters_json_is_400(self):
        response = _client().get(
            "/api/run/uid-002/frame",
            params={"cols": "cam-MaxCounts", "filters": "{not json"},
        )
        assert response.status_code == 400
        assert "filters" in response.json()["detail"]

    def test_unknown_filter_column_is_400(self):
        response = _client().get(
            "/api/run/uid-002/filter-count",
            params={
                "filters": '{"groups":[{"conditions":'
                '[{"column":"nope","low":0,"high":1}]}]}'
            },
        )
        assert response.status_code == 400

    def test_unknown_bincfg_field_is_400(self):
        response = _client().get(
            "/api/run/uid-002/binned",
            params={"cols": "cam-MaxCounts", "bincfg": '{"surprise": 1}'},
        )
        assert response.status_code == 400
        assert "unknown fields" in response.json()["detail"]

    def test_out_of_vocabulary_err_is_400(self):
        response = _client().get(
            "/api/run/uid-002/binned",
            params={"cols": "cam-MaxCounts", "bincfg": '{"err": "bogus"}'},
        )
        assert response.status_code == 400

    def test_unknown_column_is_404(self):
        response = _client().get("/api/run/uid-002/frame?cols=nope")
        assert response.status_code == 404

    def test_more_than_four_y_columns_is_400(self):
        query = "&".join(f"cols=c{i}" for i in range(5))
        response = _client().get(f"/api/run/uid-002/frame?{query}")
        assert response.status_code == 400
        assert "at most 4" in response.json()["detail"]

    def test_completed_run_json_never_immutable(self):
        # The s-file half of the union grows after the run completes
        # (ScanAnalysis appends columns) — a pinned response would hide
        # them until the next portal release.
        for path in (
            "columns",
            "frame?cols=cam-MaxCounts",
            'binned?cols=cam-MaxCounts&bincfg={"bin_col":"mono"}',
            "filter-count",
        ):
            response = _client().get(f"/api/run/uid-002/{path}")
            assert response.status_code == 200, path
            assert response.headers["cache-control"] == "no-cache", path

    def test_api_outage_is_503(self):
        response = _client(DownCatalog()).get("/api/run/uid-002/columns")
        assert response.status_code == 503

    def test_api_unknown_run_is_404(self):
        assert _client().get("/api/run/nope/columns").status_code == 404

    def test_nan_values_serialize_as_null(self):
        # cam-dead is all-NaN and unplottable, but a partial-NaN column
        # must serialize NaN as null (valid JSON), never bare NaN.
        catalog = FakeCatalog()
        detail = _detail(7)
        detail.data.loc[1, "cam-MaxCounts"] = float("nan")
        catalog.details["uid-007"] = detail
        response = _client(catalog).get("/api/run/uid-007/frame?cols=cam-MaxCounts")
        assert response.status_code == 200
        assert b"NaN" not in response.content
        assert response.json()["series"]["cam-MaxCounts"] == [10.0, None, 11.0]

    def test_wrong_typed_bincfg_fields_are_400_not_500(self):
        # BinningConfig is a plain dataclass: type/arity discipline
        # lives in parse_bincfg — these all 500'd before the review fix.
        client = _client()
        bad = [
            '{"err":"percentile","percentiles":[0.5]}',  # arity 1
            '{"percentiles":[0.1,0.2,0.3]}',  # arity 3 (was silently 2)
            '{"min_count":"2"}',  # string int
            '{"ddof":"x"}',  # non-numeric
            '{"ddof":1.5}',  # non-integral
            '{"bin_width":"wide"}',  # string float
            '{"scale_to_sigma":"no"}',  # truthy string ≠ bool
            '{"right":1}',  # int ≠ bool
            '{"bin_col":3}',  # non-string
            '{"bin_edges":[1]}',  # < 2 edges
            '{"value_cols":"v"}',  # bare string, not a list
            '{"dropna":[]}',  # unhashable into a choice field
            '{"err":{"a":1}}',  # unhashable into a choice field
            '{"agg":[1,2]}',  # unhashable into a choice field
            '{"ddof":NaN}',  # json.loads admits NaN literals
            '{"ddof":Infinity}',  # ... and Infinity
            '{"min_count":1e400}',  # float overflow to inf
            '{"quantile_bins":' + "9" * 400 + "}",  # unbounded int
            '{"bin_col":"mono","bin_width":0}',  # /0 → inf → OverflowError
            '{"bin_col":"mono","bin_width":-0.5}',  # empty edge array
        ]
        for bincfg in bad:
            response = client.get(
                "/api/run/uid-002/binned",
                params={"cols": "cam-MaxCounts", "bincfg": bincfg},
            )
            assert response.status_code == 400, bincfg
            assert "bincfg" in response.json()["detail"], bincfg

    def test_degenerate_percentile_pair_is_400(self):
        response = _client().get(
            "/api/run/uid-002/binned",
            params={
                "cols": "cam-MaxCounts",
                "bincfg": '{"bin_col":"mono","err":"percentile",'
                '"percentiles":[0.5,0.5]}',
            },
        )
        assert response.status_code == 400

    def test_na_shot_key_serializes_as_null(self):
        # A union row the event side missed carries pd.NA in the Int64
        # shot key — it must serialize as null, never the string "<NA>".
        catalog = FakeCatalog()
        detail = _detail(7)
        detail.data["scan_event_index"] = pd.array([1, pd.NA, 3], dtype="Int64")
        catalog.details["uid-007"] = detail
        response = _client(catalog).get("/api/run/uid-007/frame?cols=cam-MaxCounts")
        assert response.status_code == 200
        assert b"<NA>" not in response.content
        assert response.json()["shot"] == [1.0, None, 3.0]

    def test_union_shot_key_coalesces_from_shotnumber(self):
        # An s-file-only union row has no event index but does have a
        # Shotnumber — the shot axis must coalesce to it, not go null
        # (Plotly silently drops null-x points from the default plot).
        catalog = FakeCatalog()
        detail = _detail(7)
        detail.data["scan_event_index"] = pd.array([1, 2, pd.NA], dtype="Int64")
        detail.data["Shotnumber"] = [1.0, 2.0, 4.0]
        catalog.details["uid-007"] = detail
        payload = (
            _client(catalog).get("/api/run/uid-007/frame?cols=cam-MaxCounts").json()
        )
        assert payload["shot"] == [1.0, 2.0, 4.0]

    def test_binned_counts_are_integers(self):
        payload = (
            _client()
            .get(
                "/api/run/uid-002/binned",
                params={"cols": "cam-MaxCounts", "bincfg": '{"bin_col":"mono"}'},
            )
            .json()
        )
        assert all(isinstance(count, int) for count in payload["counts"])

    def test_explicit_null_bincfg_fields_mean_absent(self):
        # Doctrine: JSON null = field absent, the default stands — a
        # null into a non-Optional field (min_count/percentiles/ddof)
        # must never reach bin_frame (it 500'd before the fix).
        client = _client()
        for bincfg in (
            '{"bin_col":"mono","min_count":null}',
            '{"bin_col":"mono","percentiles":null}',
            '{"bin_col":"mono","err":"std","ddof":null}',
        ):
            response = client.get(
                "/api/run/uid-002/binned",
                params={"cols": "cam-MaxCounts", "bincfg": bincfg},
            )
            assert response.status_code == 200, bincfg
        # null behaves exactly like the field not being sent
        defaulted = client.get(
            "/api/run/uid-002/binned",
            params={
                "cols": "cam-MaxCounts",
                "bincfg": '{"bin_col":"mono","min_count":null}',
            },
        ).json()
        plain = client.get(
            "/api/run/uid-002/binned",
            params={"cols": "cam-MaxCounts", "bincfg": '{"bin_col":"mono"}'},
        ).json()
        assert defaulted["series"] == plain["series"]


class TestServerFigures:
    """0.10.0: the /api responses carry the server-authored figure."""

    def test_frame_figure_is_wired_and_pretty_named(self):
        payload = (
            _client().get("/api/run/uid-002/frame?cols=cam-MaxCounts&x=mono").json()
        )
        fig = payload["figure"]
        assert [t["yaxis"] for t in fig["data"]] == ["y"]
        assert fig["data"][0]["x"] == [4.0, 5.0, 6.0]
        assert fig["data"][0]["y"] == [10.0, 12.5, 11.0]
        # Run-provenance names prettify (device : variable).
        assert fig["data"][0]["name"] == "cam : MaxCounts"
        assert fig["layout"]["xaxis"]["title"]["text"] == "mono"
        assert fig["layout"]["showlegend"] is False  # single trace

    def test_frame_without_cols_serves_no_figure(self):
        payload = _client().get("/api/run/uid-002/frame").json()
        assert "figure" not in payload

    def test_display_param_reaches_the_figure_and_the_code(self):
        payload = (
            _client()
            .get(
                "/api/run/uid-002/frame",
                params={"cols": "cam-MaxCounts", "display": '{"logy":true}'},
            )
            .json()
        )
        assert payload["figure"]["layout"]["yaxis"]["type"] == "log"
        assert "shots_figure" in payload["code"]
        assert "'logy': True" in payload["code"]

    def test_display_layout_passthrough_stays_out_of_the_server_figure(self):
        # display.layout is the CLIENT-side escape hatch — the server
        # accepts the key (a shared link carries the whole display
        # JSON) but never applies or quotes it.
        payload = (
            _client()
            .get(
                "/api/run/uid-002/frame",
                params={
                    "cols": "cam-MaxCounts",
                    "display": '{"layout":{"yaxis":{"tickformat":".2e"}}}',
                },
            )
            .json()
        )
        assert "tickformat" not in json.dumps(payload["figure"])
        assert "layout" not in payload["code"]

    def test_bad_display_is_400_not_500(self):
        client = _client()
        for display in (
            "not json",
            '["list"]',
            '{"ghost":1}',
            '{"logy":"yes"}',
            '{"msize":"big"}',
            '{"width":"big"}',
            '{"height":NaN}',
            '{"ymin":NaN}',
            '{"colors":"#123456"}',
            '{"colors":[7]}',
            '{"layout":"x"}',
        ):
            response = client.get(
                "/api/run/uid-002/frame",
                params={"cols": "cam-MaxCounts", "display": display},
            )
            assert response.status_code == 400, display
            assert "display" in response.json()["detail"]

    def test_explicit_null_display_fields_mean_absent(self):
        payload = (
            _client()
            .get(
                "/api/run/uid-002/frame",
                params={"cols": "cam-MaxCounts", "display": '{"logy":null}'},
            )
            .json()
        )
        assert payload["figure"]["layout"]["yaxis"].get("type") != "log"

    def test_no_x_means_the_shot_axis_in_figure_and_code(self):
        # (An unplottable x 404s at the endpoint — pre-existing; the
        # figure-level fallback is pinned in test_figures.py.)
        payload = _client().get("/api/run/uid-002/frame?cols=cam-MaxCounts").json()
        assert payload["figure"]["data"][0]["x"] == [1.0, 2.0, 3.0]
        assert payload["figure"]["layout"]["xaxis"]["title"]["text"] == "shot #"
        assert "x=" not in payload["code"]

    def test_binned_x_places_bins_at_per_bin_mean(self):
        # Identity bins on mono (one shot each) — the per-bin mean of
        # cam-MaxCounts as X is just its own values, hand-computable.
        payload = (
            _client()
            .get(
                "/api/run/uid-002/binned",
                params={
                    "cols": "cam-MaxCounts",
                    "x": "cam-MaxCounts",
                    "bincfg": '{"bin_col":"mono"}',
                },
            )
            .json()
        )
        assert payload["x_centers"] == [10.0, 12.5, 11.0]
        fig = payload["figure"]
        assert fig["data"][0]["x"] == [10.0, 12.5, 11.0]
        assert fig["layout"]["xaxis"]["title"]["text"] == "cam : MaxCounts"
        # The snippet reproduces the placement (replace(cfg, agg='mean')).
        assert "agg='mean'" in payload["code"]
        assert "x_values=x_centers" in payload["code"]

    def test_binned_x_reindexes_onto_the_y_bins(self):
        # The x call's dropna runs over x ALONE — a y column NaN'd for
        # one scan step (a camera down) drops that bin from the y
        # result but not from the x result.  Positional zipping would
        # shift every point one bin over; reindexing pins alignment.
        catalog = FakeCatalog()
        detail = _detail(7)
        detail.data["sig"] = [float("nan"), 20.0, 30.0]
        catalog.details["uid-007"] = detail
        payload = (
            _client(catalog)
            .get(
                "/api/run/uid-007/binned",
                params={
                    "cols": "sig",
                    "x": "cam-MaxCounts",
                    "bincfg": '{"bin_col":"mono"}',
                },
            )
            .json()
        )
        assert payload["bins"] == [5.0, 6.0]  # bin 4 dropped (NaN y)
        assert payload["x_centers"] == [12.5, 11.0]  # aligned, not shifted
        assert payload["figure"]["data"][0]["x"] == [12.5, 11.0]

    def test_binned_coercible_string_columns_400_not_500(self):
        # telemetry columns are dtype-tolerant BY DESIGN: they plot in
        # per-shot view (numeric_series coerces) but bin_frame sees the
        # raw dtype — refuse honestly, never 500.
        client = _client()
        as_x = client.get(
            "/api/run/uid-002/binned",
            params={
                "cols": "cam-MaxCounts",
                "x": "telemetry_dev-val",
                "bincfg": '{"bin_col":"mono"}',
            },
        )
        assert as_x.status_code == 400
        as_y = client.get(
            "/api/run/uid-002/binned",
            params={"cols": "telemetry_dev-val", "bincfg": '{"bin_col":"mono"}'},
        )
        assert as_y.status_code == 400

    def test_binned_without_x_keeps_bin_labels(self):
        payload = (
            _client()
            .get(
                "/api/run/uid-002/binned",
                params={"cols": "cam-MaxCounts", "bincfg": '{"bin_col":"mono"}'},
            )
            .json()
        )
        assert "x_centers" not in payload
        assert payload["figure"]["data"][0]["x"] == [4.0, 5.0, 6.0]
        assert payload["figure"]["layout"]["xaxis"]["title"]["text"] == "mono"

    def test_binned_unplottable_x_is_404(self):
        response = _client().get(
            "/api/run/uid-002/binned",
            params={
                "cols": "cam-MaxCounts",
                "x": "cam-label",
                "bincfg": '{"bin_col":"mono"}',
            },
        )
        assert response.status_code == 404

    def test_binned_figure_carries_asymmetric_errors_and_bin_col(self):
        payload = (
            _client()
            .get(
                "/api/run/uid-002/binned",
                params={"cols": "cam-MaxCounts", "bincfg": '{"bin_col":"mono"}'},
            )
            .json()
        )
        fig = payload["figure"]
        trace = fig["data"][0]
        assert trace["x"] == [4.0, 5.0, 6.0]
        assert trace["y"] == [10.0, 12.5, 11.0]
        assert trace["error_y"]["symmetric"] is False
        assert trace["error_y"]["arrayminus"] == [0.0, 0.0, 0.0]
        assert fig["layout"]["xaxis"]["title"]["text"] == "mono"
        assert "binned_figure" in payload["code"]

    def test_page_injects_the_server_palette_and_marker_default(self):
        from geecs_portal.figures import MARKER_SIZE_DEFAULT, TRACE_COLORS

        text = _client().get("/run/uid-002").text
        assert f"const TRACE_COLORS = {json.dumps(list(TRACE_COLORS))};" in text
        assert f"const MSIZE_DEFAULT = {json.dumps(MARKER_SIZE_DEFAULT)};" in text

    def test_datetime_snippet_carries_kinds(self):
        # Without kinds the notebook figure would apply logx/x-ranges
        # to a date axis the page's figure guards — divergence.
        payload = _client().get("/api/run/uid-002/frame?cols=ts_cam-MaxCounts").json()
        assert "kinds={'ts_cam-MaxCounts': 'datetime'}" in payload["code"]


class TestPlotTabPolish:
    """W1e: timestamp handling, the ts_ pick flag, day-jump stepping."""

    def test_columns_flag_ts_event_timestamps(self):
        payload = _client().get("/api/run/uid-002/columns").json()
        flags = {c["name"]: c["timestamp"] for c in payload["columns"]}
        assert flags["ts_cam-MaxCounts"] is True
        assert flags["cam-MaxCounts"] is False

    def test_frame_serves_timestamps_as_local_datetimes(self):
        from datetime import datetime

        payload = _client().get("/api/run/uid-002/frame?cols=ts_cam-MaxCounts").json()
        assert payload["kinds"] == {"ts_cam-MaxCounts": "datetime"}
        expected = datetime.fromtimestamp(_epoch(9, 30)).isoformat(
            sep=" ", timespec="milliseconds"
        )
        assert payload["series"]["ts_cam-MaxCounts"][0] == expected

    def test_labview_timestamps_shift_by_the_wire_offset(self):
        from datetime import datetime

        from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET

        # An s-file-style acq_timestamp column holds LabVIEW epoch.
        catalog = FakeCatalog()
        detail = _detail(7)
        detail.data["cam acq_timestamp"] = [_LV + 1.0, _LV + 2.0, _LV + 3.0]
        catalog.details["uid-007"] = detail
        payload = (
            _client(catalog)
            .get("/api/run/uid-007/frame", params={"cols": "cam acq_timestamp"})
            .json()
        )
        assert payload["kinds"] == {"cam acq_timestamp": "datetime"}
        expected = datetime.fromtimestamp(_LV + 1.0 - LABVIEW_EPOCH_OFFSET).isoformat(
            sep=" ", timespec="milliseconds"
        )
        assert payload["series"]["cam acq_timestamp"][0] == expected

    def test_plain_columns_carry_no_kind(self):
        payload = _client().get("/api/run/uid-002/frame?cols=cam-MaxCounts").json()
        assert payload["kinds"] == {}

    def test_jump_prefers_the_same_scan_number(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get(
            f"/run/jump/{TEST_DAY.isoformat()}?prefer=1&y=cam-MaxCounts"
            "&view=bin&experiment=Undulator",
            follow_redirects=False,
        )
        assert response.status_code in (302, 307)
        location = response.headers["location"]
        assert location.startswith("/run/uid-001?")
        assert "y=cam-MaxCounts" in location and "view=bin" in location
        assert "prefer=" not in location
        assert f"day={TEST_DAY.isoformat()}" in location

    def test_jump_falls_back_to_the_newest_run(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get(
            f"/run/jump/{TEST_DAY.isoformat()}?prefer=99", follow_redirects=False
        )
        # scan 99 doesn't exist that day → the newest run (uid-002).
        assert response.headers["location"].startswith("/run/uid-002?")

    def test_jump_on_an_empty_day_lands_on_the_day_page(self):
        from geecs_data_utils.tiled_catalog import StubCatalog

        client = _client(StubCatalog())
        response = client.get(
            "/run/jump/2026-01-01?prefer=3&filter=abc", follow_redirects=False
        )
        location = response.headers["location"]
        assert location.startswith("/day/2026-01-01")
        assert "filter=abc" in location

    def test_jump_bad_date_is_404(self):
        assert _client().get("/run/jump/nope").status_code == 404

    def test_rail_has_the_scan_dropdown_and_jump_steppers(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get(f"/run/uid-001?day={TEST_DAY.isoformat()}")
        assert "<select" in response.text  # the rail scan dropdown
        assert "Scan 002" in response.text  # the sibling run is offered
        assert "/run/jump/" in response.text  # day steppers stay in-tab
        assert "prefer=1" in response.text  # carry this run's number

    def test_frame_code_snippet_mirrors_the_datetime_conversion(self):
        payload = _client().get("/api/run/uid-002/frame?cols=ts_cam-MaxCounts").json()
        assert "LABVIEW_EPOCH_OFFSET" not in payload["code"]  # unix: no shift
        assert "datetime.fromtimestamp" in payload["code"]
        catalog = FakeCatalog()
        detail = _detail(7)
        detail.data["cam acq_timestamp"] = [_LV + 1.0, _LV + 2.0, _LV + 3.0]
        catalog.details["uid-007"] = detail
        labview = (
            _client(catalog)
            .get("/api/run/uid-007/frame", params={"cols": "cam acq_timestamp"})
            .json()
        )
        assert "- LABVIEW_EPOCH_OFFSET" in labview["code"]

    def test_display_state_rides_run_view_links(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get(
            "/run/uid-001?y=cam-MaxCounts&display=%7B%22logy%22%3Atrue%7D"
        )
        stepper_line = next(
            line for line in response.text.splitlines() if "/run/uid-002?" in line
        )
        assert "display=" in stepper_line

    def test_plot_config_suite_is_on(self):
        # The free-suite pin: wheel zoom, draw tools, spike lines are
        # config (one line to maintain), never per-knob code.
        text = _client().get("/run/uid-002").text
        assert "scrollZoom: true" in text
        assert "drawrect" in text
        assert "togglespikelines" in text
        assert "deepMerge(fig.layout, d.layout)" in text  # the passthrough


class TestReverseProxy:
    """Behind a mount prefix every link, fetch base, and redirect carries it.

    The Grafana/JupyterHub convention: the proxy strips the prefix from
    the path and names it in ``X-Forwarded-Prefix``; the app derives
    its root path per request (OSPREY panel tabs mount the portal this
    way).  Served at root — no header — nothing changes.
    """

    PREFIX = {"X-Forwarded-Prefix": "/portal"}

    def test_day_links_and_forms_carry_the_prefix(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get(f"/day/{TEST_DAY.isoformat()}", headers=self.PREFIX)
        assert response.status_code == 200
        assert '"/portal/run/uid-002?' in response.text
        assert 'action="/portal/go"' in response.text
        assert 'action="/portal/day/' in response.text
        # No unprefixed portal link survives (external hrefs excluded).
        assert 'href="/run/' not in response.text
        assert 'href="/day/' not in response.text

    def test_run_page_static_fetch_base_and_steppers_carry_the_prefix(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get("/run/uid-002", headers=self.PREFIX)
        assert response.status_code == 200
        assert 'src="/portal/static/plotly-cartesian-' in response.text
        # The JS builds every /api fetch (and the scan-jump dropdown)
        # from this constant.
        assert 'const ROOT = "/portal";' in response.text
        assert '"/portal/run/uid-001?' in response.text  # prev-scan stepper
        assert 'src="/static/' not in response.text

    def test_index_and_go_redirects_carry_the_prefix(self):
        response = _client().get("/", headers=self.PREFIX, follow_redirects=False)
        assert response.headers["location"].startswith("/portal/day/")
        response = _client().get(
            "/go?day=2026-07-12", headers=self.PREFIX, follow_redirects=False
        )
        assert response.headers["location"] == "/portal/day/2026-07-12"

    def test_run_jump_redirect_carries_the_prefix(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get(
            f"/run/jump/{TEST_DAY.isoformat()}",
            headers=self.PREFIX,
            follow_redirects=False,
        )
        assert response.headers["location"].startswith("/portal/run/uid-002?")

    def test_served_at_root_nothing_changes(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get("/run/uid-002")
        assert 'const ROOT = "";' in response.text
        assert 'src="/static/plotly-cartesian-' in response.text
        assert '"/run/uid-001?' in response.text

    def test_trailing_slash_prefix_is_normalized(self):
        response = _client().get(
            "/", headers={"X-Forwarded-Prefix": "/portal/"}, follow_redirects=False
        )
        assert response.headers["location"].startswith("/portal/day/")

    def test_malformed_prefixes_are_ignored(self):
        for bad in (
            "portal",
            "//evil.example",
            "/a b",
            "/",
            "/x\\y",
            "/p?x=1",  # query/fragment/quote characters are not a path
            "/p#frag",
            '/p"x',
            "/p<q>",
        ):
            response = _client().get(
                "/", headers={"X-Forwarded-Prefix": bad}, follow_redirects=False
            )
            assert response.headers["location"].startswith("/day/"), bad

    def test_mount_name_colliding_with_a_route_head_works(self):
        # A mount literally named /run: the middleware re-prefixes the
        # path so starlette's front-strip is exact — without it this
        # whole route family double-strips to a 404.
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get("/run/uid-002", headers={"X-Forwarded-Prefix": "/run"})
        assert response.status_code == 200
        assert 'const ROOT = "/run";' in response.text
        assert '"/run/run/uid-001?' in response.text  # prev-scan stepper

    def test_trailing_slash_redirect_keeps_the_prefix(self):
        # Starlette's redirect_slashes builds the Location from the
        # scope path — re-prefixed, so the hop stays under the mount.
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get(
            "/run/uid-002/", headers=self.PREFIX, follow_redirects=False
        )
        assert response.status_code in (301, 307)
        assert "/portal/run/uid-002" in response.headers["location"]

    def test_api_still_serves_under_a_prefix(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        response = client.get("/api/run/uid-002/columns", headers=self.PREFIX)
        assert response.status_code == 200
        assert response.json()["total"] == 3

    def test_health_still_serves_under_a_prefix(self):
        response = _client().get("/health", headers=self.PREFIX)
        assert response.status_code == 200
        assert response.json()["ok"] is True


class TestBrowsingApi:
    """0.20.0: the page-shaped reads as JSON — day list, run, device, jump.

    Everything a browser shows must be readable without scraping HTML
    (the agent surface); every payload is the template's own data.
    """

    def test_day_lists_runs_newest_first_as_the_table_shows(self):
        catalog = FakeCatalog()
        client = _client(catalog, default_experiment="Undulator")
        response = client.get(f"/api/day/{TEST_DAY.isoformat()}")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
        payload = response.json()
        assert catalog.listed == ("Undulator", TEST_DAY)
        assert payload["day"] == TEST_DAY.isoformat()
        assert payload["experiment"] == "Undulator"
        assert [run["scan"] for run in payload["runs"]] == ["Scan 002", "Scan 001"]
        row = payload["runs"][0]
        assert row["uid"] == "uid-002" and row["scan_number"] == 2
        assert row["time"] == "09:02"
        assert row["started"].startswith(f"{TEST_DAY.isoformat()} 09:02")
        assert row["exit_status"] == "success" and row["running"] is False
        assert row["save_sets"] == ["Amp4In"] and row["shots"] == 10
        assert payload["prev_day"] == "2026-07-11"
        assert payload["page"] == f"/day/{TEST_DAY.isoformat()}"

    def test_day_filter_and_experiment_mirror_the_page(self):
        catalog = FakeCatalog()
        client = _client(catalog, default_experiment="Undulator")
        payload = client.get(
            f"/api/day/{TEST_DAY.isoformat()}?filter=scan 001&experiment=Thomson"
        ).json()
        assert catalog.listed[0] == "Thomson"
        assert [run["uid"] for run in payload["runs"]] == ["uid-001"]
        assert payload["filter"] == "scan 001"

    def test_day_bad_date_is_404_and_outage_is_503(self):
        assert _client().get("/api/day/not-a-date").status_code == 404
        response = _client(FakeCatalog(fail_listing=True)).get(
            f"/api/day/{TEST_DAY.isoformat()}"
        )
        assert response.status_code == 503
        assert "catalog unavailable" in response.json()["detail"]

    def test_empty_day_is_an_empty_list_not_an_error(self):
        payload = _client(StubCatalog()).get(f"/api/day/{TEST_DAY.isoformat()}").json()
        assert payload["runs"] == []

    def test_run_carries_the_overview_table_verbatim(self):
        catalog = FakeCatalog()
        client = _client(catalog, default_experiment="Undulator")
        response = client.get("/api/run/uid-002")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
        payload = response.json()
        expected = [[k, v] for k, v in metadata_rows(catalog.details["uid-002"])]
        assert payload["metadata"] == expected
        assert payload["summary"]["scan"] == "Scan 002"
        assert payload["run_day"] == TEST_DAY.isoformat()
        assert payload["event_rows"] == 3
        assert payload["start_doc"]["scan_number"] == 2
        assert payload["stop_doc"]["exit_status"] == "success"
        assert payload["page"] == "/run/uid-002"
        assert payload["portal_version"]
        assert payload["processing_options"] == []  # feature off
        assert payload["analysis_enabled"] is False  # feature off

    def test_run_neighbours_and_day_runs_are_the_steppers(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        older = client.get("/api/run/uid-001").json()
        newer = client.get("/api/run/uid-002").json()
        assert older["neighbours"] == {"prev_uid": "", "next_uid": "uid-002"}
        assert newer["neighbours"] == {"prev_uid": "uid-001", "next_uid": ""}
        assert [run["uid"] for run in newer["day_runs"]] == ["uid-002", "uid-001"]
        assert newer["prev_day"] == "2026-07-11" and newer["next_day"] == "2026-07-13"

    def test_run_neighbours_degrade_when_the_listing_fails(self):
        payload = _client(FakeCatalog(fail_listing=True)).get("/api/run/uid-002").json()
        assert payload["neighbours"] == {"prev_uid": "", "next_uid": ""}
        assert payload["day_runs"] == []

    def test_run_documents_are_json_safe(self):
        # A document key carrying NaN / a numpy scalar / a set must not
        # 500 the whole run (json.dumps refuses all three).
        import numpy as np

        catalog = FakeCatalog()
        detail = _detail(7)
        detail.start_doc["odd"] = {
            "nan": float("nan"),
            "np": np.int64(3),
            "set": {1},
            "list": [1.5, float("inf")],
        }
        catalog.details["uid-007"] = detail
        response = _client(catalog).get("/api/run/uid-007")
        assert response.status_code == 200
        assert b"NaN" not in response.content
        odd = response.json()["start_doc"]["odd"]
        assert odd["nan"] is None and odd["np"] == 3.0
        assert odd["list"] == [1.5, None]
        assert isinstance(odd["set"], str)

    def test_run_unknown_is_404_and_outage_is_503(self):
        assert _client().get("/api/run/nope").status_code == 404
        assert _client(DownCatalog()).get("/api/run/uid-002").status_code == 503

    def test_run_page_link_carries_the_proxy_prefix(self):
        payload = (
            _client()
            .get("/api/run/uid-002", headers={"X-Forwarded-Prefix": "/portal"})
            .json()
        )
        assert payload["page"] == "/portal/run/uid-002"

    def test_day_and_jump_page_links_carry_the_proxy_prefix(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        prefix = {"X-Forwarded-Prefix": "/portal"}
        day = client.get(f"/api/day/{TEST_DAY.isoformat()}", headers=prefix).json()
        assert day["page"] == f"/portal/day/{TEST_DAY.isoformat()}"
        hit = client.get(f"/api/run/jump/{TEST_DAY.isoformat()}", headers=prefix).json()
        assert hit["page"] == "/portal/run/uid-002"
        empty = (
            _client(StubCatalog())
            .get("/api/run/jump/2026-01-01", headers=prefix)
            .json()
        )
        assert empty["uid"] is None and empty["page"] == "/portal/day/2026-01-01"

    def test_jump_prefers_the_scan_number_else_the_newest(self):
        client = _client(FakeCatalog(), default_experiment="Undulator")
        hit = client.get(f"/api/run/jump/{TEST_DAY.isoformat()}?prefer=1").json()
        assert hit["uid"] == "uid-001" and hit["matched"] is True
        assert hit["page"] == "/run/uid-001" and hit["runs"] == 2
        miss = client.get(f"/api/run/jump/{TEST_DAY.isoformat()}?prefer=99").json()
        assert miss["uid"] == "uid-002" and miss["matched"] is False

    def test_jump_on_an_empty_day_names_no_run(self):
        payload = _client(StubCatalog()).get("/api/run/jump/2026-01-01?prefer=3").json()
        assert payload["uid"] is None and payload["scan_number"] is None
        assert payload["page"] == "/day/2026-01-01"

    def test_jump_bad_date_is_404_and_outage_is_503(self):
        assert _client().get("/api/run/jump/nope").status_code == 404
        response = _client(FakeCatalog(fail_listing=True)).get(
            f"/api/run/jump/{TEST_DAY.isoformat()}"
        )
        assert response.status_code == 503

    def test_device_without_a_resolvable_folder_is_404(self, monkeypatch):
        from geecs_data_utils import scan_paths as scan_paths_mod

        monkeypatch.setattr(scan_paths_mod, "daily_scan_folder", lambda *a, **k: None)
        client = _client()
        assert client.get("/api/run/uid-002/device").status_code == 400
        response = client.get("/api/run/uid-002/device?device=cam")
        assert response.status_code == 404
        assert "not resolvable" in response.json()["detail"]

    def test_openapi_lists_the_agent_surface(self):
        # docs_url is off, but the schema stays: scripts can discover
        # the deployed portal's routes instead of guessing.
        paths = _client().get("/openapi.json").json()["paths"]
        for route in (
            "/health",
            "/api/day/{day}",
            "/api/run/{uid}",
            "/api/run/{uid}/device",
            "/api/run/jump/{day}",
            "/api/run/{uid}/columns",
            "/api/run/{uid}/frame",
            "/api/run/{uid}/binned",
            "/api/run/{uid}/filter-count",
            "/api/run/{uid}/bin-images",
            "/api/run/{uid}/analysis",
            "/run/{uid}/artifact",
            "/run/{uid}/image.png",
            "/run/{uid}/bin-image.png",
            "/run/{uid}/plot.png",
        ):
            assert route in paths, route
