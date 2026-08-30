"""Portal app tests — hermetic: fake catalogs, no network, no data root."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from fastapi.testclient import TestClient

from geecs_data_utils.tiled_catalog import (
    CatalogStatus,
    RunDetail,
    StubCatalog,
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
                "scan_event_index": [0, 1, 2],  # id machinery: never a pick
                "cam-MaxCounts": [10.0, 12.5, 11.0],
                "cam-acq_timestamp": [_LV + 1.0, _LV + 2.0, _LV + 3.0],  # companion
                "cam-label": ["a", "b", "c"],  # non-numeric: never plottable
                "telemetry_dev-val": ["1.5", "2.5", "3.5"],  # dtype-tolerant
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
    def test_health_reports_probe(self):
        response = _client().get("/health")
        assert response.status_code == 200
        assert response.json() == {"ok": True, "catalog": "fake catalog"}

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
        assert "experiment=Bella%20PW" in response.text
        assert "experiment=Bella PW" not in response.text


class TestRunView:
    def test_metadata_rows_render(self):
        response = _client().get("/run/uid-002")
        assert response.status_code == 200
        assert "Scan 002" in response.text
        assert "success" in response.text

    def test_pick_list_is_schema_shared_semantics(self):
        response = _client().get("/run/uid-002")
        assert "cam-MaxCounts" in response.text
        assert "telemetry_dev-val" in response.text  # numeric strings plot
        assert "cam-label" not in response.text  # non-numeric
        assert "cam-dead" not in response.text  # all-NaN
        assert "scan_event_index" not in response.text  # id machinery
        assert "cam-acq_timestamp" not in response.text  # companion machinery

    def test_stepped_scan_defaults_x_to_scan_variable(self):
        catalog = FakeCatalog()
        catalog.details["uid-007"] = _detail(7, motor="mono")
        response = _client(catalog).get("/run/uid-007?y=cam-MaxCounts")
        assert "plot.png?y=cam-MaxCounts&amp;x=mono" in response.text

    def test_explicit_x_wins_over_stepped_default(self):
        catalog = FakeCatalog()
        catalog.details["uid-007"] = _detail(7, motor="mono")
        response = _client(catalog).get(
            "/run/uid-007?y=cam-MaxCounts&x=telemetry_dev-val"
        )
        assert "plot.png?y=cam-MaxCounts&amp;x=telemetry_dev-val" in response.text

    def test_selected_column_embeds_plot(self):
        response = _client().get("/run/uid-002?y=cam-MaxCounts")
        assert 'src="/run/uid-002/plot.png?y=cam-MaxCounts"' in response.text

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
