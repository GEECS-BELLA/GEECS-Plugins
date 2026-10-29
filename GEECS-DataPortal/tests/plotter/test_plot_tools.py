"""The Plot page's tools from the old Plot tab: shot filters, the binned view,
display settings and the logbook send, all kept in the settings set.

Scan001 of the fake share: shots 1-4, ``Bin #`` 1 1 2 2, the jet at 1 1 2 2,
``U_ICT charge`` 10, 12, NaN, 30.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from geecs_portal.plotter import figures
from geecs_portal.plotter.app import create_app
from tests.plotter.conftest import EXPERIMENT, SCAN_VARIABLE, post

SCAN = {"day": "2026-10-02", "scan": 1}
CHARGE = "U_ICT charge"
#: Keep the shots whose charge is within 11..40 (NaN fails): shots 2 and 4.
KEEP_HIGH = json.dumps(
    {
        "groups": [
            {
                "name": "high",
                "enabled": True,
                "conditions": [
                    {"column": CHARGE, "mode": "within", "low": 11, "high": 40}
                ],
            }
        ],
        "nan_policy": "exclude",
    }
)


def _figure(client: TestClient, **params: object) -> dict:
    response = client.get("/api/figure", params={**SCAN, "y": CHARGE, **params})
    assert response.status_code == 200, response.text
    return response.json()


# -- the shot filters -------------------------------------------------------


def test_the_filters_keep_only_the_shots_they_pass(client: TestClient) -> None:
    body = _figure(client, filters=KEEP_HIGH)
    assert body["figure"]["data"][0]["y"] == [12.0, 30.0]
    assert body["figure"]["data"][0]["x"] == [2, 4]  # the shot numbers kept
    assert (body["pass"], body["total"]) == (2, 4)


def test_the_live_count_is_the_filters_own(client: TestClient) -> None:
    response = client.get("/api/filter-count", params={**SCAN, "filters": KEEP_HIGH})
    assert response.json() == {"pass": 2, "total": 4}


def test_bad_filters_are_a_400_naming_them(client: TestClient) -> None:
    response = client.get(
        "/api/figure", params={**SCAN, "y": CHARGE, "filters": "{not json"}
    )
    assert response.status_code == 400
    assert "filters" in response.json()["detail"]


def test_a_fit_is_made_to_the_kept_shots(client: TestClient) -> None:
    body = _figure(client, filters=KEEP_HIGH, yfit="polynomial", yorder=1)
    assert body["fits"][0]["points"] == 2


# -- the binned view ----------------------------------------------------------


def test_the_binned_view_draws_each_bins_median_and_error_bars(
    client: TestClient,
) -> None:
    """Bin 1 holds 10 and 12 (median 11); bin 2 only 30 (the NaN shot drops)."""
    body = _figure(client, view="bin")
    trace = body["figure"]["data"][0]
    assert trace["y"] == [11.0, 30.0]
    assert trace["mode"] == "markers+lines" and "error_y" in trace
    assert body["fits"] == []  # fits are made to shots, never to bin centres
    assert body["view"] == "bin"


def test_bins_are_placed_at_the_x_columns_mean(client: TestClient) -> None:
    body = _figure(client, view="bin", x=SCAN_VARIABLE)
    assert body["figure"]["data"][0]["x"] == [1.0, 2.0]


def test_the_filters_narrow_the_bins_too(client: TestClient) -> None:
    body = _figure(client, view="bin", filters=KEEP_HIGH)
    assert body["figure"]["data"][0]["y"] == [12.0, 30.0]


def test_a_missing_bin_column_is_a_404(client: TestClient) -> None:
    response = client.get(
        "/api/figure",
        params={
            **SCAN,
            "y": CHARGE,
            "view": "bin",
            "bincfg": json.dumps({"bin_col": "nope"}),
        },
    )
    assert response.status_code == 404


def test_the_binned_code_rebuilds_the_same_figure(
    client: TestClient, share: Path
) -> None:
    body = _figure(client, view="bin", x=SCAN_VARIABLE, filters=KEEP_HIGH)
    code = body["code"].replace(
        "experiment='Testexp'", f"experiment='Testexp', base_path={str(share)!r}"
    )
    namespace: dict = {}
    exec(code.replace("fig.show()\n", ""), namespace)
    rebuilt = figures.page_figure(namespace["fig"])["data"][0]
    served = body["figure"]["data"][0]
    assert (rebuilt["x"], rebuilt["y"]) == (served["x"], served["y"])


def test_the_filtered_shot_code_rebuilds_the_same_figure(
    client: TestClient, share: Path
) -> None:
    body = _figure(client, filters=KEEP_HIGH, display=json.dumps({"msize": 8}))
    code = body["code"].replace(
        "experiment='Testexp'", f"experiment='Testexp', base_path={str(share)!r}"
    )
    namespace: dict = {}
    exec(code.replace("fig.show()\n", ""), namespace)
    rebuilt = figures.page_figure(namespace["fig"])["data"][0]
    assert rebuilt["y"] == [12.0, 30.0]
    assert rebuilt["marker"]["size"] == 8


# -- the display settings ------------------------------------------------------


def test_the_display_reaches_the_figure(client: TestClient) -> None:
    display = {"logy": True, "msize": 9, "colors": ["#ff0000"], "width": 640}
    body = _figure(client, display=json.dumps(display))
    trace, layout = body["figure"]["data"][0], body["figure"]["layout"]
    assert trace["marker"] == {"color": "#ff0000", "size": 9.0}
    assert layout["yaxis"]["type"] == "log" and layout["width"] == 640
    assert layout["yaxis"]["title"]["font"]["color"] == "#ff0000"  # the axis follows


def test_the_layout_patch_never_reaches_the_server_figure(client: TestClient) -> None:
    """The raw Plotly layout is merged in the browser only."""
    body = _figure(client, display=json.dumps({"layout": {"title": {"text": "x"}}}))
    assert "title" not in body["figure"]["layout"]


# -- kept in the set -----------------------------------------------------------


def test_the_tools_save_into_the_set(client: TestClient, settings_dir: Path) -> None:
    post(
        client,
        scan=1,
        view="bin",
        filters=KEEP_HIGH,
        bincfg='{"agg": "mean"}',
        display='{"msize": 7}',
    )
    saved = json.loads((settings_dir / "default.json").read_text())
    assert saved["xy_view"] == "bin" and saved["bin_config"] == '{"agg": "mean"}'
    assert json.loads(saved["shot_filters"])["groups"][0]["name"] == "high"
    page = client.get("/", params={"set": "default", **SCAN}).text
    assert "view=bin" in page and "filters=%7B" in page  # the figure URL carries them
    assert ">high<" in page  # the rail's chip


def test_bad_tool_settings_are_refused_not_saved(
    client: TestClient, settings_dir: Path
) -> None:
    response = client.post(
        "/",
        data={
            **{k: str(v) for k, v in SCAN.items()},
            "set": "default",
            "owner": "default",
            "bincfg": '{"agg": "mode"}',
        },
    )
    assert response.status_code == 422
    assert not (settings_dir / "default.json").exists()


# -- the logbook send ----------------------------------------------------------


def test_the_send_drawer_needs_an_absolute_logbook_url(
    share: Path, settings_dir: Path
) -> None:
    def page(logbook_url: str) -> str:
        app = create_app(
            EXPERIMENT,
            base_directory=share,
            settings_dir=settings_dir,
            logbook_url=logbook_url,
        )
        return TestClient(app).get("/", params={"set": "default", **SCAN}).text

    sending = page("http://logbook.example:8400")
    assert 'id="send-drawer"' in sending and 'data-logbook-send="true"' in sending
    assert 'data-uid="folder:Testexp:2026-10-02:1"' in sending
    for path_only in ("", "/log"):
        assert 'id="send-drawer"' not in page(path_only)


# -- filters that keep nothing, or do not fit the scan ----------------------

KEEP_NONE = KEEP_HIGH.replace('"low": 11, "high": 40', '"low": 1000, "high": 2000')


def test_a_filter_keeping_no_shot_draws_an_empty_plot(client: TestClient) -> None:
    """Zero kept shots is an answer ("0 / 4 pass"), not a missing column."""
    for params in (
        {},
        {"view": "bin"},
        {"view": "bin", "x": SCAN_VARIABLE},
        {"yfit": "gaussian"},
    ):
        body = _figure(client, filters=KEEP_NONE, **params)
        assert (body["pass"], body["total"]) == (0, 4)
        assert body["figure"]["data"][0]["y"] == []


def test_a_y_with_no_value_left_still_draws_the_others(client: TestClient) -> None:
    """Shot 3 alone: its charge is NaN, its jet is not; Y1 draws, Y2 is a gap."""
    only_3 = KEEP_HIGH.replace(CHARGE, "Shotnumber").replace(
        '"low": 11, "high": 40', '"low": 2.5, "high": 3.5'
    )
    body = _figure(client, y=SCAN_VARIABLE, y2=CHARGE, filters=only_3)
    y1, y2 = body["figure"]["data"][0], body["figure"]["data"][1]
    assert y1["y"] == [2.0] and y2["y"] == [None]
    assert body["pass"] == 1


def test_a_saved_filter_on_a_column_this_scan_lacks_says_so(client: TestClient) -> None:
    """The set follows the viewer to Scan002, which has no jet column."""
    jet = KEEP_HIGH.replace(CHARGE, SCAN_VARIABLE)
    response = client.get(
        "/api/figure",
        params={"day": "2026-10-02", "scan": 2, "y": CHARGE, "filters": jet},
    )
    assert response.status_code == 400
    assert "saved shot filter" in response.json()["detail"]


def test_a_saved_bin_column_this_scan_lacks_says_so(client: TestClient) -> None:
    response = client.get(
        "/api/figure",
        params={
            "day": "2026-10-02",
            "scan": 2,
            "y": CHARGE,
            "view": "bin",
            "bincfg": json.dumps({"bin_col": SCAN_VARIABLE}),
        },
    )
    assert response.status_code == 404
    assert "saved bin column" in response.json()["detail"]
