"""The routes over the fake share: one test per endpoint, then the refusals."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import ALIASED, SCAN_VARIABLE


def test_columns_lists_the_sfile_and_the_scan_variable(client: TestClient) -> None:
    response = client.get("/api/columns", params={"day": "2026-10-02", "scan": 1})
    assert response.status_code == 200
    body = response.json()
    assert "U_ICT charge" in body["columns"]
    assert SCAN_VARIABLE in body["columns"]
    assert body["default_x"] == SCAN_VARIABLE
    assert body["shots"] == 4
    assert response.headers["cache-control"] == "no-cache"


def test_a_noscan_has_no_default_x(client: TestClient) -> None:
    body = client.get("/api/columns", params={"day": "2026-10-02", "scan": 2}).json()
    assert body["default_x"] == ""


def test_figure_plots_y_against_x(client: TestClient) -> None:
    response = client.get(
        "/api/figure",
        params={
            "day": "2026-10-02",
            "scan": 1,
            "x": SCAN_VARIABLE,
            "y": "U_ICT charge",
        },
    )
    assert response.status_code == 200
    body = response.json()
    (trace,) = body["figure"]["data"]
    assert trace["x"] == [1.0, 1.0, 2.0, 2.0]
    assert trace["y"] == [10.0, 12.0, None, 30.0]  # the NaN shot is a gap
    assert trace["marker"]["color"] == "$tok:--trace-1"
    assert "scans.load_frame(run)" in body["code"]
    assert "experiment='Testexp'" in body["code"]


def test_figure_without_x_uses_the_shot_numbers(client: TestClient) -> None:
    body = client.get(
        "/api/figure", params={"day": "2026-10-02", "scan": 2, "y": "U_ICT charge"}
    ).json()
    assert body["figure"]["data"][0]["x"] == [1, 2]


def test_the_code_rebuilds_the_same_numbers(client: TestClient, share: Path) -> None:
    """The snippet, run against the same share, draws the served points."""
    from geecs_plotter import figures, scans

    body = client.get(
        "/api/figure",
        params={
            "day": "2026-10-02",
            "scan": 1,
            "x": SCAN_VARIABLE,
            "y": "U_ICT charge",
        },
    ).json()
    code = body["code"].replace(
        "experiment='Testexp'", f"experiment='Testexp', base_path={str(share)!r}"
    )
    code = code.replace("fig.show()\n", "")
    namespace: dict = {}
    exec(code, namespace)
    assert (
        figures.page_figure(namespace["fig"])["data"][0]["y"]
        == body["figure"]["data"][0]["y"]
    )
    assert scans  # the snippet's imports resolve


def test_an_unknown_scan_is_a_404(client: TestClient) -> None:
    response = client.get("/api/columns", params={"day": "2026-10-02", "scan": 9})
    assert response.status_code == 404
    assert "Scan009" in response.json()["detail"]


def test_an_unknown_column_is_a_404(client: TestClient) -> None:
    response = client.get(
        "/api/figure", params={"day": "2026-10-02", "scan": 1, "y": "nope"}
    )
    assert response.status_code == 404


def test_a_bad_day_is_a_422(client: TestClient) -> None:
    response = client.get("/api/columns", params={"day": "October", "scan": 1})
    assert response.status_code == 422


def test_a_lookup_never_creates_a_scan_folder(client: TestClient, share: Path) -> None:
    before = sorted(p for p in share.rglob("*"))
    client.get("/api/columns", params={"day": "2026-10-02", "scan": 9})
    client.get("/", params={"day": "2026-10-03", "scan": 1})
    assert sorted(p for p in share.rglob("*")) == before


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["ok"] is True


def test_an_unpadded_scan_folder_opens(client: TestClient, share: Path) -> None:
    """A ``Scan7`` folder (no zero padding) is still scan 7."""
    day = share / "Testexp" / "Y2026" / "10-Oct" / "26_1002"
    (day / "scans" / "Scan002").rename(day / "scans" / "Scan7")
    (day / "analysis" / "s2.txt").rename(day / "analysis" / "s7.txt")
    body = client.get("/api/columns", params={"day": "2026-10-02", "scan": 7}).json()
    assert "U_ICT charge" in body["columns"]


def test_the_alias_toggle_titles_the_axis(client: TestClient) -> None:
    params = {"day": "2026-10-02", "scan": 1, "x": SCAN_VARIABLE, "y": ALIASED}
    raw = client.get("/api/figure", params=params).json()
    assert raw["figure"]["layout"]["yaxis"]["title"]["text"] == ALIASED
    shown = client.get("/api/figure", params={**params, "yalias": "true"}).json()
    assert shown["figure"]["layout"]["yaxis"]["title"]["text"] == "Gas (psi)"
    # X has no alias: the toggle leaves it as it is.
    both = client.get(
        "/api/figure", params={**params, "xalias": "true", "yalias": "true"}
    ).json()
    assert both["figure"]["layout"]["xaxis"]["title"]["text"] == SCAN_VARIABLE
    assert "pretty={'U_Gas Pressure Alias:Gas (psi)': 'Gas (psi)'}" in shown["code"]
    assert "pretty=" not in raw["code"]
