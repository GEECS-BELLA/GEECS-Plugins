"""The Plot page's 2-D map: the portal's old Grid tab, on the set and rail.

Scan003 of the fake share steps the jet over two positions and its height
over two, one bin each (four bins, two shots per bin). Its folder names
one scan variable (the jet), so the map's X defaults to it and Y is the
viewer's pick, each bin placed at its measured position.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from geecs_portal.plotter import figures
from tests.plotter.conftest import SCAN_VARIABLE, post, write_scan

HEIGHT = "U_Jet Height"
CHARGE = "U_ICT charge"
MAP_JS = (
    Path(__file__).resolve().parents[2]
    / "geecs_portal"
    / "plotter"
    / "static"
    / "map.js"
)


@pytest.fixture
def two_d(share: Path) -> Path:
    """Add Scan003: bins 1-4 at (jet, height) (1,0) (2,0) (1,1) (2,1)."""
    jet = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]
    height = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    write_scan(
        share,
        3,
        pd.DataFrame(
            {
                "Shotnumber": list(range(1, 9)),
                "Bin #": [1, 1, 2, 2, 3, 3, 4, 4],
                SCAN_VARIABLE: jet,
                HEIGHT: height,
                CHARGE: [10.0, 12.0, 20.0, 22.0, 30.0, 32.0, 40.0, 42.0],
            }
        ),
        parameter=SCAN_VARIABLE,
    )
    return share


def _map(client: TestClient, **params: object) -> dict:
    query = {"day": "2026-10-02", "scan": 3, "value": CHARGE, "x": SCAN_VARIABLE}
    response = client.get("/api/map", params={**query, "y": HEIGHT, **params})
    assert response.status_code == 200, response.text
    return response.json()


# -- the route ----------------------------------------------------------------


def test_the_map_is_each_bins_statistic_at_its_measured_position(
    client: TestClient, two_d: Path
) -> None:
    body = _map(client)
    cells = sorted(body["cells"], key=lambda c: c["bin"])
    assert [(c["x"], c["y"], c["center"]) for c in cells] == [
        (1.0, 0.0, 11.0),
        (2.0, 0.0, 21.0),
        (1.0, 1.0, 31.0),
        (2.0, 1.0, 41.0),
    ]
    assert body["kind"] == "points"  # a folder scan has no planned grid
    assert set(body["figures"]) == {"center", "error"}
    assert body["figures"]["center"]["layout"]["paper_bgcolor"] == "$tok:--surface"
    assert body["bin_column"] == "Bin #"


def test_the_filters_narrow_the_shots_in_each_cell(
    client: TestClient, two_d: Path
) -> None:
    keep_low = json.dumps(
        {
            "groups": [
                {
                    "name": "low",
                    "enabled": True,
                    "conditions": [
                        {"column": CHARGE, "mode": "within", "low": 0, "high": 25}
                    ],
                }
            ],
            "nan_policy": "exclude",
        }
    )
    body = _map(client, filters=keep_low)
    assert (body["pass"], body["total"]) == (4, 8)
    assert sum(len(c["shots"]) for c in body["cells"]) == 4


def test_a_column_the_scan_lacks_is_a_400_naming_it(
    client: TestClient, two_d: Path
) -> None:
    response = client.get(
        "/api/map",
        params={
            "day": "2026-10-02",
            "scan": 3,
            "value": "nope",
            "x": SCAN_VARIABLE,
            "y": HEIGHT,
        },
    )
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_the_statistics_reach_the_map(client: TestClient, two_d: Path) -> None:
    body = _map(client, mapcfg=json.dumps({"average": "mean", "error": "std"}))
    assert body["config"]["average"] == "mean"
    assert body["error_label"] == "Standard deviation"


def test_the_code_rebuilds_the_same_maps(client: TestClient, two_d: Path) -> None:
    body = _map(client)
    code = body["code"].replace(
        "experiment='Testexp'", f"experiment='Testexp', base_path={str(two_d)!r}"
    )
    shown: list = []
    namespace: dict = {}
    exec(
        code.replace("    fig.show()\n", "    shown.append(fig)\n"),
        {"shown": shown, **namespace},
    )
    assert len(shown) == 2
    rebuilt = figures.page_figure(shown[0])["data"][0]
    served = body["figures"]["center"]["data"][0]
    for key in ("x", "y", "customdata"):
        assert rebuilt.get(key) == served.get(key), key


# -- the page -------------------------------------------------------------------


def test_the_map_view_defaults_x_to_the_scan_variable_and_asks_for_y(
    client: TestClient, two_d: Path
) -> None:
    page = post(client, scan=3, tab="map")
    assert 'data-value="map" aria-pressed="true"' in page
    assert f"scan variable: {SCAN_VARIABLE}" in page
    assert "scan variable: none (pick one)" in page
    assert "Pick the Y axis in the rail" in page
    # Nothing to map until Y is picked, but the host is there: the rail's
    # shot filters (plot_tools.js) read the scan from it.
    assert 'id="map" data-url="" data-day="2026-10-02" data-scan="3"' in page
    assert 'id="filters-open"' in page


def test_a_picked_y_draws_the_map(client: TestClient, two_d: Path) -> None:
    page = post(client, scan=3, tab="map", mapy=HEIGHT, mapv=CHARGE)
    assert 'id="map"' in page and "api/map?" in page
    assert "x=U_Jet+Position" in page and "y=U_Jet+Height" in page
    # The statistics: selects, and number boxes map.js checks before saving.
    assert 'data-map-stat="average"' in page
    assert 'id="map-min_count" type="number"' in page
    assert 'data-map-stat="error_view"' in page  # IQR: the error map's side
    assert 'value="index"' not in page  # equal cells need a recorded grid


def test_the_default_entries_save_as_no_pick(
    client: TestClient, two_d: Path, settings_dir: Path
) -> None:
    """A set that follows the viewer to another scan maps that scan's axes."""
    post(client, scan=3, tab="map", mapx="", mapy=HEIGHT, mapv="")
    saved = json.loads((settings_dir / "default.json").read_text())
    assert saved["map_x"] is None and saved["map_value"] is None
    assert saved["map_y"] == HEIGHT and saved["tab"] == "map"


def test_bad_map_statistics_are_refused_not_saved(
    client: TestClient, two_d: Path, settings_dir: Path
) -> None:
    response = client.post(
        "/",
        data={
            "day": "2026-10-02",
            "scan": "3",
            "set": "default",
            "owner": "default",
            "mapcfg": '{"average": "mode"}',
        },
    )
    assert response.status_code == 422
    assert not (settings_dir / "default.json").exists()


# -- map.js's number boxes ---------------------------------------------------------

#: Enough of a page for map.js to wire its controls: the form, the three
#: number boxes, and a fetch that never answers (no map is drawn).
_HARNESS = r"""
const assert = require('node:assert/strict');
let submits = 0, toasts = 0;
const form = {elements: {mapcfg: {value: CFG}, filters: {value: ""}}, requestSubmit() { submits++; }};
const handlers = {};
const box = (key, type, percent) => ({
  type, value: "", dataset: percent ? {mapStat: key, percent: ""} : {mapStat: key},
  addEventListener(kind, fn) { handlers[key] = fn; },
});
const controls = [box("lower", "number", true), box("upper", "number", true),
                  box("min_count", "number", false), box("error", "select-one", false)];
const others = {};
const document = {
  getElementById(id) {
    if (id === "pick") return form;
    if (id === "map") return {dataset: {day: "2026-10-02", scan: "3", url: "/api/map", imagesUrl: "/run/jump"}};
    return others[id] = others[id] || {value: "", textContent: "", hidden: false,
      replaceChildren() {}, addEventListener() {}, dataset: {}};
  },
  querySelectorAll(selector) { return selector === "[data-map-stat]" ? controls : []; },
};
const window = {GeecsKit: {toast() { toasts++; }}, PlotTools: {config: () => ({}), counted() {}}, addEventListener() {}};
const fetch = () => new Promise(() => {});
const ResizeObserver = class { observe() {} };
const location = {origin: "http://portal.example"};
"""


def _run_map_js(tmp_path: Path, cfg: dict, body: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to run map.js")
    script = tmp_path / "map.cjs"
    script.write_text(
        f"const CFG = {json.dumps(json.dumps(cfg))};\n"
        + _HARNESS
        + MAP_JS.read_text().replace('"use strict";', "")
        + "\n"
        + body
    )
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


_CFG = {"min_count": 3, "lower": 0.25, "upper": 0.75}


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("min_count", "", None),
        ("lower", "", None),
        ("upper", "", None),
        ("min_count", "0", None),
        ("min_count", "1.5", None),
        ("min_count", "-3", None),
        ("upper", "20", None),
        ("lower", "80", None),
        ("upper", "101", None),
        ("min_count", "5", 5),
        ("lower", "0", 0),
        ("upper", "100", 1),
    ],
)
def test_a_bad_number_is_never_saved(
    tmp_path: Path, key: str, value: str, expected: object
) -> None:
    """An empty or out-of-range box goes back to the drawn value, unsaved."""
    _run_map_js(
        tmp_path,
        _CFG,
        f"""
const key = {json.dumps(key)}, expected = {json.dumps(expected)};
const control = controls.find((c) => c.dataset.mapStat === key);
control.value = {json.dumps(value)};
handlers[key]();
if (expected === null) {{
  assert.equal(form.elements.mapcfg.value, CFG);
  assert.equal(submits, 0);
  assert.equal(Number(control.value), JSON.parse(CFG)[key] * (key === "min_count" ? 1 : 100));
}} else {{
  assert.equal(JSON.parse(form.elements.mapcfg.value)[key], expected);
  assert.equal(submits, 1);
}}
""",
    )


def test_choosing_iqr_puts_the_quartiles_back(tmp_path: Path) -> None:
    _run_map_js(
        tmp_path,
        {"error": "percentile", "lower": 0.1, "upper": 0.9},
        """
const control = controls.find((c) => c.dataset.mapStat === "error");
control.value = "iqr";
handlers.error();
const saved = JSON.parse(form.elements.mapcfg.value);
assert.deepEqual([saved.error, saved.lower, saved.upper], ["iqr", 0.25, 0.75]);
assert.equal(submits, 1);
""",
    )
