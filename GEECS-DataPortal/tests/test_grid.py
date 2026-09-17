"""Portal grid API, reproduction and figure contracts over a fake catalog."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from geecs_data_utils.tiled_catalog import RunDetail, summary_from_metadata
from geecs_portal.app import create_app
from test_app import FakeCatalog, _start_doc


def grid_client():
    start = _start_doc(
        2,
        motors=["slow", "fast"],
        plan_name="grid_scan",
        plan_pattern="outer_product",
        shape=[2, 2],
        extents=[[1, 10], [2, 4]],
        snaking=[False, True],
    )
    frame = pd.DataFrame(
        {
            "bin_number": [1, 1, 1, 2, 2, 2, 3],
            "scan_event_index": list(range(1, 8)),
            "slow": [1, 1, 1, 1, 1, 1, 10],
            "fast": [2, 2, 2, 4, 4, 4, 4],
            "signal": [1, 2, 9, 2, 4, 8, 3],
        }
    )
    catalog = FakeCatalog()
    stop = {"exit_status": "success"}
    catalog.details["uid-002"] = RunDetail(
        summary=summary_from_metadata("uid-002", start, stop),
        start_doc=start,
        stop_doc=stop,
        data=frame,
    )
    return TestClient(create_app(catalog)), catalog


def test_grid_payload_and_figures():
    client, _ = grid_client()
    r = client.get(
        "/api/run/uid-002/grid", params={"gridcfg": json.dumps({"value": "signal"})}
    )
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "no-cache"
    data = r.json()
    assert len(data["cells"]) == 4
    assert data["cells"][-1]["status"] == "unacquired"
    assert data["cells"][-2]["error"] is None
    assert data["cells"][0]["center"] == 2
    assert data["figures"]["center"]["data"][0]["type"] == "heatmap"
    assert data["figures"]["center"]["data"][0]["zsmooth"] is False
    assert "grid_scan(pf.frame, detail.start_doc" in data["code"]
    assert data["figures"]["center"]["layout"]["paper_bgcolor"] == "$tok:--surface"


def test_shared_filters_and_independent_error():
    client, _ = grid_client()
    filters = {
        "groups": [{"conditions": [{"column": "bin_number", "low": 2, "high": 2}]}]
    }
    r = client.get(
        "/api/run/uid-002/grid",
        params={
            "gridcfg": json.dumps(
                {"value": "signal", "average": "median", "error": "std"}
            ),
            "filters": json.dumps(filters),
        },
    )
    d = r.json()
    assert r.status_code == 200, r.text
    assert d["pass"] == 3 and len(d["cells"]) == 4
    assert d["cells"][0]["status"] == "filtered"
    assert d["cells"][1]["center"] == 4
    assert d["error_label"] == "Standard deviation"


@pytest.mark.parametrize(
    "cfg,code",
    [
        ({"value": "absent"}, 404),
        ({"value": "signal", "min_count": "3"}, 400),
        ({"value": "signal", "average": "mode"}, 400),
        ({"value": "signal", "visit": False}, 400),
        ({"value": "signal", "lower": 0.9, "upper": 0.1}, 400),
    ],
)
def test_grid_error_ladder(cfg, code):
    client, _ = grid_client()
    assert (
        client.get(
            "/api/run/uid-002/grid", params={"gridcfg": json.dumps(cfg)}
        ).status_code
        == code
    )


def test_grid_state_and_prefix_are_preserved():
    client, _ = grid_client()
    cfg = json.dumps({"value": "signal"})
    r = client.get(
        "/run/uid-002",
        params={"tab": "grid", "gridcfg": cfg, "gridbin": "2"},
        headers={"X-Forwarded-Prefix": "/portal"},
    )
    assert r.status_code == 200
    assert 'data-pane="grid"' in r.text
    assert 'id="grid-average"' in r.text and 'id="grid-error"' in r.text
    assert "gridcfg=" in r.text and "gridbin=2" in r.text
    assert "/portal/static/plotly" in r.text


def test_points_handle_missing_error():
    client, catalog = grid_client()
    catalog.details["uid-002"].start_doc.update(
        plan_pattern="spiral", plan_name="spiral"
    )
    r = client.get(
        "/api/run/uid-002/grid", params={"gridcfg": json.dumps({"value": "signal"})}
    )
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "points"


def test_notebook_reproduces_filtered_cells(monkeypatch):
    from geecs_data_utils.tiled_catalog import TiledScanCatalog
    from geecs_portal import analysis
    import plotly.graph_objects as go

    client, catalog = grid_client()
    monkeypatch.setattr(TiledScanCatalog, "from_config", lambda: catalog)
    monkeypatch.setattr(go.Figure, "show", lambda self: None)
    params = {
        "gridcfg": json.dumps({"value": "signal", "average": "mean", "error": "std"}),
        "filters": json.dumps(
            {"groups": [{"conditions": [{"column": "signal", "low": 2, "high": 100}]}]}
        ),
    }
    payload = client.get("/api/run/uid-002/grid", params=params).json()
    scope = {}
    exec(payload["code"], scope)
    assert (
        analysis.jsonable_document(scope["result"].cells.to_dict("records"))
        == payload["cells"]
    )


def test_cell_images_share_filtered_membership(tmp_path):
    import numpy as np
    from PIL import Image

    client, catalog = grid_client()
    folder = tmp_path / "Scan002"
    (folder / "camera").mkdir(parents=True)
    # A native file establishes the renderable device; membership is scalar-only.
    Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(
        folder / "camera" / "Scan002_camera_001.png"
    )
    catalog.details["uid-002"].start_doc["scan_folder"] = str(folder)
    filters = json.dumps(
        {"groups": [{"conditions": [{"column": "signal", "low": 2, "high": 100}]}]}
    )
    payload = client.get(
        "/api/run/uid-002/grid",
        params={"gridcfg": json.dumps({"value": "signal"}), "filters": filters},
    ).json()
    response = client.get(
        "/api/run/uid-002/bin-images",
        params={
            "device": "camera",
            "filters": filters,
            "bincfg": json.dumps({"bin_col": payload["bin_column"], "min_count": 1}),
        },
    )
    assert response.status_code == 200, response.text
    image_bins = {b["bin"]: b["shots"] for b in response.json()["bins"]}
    assert image_bins == {c["bin"]: c["shots"] for c in payload["cells"] if c["shots"]}


def test_rendered_grid_script_parses():
    from geecs_web_theme.testing import inline_scripts, javascript_syntax_error

    client, _ = grid_client()
    html = client.get("/run/uid-002", params={"tab": "grid"}).text
    for script in inline_scripts(html):
        assert javascript_syntax_error(script) is None


def test_log_and_index_coordinates():
    client, _ = grid_client()
    payload = client.get(
        "/api/run/uid-002/grid",
        params={
            "gridcfg": json.dumps(
                {"value": "signal", "yscale": "log", "xscale": "index"}
            )
        },
    ).json()
    fig = payload["figures"]["center"]
    assert fig["layout"]["yaxis"]["tickvals"] == [1, 10]
    assert fig["layout"]["xaxis"]["ticktext"] == ["2", "4"]
    assert fig["data"][0]["x"] == [-0.5, 0.5, 1.5]
    assert fig["data"][0]["y"] == pytest.approx([10**-0.5, 10**0.5, 10**1.5])
