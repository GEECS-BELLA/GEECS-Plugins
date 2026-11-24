"""The scan page over a grid scan: no grid tab, scripts that parse."""

import pandas as pd
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


def test_the_scan_page_has_no_grid_tab():
    """The map is the Plot page's view now; a tab=grid link opens the Overview."""
    client, _ = grid_client()
    r = client.get(
        "/run/uid-002",
        params={"tab": "grid"},
        headers={"X-Forwarded-Prefix": "/portal"},
    )
    assert r.status_code == 200
    assert 'data-pane="grid"' not in r.text and 'id="pane-grid"' not in r.text
    assert 'tab: p.get("tab") || "overview"' in r.text


def test_rendered_grid_script_parses():
    from geecs_web_theme.testing import inline_scripts, javascript_syntax_error

    client, _ = grid_client()
    html = client.get("/run/uid-002", params={"tab": "grid"}).text
    for script in inline_scripts(html):
        assert javascript_syntax_error(script) is None
