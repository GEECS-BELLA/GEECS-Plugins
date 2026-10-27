"""The page: on the kit, the selection in the URL, assets through ``root``."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme import PANE_STATES, STATES
from geecs_web_theme.testing import (
    bare_url_for_calls,
    inline_scripts,
    javascript_syntax_error,
    node_available,
    unknown_data_states,
)

from tests.conftest import ALIASED, SCAN_VARIABLE

_PKG = Path(__file__).resolve().parents[1] / "geecs_plotter"
_TEMPLATES = sorted((_PKG / "templates").glob("*.html"))
_SCRIPTS = sorted((_PKG / "static").glob("plotter*.js"))


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_url_for_takes_the_path(template: Path) -> None:
    assert bare_url_for_calls(template) == []


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_literal_data_state_is_a_kit_word(template: Path) -> None:
    assert unknown_data_states(template, [*STATES, *PANE_STATES]) == []


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_no_inline_scripts(template: Path) -> None:
    assert inline_scripts(template) == []


@pytest.mark.skipif(not node_available(), reason="node not installed")
@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_every_script_parses(script: Path) -> None:
    assert javascript_syntax_error(script.read_text()) is None


def test_the_empty_form(client: TestClient) -> None:
    page = client.get("/").text
    assert 'data-state="empty"' in page
    assert "data-figure-url" not in page


def test_a_scan_defaults_x_to_the_scan_variable(client: TestClient) -> None:
    page = client.get("/", params={"day": "2026-10-02", "scan": 1}).text
    assert f'<option value="{SCAN_VARIABLE}" selected>' in page
    assert '<option value="U_ICT charge" selected>' in page  # the first measurement
    assert "x=U_Jet%20Position" in page


def test_x_empty_means_the_shot_number(client: TestClient) -> None:
    page = client.get(
        "/",
        params={
            "day": "2026-10-02",
            "scan": 1,
            "x": "",
            "y": "U_ICT charge",
            "xauto": "false",
        },
    ).text
    assert '<option value="" selected>shot #</option>' in page


def test_a_missing_scan_says_so(client: TestClient) -> None:
    page = client.get("/", params={"day": "2026-10-02", "scan": 9}).text
    assert 'data-state="error"' in page
    assert "Scan009" in page


def test_links_carry_the_proxy_prefix(client: TestClient) -> None:
    page = client.get(
        "/",
        params={"day": "2026-10-02", "scan": 1},
        headers={"X-Forwarded-Prefix": "/plot"},
    ).text
    assert 'href="/plot/theme/kit.css"' in page
    assert 'src="/plot/static/plotter.js"' in page
    assert 'data-figure-url="/plot/api/figure?' in page


def test_options_carry_the_exact_column_name(client: TestClient) -> None:
    """A browser strips and collapses option text; the value keeps the name."""
    page = client.get("/", params={"day": "2026-10-02", "scan": 1}).text
    assert '<option value="U_ICT charge" selected>U_ICT charge</option>' in page


def test_only_bookkeeping_columns_still_pick_a_y(
    client: TestClient, share: Path
) -> None:
    import pandas as pd

    from tests.conftest import write_scan

    write_scan(share, 3, pd.DataFrame({"Shotnumber": [1, 2], "Bin #": [1, 1]}))
    page = client.get("/", params={"day": "2026-10-02", "scan": 3}).text
    assert "y=Shotnumber" in page


def test_each_axis_has_its_own_filter(client: TestClient) -> None:
    """LabVIEW filters the X and Y lists independently; so does the page."""
    page = client.get("/", params={"day": "2026-10-02", "scan": 1}).text
    assert 'data-filter-for="x"' in page
    assert 'data-filter-for="y"' in page
    assert "option.hidden" not in (_PKG / "static" / "plotter.js").read_text()


def test_each_axis_lists_aliases_on_its_own_toggle(client: TestClient) -> None:
    """The Y toggle relabels the Y list only; the value stays the column."""
    page = client.get(
        "/", params={"day": "2026-10-02", "scan": 1, "y": ALIASED, "xalias": "false"}
    ).text
    assert f'<option value="{ALIASED}" selected>Gas (psi)</option>' in page
    assert f'<option value="{ALIASED}">{ALIASED}</option>' in page  # the X list
    assert 'name="yalias" value="true"' in page
    assert 'name="xalias" value="false"' in page
    assert "xalias=false&amp;yalias=true" in page


def test_the_toggles_survive_a_missing_scan(client: TestClient) -> None:
    """No lists to show, but the next Open keeps the alias choice."""
    page = client.get(
        "/", params={"day": "2026-10-02", "scan": 9, "yalias": "true"}
    ).text
    assert 'data-state="error"' in page
    assert 'name="yalias" value="true"' in page


def test_alias_and_auto_x_are_the_defaults(client: TestClient) -> None:
    page = client.get("/", params={"day": "2026-10-02", "scan": 1}).text
    assert 'name="xalias" value="true"' in page
    assert 'name="yalias" value="true"' in page
    assert 'name="xauto" value="true"' in page


def test_auto_x_follows_the_scan_variable_over_a_carried_x(
    client: TestClient,
) -> None:
    """Opening another scan with xauto on takes that scan's variable."""
    params = {"day": "2026-10-02", "scan": 1, "x": "U_ICT charge"}
    auto = client.get("/", params=params).text
    assert f'<option value="{SCAN_VARIABLE}" selected>' in auto
    manual = client.get("/", params={**params, "xauto": "false"}).text
    assert '<option value="U_ICT charge" selected>' in manual


def test_the_filters_ride_the_url(client: TestClient) -> None:
    """A pick or a toggle resubmits the form; the filter text goes with it."""
    page = client.get(
        "/", params={"day": "2026-10-02", "scan": 1, "xf": "jet", "yf": "gas"}
    ).text
    assert 'name="xf" value="jet"' in page
    assert 'name="yf" value="gas"' in page
    missing = client.get("/", params={"day": "2026-10-02", "scan": 9, "yf": "gas"}).text
    assert 'name="yf" value="gas"' in missing  # kept for the next Open


def test_a_scan_with_no_shots_says_so(client: TestClient, share: Path) -> None:
    """A scan stopped before its first shot: a header-only s-file, no picks."""
    import pandas as pd

    from tests.conftest import write_scan

    write_scan(share, 4, pd.DataFrame(columns=["Shotnumber", "Bin #", "scan"]))
    response = client.get("/", params={"day": "2026-10-02", "scan": 4})
    assert response.status_code == 200
    assert "No columns to plot" in response.text


def test_the_scan_number_opens_on_change(client: TestClient) -> None:
    """The scan box is marked for plotter.js to open the scan when it changes."""
    page = client.get("/").text
    assert "data-auto-open" in page
    assert "[data-auto-open]" in (_PKG / "static" / "plotter.js").read_text()


def test_stepping_past_an_empty_scan_keeps_the_picks(
    client: TestClient, share: Path
) -> None:
    """A scan with nothing to plot still hands X and Y on to the next one."""
    import pandas as pd

    from tests.conftest import write_scan

    write_scan(share, 4, pd.DataFrame(columns=["Shotnumber"]))
    page = client.get(
        "/",
        params={"day": "2026-10-02", "scan": 4, "x": "", "y": "U_ICT charge"},
    ).text
    assert "No columns to plot" in page
    assert '<input type="hidden" name="y" value="U_ICT charge">' in page
    assert '<input type="hidden" name="x" value="">' in page
