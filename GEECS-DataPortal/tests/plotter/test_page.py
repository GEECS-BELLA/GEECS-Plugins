"""The page: on the kit, the controls in the settings set, assets through ``root``."""

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

from tests.plotter.conftest import ALIASED, SCAN_VARIABLE, post

_PKG = Path(__file__).resolve().parents[2] / "geecs_portal" / "plotter"
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
    page = client.get("/?set=default").text
    assert 'data-state="empty"' in page
    assert "data-figure-url" not in page


def test_a_scan_defaults_x_to_the_scan_variable(client: TestClient) -> None:
    page = client.get(
        "/", params={"set": "default", "day": "2026-10-02", "scan": 1}
    ).text
    assert f'<option value="{SCAN_VARIABLE}" selected>' in page
    assert '<option value="U_ICT charge" selected>' in page  # the first measurement
    assert "x=U_Jet%20Position" in page


def test_x_empty_means_the_shot_number(client: TestClient) -> None:
    page = post(client, scan=1, x="", y="U_ICT charge", xauto="false")
    assert '<option value="" selected>shot #</option>' in page


def test_a_missing_scan_says_so(client: TestClient) -> None:
    page = client.get(
        "/", params={"set": "default", "day": "2026-10-02", "scan": 9}
    ).text
    assert 'data-state="error"' in page
    assert "Scan009" in page


def test_links_carry_the_proxy_prefix(client: TestClient) -> None:
    page = client.get(
        "/",
        params={"set": "default", "day": "2026-10-02", "scan": 1},
        headers={"X-Forwarded-Prefix": "/plot"},
    ).text
    assert 'href="/plot/theme/kit.css"' in page
    assert 'src="/plot/static/plotter.js"' in page
    assert 'data-figure-url="/plot/api/figure?' in page


def test_options_carry_the_exact_column_name(client: TestClient) -> None:
    """A browser strips and collapses option text; the value keeps the name."""
    page = client.get(
        "/", params={"set": "default", "day": "2026-10-02", "scan": 1}
    ).text
    assert '<option value="U_ICT charge" selected>U_ICT charge</option>' in page


def test_only_bookkeeping_columns_still_pick_a_y(
    client: TestClient, share: Path
) -> None:
    import pandas as pd

    from tests.plotter.conftest import write_scan

    write_scan(share, 3, pd.DataFrame({"Shotnumber": [1, 2], "Bin #": [1, 1]}))
    page = client.get(
        "/", params={"set": "default", "day": "2026-10-02", "scan": 3}
    ).text
    assert "y=Shotnumber" in page


def test_each_axis_has_its_own_filter(client: TestClient) -> None:
    """LabVIEW filters the X and Y lists independently; so does the page."""
    page = client.get(
        "/", params={"set": "default", "day": "2026-10-02", "scan": 1}
    ).text
    assert 'data-filter-for="x"' in page
    assert 'data-filter-for="y"' in page
    assert "option.hidden" not in (_PKG / "static" / "plotter.js").read_text()


def test_each_axis_lists_aliases_on_its_own_toggle(client: TestClient) -> None:
    """The Y toggle relabels the Y list only; the value stays the column."""
    page = post(client, scan=1, y=ALIASED, xalias="false")
    assert f'<option value="{ALIASED}" selected>Gas (psi)</option>' in page
    assert f'<option value="{ALIASED}">{ALIASED}</option>' in page  # the X list
    assert 'name="yalias" value="true"' in page
    assert 'name="xalias" value="false"' in page
    assert "xalias=false&amp;yalias=true" in page


def test_the_toggles_survive_a_missing_scan(client: TestClient) -> None:
    """No lists to show, and the saved alias choice is kept for the next scan."""
    post(client, scan=1, yalias="false")
    page = post(client, scan=9, yalias="false")
    assert 'data-state="error"' in page
    assert 'name="yalias" value="false"' in page


def test_alias_and_auto_x_are_the_defaults(client: TestClient) -> None:
    page = client.get(
        "/", params={"set": "default", "day": "2026-10-02", "scan": 1}
    ).text
    assert 'name="xalias" value="true"' in page
    assert 'name="yalias" value="true"' in page
    assert 'name="xauto" value="true"' in page


def test_auto_x_follows_the_scan_variable_over_a_saved_x(
    client: TestClient,
) -> None:
    """Opening another scan with xauto on takes that scan's variable."""
    auto = post(client, scan=1, x="U_ICT charge")
    assert f'<option value="{SCAN_VARIABLE}" selected>' in auto
    manual = post(client, scan=1, x="U_ICT charge", xauto="false")
    assert '<option value="U_ICT charge" selected>' in manual


def test_the_filters_are_saved(client: TestClient) -> None:
    """A pick or a toggle saves the filter text; a scan without lists keeps it."""
    page = post(client, scan=1, xf="jet", yf="gas")
    assert 'name="xf" value="jet"' in page
    assert 'name="yf" value="gas"' in page
    post(client, scan=9)  # no lists, so no filter fields sent
    page = client.get(
        "/", params={"set": "default", "day": "2026-10-02", "scan": 1}
    ).text
    assert 'name="yf" value="gas"' in page


def test_a_scan_with_no_shots_says_so(client: TestClient, share: Path) -> None:
    """A scan stopped before its first shot: a header-only s-file, no picks."""
    import pandas as pd

    from tests.plotter.conftest import write_scan

    write_scan(share, 4, pd.DataFrame(columns=["Shotnumber", "Bin #", "scan"]))
    response = client.get(
        "/", params={"set": "default", "day": "2026-10-02", "scan": 4}
    )
    assert response.status_code == 200
    assert "No columns to plot" in response.text


def test_the_scan_number_opens_on_change(client: TestClient) -> None:
    """The scan box is marked for plotter.js to open the scan when it changes."""
    page = client.get("/?set=default").text
    assert "data-auto-open" in page
    assert "[data-auto-open]" in (_PKG / "static" / "plotter.js").read_text()


def test_stepping_past_an_empty_scan_keeps_the_picks(
    client: TestClient, share: Path
) -> None:
    """A scan with nothing to plot leaves X and Y saved for the next one."""
    import pandas as pd

    from tests.plotter.conftest import write_scan

    write_scan(share, 4, pd.DataFrame(columns=["Shotnumber"]))
    post(client, scan=1, x="", y="U_ICT charge", xauto="false")
    assert "No columns to plot" in post(client, scan=4, xauto="false")
    page = post(client, scan=1, xauto="false")
    assert '<option value="U_ICT charge" selected>' in page
    assert '<option value="" selected>shot #</option>' in page


def test_the_count_shows_one_two_or_three_y_lists(client: TestClient) -> None:
    """GEECSplotter's choice of one, two or three Y plots."""
    one = client.get(
        "/", params={"set": "default", "day": "2026-10-02", "scan": 1}
    ).text
    assert 'id="y2"' not in one and 'id="y3"' not in one
    assert 'data-toggle-for="ycount" data-value="1" aria-pressed="true"' in one
    assert "&amp;y2=&amp;y3=&amp;" in one  # one trace
    two = post(client, scan=1, ycount=2)
    assert 'id="y2"' in two and 'id="y3"' not in two
    assert 'class="swatch t2"' in two
    assert "— off —" not in two  # the count, not an "off" pick, says what draws
    three = post(client, scan=1, ycount=3, y2=ALIASED)
    assert 'id="y3"' in three
    assert f'<option value="{ALIASED}" selected>Gas (psi)</option>' in three
    assert "&amp;y2=U_Gas%20Pressure%20Alias%3AGas%20%28psi%29" in three


def test_new_y_lists_default_to_the_next_measurements(client: TestClient) -> None:
    page = post(client, scan=1, ycount=3)
    y = page.split('id="y"')[1].split("</select>")[0]
    y2 = page.split('id="y2"')[1].split("</select>")[0]
    y3 = page.split('id="y3"')[1].split("</select>")[0]
    picked = [
        part.split(" selected>")[0].rsplit('value="', 1)[1] for part in (y, y2, y3)
    ]
    assert len(set(picked)) == 3  # three different columns, none bookkeeping
    assert "Shotnumber" not in picked


def test_a_lower_count_keeps_the_hidden_picks(
    client: TestClient, settings_dir: Path
) -> None:
    import json

    post(client, scan=1, ycount=2, y2=ALIASED)
    page = post(client, scan=1, ycount=1)
    assert 'id="y2"' not in page
    assert json.loads((settings_dir / "default.json").read_text())["y2"] == ALIASED
    back = post(client, scan=1, ycount=2)
    assert f'<option value="{ALIASED}" selected>Gas (psi)</option>' in back


def test_y2_off_on_a_scan_without_its_column_but_kept(
    client: TestClient, settings_dir: Path
) -> None:
    import json

    post(client, scan=1, ycount=2, y2=ALIASED)
    page = post(client, scan=2)  # Scan002 has no gas pressure column
    assert "&amp;y2=&amp;" in page  # nothing drawn
    # The list still holds the pick, so the next click here posts it back.
    assert (
        f'<option value="{ALIASED}" selected>{ALIASED} (not in this scan)</option>'
        in page
    )
    saved = json.loads((settings_dir / "default.json").read_text())
    assert saved["y2"] == ALIASED
