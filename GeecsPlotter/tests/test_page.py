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

from tests.conftest import SCAN_VARIABLE

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
        "/", params={"day": "2026-10-02", "scan": 1, "x": "", "y": "U_ICT charge"}
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
