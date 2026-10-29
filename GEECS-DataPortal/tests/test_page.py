"""The pages are on the GeecsWebTheme kit: its status words, its glue."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from geecs_web_theme import PANE_STATES, STATES
from geecs_web_theme.testing import bare_url_for_calls, unknown_data_states

from geecs_portal import app as portal_app
from geecs_portal.app import run_state

_TEMPLATES = sorted(
    (Path(__file__).resolve().parents[1] / "geecs_portal" / "templates").glob("*.html")
)


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_url_for_takes_the_path(template: Path) -> None:
    assert bare_url_for_calls(template) == []


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_literal_data_state_is_a_kit_word(template: Path) -> None:
    # A value the page script builds (`${kind}`, `${JOB_STATES[...]}`)
    # is not a literal; JOB_STATES is pinned below, and paneState's kinds
    # are literals at each call.
    unknown = unknown_data_states(template, [*STATES, *PANE_STATES])
    assert [v for v in unknown if not v.startswith("${")] == []


def test_the_page_scripts_state_words_are_kit_words() -> None:
    """``JOB_STATES`` (an analysis job's state as a chip) and every
    ``paneState(kind, …)`` call name words the kit styles."""
    run = (_TEMPLATES[0].parent / "run.html").read_text()
    table = re.search(r"const JOB_STATES = (\{[^}]*\});", run).group(1)
    jobs = json.loads(re.sub(r"(\w+):", r'"\1":', table))
    assert set(jobs.values()) <= set(STATES)
    kinds = set(re.findall(r'paneState\("(\w+)"', run))
    assert kinds and kinds <= set(PANE_STATES)


@pytest.mark.parametrize(
    ("exit_status", "state"),
    [
        ("success", "ok"),
        ("abort", "degraded"),
        ("fail", "failed"),
        (None, "unknown"),
        ("", "unknown"),
        ("unknown", "unknown"),
        ("something new", "unknown"),
    ],
)
def test_a_runs_exit_status_is_a_kit_status_word(exit_status, state) -> None:
    assert run_state(exit_status) == state
    assert state in STATES


def test_the_glue_is_the_themes_own() -> None:
    """The portal's copies of the prefix middleware and ``root`` are gone."""
    assert not hasattr(portal_app, "_ForwardedPrefixMiddleware")
    assert not hasattr(portal_app, "_clean_prefix")


# -- the Plot tab is GEECSplotter's page, mounted at /plot ------------------


def _with_plotter(tmp_path):
    """The portal over the fake catalog, with the plotter over a fake share."""
    from fastapi.testclient import TestClient

    from geecs_portal.plotter.app import create_app as create_plotter
    from tests.plotter.conftest import EXPERIMENT
    from tests.test_app import FakeCatalog

    from geecs_portal.app import create_app

    plotter = create_plotter(
        EXPERIMENT,
        base_directory=tmp_path,
        settings_dir=tmp_path / "sets",
        mount="/plot",
    )
    return TestClient(create_app(FakeCatalog(), plotter=plotter))


def test_the_plot_tab_links_to_the_plotter_page(tmp_path) -> None:
    client = _with_plotter(tmp_path)
    page = client.get("/run/uid-002").text
    assert 'id="tab-plot"' in page and 'href="/plot/?day=' in page
    assert "const PLOT_PAGE = true;" in page
    # The scan page opens on the Overview; Plot is the other page now.
    assert client.get("/plot/health").json()["ok"] is True


def test_the_day_list_opens_a_scan_on_its_plot(tmp_path) -> None:
    from tests.test_app import TEST_DAY

    page = _with_plotter(tmp_path).get(f"/day/{TEST_DAY.isoformat()}").text
    assert f'href="/plot/?day={TEST_DAY.isoformat()}&amp;scan=2"' in page


def test_the_plot_page_carries_the_portals_tabs_behind_a_proxy(tmp_path) -> None:
    """Mounted, the page names the portal's prefix for its tabs and Plotly."""
    page = (
        _with_plotter(tmp_path)
        .get(
            "/plot/",
            params={"set": "default", "day": "2026-10-02", "scan": 1},
            headers={"X-Forwarded-Prefix": "/portal"},
        )
        .text
    )
    assert 'href="/portal/run/jump/2026-10-02?prefer=1&amp;tab=overview' in page
    assert 'src="/portal/static/plotly-cartesian-' in page
    assert 'src="/portal/plot/static/plotter.js"' in page
    assert 'action="/portal/plot/"' in page


def test_without_the_plotter_the_scan_page_keeps_its_plot_pane() -> None:
    from fastapi.testclient import TestClient

    from tests.test_app import FakeCatalog

    from geecs_portal.app import create_app

    page = TestClient(create_app(FakeCatalog())).get("/run/uid-002").text
    assert "const PLOT_PAGE = false;" in page and 'data-pane="plot"' in page
