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
    assert 'id="pane-plot"' not in page  # the scan page has no plot of its own
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


def test_without_the_plotter_the_scan_page_has_no_plot_tab() -> None:
    from fastapi.testclient import TestClient

    from tests.test_app import FakeCatalog

    from geecs_portal.app import create_app

    page = TestClient(create_app(FakeCatalog())).get("/run/uid-002").text
    assert 'id="tab-plot"' not in page and 'id="pane-plot"' not in page


def test_another_experiments_run_has_no_plot_tab(tmp_path) -> None:
    """The Plot page opens the portal's own experiment by day and number only."""
    from tests.test_app import TEST_DAY

    client = _with_plotter(tmp_path)
    page = client.get("/run/uid-002", params={"experiment": "Thomson"}).text
    assert 'id="tab-plot"' not in page and 'id="pane-plot"' not in page
    day = client.get(
        f"/day/{TEST_DAY.isoformat()}", params={"experiment": "Thomson"}
    ).text
    assert 'href="/plot/' not in day and 'href="/run/uid-002?' in day


#: Every browser global the scan page touches, as a stub that answers any
#: property, call or `new` with itself: enough to run the whole script in
#: strict mode, so a name the page reads, or assigns, but never defines
#: throws — including inside the columns bootstrap, whose catch fails the run.
_BOOT_HARNESS = r"""
"use strict";
const any = new Proxy(function () {}, {
  get: (t, k) => k === Symbol.toPrimitive ? () => "" : k === "then" ? undefined : any,
  apply: () => any, construct: () => any, set: () => true,
});
const fail = err => {
  if (err instanceof ReferenceError) { console.error(err); process.exit(1); }
};
process.on("uncaughtException", fail);
process.on("unhandledRejection", fail);
if (typeof navigator === "undefined") globalThis.navigator = {};  // node < 21
// The columns bootstrap catches its own errors into the rail's provider
// line, so a ReferenceError there would never reach the hooks above.
const providers = { set innerHTML(html) {
  if (String(html).includes("columns:")) { console.error(html); process.exit(1); }
} };
const document = new Proxy({}, {
  get: (t, k) => k === "getElementById"
    ? id => id === "providers" ? providers : any : any,
});
Object.assign(globalThis, {
  document, window: any, history: any, localStorage: any, Option: any,
  Plotly: any, GeecsKit: any, ResizeObserver: any, ClipboardItem: undefined,
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  fetch: async () => ({ ok: true, json: async () => ({ columns: [], grid_axes: [] }) }),
});
"""


@pytest.mark.parametrize("tab", ["overview", "grid", "images"])
def test_the_scan_page_script_boots_without_an_undefined_name(
    tab: str, tmp_path: Path
) -> None:
    """A deletion that leaves a reader behind (the Grid read the old Plot
    pane's ``BOOTED``) passes the syntax check and breaks in the browser."""
    import shutil
    import subprocess

    from fastapi.testclient import TestClient
    from geecs_web_theme.testing import inline_scripts

    from geecs_portal.app import create_app
    from tests.test_app import FakeCatalog

    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to boot the page script")
    page = TestClient(create_app(FakeCatalog())).get(f"/run/uid-002?tab={tab}").text
    location = (
        "globalThis.location = new URL("
        + json.dumps(f"http://portal.example/run/uid-002?tab={tab}")
        + ");\n"
    )
    script = tmp_path / "boot.cjs"
    script.write_text(
        _BOOT_HARNESS
        + location
        + "\n".join(inline_scripts(page)).replace('"use strict";', "")
        + "\nsetTimeout(() => {}, 50);\n"
    )
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
