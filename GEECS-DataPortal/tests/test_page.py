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


def test_the_images_tab_is_the_plotters_images_page(tmp_path) -> None:
    """The scan page links its Images tab there, and an old tab=images link
    is sent on (the scan page has no Images pane)."""
    client = _with_plotter(tmp_path)
    page = client.get("/run/uid-002").text
    assert 'id="tab-images"' in page and 'href="/plot/images?day=' in page
    assert 'id="pane-images"' not in page
    moved = client.get("/run/uid-002", params={"tab": "images"}, follow_redirects=False)
    assert moved.status_code == 303
    assert moved.headers["location"].startswith("/plot/images?day=")


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
    assert 'src="/portal/plot/theme/plotly-cartesian-' in page
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
// A test may hand the page real elements (``__elements`` by id) and
// document members (``__document``); everything else is the stub.
const document = new Proxy({}, {
  get: (t, k) => k === "getElementById"
    ? id => id === "providers" ? providers : (globalThis.__elements || {})[id] || any
    : (globalThis.__document || {})[k] || any,
});
Object.assign(globalThis, {
  document, window: any, history: any, localStorage: any, Option: any,
  Plotly: any, GeecsKit: any, ResizeObserver: any, CSS: { escape: String }, ClipboardItem: undefined,
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


def _run_page_script(tmp_path: Path, test: str) -> None:
    """Boot the scan page's script under node, then run *test* against it."""
    import shutil
    import subprocess

    from fastapi.testclient import TestClient
    from geecs_web_theme.testing import inline_scripts

    from geecs_portal.app import create_app
    from tests.test_app import FakeCatalog

    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to run the page script")
    page = TestClient(create_app(FakeCatalog())).get("/run/uid-002").text
    script = tmp_path / "page.cjs"
    script.write_text(
        _BOOT_HARNESS
        + 'globalThis.location = new URL("http://portal.example/run/uid-002");\n'
        + "\n".join(inline_scripts(page)).replace('"use strict";', "")
        + "\nconst check = (ok, what) => { if (!ok) { console.error(what); process.exit(1); } };\n"
        + test
        + "\nprocess.exit(0);\n"
    )
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


#: A listing with one analyzer, running at ``done`` of 11 shots (or done).
_ANALYSIS_LISTING = r"""
const listing = (done, state = "running") => ({
  running: state === "running" ? "PW_SpecFringes" : null,
  analyzers: [{
    id: "PW_SpecFringes", device: "Thorlabs", applicable: true, destructive: false,
    files: [], artifacts: [],
    job: { state, error: null, log: [], finished: null,
           progress: { done, total: 11, phase: "shot" } },
  }],
});
"""


def test_a_running_analysis_draws_shot_k_of_n_on_the_kit_meter(tmp_path) -> None:
    """The Analysis tab's row carries a determinate meter while the run lasts;
    a progress-only poll moves it without rebuilding the list (which would
    re-fetch every inline artifact); the settled row has none."""
    _run_page_script(
        tmp_path,
        _ANALYSIS_LISTING
        + r"""
const drawn = [];
globalThis.__elements = { analysisList: { set innerHTML(html) { drawn.push(html); } } };
const meter = { outerHTML: "" };
const ROW = '#analysisList .arow[data-row="PW_SpecFringes"]';
globalThis.__document = { querySelector: sel => sel === ROW
  ? { querySelector: s => s === ":scope > .aprog" ? meter : null } : null };
renderAnalysis(listing(3), false);
check(drawn.length === 1, "first listing draws the list");
check(drawn[0].includes('class="meter aprog"'), "a running row carries the kit meter");
check(drawn[0].includes("shot 3 of 11"), "the meter says shot k of N: " + drawn[0]);
check(drawn[0].includes('aria-valuenow="3"') && drawn[0].includes('aria-valuemax="11"'),
      "the meter is a progressbar with its numbers");
check(drawn[0].includes("width:27%"), "the bar is done/total wide");
renderAnalysis(listing(4), false);
check(drawn.length === 1, "a progress tick does not rebuild the list");
check(meter.outerHTML.includes("shot 4 of 11"), "a progress tick moves the meter");
renderAnalysis(listing(11, "done"), false);
check(drawn.length === 2, "the settled run redraws the list");
check(!drawn[1].includes("aprog"), "a settled row carries no meter");
check(drawn[1].includes('data-state="ok"'), "and its final chip");
""",
    )


def test_a_first_progress_report_inserts_the_meter(tmp_path) -> None:
    """A row drawn before the run reported anything has no meter; the first
    report adds one after the row's head rather than rebuilding the list."""
    _run_page_script(
        tmp_path,
        _ANALYSIS_LISTING
        + r"""
const drawn = [];
globalThis.__elements = { analysisList: { set innerHTML(html) { drawn.push(html); } } };
const head = { inserted: null, insertAdjacentHTML(where, html) { this.inserted = [where, html]; } };
const row = { querySelector: sel => sel === ":scope > .head" ? head : null };
globalThis.__document = { querySelector: sel =>
  sel === '#analysisList .arow[data-row="PW_SpecFringes"]' ? row : null };
const silent = listing(0);
silent.analyzers[0].job.progress = null;
renderAnalysis(silent, false);
check(drawn.length === 1 && !drawn[0].includes("aprog"), "no report, no meter");
renderAnalysis(listing(1), false);
check(drawn.length === 1, "the first report does not rebuild the list");
check(head.inserted && head.inserted[0] === "afterend"
      && head.inserted[1].includes("shot 1 of 11"), "the meter is inserted after the head");
""",
    )


def test_the_day_page_plots_the_scans_it_shows(tmp_path) -> None:
    """ "Plot these scans" posts the shown rows into the viewer's set."""
    import json

    client = _with_plotter(tmp_path)
    page = client.get("/day/2026-07-12").text
    assert 'id="plot-shown"' in page
    assert 'data-scan="1"' in page and 'data-scan="2"' in page
    # what the button posts for nine shown scans: the lowest the main one,
    # the rest added, merged (the overlay draws at most eight)
    rest = ",".join(f"2026-07-12:{n}" for n in range(2, 10))
    saved = client.post(
        "/plot/",
        data={
            "day": "2026-07-12",
            "scan": "1",
            "set": "ann",
            "owner": "ann",
            "page": "plot",
            "scans": rest,
            "multi": "merged",
        },
        follow_redirects=False,
    )
    assert saved.status_code == 303
    body = json.loads((tmp_path / "sets" / "ann.json").read_text())
    assert body["scan"] == 1 and len(body["extra_scans"]) == 8
    assert body["scan_view"] == "merged"


def test_a_degraded_run_shows_the_kits_degraded_status_and_its_counts(
    tmp_path,
) -> None:
    """A done run that continued past failures is not drawn green, its counts
    (NaN values included) are on its row, and listed files say they may
    predate the run (``freshness unknown`` with no run record)."""
    _run_page_script(
        tmp_path,
        r"""
const drawn = [];
globalThis.__elements = { analysisList: { set innerHTML(html) { drawn.push(html); } } };
const art = [{ path: "D/x.png", servable: true, inline: true, kind: "summary", bin: null }];
const counts = (o) => Object.assign({ units_attempted: 6, units_succeeded: 6, units_failed: 0,
  shots_load_failed: 0, nan_values: 0, product_write_failures: 0,
  scalar_write_failures: 0, fold_errors: 0 }, o);
const row = (job) => ({ running: null, analyzers: [{ id: "D", device: "cam", applicable: true,
  destructive: false, files: ["D/x.png"], artifacts: art, job }] });
const job = (o) => Object.assign({ state: "done", error: null, log: [], finished: 1700000000,
  progress: null, degraded: false, outcome: counts({}) }, o);
renderAnalysis(row(job({ degraded: true,
  outcome: counts({ units_succeeded: 4, units_failed: 2, nan_values: 18 }) })), false);
check(drawn[0].includes('data-state="degraded"'), "degraded chip: " + drawn[0]);
check(drawn[0].includes("done · degraded"), "says done, degraded");
check(!drawn[0].includes('data-state="ok"'), "never the ok chip");
check(drawn[0].includes("4 of 6 units succeeded"), "the units");
check(drawn[0].includes("<b>2</b> failed") && drawn[0].includes("<b>18</b> NaN values"),
      "the failures and the NaN values");
check(drawn[0].includes("files may predate this run (finished "), "the file-list wording");
renderAnalysis(row(job({ outcome: counts({ nan_values: 3 }) })), false);
check(drawn[1].includes('data-state="ok"'), "NaN values alone leave the run done");
check(drawn[1].includes("<b>3</b> NaN values"), "but are shown");
renderAnalysis(row(null), false);
check(drawn[2].includes("freshness unknown"), "no record: freshness unknown");
""",
    )


def test_the_outcome_terms_name_real_run_outcome_fields() -> None:
    """``OUTCOME_TERMS`` repeats ScanAnalysis' count names: a rename there fails here."""
    base = pytest.importorskip("scan_analysis.base")
    run = (_TEMPLATES[0].parent / "run.html").read_text()
    block = re.search(r"const OUTCOME_TERMS = \[(.*?)\];", run, re.S).group(1)
    names = re.findall(r'\["(\w+)", "', block)
    assert names
    fields = set(base.RunOutcome.model_fields)
    assert set(names) <= fields
    # every count but the two the line leads with is drawn
    assert set(names) == fields - {"units_attempted", "units_succeeded"}
