"""The page: it renders on the kit, addresses its assets correctly, and its scripts parse.

The three template guards — ``url_for(...)`` always takes ``.path``, every
literal ``data-state`` is a kit word, every inline script parses under
``node --check`` — are ``geecs_web_theme.testing``'s helpers; this file
only asserts over their findings.  What is the scanner's own stays here:
the script's ``K`` table of kit words and its ``setChip`` literals, the
page's own script file parsing, and "every class the page uses is styled".
"""

from __future__ import annotations

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme.testing import (
    bare_url_for_calls,
    classes_used,
    inline_scripts,
    javascript_syntax_error,
    node_available,
    styled_classes,
    unknown_data_states,
)

_PKG = Path(__file__).resolve().parents[1] / "geecs_scanner"
_TEMPLATES = sorted((_PKG / "templates").glob("*.html"))
_SCRIPTS = sorted((_PKG / "static").glob("*.js"))


def test_page_renders_on_the_kit(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    assert '<body class="kit"' in html
    assert 'src="/theme/theme-boot.js"' in html and 'href="/theme/kit.css"' in html
    assert 'href="/static/scanner.css"' in html and 'src="/static/scanner.js"' in html
    assert "data-density-picker" in html and "data-theme-picker" in html
    assert "GEECS Scanner" in html and "Demo" in html
    # the assets the page names are actually served
    assert client.get("/static/scanner.js").status_code == 200
    assert client.get("/static/scanner.css").status_code == 200
    # the JSON pointer moved under /api
    assert client.get("/api").json()["events"] == "/api/events"


def test_page_carries_the_proxy_prefix(client: TestClient) -> None:
    html = client.get("/", headers={"X-Forwarded-Prefix": "/scan"}).text
    assert 'src="/scan/theme/theme-boot.js"' in html
    assert 'href="/scan/static/scanner.css"' in html
    assert 'data-root="/scan"' in html


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_url_for_takes_the_path(template: Path) -> None:
    bare = bare_url_for_calls(template)
    assert not bare, f"{template.name}: url_for without .path — {bare[:3]}"


def test_every_literal_data_state_is_a_kit_state() -> None:
    from geecs_web_theme import PANE_STATES, STATES

    allowed = (*STATES, *PANE_STATES)
    problems = []
    for template in _TEMPLATES:
        for value in unknown_data_states(template, allowed):
            problems.append(f"{template.name}: {value}")
    # the script writes states too: every word lives in its K table, pinned
    # here, and no setChip call may pass a literal instead
    for script in _SCRIPTS:
        text = script.read_text()
        k = re.search(r"var K = \{([^}]*)\}", text)
        assert k, f"{script.name}: no K table of kit words"
        for m in re.finditer(r'"([a-z_]+)"', k.group(1)):
            if m.group(1) not in STATES:
                problems.append(f"{script.name} K: {m.group(1)}")
        keys = set(re.findall(r"([a-z_]+):\s*\"", k.group(1)))
        for m in re.finditer(r"\bK\.([A-Za-z_]+)", text):
            if m.group(1) not in keys:
                problems.append(f"{script.name}: K.{m.group(1)} is not in the K table")
        for m in re.finditer(r"setChip\(([^;]*?)\);", text, re.S):
            if re.search(r'^\s*[^,]+,\s*"', m.group(1)) or re.search(
                r'\?\s*"[a-z_]+"\s*:', m.group(1)
            ):
                problems.append(
                    f"{script.name}: literal state in setChip({m.group(1)[:60]}…)"
                )
    assert not problems, f"data-state values the kit does not colour: {problems}"


def _need_node() -> None:
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available to parse JavaScript")


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_static_scripts_parse(script: Path) -> None:
    # The page's own script FILE — the shared helper covers inline blocks.
    _need_node()
    problem = javascript_syntax_error(script.read_text())
    assert problem is None, f"{script.name} does not parse:\n{problem}"


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_inline_scripts_parse(template: Path) -> None:
    scripts = inline_scripts(template)
    if not scripts:
        pytest.skip("no inline script in this template")
    _need_node()
    for i, block in enumerate(scripts):
        problem = javascript_syntax_error(block)
        assert problem is None, f"{template.name} inline script #{i + 1}:\n{problem}"


def test_page_uses_only_kit_or_page_classes() -> None:
    """Every class the template uses is styled by the kit, the theme or scanner.css."""
    from geecs_web_theme import kit_css, theme_css

    used: set[str] = set()
    for template in _TEMPLATES:
        used |= classes_used(template)
    styled = styled_classes(
        Path(kit_css()).read_text(),
        Path(theme_css()).read_text(),
        (_PKG / "static" / "scanner.css").read_text(),
    )
    missing = sorted(used - styled)
    assert not missing, f"console.html uses {missing} but nothing styles them"


def _script_function(script: str, name: str) -> str:
    """The body of one top-level ``function name() {...}`` of the page's IIFE."""
    m = re.search(rf"\n  function {name}\([^)]*\) \{{\n(.*?)\n  \}}\n", script, re.S)
    assert m, f"scanner.js: no function {name}()"
    return m.group(1)


def test_start_gate_needs_a_valid_form_not_a_loaded_preset() -> None:
    """#900: presets are optional — a scan composed from scratch can Start.

    The gate reads form validity only: neither ``recalc`` (which computes
    it) nor ``updateStartGate`` (which applies it) may consult the loaded
    preset document.  Then the gate itself runs under node over a stub
    DOM with a valid form and no preset ever loaded: an idle manager →
    Start enabled with no hover excuse, Save as preset enabled, the
    provenance note empty; a RUNNING plan → still enabled, the next scan
    queues behind it (#905); a PAUSED plan → disabled, since an item
    added then would be removed by the client.
    """
    script = (_PKG / "static" / "scanner.js").read_text()
    recalc, gate = (
        _script_function(script, "recalc"),
        _script_function(script, "updateStartGate"),
    )
    assert "presetDoc" not in recalc, "the form's validity must not depend on a preset"
    assert "presetDoc" not in gate, "the Start gate must not depend on a preset"
    assert "load a preset" not in script
    _need_node()
    harness = "\n".join(
        [
            "var els = {};",
            'function $(id) { return els[id] || (els[id] = { disabled: true, title: "x", textContent: "x" }); }',
            'var S = { status: null, formable: true, formableNote: "", presetDoc: null, presetName: null };',
            "var valid = true;",
            f"function updateStartGate() {{\n{gate}\n}}",
            "var out = {};",
            '["idle", "running", "paused"].forEach(function (re) {',
            "  els = {}; S.status = { connected: true, re_state: re }; updateStartGate();",
            '  out[re] = { start: $("btn-start").disabled, title: $("btn-start").title,',
            '    save: $("btn-save-preset").disabled, note: $("preset-name").textContent };',
            "});",
            "console.log(JSON.stringify(out));",
        ]
    )
    out = subprocess.run(
        ["node", "-"], input=harness, capture_output=True, text=True, check=True
    )
    got = json.loads(out.stdout)
    enabled = {"start": False, "title": "", "save": False, "note": ""}
    assert got["idle"] == enabled, got
    assert got["running"] == enabled, got
    assert got["paused"]["start"] is True and "paused" in got["paused"]["title"], got


def test_presets_and_actions_are_dropdowns(client: TestClient) -> None:
    """PR 5a: the rail's preset picklist and the actions picklist became selects."""
    html = client.get("/").text
    assert '<select id="preset"' in html and '<select id="action">' in html
    assert 'id="presets"' not in html and 'id="actions-list"' not in html
    # the preview the action dropdown drives is still there
    assert 'id="action-steps"' in html


def test_move_panel_carries_the_kit_live_row(client: TestClient) -> None:
    """PR 5b: the picked variable's readback sits in a kit .live row, hidden until a pick."""
    html = client.get("/").text
    assert '<div class="live" id="mv-live" hidden>' in html
    for span in ('id="mv-k"', 'id="mv-rb"', 'id="mv-age"'):
        assert span in html


def test_form_starts_at_the_mode_segment(client: TestClient) -> None:
    """#896: the preset picker is optional, so it lives in the footer beside Save as preset;

    the panel's body opens on its mode-specific controls, and the
    provenance note stays in the same row as the picker.
    """
    html = client.get("/").text
    sub = html[html.index('id="submit"') : html.index('id="queue"')]
    body = sub[sub.index('<div class="body">') : sub.index('id="optimizer-form"')]
    assert "<select" not in body and "<input" not in body, body
    footer = sub[sub.index("<footer>") :]
    for piece in (
        '<select id="preset"',
        'id="presets-note"',
        'id="preset-name"',
        'id="btn-save-preset"',
    ):
        assert piece in footer, piece


def _hint_texts(html: str) -> list[str]:
    """The inner text of every ``span.hint`` — through the stdlib parser, not a regex."""

    class Hints(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.texts: list[str] = []
            self._depth = 0

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            if self._depth:
                self._depth += 1
            elif tag == "span" and "hint" in (dict(attrs).get("class") or "").split():
                self._depth = 1
                self.texts.append("")

        def handle_endtag(self, tag: str) -> None:
            if self._depth:
                self._depth -= 1

        def handle_data(self, data: str) -> None:
            if self._depth:
                self.texts[-1] += data

    p = Hints()
    p.feed(html)
    return p.texts


def test_hints_carry_state_or_a_unit_never_prose(client: TestClient) -> None:
    """#895: the static explanatory hints are gone; the computed ones and the unit stay."""
    html = client.get("/").text
    hints = " | ".join(_hint_texts(html))
    for prose in (
        "catalog name",
        "seeds the form below",
        "stepped inside each axis-1 point",
        "shot_control_configurations/",  # survives only as the field's hover text
        "goes to ScanInfo",
        "measure only",
        "one YAML under presets/",
        "names the manager resolves",
    ):
        assert prose not in hints, prose
    assert 'id="mode-note"' not in html
    script = (_PKG / "static" / "scanner.js").read_text()
    for prose in (
        "var NOTES",
        "mode-note",
        "seeds the form below",
        "pick one to preview",
        "listed first",
    ):
        assert prose not in script, prose
    for kept in (
        'id="pts1"',
        'id="pts2"',
        'id="shots-hint"',
        'id="presets-note"',
        'id="actions-note"',
        'id="mv-hint"',
        'id="devq-hint"',
        '<span class="hint">seconds</span>',
        "Loading presets…",
    ):
        assert kept in html, kept


def test_optimizer_availability_preserves_failures_and_exclusions():
    _need_node()
    body = _script_function(
        (_PKG / "static/scanner.js").read_text(), "renderOptimizerAvailability"
    )
    harness = (
        """
var els = {}, button = {};
function $(id) { return els[id] || (els[id] = {}); }
var document = { querySelector: () => button };
"""
        + f"function render(listing) {{\n{body}\n}}"
        + """
var result = [];
[
 {names: [], detail: "server unreachable"},
 {names: ["good"], unavailable: {broken: "vocs missing", legacy: "retired dialect"}},
 {names: [], unavailable: {legacy: "retired dialect"}},
 {names: ["good"], unavailable: {}}
].forEach(listing => {
 render(listing);
 result.push({disabled: button.disabled, text: $("optimizer-availability").textContent, hidden: $("optimizer-availability").hidden});
});
console.log(JSON.stringify(result));
"""
    )
    result = subprocess.run(
        ["node", "-"], input=harness, text=True, capture_output=True, check=True
    )
    error, partial, legacy, healthy = json.loads(result.stdout)
    assert error["disabled"] and "server unreachable" in error["text"]
    assert "No compatible" not in error["text"]
    assert not partial["disabled"] and "broken: vocs missing" in partial["text"]
    assert "legacy: retired dialect" in partial["text"]
    assert legacy["disabled"] and not legacy["hidden"]
    assert healthy["hidden"] and not healthy["disabled"]


def test_physical_best_targets_stay_visible_after_move_is_queued():
    _need_node()
    body = _script_function(
        (_PKG / "static/scanner.js").read_text(), "renderOptimization"
    )
    harness = (
        """
var els = {};
function el() { return {children: [], appendChild(child) {this.children.push(child);},
 set textContent(value) {this.text = value; this.children = [];}}; }
function $(id) { return els[id] || (els[id] = el()); }
function td(value) { return {text: value}; }
var document = {createElement: el};
var S = {status: {connected: true, re_state: "idle", items_in_queue: 0}, optimization: {
 run_uid: "run", config: "test", iteration: 1, max_iterations: 1, exit_status: "success", finished: true,
 measured: {bump: 0}, outputs: {}, best: {bump: 999}, valid_shots: {},
 best_moves: {"Motor1:Current": 0.123456789, "Motor2:Current": -0.5}
}};
"""
        + f"function renderOptimization() {{\n{body}\n}}"
        + """
renderOptimization();
var before = $("optimization-targets").children.map(row => row.children.map(cell => cell.text));
S.optimization.invalidated_reason = "best-settings move was queued";
renderOptimization();
console.log(JSON.stringify({before, after: $("optimization-targets").children.map(row => row.children.map(cell => cell.text)), disabled: $("btn-set-best").disabled}));
"""
    )
    result = subprocess.run(
        ["node", "-"], input=harness, text=True, capture_output=True, check=True
    )
    state = json.loads(result.stdout)
    assert (
        state["before"]
        == state["after"]
        == [["Motor1:Current", "0.123456789"], ["Motor2:Current", "-0.5"]]
    )
    assert state["disabled"]


def test_single_position_range_round_trips_through_compatibility_form():
    _need_node()
    script = (_PKG / "static/scanner.js").read_text()
    functions = "\n".join(
        "function "
        + name
        + "("
        + args
        + ") {\n"
        + _script_function(script, name)
        + "\n}"
        for name, args in [
            ("formShape", "plan"),
            ("fillFormFromPreset", "doc"),
            ("stepFor", "start,stop,num"),
            ("points", "a,b,s"),
        ]
    )
    harness = (
        """
var els = {}, S = {};
function $(id) {return els[id] || (els[id] = {value: "", appendChild() {}});}
function setSelect(id,v) {$(id).value = v;} function setMode() {} function setAcq() {}
function noDevicesNote() {} function recalc() {} function renderCalibration() {}
"""
        + functions
        + """
var result = [2,5].map(stop => {
 var axis = {kind: "range", axis: "M", start: 2, stop: stop, num: points(2,stop,10)};
 var plan = {name: "sweep", kwargs: {sweep: {trajectory: {kind: "axes", axes: [axis]}}}};
 fillFormFromPreset({plan: plan});
 return {shape: formShape(plan), formable: S.formable, start: $("start1").value, stop: $("stop1").value, step: $("step1").value, count: points(Number($("start1").value),Number($("stop1").value),Number($("step1").value))};
});
console.log(JSON.stringify(result));
"""
    )
    result = subprocess.run(
        ["node", "-"], input=harness, text=True, capture_output=True, check=True
    )
    assert (
        json.loads(result.stdout)
        == [
            {
                "shape": "scan",
                "formable": True,
                "start": 2,
                "stop": 2,
                "step": 1,
                "count": 1,
            }
        ]
        * 2
    )
