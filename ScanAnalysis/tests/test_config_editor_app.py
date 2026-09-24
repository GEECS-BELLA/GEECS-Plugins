"""The config editor router over a tmp tree (fastapi TestClient; skipped without the editor extra)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from scan_analysis.config_editor import create_editor_router  # noqa: E402
from scan_analysis.config_store import ConfigStore  # noqa: E402


def _doc(name="UC_A"):
    return {
        "schema_version": 2,
        "name": name,
        "analyzer": {"kind": "beam"},
        "image": {"type": "camera", "bit_depth": 16},
        "scan": {"priority": 5},
    }


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "analyzers" / "HTU").mkdir(parents=True)
    (tmp_path / "analyzers" / "HTU" / "UC_A.yaml").write_text(yaml.safe_dump(_doc()))
    (tmp_path / "groups" / "HTU").mkdir(parents=True)
    (tmp_path / "groups" / "HTU" / "baseline.yaml").write_text(
        yaml.safe_dump({"name": "b", "analyzers": ["UC_A"]})
    )
    return tmp_path


def _host(tree: Path) -> TestClient:
    """The editor router on a bare app, the way the portal hosts it."""
    app = fastapi.FastAPI()
    app.include_router(create_editor_router(ConfigStore(tree)))
    return TestClient(app)


@pytest.fixture
def client(tree):
    return _host(tree)


class TestPagesAndStatic:
    def test_page_and_assets(self, client):
        assert client.get("/").status_code == 200
        assert "ConfigEditor.mount" in client.get("/").text
        js = client.get("/static/editor.js")
        assert js.status_code == 200
        assert (
            js.headers["cache-control"] == "no-cache"
        )  # ships with ScanAnalysis, not the host
        assert client.get("/static/editor.css").status_code == 200
        assert client.get("/static/other.js").status_code == 404

    def test_a_malformed_file_never_takes_the_editor_down(self, tree):
        """A tab-indented file is an invalid entry in the listing and an invalid document on read."""
        (tree / "analyzers" / "HTU" / "Tabbed.yaml").write_text("name: T\n\tx: 1\n")
        client = _host(tree)
        listing = client.get("/api/list")
        assert listing.status_code == 200
        bad = [a for a in listing.json()["analyzers"] if a["id"] == "Tabbed"]
        assert (
            bad and not bad[0]["valid"] and "cannot start any token" in bad[0]["error"]
        )
        r = client.get("/api/analyzers/Tabbed")
        assert r.status_code == 200
        body = r.json()
        assert not body["valid"] and body["document"] == {} and "\t" in body["yaml"]


class TestApi:
    def test_list_and_schema(self, client):
        body = client.get("/api/list").json()
        assert [a["id"] for a in body["analyzers"]] == ["UC_A"]
        assert body["groups"][0]["id"] == "baseline"
        assert body["namespaces"]["analyzer"] == ["HTU"]
        assert body["preview"] is False
        schema = client.get("/api/schema/analyzer").json()
        # the recipe's schema, its step vocabulary from the analysis core
        assert schema["properties"]["input"]["discriminator"]["propertyName"] == "kind"
        steps = schema["properties"]["steps"]["items"]["discriminator"]
        assert steps["propertyName"] == "step" and "median" in steps["mapping"]
        assert client.get("/api/schema/nope").status_code == 404

    def test_recipe_roundtrip_and_a_v2_file_still_reads(self, client):
        recipe = {
            "schema_version": 3,
            "device": "UC_New",
            "input": {"kind": "camera"},
            "steps": [{"step": "median", "kernel": 5}],
            "measure": {"kind": "beam"},
            "figure": {"imshow": {"cmap": "viridis"}},
            "summaries": [{"kind": "image_grid"}, {"kind": "average"}],
        }
        ok = client.post("/api/validate/analyzer", json={"document": recipe}).json()
        assert ok["ok"], ok["errors"]
        assert "step: median" in ok["yaml"]
        bad = dict(recipe, steps=[{"step": "median", "kernel": 4}])
        report = client.post("/api/validate/analyzer", json={"document": bad}).json()
        assert not report["ok"] and report["errors"][0]["loc"] == "steps.0.kernel"
        r = client.put(
            "/api/analyzers/HTU/UC_New", json={"document": recipe, "etag": None}
        )
        assert r.status_code == 201, r.text
        loaded = client.get("/api/analyzers/UC_New").json()
        assert loaded["valid"] and loaded["document"]["schema_version"] == 3
        listing = client.get("/api/list").json()["analyzers"]
        versions = {a["id"]: a["schema_version"] for a in listing}
        assert versions == {"UC_A": 2, "UC_New": 3}
        # the v2 file: still read and valid (the page shows it read-only)
        old = client.get("/api/analyzers/UC_A").json()
        assert old["valid"] and old["document"]["schema_version"] == 2

    def test_read_validate_save_roundtrip(self, client):
        loaded = client.get("/api/analyzers/UC_A").json()
        assert loaded["etag"] and loaded["valid"]
        doc = loaded["document"]
        doc["analyzer"] = {"kind": "beam", "compute_slopes": True}
        report = client.post("/api/validate/analyzer", json={"document": doc}).json()
        assert report["ok"] and "compute_slopes: true" in report["yaml"]
        saved = client.put(
            "/api/analyzers/HTU/UC_A", json={"document": doc, "etag": loaded["etag"]}
        )
        assert saved.status_code == 200
        assert saved.json()["etag"] != loaded["etag"]
        stale = client.put(
            "/api/analyzers/HTU/UC_A", json={"document": doc, "etag": loaded["etag"]}
        )
        assert stale.status_code == 409

    def test_invalid_document_is_422_with_locations(self, client):
        loaded = client.get("/api/analyzers/UC_A").json()
        bad = dict(loaded["document"], analyzer={"kind": "beam", "compute_slope": True})
        r = client.put(
            "/api/analyzers/HTU/UC_A", json={"document": bad, "etag": loaded["etag"]}
        )
        assert r.status_code == 422
        assert any("compute_slope" in e["loc"] for e in r.json()["errors"])

    def test_create_and_delete(self, client):
        r = client.put(
            "/api/analyzers/PW/PW_New", json={"document": _doc("PW_New"), "etag": None}
        )
        assert r.status_code == 201 and r.json()["created"]
        etag = r.json()["etag"]
        assert (
            client.delete("/api/analyzers/PW_New", params={"etag": "0-0"}).status_code
            == 409
        )
        assert (
            client.delete("/api/analyzers/PW_New", params={"etag": etag}).status_code
            == 204
        )
        assert client.get("/api/analyzers/PW_New").status_code == 404

    def test_group_ref_cross_check(self, client):
        r = client.post(
            "/api/validate/group",
            json={"document": {"name": "g", "analyzers": ["Nope"]}},
        )
        assert r.json()["ok"] is False

    def test_preview_404_without_host(self, client):
        r = client.post("/api/preview", json={"document": _doc(), "params": {}})
        assert r.status_code == 404


class TestHostedRouter:
    def test_read_only_refuses_writes(self, tree):
        app = fastapi.FastAPI()
        app.include_router(
            create_editor_router(ConfigStore(tree), read_only=True), prefix="/configs"
        )
        c = TestClient(app)
        assert c.get("/configs/api/list").json()["read_only"] is True
        r = c.put(
            "/configs/api/analyzers/HTU/UC_A", json={"document": _doc(), "etag": None}
        )
        assert r.status_code == 405

    def test_preview_ladder(self, tree):
        calls = []

        def preview(document, params):
            calls.append((document["name"], dict(params)))
            if params.get("shot") == 99:
                raise LookupError("no such shot")
            if params.get("device") == "bad":
                raise ValueError("analyzer refused")
            return b"\x89PNG..."

        app = fastapi.FastAPI()
        app.include_router(
            create_editor_router(ConfigStore(tree), preview=preview), prefix="/configs"
        )
        c = TestClient(app)
        ok = c.post(
            "/configs/api/preview",
            json={"document": _doc(), "params": {"shot": 1, "device": "cam"}},
        )
        assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
        assert calls[-1][0] == "UC_A"
        assert (
            c.post(
                "/configs/api/preview",
                json={"document": _doc(), "params": {"shot": 99}},
            ).status_code
            == 404
        )
        assert (
            c.post(
                "/configs/api/preview",
                json={"document": _doc(), "params": {"device": "bad"}},
            ).status_code
            == 400
        )
        invalid = c.post(
            "/configs/api/preview", json={"document": {"name": "x"}, "params": {}}
        )
        assert invalid.status_code == 422


class TestSidebarCollapseState:
    """The tree's collapse rules, run as the browser runs them.

    The sidebar is page JavaScript, so these extract the real functions out
    of ``editor.js`` and execute them under node — the technique
    ``GeecsScanner/tests/test_page.py`` uses for the console's own logic.
    """

    @staticmethod
    def _run(expression: str, stored: str | None = None) -> object:
        """Evaluate *expression* with the tree's state helpers in scope.

        *stored* seeds what ``localStorage`` already holds for the sidebar.
        """
        from geecs_web_theme.testing import node_available

        if not node_available():  # pragma: no cover - CI and dev machines have it
            pytest.skip("node not available to run JavaScript")
        source = (
            Path(__file__).resolve().parents[1]
            / "scan_analysis/config_editor/static/editor.js"
        ).read_text()
        key = re.search(r'\n    const OPEN_KEY = "([^"]+)";', source)
        assert key, "editor.js: no OPEN_KEY"
        bodies = [f'var OPEN_KEY = "{key.group(1)}";']
        for name in ("openState", "rememberOpen", "wantOpen"):
            m = re.search(
                rf"\n    function {name}\([^)]*\) \{{\n(.*?)\n    \}}\n", source, re.S
            )
            assert m, f"editor.js: no function {name}()"
            args = re.search(rf"function {name}\(([^)]*)\)", source).group(1)
            bodies.append(f"function {name}({args}) {{\n{m.group(1)}\n}}")
        harness = (
            "var __store = "
            + json.dumps({} if stored is None else {key.group(1): stored})
            + ";\n"
            "var window = {localStorage: {\n"
            "  getItem: function (k) { return k in __store ? __store[k] : null; },\n"
            "  setItem: function (k, v) { __store[k] = v; },\n"
            "}};\n"
            + "\n".join(bodies)
            + f"\nconsole.log(JSON.stringify({expression}));"
        )
        result = subprocess.run(
            ["node", "-"], input=harness, text=True, capture_output=True, check=True
        )
        return json.loads(result.stdout)

    def test_an_untouched_node_follows_the_open_document(self) -> None:
        """Nothing said about it: the node holding what is on screen expands, the rest stay shut."""
        assert self._run(
            "[wantOpen({}, 'analyzer', true), wantOpen({}, 'group', false)]"
        ) == [
            True,
            False,
        ]

    def test_closing_the_node_that_holds_the_document_sticks(self) -> None:
        """The bug a set of open keys cannot express: Save re-renders, and the collapse must survive it."""
        assert (
            self._run(
                "(rememberOpen('analyzer', false), wantOpen(openState(), 'analyzer', true))"
            )
            is False
        )

    def test_opening_a_node_outlives_the_render(self) -> None:
        """An expansion the user asked for is remembered even when nothing on screen needs it."""
        assert (
            self._run(
                "(rememberOpen('analyzer/HTU', true), wantOpen(openState(), 'analyzer/HTU', false))"
            )
            is True
        )

    def test_unreadable_storage_is_not_a_broken_sidebar(self) -> None:
        """A private window, blocked site data, or a stale format: the tree opens fresh, never throws.

        Each case reaches a different branch — empty storage, the JSON that
        does not parse, and the array the previous format wrote, which has to
        be discarded rather than indexed into.
        """
        assert self._run("openState()") == {}
        assert self._run("openState()", stored="not json at all") == {}
        assert self._run("openState()", stored='["analyzer", "group/HTU"]') == {}
        assert self._run("openState()", stored="null") == {}
        # and the stale array does not survive the next write
        assert self._run(
            "(rememberOpen('group', true), openState())",
            stored='["analyzer", "group/HTU"]',
        ) == {"group": True}


def test_retired_upload_fields_are_hidden_and_preserved():
    """Run the actual object form: no upload control, no loss on save.

    The group form only: a v2 diagnostic (the other document with a retired
    upload field) is shown read-only, never rendered as a form.
    """
    from geecs_web_theme.testing import node_available

    if not node_available():
        pytest.skip("node not available to run JavaScript")
    kind, value = "group", False
    schema = ConfigStore.schema(kind)
    key = "upload_to_scanlog"
    field = schema["properties"][key]
    assert field["deprecated"] is True
    source = (
        Path(__file__).resolve().parents[1]
        / "scan_analysis/config_editor/static/editor.js"
    ).read_text()
    start = source.index("    object(n, value, path, optional, opts) {")
    end = source.index("    renderOptionalSection(", start)
    method = source[start:end]
    harness = (
        """
const el = () => ({append() {}});
const form = {
  schema: {
    unwrapOptional: n => ({inner: n}),
    kindOf: () => 'number',
  },
  render() { throw Error('Retired field was rendered'); },
"""
        + method
        + "};\n"
    )
    harness += "const schema = " + json.dumps({"properties": {key: field}}) + ";\n"
    harness += "const doc = " + json.dumps({key: value}) + ";\n"
    harness += "console.log(JSON.stringify([form.object(schema, doc, [], false).get(), form.object(schema, {}, [], false).get()]));"
    result = subprocess.run(
        ["node", "-"], input=harness, text=True, capture_output=True, check=True
    )
    assert json.loads(result.stdout) == [{key: value}, {}]


_FAKE_DOM = r"""
class Node {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.attrs = {}; this.listeners = {}; this.parent = null; this._class = ""; this.value = ""; this.checked = false; this.disabled = false; }
  get textContent() { return this._text !== undefined ? this._text : this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get className() { return this._class; } set className(v) { this._class = v; }
  get classList() { const self = this; return {
    contains(c) { return self._class.split(/\s+/).includes(c); },
    add(c) { if (!this.contains(c)) self._class = (self._class + " " + c).trim(); },
    remove(c) { self._class = self._class.split(/\s+/).filter((x) => x && x !== c).join(" "); },
    toggle(c, on) { if (on === undefined) on = !this.contains(c); on ? this.add(c) : this.remove(c); } }; }
  set innerHTML(v) { this.children = []; this._html = v; } get innerHTML() { return this._html || ""; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "value") this.value = String(v); if (k === "disabled") this.disabled = true; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t) { for (const f of this.listeners[t] || []) f({ target: this, preventDefault() {} }); }
  _adopt(c) { if (typeof c === "string") { const tn = new Node("#text"); tn.textContent = c; c = tn; } if (c.parent) c.parent.children = c.parent.children.filter((x) => x !== c); c.parent = this; return c; }
  append(...cs) { for (const c of cs) this.children.push(this._adopt(c)); }
  prepend(...cs) { this.children.unshift(...cs.map((c) => this._adopt(c))); }
  insertBefore(c, ref) { c = this._adopt(c); const i = this.children.indexOf(ref); if (i < 0) this.children.push(c); else this.children.splice(i, 0, c); }
  remove() { if (this.parent) { this.parent.children = this.parent.children.filter((x) => x !== this); this.parent = null; } }
  focus() {}
  *walk() { for (const c of this.children) { yield c; yield* c.walk(); } }
  matches(sel) {
    const m = sel.match(/^([a-z]*)((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/i); if (!m) return false;
    if (m[1] && this.tagName !== m[1].toUpperCase()) return false;
    for (const c of (m[2].match(/\.[\w-]+/g) || [])) if (!this.classList.contains(c.slice(1))) return false;
    for (const a of (m[3].match(/\[[^\]]+\]/g) || [])) { const am = a.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/); if (!am) return false; const v = this.getAttribute(am[1]); if (v === null) return false; if (am[2] !== undefined && v !== am[2]) return false; }
    return true; }
  querySelectorAll(sel) { const parts = sel.split(",").map((s) => s.trim()); const out = []; for (const n of this.walk()) if (parts.some((p) => n.matches(p))) out.push(n); return out; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}
const document = { createElement: (t) => new Node(t), getElementById: () => null };
const window = { document };
"""

_BEAM_RECIPE = {
    "schema_version": 3,
    "device": "UC_TopView",
    "description": "IR mode at the input to amp3",
    # "532" and "true" below are STRINGS that would parse as JSON: the rows must
    # show them quoted and read them back as strings (review #996 finding 1)
    "metadata": {
        "location": "Room 148",
        "spatial_calibration": 2.44e-05,
        "notes": "532",
    },
    "input": {"kind": "camera"},
    "inputs": {"bg": {"path": "{scan_dir}/bg.png", "fallback_level": 3}},
    "steps": [
        {"step": "background_frame", "source": "bg"},
        {"step": "roi", "bounds": [[350, 600], [10, 750]]},
        {"step": "median", "kernel": 5},
        {"step": "circular_mask", "center": [200, 300], "radius": 50, "units": "axis"},
    ],
    "measure": {"kind": "beam", "compute_slopes": True},
    "scan": {"priority": 10, "average_frames_first": True},
    "figure": {
        "imshow": {"cmap": "viridis", "vmin": 0},
        "fig": {"dpi": 150},
        "axes": {"title": "top view"},
        "overlays": {
            # "0.5" is a legal matplotlib grey — as a STRING; retyped to 0.5 it breaks the draw
            "com": {"color": "0.5", "hidden": False},
            "projection_x": {"scale": 0.2, "label": "true"},
        },
    },
    "summaries": [
        {"kind": "image_grid", "panel_size": [6.0, 6.0], "columns": 4},
        {"kind": "average"},
    ],
}

_LINE_RECIPE = {
    "schema_version": 3,
    "device": "U_BCaveMagSpec",
    "output_name": "U_BCaveMagSpec-interpSpec",
    "input": {
        "kind": "line",
        "folder": "U_BCaveMagSpec-interpSpec",
        "file_tail": ".txt",
        "format": "device_hdf5",
        "loading": {"data_type": "pva_stack"},
        "x_scale": 1000.0,
        "x_unit": "MeV",
        "label": "Charge density vs Energy",
    },
    "steps": [
        {"step": "background_constant", "level": 0},
        {"step": "roi", "bounds": [[60, 160]], "units": "axis"},
        {"step": "interpolate", "count": 400, "lower": 70},
        {"step": "clip_below", "level": -10},
    ],
    "measure": {"kind": "line"},
    "summaries": [
        {"kind": "waterfall", "sort_key": "U_S1:charge", "scale": "sequential"},
        {"kind": "average"},
    ],
}


def test_recipe_form_round_trips_and_reorders(tree):
    """Run the real recipe form under node on a fake DOM.

    What the form reads back is what was loaded (a corpus beam recipe with a
    frame input and overlay styles; a line recipe), a step the registry does
    not know is kept as written (never swapped for the first kind), and the
    card's "move up" swaps the steps while every field path stays true.
    """
    from geecs_web_theme.testing import node_available

    if not node_available():
        pytest.skip("node not available to run JavaScript")
    source = (
        Path(__file__).resolve().parents[1]
        / "scan_analysis/config_editor/static/editor.js"
    ).read_text()
    utils = source[
        source.index("  const esc = ") : source.index("  async function api(")
    ]
    classes = source[
        source.index("  class Schema {") : source.index(
            "  // -------------------------------------------------------------- editor"
        )
    ]
    sections = re.search(
        r"const RECIPE_SECTIONS = \(\) => (\[.*?\n    \]);", source, re.S
    )
    assert sections, "editor.js: no RECIPE_SECTIONS"
    docs = {
        "beam": _BEAM_RECIPE,
        "line": _LINE_RECIPE,
        "typo": dict(_BEAM_RECIPE, inputs={}, steps=[{"step": "medain", "kernel": 3}]),
    }
    harness = (
        _FAKE_DOM
        + utils
        + classes
        + f"const SCHEMA = {json.dumps(ConfigStore.schema('analyzer'))};\n"
        + f"const DOCS = {json.dumps(docs)};\n"
        + f"const SECTIONS = () => {sections.group(1)};\n"
        + """
const schema = new Schema(SCHEMA);
const form = new Form(schema, () => {});
const out = {};
for (const [name, doc] of Object.entries(DOCS)) {
  const r = form.object(schema.resolve(schema.root), doc, [], false, { sections: SECTIONS(), hidden: new Set(["schema_version"]) });
  const paths = r.node.querySelectorAll(".field").map((f) => f.getAttribute("data-path"));
  const ups = r.node.querySelectorAll('button[title="move up"]');
  out[name] = { roundtrip: r.get(), paths, adder: r.node.querySelectorAll("select.add").map((s) => s.children.map((o) => o.textContent)) };
  if (ups.length > 1) { ups[1].fire("click"); out[name].after_up = r.get().steps; out[name].paths_after = r.node.querySelectorAll(".field").map((f) => f.getAttribute("data-path")); }
}
console.log(JSON.stringify(out));
"""
    )
    result = subprocess.run(
        ["node", "-"], input=harness, text=True, capture_output=True, check=True
    )
    out = json.loads(result.stdout)
    store = ConfigStore(tree)
    for name in ("beam", "line"):
        expected = store.validate("analyzer", docs[name])
        assert expected.ok, expected.errors
        got = store.validate("analyzer", out[name]["roundtrip"])
        assert got.ok, got.errors
        assert got.canonical == expected.canonical, name
        # every step's fields are addressable by the server's error locations
        assert "steps.1.units" in out[name]["paths"]  # the roi step's field
        assert "steps.1.bounds.0" in out[name]["paths"]  # its first bounds pair
    # the corpus beam recipe's frame input and overlay rows came back too —
    # the JSON-looking strings as strings, the typed values typed
    assert out["beam"]["roundtrip"]["inputs"] == _BEAM_RECIPE["inputs"]
    assert out["beam"]["roundtrip"]["figure"]["overlays"]["com"]["color"] == "0.5"
    assert out["beam"]["roundtrip"]["metadata"]["notes"] == "532"
    assert out["beam"]["roundtrip"]["figure"]["overlays"]["projection_x"] == {
        "scale": 0.2,
        "label": "true",
    }
    assert (
        out["beam"]["roundtrip"]["figure"]["overlays"]
        == _BEAM_RECIPE["figure"]["overlays"]
    )
    # the step list offers the registry, with the frame-shape hints
    steps_adder = out["beam"]["adder"][0]
    assert "roi" in steps_adder and "interpolate (traces)" in steps_adder
    assert "circular_mask (images)" in steps_adder
    # move up on the second card swaps the first two steps; paths renumber
    assert out["beam"]["after_up"][:2] == [
        _BEAM_RECIPE["steps"][1],
        _BEAM_RECIPE["steps"][0],
    ]
    assert "steps.0.units" in out["beam"]["paths_after"]
    assert "steps.1.units" not in out["beam"]["paths_after"]
    assert "steps.1.source" in out["beam"]["paths_after"]
    # an unknown step name is kept as written, for the server to refuse by location
    assert out["typo"]["roundtrip"]["steps"] == [{"step": "medain", "kernel": 3}]
    assert not store.validate("analyzer", out["typo"]["roundtrip"]).ok
