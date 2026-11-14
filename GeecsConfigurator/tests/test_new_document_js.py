"""A new recipe in the browser: ``embed.js``'s ``create`` and its draft, under node, against this app with the portal's.

The answers are the real ones (this app over a commit path, the portal's
app beside it, the drafts its blank route serves), keyed by URL for a stub
``fetch``. What is pinned: the start form offers this box's folders and the
schema's frame kinds, asks a trace's file format (read off the schema), opens
the portal's draft as a new document that says nothing is written until Save,
and Save is the save route's create at ``<folder>/<name>.yaml`` with the
draft as served. That PUT is then sent to the app itself, which commits it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import git
from fastapi.testclient import TestClient
from geecs_web_theme import static_dir
from geecs_web_theme.testing import node_available
from test_contract_portal import PORTAL, UC_CROP, RECIPE, _portal

from geecs_configurator.app import create_app
from geecs_configurator.settings import Settings

_EMBED = (
    Path(__file__).resolve().parents[1] / "geecs_configurator" / "static" / "embed.js"
)
API = "http://cfg.test"
LINE = {"loading": {"data_type": "tsv"}}

_HARNESS = r"""
function el(tag) {
  const n = { tagName: tag.toUpperCase(), children: [], attrs: {}, on: {}, value: "", parentNode: null, _t: "", disabled: false,
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === "value") this.value = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    appendChild(c) { c.parentNode = this; this.children.push(c); if (this.tagName === "SELECT" && c.attrs.selected !== undefined) this.value = c.attrs.value; return c; },
    insertBefore(c, ref) { c.parentNode = this; const i = this.children.indexOf(ref); this.children.splice(i < 0 ? this.children.length : i, 0, c); return c; },
    removeChild(c) { this.children = this.children.filter((x) => x !== c); },
    addEventListener(k, fn) { this.on[k] = fn; },
    get firstChild() { return this.children[0] || null; },
    get textContent() { return this._t + this.children.map((c) => c.textContent).join(""); },
    set textContent(v) { this._t = String(v); this.children = []; } };
  return n;
}
global.document = { createElement: el, createTextNode: (t) => Object.assign(el("#text"), { _t: t }), body: el("body") };
const walk = (n, f, out = []) => { if (f(n)) out.push(n); n.children.forEach((c) => walk(c, f, out)); return out; };
const button = (scope, text) => walk(scope, (n) => n.tagName === "BUTTON" && n.textContent === text);
const byLabel = (scope, label) => walk(scope, (n) => n.attrs["aria-label"] === label)[0];
const settle = () => new Promise((r) => setTimeout(r, 5));
global.GeecsForm = require(process.argv[1]);
const G = require(process.argv[2]);
const C = JSON.parse(require("fs").readFileSync(0, "utf8"));
const sent = [], held = [];
let hold = null;  // (method, url) -> true: the answer waits in `held` until released
global.fetch = (url, init) => {
  const method = (init && init.method) || "GET", body = init && init.body ? JSON.parse(init.body) : undefined;
  sent.push([method, decodeURIComponent(url), body]);
  if (hold && hold(method, decodeURIComponent(url))) { const go = global.fetch.bind(null, url, init); return new Promise((resolve) => held.push(() => { hold = null; resolve(go()); })); }
  if (method === "PUT") return Promise.resolve({ status: 200, text: () => Promise.resolve(JSON.stringify({ outcome: "committed", author: "Ada via configurator", etag: "e1", tier: "operator", changes: [], path: "x" })) });
  const [status, out] = C.answers[method + " " + decodeURIComponent(url)] || [404, { detail: "unknown " + url }];
  return Promise.resolve({ status, text: () => Promise.resolve(JSON.stringify(out)) });
};
(async () => {
  const out = {}, heard = [];
  out.choices = G.startChoices(C.schema).map((c) => [c.kind, c.fields.map((f) => f.label)]);
  G.create({ api: C.api, kind: "analysis_recipe", experiment: "", operator: "Ada", onSaved: (b, id) => heard.push([b.outcome, id]) });
  await settle();
  const drawer = document.body.children[0];
  out.folders = walk(byLabel(drawer, "the folder"), (n) => n.tagName === "OPTION").map((n) => n.attrs.value);
  byLabel(drawer, "the new document's name").value = "-bad";
  button(drawer, "Open the draft")[0].on.click(); await settle();
  out.badName = drawer.textContent.includes("letters, digits");
  byLabel(drawer, "the new document's name").value = "UC_New";
  const frames = byLabel(drawer, "the frame kind"); frames.value = "line"; frames.on.change();
  out.asksFormat = !!byLabel(drawer, "loading.data_type");
  byLabel(drawer, "loading.data_type").value = "tsv";
  button(drawer, "Open the draft")[0].on.click(); await settle();
  out.title = walk(drawer, (n) => n.attrs.id === "geecs-configurator-title")[0].textContent;
  out.says = drawer.textContent.includes("nothing is written until Save");
  out.noCopyOrDelete = button(drawer, "Delete…").length === 0;
  out.readsNothingAtTheId = sent.filter((s) => s[0] === "GET" && s[1].indexOf("/api/documents/analysis_recipe/Ns/UC_New.yaml") >= 0).length;
  button(drawer, "Save")[0].on.click(); await settle();
  const confirm = button(drawer, "Save"); if (confirm.length > 1) confirm.slice(-1)[0].on.click(); await settle();
  out.put = sent.filter((s) => s[0] === "PUT").map((s) => [s[1], s[2]]);
  out.heard = heard;
  out.reopened = sent.filter((s) => s[0] === "GET" && s[1].indexOf("/api/documents/analysis_recipe/Ns/UC_New.yaml") >= 0).length;
  // late answers: the drawer moved to another document before the blank, or the create, answered
  const title = () => walk(document.body.children[0], (n) => n.attrs.id === "geecs-configurator-title")[0].textContent;
  const startDraft = async () => {
    G.create({ api: C.api, kind: "analysis_recipe", experiment: "", operator: "Ada", onSaved: (b, id) => heard.push([b.outcome, id]) }); await settle();
    const d = document.body.children[0];
    byLabel(d, "the new document's name").value = "UC_New";
    const f = byLabel(d, "the frame kind"); f.value = "line"; f.on.change(); byLabel(d, "loading.data_type").value = "tsv";
    button(d, "Open the draft")[0].on.click(); await settle();
  };
  const other = { api: C.api, kind: "analysis_recipe", experiment: "", id: "Ns/Elsewhere.yaml", operator: "Ada" };
  // the start form's own answer (folders, schema) after another document was opened
  hold = (m, u) => m === "GET" && u.indexOf("/api/folders/") >= 0;
  G.create({ api: C.api, kind: "analysis_recipe", experiment: "", operator: "Ada" }); await settle();
  G.open(Object.assign({}, other, { id: "Ns/Elsewhere0.yaml" })); await settle(); held.shift()(); await settle();
  out.lateStart = [title(), button(document.body.children[0], "Open the draft").length];
  hold = (m, u) => m === "GET" && u.indexOf("/api/blank/") >= 0;
  await startDraft();
  G.open(other); await settle(); held.shift()(); await settle();
  out.lateBlank = title();
  heard.length = 0;
  await startDraft();
  hold = (m) => m === "PUT";
  button(document.body.children[0], "Save")[0].on.click(); await settle();
  const again = button(document.body.children[0], "Save"); if (again.length > 1) again.slice(-1)[0].on.click(); await settle();
  G.open(Object.assign({}, other, { id: "Ns/Elsewhere2.yaml" })); await settle(); held.shift()(); await settle();
  out.lateCreate = [title(), heard.slice()];
  // the configurator's own page passes its root, "" when served at /: a same-origin api
  out.sameOrigin = (() => { try { G.create({ api: "", kind: "analysis_recipe" }); return true; } catch (e) { return String(e); } })();
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture
def root(root: Path) -> Path:
    (root / RECIPE).write_text(UC_CROP, encoding="utf-8")
    return root


@pytest.fixture
def committing(origin: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Any:  # fmt: skip
    monkeypatch.setattr("geecs_configurator.app.FRESH_S", -1.0)
    state = tmp_path_factory.mktemp("service")
    env = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Bench", "GEECS_CONFIGURATOR_ANALYSIS_FOLDERS": "Ns", "GEECS_CONFIGURATOR_STATE_DIR": str(state), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_EXPERIMENT": "Bench", "GEECS_CONFIGURATOR_PORTAL_URL": PORTAL}  # fmt: skip
    with _portal(state / "live" / "scan_analysis_configs") as portal:
        with TestClient(create_app(Settings.from_env(env), portal_http=portal)) as client:  # fmt: skip
            yield client


def test_a_new_recipe_from_the_start_form_is_the_save_routes_create(committing: TestClient, origin: Path) -> None:  # fmt: skip
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    blank = f"/api/blank/analysis_recipe/Ns/UC_New.yaml?input=line&device=&fields={json.dumps(LINE, separators=(',', ':'))}"  # fmt: skip
    answers = {}
    for path in ("/api/folders/analysis_recipe", "/api/schema/analysis_recipe", "/api/kinds", blank):  # fmt: skip
        r = committing.get(path)
        assert r.status_code == 200, (path, r.text)
        answers[f"GET {API}{path}"] = [200, r.json()]
    draft = answers[f"GET {API}{blank}"][1]["document"]
    dry = "/api/dry-run/analysis_recipe?experiment=&id=Ns/UC_New.yaml"
    answers[f"POST {API}{dry}"] = [200, committing.post(dry, json={"document": draft}).json()]  # fmt: skip
    stdin = {"api": API, "answers": answers, "schema": answers[f"GET {API}/api/schema/analysis_recipe"][1]}  # fmt: skip
    done = subprocess.run(["node", "-e", _HARNESS, str(static_dir() / "form.js"), str(_EMBED)], input=json.dumps(stdin), text=True, capture_output=True, timeout=30)  # fmt: skip
    assert done.returncode == 0, done.stderr
    ran = json.loads(done.stdout)
    # the required format of a trace, then each kind's optional enumerated fields (a stack's `format`)
    assert dict(ran["choices"]) == {
        "camera": ["format"],
        "line": ["loading.data_type", "format", "storage_dtype"],
    }
    # a late answer never replaces the document opened since: the blank's, and the create's (the host still hears it)
    assert ran["lateStart"] == [
        "analysis_recipe · Ns/Elsewhere0.yaml",
        0,
    ]  # no start form drawn over it
    assert ran["lateBlank"] == "analysis_recipe · Ns/Elsewhere.yaml"
    assert ran["lateCreate"] == [
        "analysis_recipe · Ns/Elsewhere2.yaml",
        [["committed", "Ns/UC_New.yaml"]],
    ]
    assert (
        ran["folders"] == ["Ns"]
        and ran["badName"]
        and ran["asksFormat"]
        and ran["sameOrigin"] is True
    )
    assert (ran["title"], ran["says"], ran["noCopyOrDelete"], ran["readsNothingAtTheId"]) == ("analysis_recipe · Ns/UC_New.yaml (new)", True, True, 0)  # fmt: skip
    [(url, body)] = ran["put"]
    assert url == f"{API}/api/documents/analysis_recipe/Ns/UC_New.yaml?experiment="
    assert (body["create"], body["etag"], body["operator"], body["document"]) == (True, "", "Ada", draft)  # fmt: skip
    assert ran["heard"] == [["committed", "Ns/UC_New.yaml"]] and ran["reopened"] == 1  # the file opens as stored  # fmt: skip
    # what the browser sent, sent to the app: the create commits
    r = committing.put("/api/documents/analysis_recipe/Ns/UC_New.yaml", json=body)
    assert r.status_code == 200 and r.json()["outcome"] == "committed", r.text
    assert "scan_analysis_configs/analyzers/Ns/UC_New.yaml" in git("ls-tree", "-r", "--name-only", "main", cwd=origin)  # fmt: skip
