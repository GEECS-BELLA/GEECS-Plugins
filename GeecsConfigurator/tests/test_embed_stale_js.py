"""What the drawer shows answers the draft in it: a preview or a dry run of an earlier draft never stands for the edited one.

Under node, ``embed.js`` and the theme's ``form.js`` against this app's own
answers (``test_embed_js._answers``), with the host's preview and the dry
run held and released out of order: the later Preview's picture is the one
shown whatever order they resolve in; an edit after a preview marks it
stale; a dry run answered for an earlier draft asks again rather than
offering its Save, and its Save button never saves an edited draft.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme import static_dir
from geecs_web_theme.testing import node_available
from test_embed_js import API, Q, _answers

_EMBED = (
    Path(__file__).resolve().parents[1] / "geecs_configurator" / "static" / "embed.js"
)

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
const settle = () => new Promise((r) => setTimeout(r, 5));
global.GeecsForm = require(process.argv[1]);
const G = require(process.argv[2]);
const C = JSON.parse(require("fs").readFileSync(0, "utf8"));
const kinds = C.kinds.map((k) => k.kind === "setup" ? Object.assign({}, k, { dry_run: "a check" }) : k);
const sent = [], held = [];
global.fetch = (url, init) => {
  const method = (init && init.method) || "GET", body = init && init.body ? JSON.parse(init.body) : undefined;
  sent.push([method, url, body]);
  if (url.endsWith("/api/kinds")) return Promise.resolve({ status: 200, text: () => Promise.resolve(JSON.stringify(kinds)) });
  if (url.indexOf("/api/dry-run/") >= 0) return new Promise((resolve) => held.push({ body, resolve: () => resolve({ status: 200, text: () => Promise.resolve(JSON.stringify({ kind: "setup", by: "scanner", ran: true, detail: "fine for " + body.document.description })) }) }));
  const [status, out] = C.answers[method + " " + url] || [404, { detail: "unknown" }];
  return Promise.resolve({ status, text: () => Promise.resolve(JSON.stringify(out)) });
};
(async () => {
  const out = {}, pictures = [];
  const preview = (doc) => new Promise((resolve) => pictures.push({ doc, resolve }));
  G.open({ api: C.api, kind: "setup", experiment: "Bench", id: "low_power.yaml", fields: ["description"], operator: "Ada", preview });
  await settle();
  const drawer = document.body.children[0];
  const box = walk(drawer, (n) => n.attrs["data-path"] === "description")[0];
  const input = walk(box, (n) => ["INPUT", "TEXTAREA"].includes(n.tagName))[0];
  const edit = (text) => { input.value = text; input.on.input({ target: input }); };
  const shown = () => walk(drawer, (n) => n.tagName === "IMG").map((n) => n.attrs.alt);
  const stale = () => walk(drawer, (n) => n.attrs["data-stale"] !== undefined)[0] || { attrs: {} };
  // Preview A, then B; B resolves first, A last: B stays shown
  edit("A"); button(drawer, "Preview")[0].on.click();
  edit("B"); button(drawer, "Preview")[0].on.click();
  const img = (alt) => Object.assign(el("img"), { attrs: { alt } });
  pictures[1].resolve(img("B")); await settle();
  pictures[0].resolve(img("A")); await settle();
  out.outOfOrder = [pictures.map((p) => p.doc.description), shown()];
  // an edit after it marks the shown preview stale
  out.beforeEdit = stale().attrs["data-stale"];
  edit("C"); await settle();
  out.afterEdit = [stale().attrs["data-stale"], drawer.textContent.includes("earlier edit")];
  // a dry run asked for C, edited to D while it ran: it asks again for D, and offers no Save for C
  button(drawer, "Save")[0].on.click(); await settle();
  edit("D"); held[0].resolve(); await settle();
  out.reAsked = held.map((h) => h.body.document.description);
  if (held[1]) held[1].resolve(); await settle();
  // the answer for D is shown; edit to E, then press its Save: it asks again, nothing is PUT
  edit("E"); const saves = button(drawer, "Save"); if (saves.length > 1) saves.slice(-1)[0].on.click(); await settle();
  out.saveOfEarlier = [held.map((h) => h.body.document.description), sent.filter((s) => s[0] === "PUT").length];
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    from conftest import FILES

    from geecs_configurator.app import create_app
    from geecs_configurator.settings import Settings

    root = tmp_path_factory.mktemp("root")
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    with TestClient(create_app(Settings(configs_root=root, experiment="Bench"))) as client:  # fmt: skip
        stdin = {"api": API, "answers": _answers(client), "kinds": client.get("/api/kinds").json()}  # fmt: skip
    assert f"GET {API}/api/documents/setup/low_power.yaml{Q}" in stdin["answers"]
    done = subprocess.run(["node", "-e", _HARNESS, str(static_dir() / "form.js"), str(_EMBED)], input=json.dumps(stdin), text=True, capture_output=True, timeout=30)  # fmt: skip
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_later_preview_is_shown_whatever_order_they_resolve_in(ran: dict) -> None:
    assert ran["outOfOrder"] == [["A", "B"], ["B"]]


def test_an_edit_after_a_preview_marks_it_stale(ran: dict) -> None:
    assert ran["beforeEdit"] == "false" and ran["afterEdit"] == ["true", True]


def test_a_dry_run_of_an_earlier_draft_asks_again_and_never_saves_the_edit(ran: dict) -> None:  # fmt: skip
    assert ran["reAsked"] == ["C", "D"]
    assert ran["saveOfEarlier"] == [["C", "D", "E"], 0]
