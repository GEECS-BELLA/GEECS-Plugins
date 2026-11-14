"""A late answer touches the drawer only while the document that asked is still the one in it.

The drawer is shared: an operator may start a delete, a copy or a save, close
the drawer, open another document and edit it while the answer is on its
way. Under node, ``embed.js`` and the theme's ``form.js`` against this app's
own answers (``test_embed_js._answers``), with each answer held until the
drawer has moved to the other document: the other document's form and its
unsaved edit stay; the host still hears of what was saved or deleted.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme import static_dir
from geecs_web_theme.testing import node_available
from test_embed_js import API, _answers

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
const sent = [], held = [];
const reply = (status, out) => ({ status, text: () => Promise.resolve(JSON.stringify(out)) });
global.fetch = (url, init) => {
  const method = (init && init.method) || "GET", body = init && init.body ? JSON.parse(init.body) : undefined;
  sent.push([method, url, body]);
  if (url.endsWith("/api/kinds")) return Promise.resolve(reply(200, C.kinds));
  if (url.indexOf("/api/delete-check/") >= 0) return Promise.resolve(reply(200, { path: "p/low_power.yaml", deletable: true, referrers: [], reason: "" }));
  if (method === "DELETE") return new Promise((resolve) => held.push(() => resolve(reply(200, { outcome: "committed", path: "p/low_power.yaml", author: "Ada via configurator", commit: "abc1234567", deleted: true }))));
  if (method === "PUT") return new Promise((resolve) => held.push(() => resolve(reply(200, { outcome: "committed", author: "Ada via configurator", etag: "e9", tier: "operator", changes: [] }))));
  const [status, out] = C.answers[method + " " + url] || [404, { detail: "unknown" }];
  return Promise.resolve(reply(status, out));
};
(async () => {
  const out = {}, heard = [];
  const drawer = () => document.body.children[0];
  const title = () => walk(drawer(), (n) => n.attrs.id === "geecs-configurator-title")[0].textContent;
  const field = () => { const box = walk(drawer(), (n) => n.attrs["data-path"] === "description")[0]; return box && walk(box, (n) => ["INPUT", "TEXTAREA"].includes(n.tagName))[0]; };
  const edit = (text) => { const input = field(); if (input) { input.value = text; input.on.input({ target: input }); } };
  const state = () => [title(), field() ? field().value : null, heard.slice()];
  // each case runs on its own: one that breaks the drawer is its own failure, not the next case's
  const fresh = async () => { held.length = 0; heard.length = 0; G.open(opts("dated.yaml")); await settle(); };
  const opts = (id) => ({ api: C.api, kind: "setup", experiment: "Bench", id, fields: ["description"], operator: "Ada", onSaved: (b) => heard.push(b.outcome) });
  const release = async () => { if (held.length) held.shift()(); await settle(); };
  // a delete of A confirmed, then B opened and edited before A's answer comes
  G.open(opts("low_power.yaml")); await settle();
  button(drawer(), "Delete…")[0].on.click(); await settle();
  const dlg = walk(document.body, (n) => n.tagName === "DIALOG").slice(-1)[0];
  const typed = walk(dlg, (n) => n.tagName === "INPUT")[0]; typed.value = "low_power"; typed.on.input();
  button(dlg, "Delete low_power")[0].on.click(); await settle();
  G.open(opts("far.yaml")); await settle(); edit("B, unsaved");
  await release();
  out.afterDelete = state();
  // a copy of B sent, then A opened and edited before the copy's answer comes
  await fresh(); G.open(opts("far.yaml")); await settle();
  walk(drawer(), (n) => n.tagName === "INPUT" && n.attrs["aria-label"] === "the copy's name")[0].value = "far_copy";
  button(drawer(), "Save as a copy")[0].on.click(); await settle();
  G.open(opts("low_power.yaml")); await settle(); edit("A, unsaved");
  await release();
  out.afterCopy = state().concat([sent.filter((s) => s[0] === "GET" && s[1].indexOf("far_copy") >= 0).length]);
  // a save of A sent, then B opened and edited before the save's answer comes
  await fresh(); G.open(opts("low_power.yaml")); await settle(); edit("A, to save");
  button(drawer(), "Save")[0].on.click(); await settle();
  const ok = button(drawer(), "Save without it").concat(button(drawer(), "Save").slice(1)); if (ok.length) { ok[0].on.click(); await settle(); }
  const puts = held.length;
  G.open(opts("far.yaml")); await settle(); edit("B again");
  await release();
  out.afterSave = state().concat([puts]);
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
    done = subprocess.run(["node", "-e", _HARNESS, str(static_dir() / "form.js"), str(_EMBED)], input=json.dumps(stdin), text=True, capture_output=True, timeout=30)  # fmt: skip
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_late_delete_leaves_the_document_opened_since_and_its_edit(ran: dict) -> None:
    assert ran["afterDelete"] == ["setup · far.yaml", "B, unsaved", ["committed"]]


def test_a_late_copy_opens_nothing_over_the_document_opened_since(ran: dict) -> None:
    assert ran["afterCopy"] == ["setup · low_power.yaml", "A, unsaved", ["committed"], 0]  # fmt: skip


def test_a_late_save_leaves_the_document_opened_since(ran: dict) -> None:
    """No fix needed (the save's answer reaches only its own form and status); pinned with the others."""
    assert ran["afterSave"] == ["setup · far.yaml", "B again", ["committed"], 1]  # fmt: skip
