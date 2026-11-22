"""A write's answer never takes an edit made after the write was sent: the form keeps it, said to be unsaved.

The drawer's form stays editable while a save, a copy or a delete is on its
way. Under node, ``embed.js`` and the theme's ``form.js``
against this app's own answers (``test_embed_js._answers``), each write's
answer is held while the operator edits the form, then released: the edit is
still in the form, the drawer says it is unsaved, and the host is told
(``onSaved``'s ``unsaved``), so it does not navigate away. A create's answer
is the same rule's last case (``test_new_document_js.py``).
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
  // each case on its own: one that breaks the drawer is its own failure (the next case reopens the document)
  const run = async (name, fn) => { try { await fn(); } catch (e) { out[name] = "error: " + e.message; } };
  const reopen = async () => { held.length = 0; G.open({ api: C.api, kind: "setup", experiment: "Bench", id: "far.yaml", operator: "Ada" }); await settle(); G.open(opts()); await settle(); };
  const release = async () => { held.shift()(); await settle(); };
  const unsavedSaid = () => drawer().textContent.includes("still in");
  const state = () => [title(), field() ? field().value : null, unsavedSaid(), heard.slice(-1)[0] || null];
  const pressSave = async () => {
    button(drawer(), "Save")[0].on.click(); await settle();
    const ok = button(drawer(), "Save without it").concat(button(drawer(), "Save").slice(1)); if (ok.length) { ok.slice(-1)[0].on.click(); await settle(); }
  };
  const opts = () => ({ api: C.api, kind: "setup", experiment: "Bench", id: "low_power.yaml", fields: ["description"], operator: "Ada", onSaved: (b, id, how) => heard.push([b.outcome, id, !!(how && how.unsaved)]) });
  G.open(opts()); await settle();
  // save: sent, edited while it was on its way, answered
  await run("save", async () => { edit("sent"); await pressSave(); edit("later, the save"); await release(); out.save = state(); });
  // copy: asked, edited, answered (the drawer stays on this document)
  await run("copy", async () => { await reopen();
    walk(drawer(), (n) => n.tagName === "INPUT" && n.attrs["aria-label"] === "the copy's name")[0].value = "low_copy";
    button(drawer(), "Save as a copy")[0].on.click(); await settle(); edit("later, the copy"); await release();
    out.copy = state().concat([sent.filter((s) => s[0] === "GET" && s[1].indexOf("low_copy") >= 0).length]); });
  // delete: confirmed, edited, answered (the form becomes a new document at the same id)
  await run("delete", async () => { await reopen();
    button(drawer(), "Delete…")[0].on.click(); await settle();
    const dlg = walk(document.body, (n) => n.tagName === "DIALOG").slice(-1)[0];
    const typed = walk(dlg, (n) => n.tagName === "INPUT")[0]; typed.value = "low_power"; typed.on.input();
    button(dlg, "Delete low_power")[0].on.click(); await settle(); edit("later, the delete"); await release();
    out.delete = state();
    await pressSave();
    const put = sent.filter((s) => s[0] === "PUT").slice(-1)[0];
    out.saveAfterDelete = [put[2].create, put[2].etag, put[2].document.description]; });
  // with no edit since, each write's answer follows it: the copy opens, the deleted form is gone
  await run("copyUnedited", async () => { await reopen();
    walk(drawer(), (n) => n.tagName === "INPUT" && n.attrs["aria-label"] === "the copy's name")[0].value = "low_copy2";
    button(drawer(), "Save as a copy")[0].on.click(); await settle(); await release();
    out.copyUnedited = [title(), heard.slice(-1)[0]]; });
  await run("deleteUnedited", async () => { await reopen();
    button(drawer(), "Delete…")[0].on.click(); await settle();
    const dlg = walk(document.body, (n) => n.tagName === "DIALOG").slice(-1)[0];
    const typed = walk(dlg, (n) => n.tagName === "INPUT")[0]; typed.value = "low_power"; typed.on.input();
    button(dlg, "Delete low_power")[0].on.click(); await settle(); await release();
    out.deleteUnedited = [field() ? "form kept" : "form gone", heard.slice(-1)[0]]; });
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


def test_an_edit_made_while_a_save_was_on_its_way_stays_and_is_unsaved(ran: dict) -> None:  # fmt: skip
    assert ran["save"] == ["setup · low_power.yaml", "later, the save", True, ["committed", "low_power.yaml", True]]  # fmt: skip


def test_an_edit_made_while_a_copy_was_on_its_way_keeps_the_drawer_on_it(ran: dict) -> None:  # fmt: skip
    assert ran["copy"] == ["setup · low_power.yaml", "later, the copy", True, ["committed", "low_copy.yaml", True], 0]  # fmt: skip


def test_an_edit_made_while_a_delete_was_on_its_way_stays_as_a_new_document(ran: dict) -> None:  # fmt: skip
    assert ran["delete"] == ["setup · low_power.yaml (new)", "later, the delete", True, ["committed", "low_power.yaml", True]]  # fmt: skip
    assert ran["saveAfterDelete"] == [
        True,
        "",
        "later, the delete",
    ]  # Save creates it again, with the edit


def test_with_no_edit_since_the_answer_is_followed(ran: dict) -> None:
    """The other half of each comparison: an unedited copy opens the copy, an unedited delete clears the form."""
    assert ran["copyUnedited"] == ["setup · low_copy2.yaml", ["committed", "low_copy2.yaml", False]]  # fmt: skip
    assert ran["deleteUnedited"] == [
        "form gone",
        ["committed", "low_power.yaml", False],
    ]
