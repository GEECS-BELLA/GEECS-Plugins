"""The embedded editor's mount contract: ``embed.js`` and the theme's ``form.js`` under node, against this app.

The configurator's own answers (the kind's list, the schema, the document,
the catalogue) are read from the real app and handed to a stub ``fetch``,
so a reshaped route fails here; the host is a stub DOM. What is pinned is
the contract a host relies on: the drawer draws only ``fields``, the save
sends the whole document (the undrawn fields as stored), the etag it was
served and the host's operator, never saves without one, asks the dry run
first and stops on its refusal, and hands ``preview`` the document.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme import static_dir
from geecs_web_theme.testing import node_available

from geecs_configurator.save import SaveRequest

_EMBED = (
    Path(__file__).resolve().parents[1] / "geecs_configurator" / "static" / "embed.js"
)
API = "http://cfg.test"
Q = "?experiment=Bench"

_HARNESS = r"""
let ids = 0;
function el(tag) {
  const n = { __id: ++ids, tagName: tag.toUpperCase(), children: [], attrs: {}, on: {}, value: "", parentNode: null, _t: "", disabled: false,
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === "value") this.value = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    appendChild(c) { c.parentNode = this; this.children.push(c); if (this.tagName === "SELECT" && c.attrs.selected !== undefined) this.value = c.attrs.value; return c; },
    removeChild(c) { this.children = this.children.filter((x) => x !== c); },
    addEventListener(k, fn) { this.on[k] = fn; },
    get firstChild() { return this.children[0] || null; },
    get textContent() { return this._t + this.children.map((c) => c.textContent).join(""); },
    set textContent(v) { this._t = String(v); this.children = []; } };
  return n;
}
global.document = { createElement: el, createTextNode: (t) => Object.assign(el("#text"), { _t: t }), body: el("body") };
const walk = (n, f, out = []) => { if (f(n)) out.push(n); n.children.forEach((c) => walk(c, f, out)); return out; };
const button = (scope, text) => walk(scope, (n) => n.tagName === "BUTTON" && n.textContent === text)[0];
const settle = () => new Promise((r) => setTimeout(r, 5));
global.GeecsForm = require(process.argv[1]);
const G = require(process.argv[2]);
const C = JSON.parse(require("fs").readFileSync(0, "utf8"));
const seen = [];
let kinds = C.kinds;
global.fetch = (url, init) => {
  const method = (init && init.method) || "GET", body = init && init.body ? JSON.parse(init.body) : undefined;
  seen.push([method, url, body]);
  const key = method + " " + url;
  const [status, out] = key in C.answers ? C.answers[key] : url.endsWith("/api/kinds") ? [200, kinds] : [404, { detail: "unknown " + key }];
  return Promise.resolve({ status, text: () => Promise.resolve(JSON.stringify(out)) });
};
(async () => {
  const out = {}, previews = [], saved = [];
  let who = "";
  const opts = { api: C.api + "/", kind: "setup", experiment: "Bench", id: "low_power", fields: ["description"], operator: () => who,
                 preview: (d) => previews.push(d), onSaved: (b) => saved.push(b.outcome) };
  G.open(opts); await settle();
  const drawer = document.body.children[0];
  out.drawer = [drawer.attrs.class, drawer.attrs["data-open"]];
  out.drawn = walk(drawer, (n) => n.attrs["data-path"] !== undefined).map((n) => n.attrs["data-path"]);
  const desc = walk(drawer, (n) => n.attrs["data-path"] === "description")[0];
  const box = walk(desc, (n) => ["INPUT", "TEXTAREA"].includes(n.tagName))[0];
  box.value = "edited"; box.on.input({ target: box });
  button(drawer, "Save").on.click(); await settle();
  out.noOperator = [seen.filter((s) => s[0] !== "GET").length, drawer.textContent.includes("Type your name")];
  who = "Ada";
  button(drawer, "Save").on.click(); await settle();
  out.sent = seen.filter((s) => s[0] !== "GET");
  out.saved = saved;
  button(drawer, "Preview").on.click();
  out.preview = previews[0];
  // the same document again: the drawer as it was, nothing fetched
  const before = seen.length;
  G.open(opts); await settle();
  out.reopenFetched = seen.length - before;
  // a kind with a dry run that refuses: nothing is PUT
  kinds = kinds.map((k) => k.kind === "setup" ? Object.assign({}, k, { dry_run: "a check" }) : k);
  C.answers["POST " + C.api + "/api/dry-run/setup?experiment=Bench"] = [200, { kind: "setup", by: "scanner", ran: true, refusal: "no such device" }];
  const n = seen.length;
  G.open(Object.assign({}, opts, { fields: null })); await settle();
  button(document.body.children[0], "Save").on.click(); await settle();
  out.refused = [seen.slice(n).filter((s) => s[0] !== "GET").map((s) => s[0]), document.body.children[0].textContent.includes("refused your edit")];
  out.drawers = document.body.children.length;
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _answers(client: TestClient) -> dict:
    """The app's own answers for what the embed reads, keyed as the stub's fetch sees them."""
    out = {}
    for path in ("/api/documents/setup" + Q, "/api/schema/setup", "/api/documents/setup/low_power.yaml" + Q):  # fmt: skip
        r = client.get(path)
        assert r.status_code == 200, path
        out[f"GET {API}{path}"] = [200, r.json()]
    doc = client.get("/api/documents/setup/low_power.yaml" + Q).json()["document"]
    out[f"POST {API}/api/dry-run/setup{Q}"] = [200, client.post("/api/dry-run/setup" + Q, json={"document": doc}).json()]  # fmt: skip
    out[f"PUT {API}/api/documents/setup/low_power.yaml{Q}"] = [200, {"outcome": "committed", "author": "Ada via configurator", "etag": "e2", "tier": "operator", "changes": []}]  # fmt: skip
    return out


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
        etag = client.get(f"/api/documents/setup/low_power.yaml{Q}").json()["etag"]
        stored = client.get(f"/api/documents/setup/low_power.yaml{Q}").json()["document"]  # fmt: skip
    done = subprocess.run(["node", "-e", _HARNESS, str(static_dir() / "form.js"), str(_EMBED)], input=json.dumps(stdin), text=True, capture_output=True, timeout=30)  # fmt: skip
    assert done.returncode == 0, done.stderr
    return {**json.loads(done.stdout), "etag": etag, "stored": stored}


def test_the_drawer_draws_only_the_fields_asked(ran: dict) -> None:
    assert ran["drawer"] == ["drawer", "true"]
    assert ran["drawn"] == ["description"]


def test_no_save_without_the_hosts_operator(ran: dict) -> None:
    assert ran["noOperator"] == [0, True]


def test_the_save_sends_the_whole_document_its_etag_and_the_operator(ran: dict) -> None:
    # the dry run first (this app has no consumer check for a setup: by "none", so straight on), then the PUT
    (dry, _, _), (method, url, body) = ran["sent"]
    assert dry == "POST" and (method, url) == (
        "PUT",
        f"{API}/api/documents/setup/low_power.yaml{Q}",
    )
    SaveRequest.model_validate(body)  # the save route's own model
    assert body["etag"] == ran["etag"] and body["operator"] == "Ada"
    assert body["document"] == {
        **ran["stored"],
        "description": "edited",
    }  # the undrawn steps kept as stored
    assert ran["saved"] == ["committed"]


def test_preview_gets_the_document(ran: dict) -> None:
    assert ran["preview"]["description"] == "edited" and ran["preview"]["steps"] == ran["stored"]["steps"]  # fmt: skip


def test_the_same_document_reopens_as_left_in_one_drawer(ran: dict) -> None:
    assert ran["reopenFetched"] == 0 and ran["drawers"] == 1


def test_a_dry_run_refusal_stops_the_save(ran: dict) -> None:
    assert ran["refused"] == [["POST"], True]
